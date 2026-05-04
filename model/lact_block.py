"""LaCT block: LayerNorm + FastWeightGluMLPMultihead + residual.

Replaces VGGT global attention. Cross-chunk state lives in detached fast
weights ``w0,w1,w2``. Optional 2D RoPE on q,k (matches VGGT global attn).
Optional TBPTT window keeps gradients flowing across the last K chunks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lact_ttt_glu import TTTOperator, ar_ttt_op, FastWeightGluMLPMultihead


class LaCTBlock(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        fast_weight_lr: float = 0.01,
        ff_expand: int = 4,  # unused; kept for API
        inter_multi: int = 4,
        muon_update_steps: int = 0,
        ttt_update_cap: int = 256,
        ttt_apply_cap: int = 512,
        rope: Optional[nn.Module] = None,
        tbptt_window: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.glu = FastWeightGluMLPMultihead(
            dim=dim, head_dim=head_dim, inter_multi=inter_multi, bias=False,
            use_o_norm=True, base_lr=fast_weight_lr,
            muon_update_steps=muon_update_steps, elastic_lambda=0.0,
            rope=rope,
        )
        # near-identity start so frozen heads stay stable before distillation
        nn.init.zeros_(self.glu.c_proj.weight)
        if self.glu.c_proj.bias is not None:
            nn.init.zeros_(self.glu.c_proj.bias)

        self.ttt_update_cap = int(ttt_update_cap)
        self.ttt_apply_cap = int(ttt_apply_cap)
        self.tbptt_window = int(tbptt_window)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        # runtime state (per scene)
        self._w0: Optional[torch.Tensor] = None
        self._w1: Optional[torch.Tensor] = None
        self._w2: Optional[torch.Tensor] = None
        self._chunks_since_detach: int = 0

    def reset_memory(self):
        self._w0 = self._w1 = self._w2 = None
        self._chunks_since_detach = 0

    def _init_runtime_weights(self, batch: int, device, dtype):
        nh = self.glu.num_heads; d = self.glu.head_dim
        w0 = self.glu.w0.detach().to(device=device, dtype=dtype)
        w1 = self.glu.w1.detach().to(device=device, dtype=dtype)
        w2 = self.glu.w2.detach().to(device=device, dtype=dtype)
        self._w0 = w0.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * nh, d, w0.shape[-1]).contiguous()
        self._w1 = w1.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * nh, w1.shape[1], d).contiguous()
        self._w2 = w2.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * nh, d, w2.shape[-1]).contiguous()
        self._chunks_since_detach = 0

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """x: (B, N, D). positions: (B, N, 2) for RoPE (optional)."""
        bsz, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")
        residual = x
        x = self.norm(x)

        nh = self.glu.num_heads
        if self._w0 is None or self._w0.shape[0] != bsz * nh:
            self._init_runtime_weights(bsz, x.device, x.dtype)

        # pad seq_len up to a multiple of update_minibatch (no divisor search)
        umb = min(self.ttt_update_cap, seq_len) if update_memory else seq_len
        umb = max(umb, 1)
        pad = (-seq_len) % umb if update_memory else 0
        if pad:
            x_pad = F.pad(x, (0, 0, 0, pad))
            if positions is not None:
                positions = F.pad(positions, (0, 0, 0, pad))
        else:
            x_pad = x
        L = seq_len + pad

        if update_memory:
            apply_chunk = self.ttt_apply_cap if L > self.ttt_apply_cap else None
            ttt_cfg = ar_ttt_op(update_minibatch=umb, length=L, update_length=L, apply_chunk=apply_chunk)
        else:
            ttt_cfg = [TTTOperator(0, L, False, False, True)]

        shape_info = {
            "ttt_config": ttt_cfg,
            "w0": self._w0, "w1": self._w1, "w2": self._w2,
            "positions": positions,
        }
        # gradient checkpoint at the GLU boundary: state mutation happens *after* and
        # is unaffected; the recompute observes the same shape_info captured here.
        if self.gradient_checkpoint and self.training and x_pad.requires_grad:
            from torch.utils.checkpoint import checkpoint as _ckpt
            def _run(xp, si):
                return self.glu(xp, shape_info=si)
            out, tail = _ckpt(_run, x_pad, shape_info, use_reentrant=False)
        else:
            out, tail = self.glu(x_pad, shape_info=shape_info)

        if update_memory:
            new_w0, new_w1, new_w2 = tail["w0"], tail["w1"], tail["w2"]
            if self.tbptt_window > 0 and self._chunks_since_detach + 1 < self.tbptt_window:
                self._w0, self._w1, self._w2 = new_w0, new_w1, new_w2
                self._chunks_since_detach += 1
            else:
                self._w0 = new_w0.detach()
                self._w1 = new_w1.detach()
                self._w2 = new_w2.detach()
                self._chunks_since_detach = 0

        if pad:
            out = out[:, :seq_len, :]
        return residual + out
