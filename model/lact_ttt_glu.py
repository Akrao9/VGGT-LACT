# SPDX-License-Identifier: Apache-2.0
"""tttLRM-style fast-weight GLU TTT (FastWeightGluMLPMultihead) with optional 2D RoPE on q,k."""

from __future__ import annotations

import collections
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TTTOperator = collections.namedtuple(
    "TTTOperator", ["start", "end", "fast_weight", "update", "apply"]
)


def ar_ttt_op(
    update_minibatch: int = 1024,
    length: int = 10240,
    update_length: Optional[int] = None,
    apply_chunk: Optional[int] = None,
):
    if update_length is None:
        update_length = length
    assert update_length % update_minibatch == 0, (
        f"update_length={update_length} not divisible by update_minibatch={update_minibatch}"
    )
    config: list[TTTOperator] = []
    for start in range(0, update_length, update_minibatch):
        config.append(TTTOperator(start, start + update_minibatch, True, True, False))
    if apply_chunk is None or apply_chunk <= 0 or apply_chunk >= length:
        config.append(TTTOperator(0, length, False, False, True))
    else:
        a = 0
        while a < length:
            b = min(a + apply_chunk, length)
            config.append(TTTOperator(a, b, False, False, True))
            a = b
    return config


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.special.expm1(-x))


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Muon NS-5 runs in bf16 for speed; convert if needed, leave alone if already bf16.
    X = G if G.dtype == torch.bfloat16 else G.bfloat16()
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(max(steps, 0)):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X.to(dtype=G.dtype)


def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    ttt_config, muon_update_steps: int = 0,
    elastic_lambda: float = 0.0, fisher_alpha: float = 0.1, anchor_beta: float = 0.99,
):
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    use_elastic = elastic_lambda > 0.0
    if use_elastic:
        w0_anchor, w1_anchor, w2_anchor = w0.clone(), w1.clone(), w2.clone()
        F0 = torch.zeros_like(w0); F1 = torch.zeros_like(w1); F2 = torch.zeros_like(w2)

    output: list[torch.Tensor] = []
    for start, end, fast_weight, update, apply in ttt_config:
        w0_now, w1_now, w2_now = w0, w1, w2

        if fast_weight:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i, lr1i, lr2i = lr0[:, start:end, :], lr1[:, start:end, :], lr2[:, start:end, :]

            gate_pre = ki @ w0_now
            hidden_pre = ki @ w2_now
            gate = F.silu(gate_pre, inplace=False)
            hidden = gate * hidden_pre

            dhidden = vi @ w1_now.transpose(-1, -2)
            dh_pre = dhidden * gate
            dgate = dhidden * hidden_pre
            dgate_pre = silu_backprop(dgate, gate_pre)

            w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi
            w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_pre
            w2_grad = (ki * lr2i).transpose(-1, -2) @ dh_pre

            w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            if use_elastic:
                with torch.no_grad():
                    F0 = fisher_alpha * F0 + (1.0 - fisher_alpha) * w0_grad.detach().square()
                    F1 = fisher_alpha * F1 + (1.0 - fisher_alpha) * w1_grad.detach().square()
                    F2 = fisher_alpha * F2 + (1.0 - fisher_alpha) * w2_grad.detach().square()
                    inv_imp0 = 1.0 - F0 / (F0.max() + 1e-8)
                    inv_imp1 = 1.0 - F1 / (F1.max() + 1e-8)
                    inv_imp2 = 1.0 - F2 / (F2.max() + 1e-8)
                w0_now = w0_now - elastic_lambda * inv_imp0 * (w0_now - w0_anchor)
                w1_now = w1_now - elastic_lambda * inv_imp1 * (w1_now - w1_anchor)
                w2_now = w2_now - elastic_lambda * inv_imp2 * (w2_now - w2_anchor)

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            if update:
                w0, w1, w2 = w0_now, w1_now, w2_now
                if use_elastic:
                    w0_anchor = anchor_beta * w0_anchor + (1.0 - anchor_beta) * w0.detach()
                    w1_anchor = anchor_beta * w1_anchor + (1.0 - anchor_beta) * w1.detach()
                    w2_anchor = anchor_beta * w2_anchor + (1.0 - anchor_beta) * w2.detach()

        if apply:
            qi = q[:, start:end, :]
            # No inplace silu: the input (qi @ w0_now) is needed for backward.
            oi = (F.silu(qi @ w0_now) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    return torch.cat(output, dim=1), w0, w1, w2


class FastWeightGluMLPMultihead(nn.Module):
    """tttLRM FastWeightGluMLPMultihead with optional 2D RoPE on q,k.

    Pass ``rope`` (a VGGT ``RotaryPositionEmbedding2D``) and per-call ``positions``
    of shape ``(B, L, 2)`` via ``shape_info["positions"]`` to enable RoPE.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 4,
        bias: bool = False,
        use_o_norm: bool = True,
        base_lr: float = 0.01,
        muon_update_steps: int = 0,
        elastic_lambda: float = 0.0,
        fisher_alpha: float = 0.5,
        anchor_beta: float = 0.8,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim, self.head_dim = dim, head_dim
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.elastic_lambda = elastic_lambda
        self.fisher_alpha = fisher_alpha
        self.anchor_beta = anchor_beta
        self.rope = rope

        d_h = int(head_dim * inter_multi)
        gain = math.sqrt(2)
        self.w0 = nn.Parameter(torch.randn(self.num_heads, head_dim, d_h) * gain / math.sqrt(head_dim))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_h, head_dim) * gain / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(self.num_heads, head_dim, d_h) * gain / math.sqrt(head_dim))

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.lr_fc = nn.Linear(dim, self.num_heads * 3)
        self.base_lr_inv = nn.Parameter(inv_softplus(torch.tensor(float(base_lr), dtype=torch.float32)))
        self.use_o_norm = use_o_norm
        self.o_norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, vis_dict=None, shape_info=None):
        if shape_info is None:
            raise ValueError("shape_info is required")
        bsz, seq_len, _ = x.shape
        h, d = self.num_heads, self.head_dim

        qkv = F.silu(self.to_qkv(x), inplace=False)
        t = qkv.view(bsz, seq_len, 3, h, d).permute(2, 0, 3, 1, 4)  # (3,B,H,L,d)
        q, k, v = t[0], t[1], t[2]  # (B,H,L,d) each

        positions = shape_info.get("positions")
        if self.rope is not None and positions is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        q = q.reshape(bsz * h, seq_len, d)
        k = k.reshape(bsz * h, seq_len, d)
        v = v.reshape(bsz * h, seq_len, d)
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            lr = self.lr_fc(x.to(self.lr_fc.weight.dtype))
        lr = F.softplus(lr.float() + self.base_lr_inv)
        lr = lr.view(bsz, seq_len, 3, h, 1).permute(2, 0, 3, 1, 4)
        lr0 = lr[0].reshape(bsz * h, seq_len, 1)
        lr1 = lr[1].reshape(bsz * h, seq_len, 1)
        lr2 = lr[2].reshape(bsz * h, seq_len, 1)

        if "w0" in shape_info:
            w0, w1, w2 = shape_info["w0"], shape_info["w1"], shape_info["w2"]
        else:
            w0 = self.w0.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * h, d, self.w0.shape[-1])
            w1 = self.w1.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * h, self.w1.shape[1], d)
            w2 = self.w2.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * h, d, self.w2.shape[-1])

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2,
            shape_info["ttt_config"],
            muon_update_steps=self.muon_update_steps,
            elastic_lambda=self.elastic_lambda,
            fisher_alpha=self.fisher_alpha,
            anchor_beta=self.anchor_beta,
        )

        output = output.view(bsz, h, seq_len, d).permute(0, 2, 1, 3).reshape(bsz, seq_len, h * d)
        if self.use_o_norm:
            output = self.o_norm(output)
        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}
