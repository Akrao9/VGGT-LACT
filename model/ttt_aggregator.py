"""TTTAggregator: VGGT aggregator with global attention replaced by LaCT.

Adds:
  * 2D RoPE on the LaCT (cross-frame) path, matching VGGT global attention.
  * Optional ``virtual_tokens`` (Plücker-encoded target rays for tttLRM-style
    novel-view rendering). They are concatenated to source tokens, routed through
    BOTH frame-wise and LaCT layers, and split back at the output.
  * Padding-aware LaCT mini-batches (in ``LaCTBlock``); no divisor surprises.
  * Shared ``frame_wise_blocks`` (no copy) when built from a teacher VGGT.

Output format matches VGGT exactly: ``(aggregated_tokens_list, patch_start_idx)``
where each entry is ``(B, N, P_total, 2C)``. When ``virtual_tokens`` is given,
a parallel ``virtual_tokens_list`` of shape ``(B, M, 2C)`` is also returned.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    from vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
except ImportError:
    PositionGetter = None
    RotaryPositionEmbedding2D = None

from .lact_block import LaCTBlock


def _slice_expand_and_flatten(token_tensor, B, S, include_first=True):
    if include_first and S >= 1:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        if S > 1:
            others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
            combined = torch.cat([query, others], dim=1)
        else:
            combined = query
    else:
        combined = token_tensor[:, 1:, ...].expand(B, S, *token_tensor.shape[2:])
    return combined.reshape(B * S, *combined.shape[2:])


class TTTAggregator(nn.Module):
    def __init__(
        self,
        frame_wise_blocks: nn.ModuleList,
        lact_blocks: nn.ModuleList,
        patch_embed: nn.Module,
        camera_token: nn.Parameter,
        register_token: nn.Parameter,
        norm: Optional[nn.Module] = None,
        rope: Optional[nn.Module] = None,
        position_getter=None,
        patch_start_idx: int = 5,
        patch_size: int = 14,
        resnet_mean: Optional[torch.Tensor] = None,
        resnet_std: Optional[torch.Tensor] = None,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.frame_wise_blocks = frame_wise_blocks
        self.lact_blocks = lact_blocks
        self.norm = norm
        self.rope = rope
        if position_getter is None and PositionGetter is not None:
            position_getter = PositionGetter()
        self.position_getter = position_getter
        self.patch_start_idx = patch_start_idx
        self.patch_size = patch_size

        self.camera_token = camera_token
        self.register_token = register_token

        if resnet_mean is None:
            resnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        if resnet_std is None:
            resnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("_resnet_mean", resnet_mean, persistent=False)
        self.register_buffer("_resnet_std", resnet_std, persistent=False)

        assert len(frame_wise_blocks) == len(lact_blocks)
        self.gradient_checkpoint = gradient_checkpoint
        self.use_reentrant = False

    @classmethod
    def from_vggt_aggregator(
        cls,
        vggt_aggregator: nn.Module,
        lact_dim: int = 1024,
        lact_heads: int = 16,
        fast_weight_lr: float = 0.01,
        lact_ttt_update_cap: int = 256,
        lact_ttt_apply_cap: int = 512,
        share_frame_blocks: bool = True,
        tbptt_window: int = 0,
        gradient_checkpoint: bool = False,
    ) -> "TTTAggregator":
        if hasattr(vggt_aggregator, "frame_blocks"):
            src = list(vggt_aggregator.frame_blocks)
        elif hasattr(vggt_aggregator, "blocks"):
            src = list(vggt_aggregator.blocks)[0::2]
        else:
            raise AttributeError("VGGT aggregator missing frame_blocks")

        if share_frame_blocks:
            frame_wise_blocks = nn.ModuleList(src)
        else:
            import copy
            frame_wise_blocks = nn.ModuleList([copy.deepcopy(b) for b in src])

        rope = getattr(vggt_aggregator, "rope", None)
        position_getter = getattr(vggt_aggregator, "position_getter", None)

        lact_blocks = nn.ModuleList([
            LaCTBlock(
                dim=lact_dim, num_heads=lact_heads, fast_weight_lr=fast_weight_lr,
                ttt_update_cap=lact_ttt_update_cap, ttt_apply_cap=lact_ttt_apply_cap,
                rope=rope, tbptt_window=tbptt_window,
                gradient_checkpoint=gradient_checkpoint,
            )
            for _ in src
        ])

        return cls(
            frame_wise_blocks=frame_wise_blocks,
            lact_blocks=lact_blocks,
            patch_embed=getattr(vggt_aggregator, "patch_embed", None),
            camera_token=vggt_aggregator.camera_token,
            register_token=vggt_aggregator.register_token,
            norm=getattr(vggt_aggregator, "norm", None),
            rope=rope,
            position_getter=position_getter,
            patch_start_idx=getattr(vggt_aggregator, "patch_start_idx", 5),
            patch_size=getattr(vggt_aggregator, "patch_size", 14),
            resnet_mean=getattr(vggt_aggregator, "_resnet_mean", None),
            resnet_std=getattr(vggt_aggregator, "_resnet_std", None),
            gradient_checkpoint=gradient_checkpoint,
        )

    def reset_memory(self):
        for blk in self.lact_blocks:
            blk.reset_memory()

    def _build_positions(self, BNs: int, h: int, w: int, device, n_special: int, n_extra: int = 0):
        """Returns positions of shape (BNs, n_special + h*w + n_extra, 2). Special / extra at index 0."""
        if self.position_getter is None:
            return None
        pos = self.position_getter(BNs, h, w, device=device)  # (BNs, h*w, 2)
        if n_special > 0:
            pos = pos + 1
            pos = torch.cat([torch.zeros(BNs, n_special, 2, device=device, dtype=pos.dtype), pos], dim=1)
        if n_extra > 0:
            pos = torch.cat([pos, torch.zeros(BNs, n_extra, 2, device=device, dtype=pos.dtype)], dim=1)
        return pos

    def forward(
        self,
        images: torch.Tensor,
        chunk_size: Optional[int] = None,
        keep_layer_indices: Optional[set] = None,
        reset_memory: bool = True,
        virtual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int] | Tuple[List[torch.Tensor], int, List[torch.Tensor]]:
        """
        images: (B, N, C, H, W).
        virtual_tokens: optional (B, M, D) — appended to source tokens, routed
            through both frame-wise (broadcast per chunk) and LaCT layers, and
            split back at the output. Used by tttLRM-style decoder.
        """
        B, N, C, H, W = images.shape
        chunk_size = chunk_size or 1
        if reset_memory:
            self.reset_memory()

        h_p, w_p = H // self.patch_size, W // self.patch_size
        num_layers = len(self.frame_wise_blocks)
        if keep_layer_indices is None:
            keep_layer_indices = set(range(num_layers))
        else:
            keep_layer_indices = {idx % num_layers for idx in keep_layer_indices}

        images_n = (images - self._resnet_mean) / self._resnet_std

        all_frame = {idx: [] for idx in keep_layer_indices}
        all_lact = {idx: [] for idx in keep_layer_indices}
        track_virtual = virtual_tokens is not None
        all_virtual_frame = {idx: [] for idx in keep_layer_indices} if track_virtual else None
        all_virtual_lact = {idx: [] for idx in keep_layer_indices} if track_virtual else None
        M = 0 if virtual_tokens is None else virtual_tokens.shape[1]
        D_emb = None

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sN = end - start
            chunk = images_n[:, start:end].reshape(B * sN, C, H, W)
            patch_tokens = self.patch_embed(chunk)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            _, P, D = patch_tokens.shape
            D_emb = D

            is_first = (start == 0)
            cam_tok = _slice_expand_and_flatten(self.camera_token, B, sN, include_first=is_first)
            reg_tok = _slice_expand_and_flatten(self.register_token, B, sN, include_first=is_first)
            tokens = torch.cat([cam_tok, reg_tok, patch_tokens], dim=1)  # (B*sN, P_total, D)
            P_total = tokens.shape[1]

            pos_frame = self._build_positions(B * sN, h_p, w_p, tokens.device, n_special=self.patch_start_idx)
            # LaCT positions: same per-frame layout repeated; virtual tokens get 0 pos
            pos_lact = None
            if pos_frame is not None:
                pos_lact = pos_frame.view(B, sN, P_total, 2).reshape(B, sN * P_total, 2)
                if M > 0:
                    pos_lact = torch.cat(
                        [pos_lact, torch.zeros(B, M, 2, device=pos_lact.device, dtype=pos_lact.dtype)],
                        dim=1,
                    )

            for layer_idx, (fw, lact) in enumerate(zip(self.frame_wise_blocks, self.lact_blocks)):
                # frame-wise: per-frame attention. Virtual tokens are broadcast across frames here.
                if M > 0:
                    vt_per_frame = virtual_tokens.unsqueeze(1).expand(B, sN, M, D).reshape(B * sN, M, D)
                    fw_in = torch.cat([tokens, vt_per_frame], dim=1)
                    pos_fw = None
                    if pos_frame is not None:
                        pos_fw = torch.cat(
                            [pos_frame, torch.zeros(B * sN, M, 2, device=pos_frame.device, dtype=pos_frame.dtype)],
                            dim=1,
                        )
                else:
                    fw_in = tokens
                    pos_fw = pos_frame

                if self.training and self.gradient_checkpoint and fw_in.requires_grad:
                    fw_out = checkpoint(fw, fw_in, pos_fw, use_reentrant=self.use_reentrant)
                else:
                    fw_out = fw(fw_in, pos=pos_fw)

                if M > 0:
                    tokens = fw_out[:, :P_total, :]
                    # average virtual updates across frames so each virtual token has one state
                    vt_updated = fw_out[:, P_total:, :].view(B, sN, M, D).mean(dim=1)
                else:
                    tokens = fw_out

                if layer_idx in keep_layer_indices:
                    all_frame[layer_idx].append(tokens.reshape(B, sN, P_total, D))

                # LaCT path: cross-frame (B, sN*P_total + M, D)
                lact_in = tokens.reshape(B, sN * P_total, D)
                if M > 0:
                    lact_in = torch.cat([lact_in, vt_updated], dim=1)
                lact_out = lact(lact_in, positions=pos_lact, update_memory=True)
                if M > 0:
                    tokens = lact_out[:, : sN * P_total, :].reshape(B * sN, P_total, D)
                    virtual_tokens = lact_out[:, sN * P_total :, :]
                else:
                    tokens = lact_out.reshape(B * sN, P_total, D)

                if layer_idx in keep_layer_indices:
                    if M > 0:
                        all_lact[layer_idx].append(tokens.reshape(B, sN, P_total, D))
                        all_virtual_frame[layer_idx].append(vt_updated)
                        all_virtual_lact[layer_idx].append(virtual_tokens)
                    else:
                        all_lact[layer_idx].append(lact_out.reshape(B, sN, P_total, D))

        output_list: List[Optional[torch.Tensor]] = [None] * num_layers
        virtual_list: List[Optional[torch.Tensor]] = [None] * num_layers if track_virtual else None
        for idx in sorted(keep_layer_indices):
            frame_cat = torch.cat(all_frame[idx], dim=1)
            lact_cat = torch.cat(all_lact[idx], dim=1)
            output_list[idx] = torch.cat([frame_cat, lact_cat], dim=-1)
            if virtual_list is not None:
                # take the last chunk's frame-virtual and lact-virtual outputs (no frame dim).
                vf = all_virtual_frame[idx][-1]
                vl = all_virtual_lact[idx][-1]
                virtual_list[idx] = torch.cat([vf, vl], dim=-1)

        if virtual_list is not None:
            return output_list, self.patch_start_idx, virtual_list
        return output_list, self.patch_start_idx
