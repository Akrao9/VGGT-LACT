"""VGGT-TTT — VGGT with tttLRM-style LaCT aggregator and optional virtual-token decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .ttt_aggregator import TTTAggregator


def _aggregated_tokens_for_heads(
    aggregated_tokens_list: list,
    target_dtype: torch.dtype = torch.float32,
) -> list:
    """VGGT heads run with autocast off; LayerNorm requires input.dtype == weight.dtype.
    Casts each layer's tokens to the head's actual weight dtype (default fp32)."""
    out: list = []
    for t in aggregated_tokens_list:
        if t is None:
            out.append(None)
        elif t.dtype != target_dtype:
            out.append(t.to(target_dtype))
        else:
            out.append(t)
    return out


class VGGT_TTT(nn.Module):
    def __init__(
        self,
        vggt_model: nn.Module,
        lact_dim: int = 1024,
        lact_heads: int = 16,
        fast_weight_lr: float = 0.01,
        chunk_size: int = 1,
        lact_ttt_update_cap: int = 256,
        lact_ttt_apply_cap: int = 512,
        share_frame_blocks: bool = True,
        tbptt_window: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.aggregator = TTTAggregator.from_vggt_aggregator(
            vggt_model.aggregator,
            lact_dim=lact_dim, lact_heads=lact_heads,
            fast_weight_lr=fast_weight_lr,
            lact_ttt_update_cap=lact_ttt_update_cap,
            lact_ttt_apply_cap=lact_ttt_apply_cap,
            share_frame_blocks=share_frame_blocks,
            tbptt_window=tbptt_window,
            gradient_checkpoint=gradient_checkpoint,
        )
        self.camera_head = vggt_model.camera_head
        self.depth_head = vggt_model.depth_head
        self.point_head = vggt_model.point_head
        self.track_head = getattr(vggt_model, "track_head", None)
        self.chunk_size = chunk_size
        # Force prediction heads to fp32 regardless of how the teacher was cast.
        # VGGT's heads run with autocast disabled and have multiple internal layers
        # (conv-transpose, layernorm, qr, …) that crash or degrade in bf16. Heads are
        # small (~30M params total) so the fp32 footprint is negligible.
        for head in (self.camera_head, self.depth_head, self.point_head, self.track_head):
            if head is not None:
                head.to(torch.float32)

    @classmethod
    def from_pretrained(cls, vggt_model_id: str = "facebook/VGGT-1B", **kwargs) -> "VGGT_TTT":
        from vggt.models.vggt import VGGT
        vggt = VGGT.from_pretrained(vggt_model_id)
        return cls(vggt, **kwargs)

    def freeze_pretrained(self):
        if self.aggregator.patch_embed is not None:
            for p in self.aggregator.patch_embed.parameters(): p.requires_grad = False
        self.aggregator.camera_token.requires_grad = False
        self.aggregator.register_token.requires_grad = False
        for p in self.aggregator.frame_wise_blocks.parameters(): p.requires_grad = False
        for head in [self.camera_head, self.depth_head, self.point_head, self.track_head]:
            if head is None: continue
            for p in head.parameters(): p.requires_grad = False
        for p in self.aggregator.lact_blocks.parameters(): p.requires_grad = True

    def reset_memory(self):
        self.aggregator.reset_memory()

    def lact_state_dict(self) -> Dict[str, torch.Tensor]:
        """Slim checkpoint: only the LaCT blocks (the only trainable params in stage 1)."""
        return {f"aggregator.lact_blocks.{k}": v for k, v in self.aggregator.lact_blocks.state_dict().items()}

    def load_lact_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True):
        prefix = "aggregator.lact_blocks."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.aggregator.lact_blocks.load_state_dict(sub, strict=strict)

    def _heads_keep_indices(self) -> set:
        keep = {len(self.aggregator.frame_wise_blocks) - 1}
        for head in (self.depth_head, self.point_head):
            if head is not None and hasattr(head, "intermediate_layer_idx"):
                keep.update(head.intermediate_layer_idx)
        return keep

    def forward(
        self,
        images: torch.Tensor,
        chunk_size: Optional[int] = None,
        return_tokens: bool = False,
        strict_heads: bool = True,
        reset_memory: bool = True,
        virtual_tokens: Optional[torch.Tensor] = None,
        skip_heads: Optional[set] = None,
    ) -> Dict[str, Any]:
        """skip_heads: subset of {'camera','depth','point'} to bypass — saves VRAM
        when downstream losses don't need that head's outputs."""
        chunk_size = chunk_size or self.chunk_size
        skip_heads = set(skip_heads) if skip_heads else set()
        keep = self._heads_keep_indices()

        result = self.aggregator(
            images, chunk_size=chunk_size, keep_layer_indices=keep,
            reset_memory=reset_memory, virtual_tokens=virtual_tokens,
        )
        if virtual_tokens is None:
            aggregated_tokens_list, ps_idx = result
            virtual_list = None
        else:
            aggregated_tokens_list, ps_idx, virtual_list = result

        outputs: Dict[str, Any] = {}

        def handle(name: str, exc: Exception):
            if isinstance(exc, torch.OutOfMemoryError): raise
            if strict_heads: raise RuntimeError(f"{name} head failed") from exc
            print(f"{name} head skipped: {exc}")

        # match the head's actual weight dtype so LayerNorm/Conv don't error.
        head_dtype = next(self.camera_head.parameters()).dtype
        tokens_fp32 = _aggregated_tokens_for_heads(aggregated_tokens_list, target_dtype=head_dtype)
        images_for_heads = images if images.dtype == head_dtype else images.to(head_dtype)
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if "camera" not in skip_heads:
                try:
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    pose_enc = self.camera_head(tokens_fp32)[-1]
                    # pose decode involves quaternion math + small matrix inverses; force fp32
                    # so bf16 heads don't degrade extrinsic/intrinsic numerics.
                    extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), images.shape[-2:])
                    outputs.update(extrinsic=extr, intrinsic=intr, pose_enc=pose_enc)
                except Exception as e: handle("Camera", e)

            if "depth" not in skip_heads:
                try:
                    d, dc = self.depth_head(tokens_fp32, images=images_for_heads, patch_start_idx=ps_idx)
                    outputs.update(depth=d, depth_conf=dc)
                except Exception as e: handle("Depth", e)

            if "point" not in skip_heads:
                try:
                    wp, wpc = self.point_head(tokens_fp32, images=images_for_heads, patch_start_idx=ps_idx)
                    outputs.update(world_points=wp, world_points_conf=wpc)
                except Exception as e: handle("Point", e)

        if return_tokens:
            outputs["aggregated_tokens_list"] = aggregated_tokens_list
            outputs["patch_start_idx"] = ps_idx
        if virtual_list is not None:
            outputs["virtual_tokens_list"] = virtual_list
            # the final-layer virtual tokens are what the tttLRM decoder reads
            outputs["virtual_tokens_final"] = virtual_list[max(keep)]
        return outputs
