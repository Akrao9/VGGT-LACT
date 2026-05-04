"""tttLRM-style decoder for VGGT-TTT.

Virtual query tokens (Plücker-encoded target rays) are seeded from camera
intrinsics+extrinsics, fed through ``TTTAggregator`` alongside source tokens,
and then projected to per-token Gaussian parameters by a thin MLP head.

This matches tttLRM's design: the trunk does the heavy mixing, the decoder is
a small head. Anchored to virtual rays, not source pixels — supports arbitrary
novel-view rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GaussianBundle:
    xyz: torch.Tensor       # (B, M, 3)
    scale: torch.Tensor     # (B, M, 3)
    rotation: torch.Tensor  # (B, M, 4) quaternion (xyzw)
    opacity: torch.Tensor   # (B, M, 1)
    color: torch.Tensor     # (B, M, 3)


def compute_plucker_rays(c2w: torch.Tensor, K: torch.Tensor, H: int, W: int, sample_grid: int) -> torch.Tensor:
    """
    c2w: (B, V, 4, 4)   K: (B, V, 3, 3)
    Returns Plücker rays of shape (B, V * sample_grid * sample_grid, 6) = [origin, direction].
    Samples a uniform sample_grid x sample_grid grid per view.
    """
    B, V = c2w.shape[:2]
    device, dtype = c2w.device, c2w.dtype
    ys = torch.linspace(0.5, H - 0.5, sample_grid, device=device, dtype=dtype)
    xs = torch.linspace(0.5, W - 0.5, sample_grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)  # (P, 3)

    # Solve K @ rays_cam = pix per (B,V); falls back to pinv on singular intrinsics.
    try:
        rays_cam = torch.linalg.solve(K, pix.t().unsqueeze(0).unsqueeze(0).expand(B, V, 3, -1))
        rays_cam = rays_cam.transpose(-1, -2)  # (B, V, P, 3)
    except RuntimeError:
        K_inv = torch.linalg.pinv(K)
        rays_cam = torch.einsum("bvij,pj->bvpi", K_inv, pix)
    rays_cam = F.normalize(rays_cam, dim=-1)

    R = c2w[..., :3, :3]  # (B, V, 3, 3)
    t = c2w[..., :3, 3]   # (B, V, 3)
    rays_world = torch.einsum("bvij,bvpj->bvpi", R, rays_cam)  # (B, V, P, 3)
    origins = t.unsqueeze(2).expand(-1, -1, rays_cam.shape[2], -1)
    plucker = torch.cat([origins, rays_world], dim=-1)  # (B, V, P, 6)
    return plucker.reshape(B, V * rays_cam.shape[2], 6)


class TTTLRMDecoder(nn.Module):
    """
    Seeds virtual tokens from Plücker rays + thin MLP head over the trunk's
    virtual-token output to predict 14-dim Gaussian params per token.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 256,
        plucker_dim: int = 6,
        sh_degree: int = 0,
    ):
        super().__init__()
        if sh_degree != 0:
            raise NotImplementedError(
                "sh_degree>0 not wired to render_views (which calls gsplat with sh_degree=0). "
                "Either keep sh_degree=0 or update both this head and scripts/train_decoder.render_views."
            )
        out_color = 3
        self.sh_degree = sh_degree
        self.token_dim = token_dim

        self.ray_embed = nn.Sequential(
            nn.Linear(plucker_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # head reads concatenated frame+lact (2C) virtual token outputs
        in_dim = 2 * token_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1 + 3 + 4 + 1 + out_color),
        )

    def init_virtual_tokens(self, plucker: torch.Tensor) -> torch.Tensor:
        """plucker: (B, M, 6) -> (B, M, token_dim)"""
        return self.ray_embed(plucker)

    def decode(self, virtual_out: torch.Tensor, plucker: torch.Tensor) -> GaussianBundle:
        """
        virtual_out: (B, M, 2C) — final-layer virtual tokens from TTTAggregator.
        plucker:     (B, M, 6) — origins + directions of the virtual rays.

        Predicts a depth-along-ray (softplus) so xyz = origin + depth * dir, plus
        scale/rotation/opacity/color.
        """
        params = self.head(virtual_out)  # (B, M, 11+color)
        depth_logit, scale, rot, opacity, color = torch.split(
            params, [1, 3, 4, 1, params.shape[-1] - 9], dim=-1,
        )
        origins, dirs = plucker[..., :3], plucker[..., 3:]
        depth = F.softplus(depth_logit)
        xyz = origins + depth * dirs
        scale = F.softplus(scale - 2.0)  # init small
        rot = F.normalize(rot, dim=-1)
        opacity = torch.sigmoid(opacity)
        color = torch.sigmoid(color)
        return GaussianBundle(xyz=xyz, scale=scale, rotation=rot, opacity=opacity, color=color)
