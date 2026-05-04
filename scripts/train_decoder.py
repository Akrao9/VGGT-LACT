"""train_decoder.py — train the tttLRM-style virtual-token Gaussian decoder on top of frozen VGGT-TTT.

Workflow per scene:
  1. sample S source views and T target views (with c2w + K)
  2. build Plücker rays for the targets, pass through ``TTTLRMDecoder.init_virtual_tokens``
  3. run VGGT-TTT with ``virtual_tokens=...`` so the trunk mixes them with source tokens
  4. decode the final virtual tokens into Gaussians and rasterize them at target views
  5. photometric loss against target images

Assumes a NeRF-Studio-style ``transforms.json`` (``frames[*].file_path``,
``transform_matrix``, fl_x, fl_y, cx, cy) per scene.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.io_utils import torch_load_checkpoint
from model.tttlrm_decoder import TTTLRMDecoder, compute_plucker_rays
from model.vggt_ttt import VGGT_TTT
from pipeline.input_pipeline import load_images
from scripts.script_common import (
    autocast_context, configure_cuda_backend, fused_adamw,
    maybe_compile, resolve_amp_dtype, set_inductor_cache, unwrap_compiled,
)


# ─────────────────────── data ───────────────────────

def load_transforms(scene_dir: Path) -> dict:
    for cand in ("transforms.json", "transforms_train.json"):
        p = scene_dir / cand
        if p.exists():
            return json.loads(p.read_text())
        for sub in scene_dir.iterdir():
            if sub.is_dir() and (sub / cand).exists():
                return json.loads((sub / cand).read_text())
    raise FileNotFoundError(f"transforms.json not found under {scene_dir}")


def parse_intrinsics(meta: dict) -> torch.Tensor:
    K = torch.eye(3)
    K[0, 0] = meta.get("fl_x", meta.get("focal", 500.0))
    K[1, 1] = meta.get("fl_y", meta.get("focal", 500.0))
    K[0, 2] = meta.get("cx", 0.5 * meta.get("w", 1.0))
    K[1, 2] = meta.get("cy", 0.5 * meta.get("h", 1.0))
    return K


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, n_source: int = 16, n_target: int = 4, target_size: int = 518):
        self.root = root
        self.n_source = n_source
        self.n_target = n_target
        self.target_size = target_size
        self.scenes: list[Path] = []
        for p in sorted(root.iterdir()) if root.is_dir() else []:
            if p.is_dir() and any(p.rglob("transforms*.json")):
                self.scenes.append(p)
        if not self.scenes:
            raise RuntimeError(f"no scenes with transforms under {root}")

    def __len__(self): return len(self.scenes)

    def __getitem__(self, idx) -> dict[str, Any]:
        scene = self.scenes[idx]
        meta = load_transforms(scene)
        K = parse_intrinsics(meta)
        frames = meta["frames"]
        random.shuffle(frames)
        pick = frames[: self.n_source + self.n_target]
        if len(pick) < self.n_source + self.n_target:
            raise RuntimeError(f"scene {scene} has only {len(pick)} frames")
        src, tgt = pick[: self.n_source], pick[self.n_source :]

        def gather(items):
            imgs = load_images([str(scene / f["file_path"]) for f in items], self.target_size)
            c2w = torch.stack([torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in items])
            return imgs, c2w

        src_img, src_c2w = gather(src)
        tgt_img, tgt_c2w = gather(tgt)
        H, W = src_img.shape[-2:]
        Kbatch = K.unsqueeze(0).expand(len(src) + len(tgt), -1, -1).clone()
        # rescale K to match target_size crop
        sx = W / meta.get("w", W); sy = H / meta.get("h", H)
        Kbatch[:, 0] *= sx; Kbatch[:, 1] *= sy
        return dict(
            src_img=src_img, src_c2w=src_c2w, src_K=Kbatch[: len(src)],
            tgt_img=tgt_img, tgt_c2w=tgt_c2w, tgt_K=Kbatch[len(src) :],
            H=H, W=W,
        )


def collate(batch: list[dict]) -> dict[str, Any]:
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = batch[0][k]
    return out


# ─────────────────────── render ───────────────────────

def render_views(gaussians, c2w: torch.Tensor, K: torch.Tensor, H: int, W: int):
    """Per-view gsplat rasterization. Inputs in fp32 for stability."""
    from gsplat import rasterization
    B, V = c2w.shape[:2]
    out = torch.zeros(B, V, H, W, 3, device=c2w.device)
    for b in range(B):
        xyz = gaussians.xyz[b].float()
        rot = gaussians.rotation[b].float()
        scl = gaussians.scale[b].float()
        opa = gaussians.opacity[b].squeeze(-1).float()
        col = gaussians.color[b].float()
        for v in range(V):
            w2c = torch.linalg.inv(c2w[b, v].float()).unsqueeze(0)
            Kv = K[b, v].float().unsqueeze(0)
            img, _, _ = rasterization(
                xyz, rot, scl, opa, col, w2c, Kv, W, H,
                sh_degree=0, render_mode="RGB",
                backgrounds=torch.ones(1, 3, device=c2w.device),
                rasterize_mode="classic",
            )
            out[b, v] = img.squeeze(0)
    return out


# ─────────────────────── train ───────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--lact_ckpt", default="", help="Slim LaCT checkpoint from finetune.py.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n_source", type=int, default=16)
    p.add_argument("--n_target", type=int, default=4)
    p.add_argument("--virtual_grid", type=int, default=24,
                   help="Sample virtual_grid x virtual_grid rays per target view. M=V*g*g virtual tokens.")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    set_inductor_cache(); configure_cuda_backend()
    dtype = resolve_amp_dtype(args.device, args.precision)

    print("loading frozen VGGT-TTT...")
    student = VGGT_TTT.from_pretrained(share_frame_blocks=True).to(args.device)
    if args.lact_ckpt and os.path.exists(args.lact_ckpt):
        state = torch_load_checkpoint(args.lact_ckpt, map_location=args.device)
        student.load_lact_state_dict(state, strict=True)
        print(f"loaded LaCT weights from {args.lact_ckpt}")
    student.freeze_pretrained()
    for p_ in student.aggregator.lact_blocks.parameters():
        p_.requires_grad = False
    student.eval()
    if dtype is not None:
        student.to(dtype)

    decoder = TTTLRMDecoder(token_dim=student.aggregator.lact_blocks[0].dim).to(args.device)
    if dtype is not None:
        decoder.to(dtype)

    opt = fused_adamw(decoder.parameters(), lr=args.lr)
    student = maybe_compile(student, args.compile)
    decoder = maybe_compile(decoder, args.compile)

    ds = SceneDataset(Path(args.data_dir), args.n_source, args.n_target)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collate,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            t0 = time.perf_counter()
            src = batch["src_img"].to(args.device, non_blocking=True)
            tgt_img = batch["tgt_img"].to(args.device, non_blocking=True)
            tgt_c2w = batch["tgt_c2w"].to(args.device, non_blocking=True)
            tgt_K = batch["tgt_K"].to(args.device, non_blocking=True)
            H, W = batch["H"], batch["W"]

            plucker = compute_plucker_rays(tgt_c2w, tgt_K, H, W, args.virtual_grid)  # (B, M, 6)
            with autocast_context(args.device, dtype):
                vt = decoder.init_virtual_tokens(plucker.to(dtype) if dtype is not None else plucker)
                unwrap_compiled(student).reset_memory()
                outputs = student(src, virtual_tokens=vt, strict_heads=False)
                vt_out = outputs["virtual_tokens_final"]  # (B, M, 2C)
                gaussians = decoder.decode(vt_out, plucker)

            # render in fp32 (gsplat is unstable in bf16)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                rendered = render_views(gaussians, tgt_c2w, tgt_K, H, W)
                target = tgt_img.permute(0, 1, 3, 4, 2)  # (B,V,H,W,3)
                loss = F.mse_loss(rendered, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()

            if step % 10 == 0:
                psnr = -10 * torch.log10(loss.detach().clamp_min(1e-10))
                print(f"e{epoch+1} s{step} {time.perf_counter()-t0:.1f}s mse={loss.item():.4f} psnr={psnr.item():.2f}",
                      flush=True)

        path = os.path.join(args.save_dir, f"tttlrm_decoder_e{epoch+1}.pt")
        torch.save(unwrap_compiled(decoder).state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
