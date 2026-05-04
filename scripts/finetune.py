"""finetune.py — distill VGGT into LaCT blocks (stage 1) or run consistency (stage 2).

``configure_cuda_backend()`` (via ``script_common``) enables Flash and memory-efficient
SDP backends for VGGT frame attention, and sets ``PYTORCH_CUDA_ALLOC_CONF`` fragmentation
defaults when unset.

Optimised for one Blackwell GPU (RTX Pro 6000 96GB) + 175GB CPU RAM. By default both
teacher (frozen VGGT) and student (VGGT-TTT) use the same GPU; the student *shares*
the teacher's frame_blocks (no duplicate weights). Use ``--teacher_device cpu`` when
VRAM is tight: the teacher runs on CPU (slower) and the student keeps its own GPU
copy of the frame-wise blocks. Saves slim LaCT-only checkpoints to Drive.

Usage:
  # local
  python scripts/finetune.py --data_dir /path/to/scenes --stage 1 --epochs 5

  # DL3DV streaming (480P, 8 scenes/epoch)
  python scripts/finetune.py --stream_dl3dv --stage 1 --epochs 5 --seq_len 32 \\
      --save_dir /content/drive/MyDrive/vggt_ttt_ckpts
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.io_utils import torch_load_checkpoint
from model.losses import consistency_loss, distillation_loss
from model.vggt_ttt import VGGT_TTT
from pipeline.input_pipeline import load_images
from scripts.dl3dv_streaming import (
    StreamingDl3dvPrefetcher, build_dl3dv_download_items,
    download_scene_input, downloaded_scene_root,
)
from scripts.script_common import (
    autocast_context, configure_cuda_backend, fused_adamw,
    maybe_compile, resolve_amp_dtype, set_inductor_cache, unwrap_compiled,
)


# ─────────────────────── checkpoints (slim) ───────────────────────

def save_lact_ckpt(model: nn.Module, save_dir: str, stage: int, suffix: str = "") -> str:
    os.makedirs(save_dir, exist_ok=True)
    name = f"vggt_ttt_lact_stage{stage}{suffix}.pt"
    path = os.path.join(save_dir, name)
    tmp = path + ".tmp"
    torch.save(unwrap_compiled(model).lact_state_dict(), tmp)
    os.replace(tmp, path)
    print(f"saved {path}")
    return path


def load_lact_ckpt_if_present(model: nn.Module, save_dir: str, stage: int, device: str) -> bool:
    """Resume order: stage{N}.pt → stage{N}_final.pt → stage{N-1}.pt → stage{N-1}_final.pt."""
    candidates = [f"vggt_ttt_lact_stage{stage}.pt", f"vggt_ttt_lact_stage{stage}_final.pt"]
    if stage > 1:
        candidates += [f"vggt_ttt_lact_stage{stage-1}.pt", f"vggt_ttt_lact_stage{stage-1}_final.pt"]
    for name in candidates:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            state = torch_load_checkpoint(path, map_location=device)
            unwrap_compiled(model).load_lact_state_dict(state, strict=True)
            print(f"resumed LaCT weights from {path}")
            return True
    return False


# ─────────────────────── dataset ───────────────────────

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, seq_len: int = 16, target_size: int = 518):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.target_size = target_size
        self.sequences: list[list[str]] = []

        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(root)
        flat = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
        scene_dirs = [root] if flat else [p for p in sorted(root.iterdir()) if p.is_dir()]

        for scene in scene_dirs:
            frames = sorted(scene.rglob("*.jpg")) + sorted(scene.rglob("*.png"))
            frames = [str(f) for f in frames]
            if not frames:
                continue
            if len(frames) < seq_len:
                rep = (seq_len // len(frames)) + 1
                self.sequences.append((frames * rep)[:seq_len])
            else:
                step = max(1, seq_len // 2)
                for s in range(0, len(frames) - seq_len + 1, step):
                    self.sequences.append(frames[s : s + seq_len])

        if not self.sequences:
            raise RuntimeError(f"no sequences under {root}")
        print(f"dataset: {len(self.sequences)} sequences x {seq_len} frames")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return load_images(self.sequences[idx], target_size=self.target_size)


def make_dataloader(
    data_dir: str, seq_len: int, batch_size: int, num_workers: int, prefetch_factor: int,
    target_size: int = 518,
):
    ds = VideoFrameDataset(data_dir, seq_len=seq_len, target_size=target_size)
    kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(ds, **kw)


# ─────────────────────── model build ───────────────────────

def build_stage2_student(args, device: str, dtype) -> nn.Module:
    """Build the student for stage 2 (consistency, no teacher needed at runtime).

    Always uses ``share_frame_blocks=False`` so the student owns its frame-block
    weights on GPU. The teacher is constructed temporarily only to satisfy the
    VGGT_TTT constructor, then discarded immediately — the student is fully
    self-contained and all teacher VRAM is reclaimed.
    """
    import gc
    from vggt.models.vggt import VGGT

    print("stage 2: loading teacher VGGT-1B on CPU temporarily to init student weights...")
    # Build teacher on CPU so we never have two backbones on the training GPU at once.
    try:
        teacher = VGGT.from_pretrained("facebook/VGGT-1B", map_location="cpu")
    except TypeError:
        teacher = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if dtype is not None:
        teacher = teacher.to(dtype)

    print("building student VGGT-TTT on CPU (own copy of frame_blocks; share_frame_blocks=False)...")
    student = VGGT_TTT(
        teacher,
        chunk_size=args.chunk_size,
        fast_weight_lr=args.fast_weight_lr,
        lact_ttt_update_cap=args.lact_ttt_update_cap,
        lact_ttt_apply_cap=args.lact_ttt_apply_cap,
        share_frame_blocks=False,
        tbptt_window=args.tbptt_window,
        gradient_checkpoint=args.grad_ckpt,
    )
    if dtype is not None:
        student.aggregator.lact_blocks.to(dtype)
    student.freeze_pretrained()

    # Teacher fully released — student holds its own copies of all submodules.
    del teacher
    gc.collect()
    # Now move the lone backbone to the training device.
    student.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"stage 2: teacher VGGT freed; student moved to {device} (self-contained).")

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"trainable {n_train/1e6:.1f}M / total {n_total/1e6:.1f}M params")
    return student


def build_teacher_and_student(args, device: str, dtype, teacher_device: str):
    """Build VGGT teacher and VGGT-TTT student.

    When ``teacher_device`` is ``cpu``, the teacher is loaded on CPU only (so VRAM
    is not spent on the frozen VGGT stack). The student cannot share frame blocks
    with a CPU teacher, so ``share_frame_blocks=False`` and the student keeps its
    own GPU copy of the frame-wise blocks.
    """
    from vggt.models.vggt import VGGT
    t_dev = (teacher_device or device).strip()
    use_cpu_teacher = t_dev.lower() == "cpu"

    if use_cpu_teacher:
        print("loading teacher VGGT-1B on CPU (frees GPU for student forward/backward)...")
        try:
            teacher = VGGT.from_pretrained("facebook/VGGT-1B", map_location="cpu")
        except TypeError:
            teacher = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if dtype is not None:
            teacher = teacher.to(dtype)
        print("building student VGGT-TTT (own copy of frame_blocks on GPU; teacher on CPU)...")
        student = VGGT_TTT(
            teacher,
            chunk_size=args.chunk_size,
            fast_weight_lr=args.fast_weight_lr,
            lact_ttt_update_cap=args.lact_ttt_update_cap,
            lact_ttt_apply_cap=args.lact_ttt_apply_cap,
            share_frame_blocks=False,
            tbptt_window=args.tbptt_window,
            gradient_checkpoint=args.grad_ckpt,
        )
        if dtype is not None:
            student.aggregator.lact_blocks.to(dtype)
        student.to(device)
        student.freeze_pretrained()
    else:
        print("loading teacher VGGT-1B...")
        teacher = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if dtype is not None:
            teacher = teacher.to(dtype)
        print("building student VGGT-TTT (sharing frame_blocks with teacher)...")
        student = VGGT_TTT(
            teacher,
            chunk_size=args.chunk_size,
            fast_weight_lr=args.fast_weight_lr,
            lact_ttt_update_cap=args.lact_ttt_update_cap,
            lact_ttt_apply_cap=args.lact_ttt_apply_cap,
            share_frame_blocks=True,
            tbptt_window=args.tbptt_window,
            gradient_checkpoint=args.grad_ckpt,
        )
        if dtype is not None:
            student.aggregator.lact_blocks.to(dtype)
        student.to(device)
        student.freeze_pretrained()

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"trainable {n_train/1e6:.1f}M / total {n_total/1e6:.1f}M params")
    return teacher, student


def _move_tensors_to_device(obj, device: torch.device | str):
    """Recursively move tensor leaves to ``device`` (e.g. teacher outputs for loss on GPU)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_move_tensors_to_device(x, device) for x in obj]
        return type(obj)(seq)
    return obj


def build_optim(student: nn.Module, lr: float, epochs: int):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = fused_adamw(params, lr=lr)
    if epochs <= 1:
        sched = CosineAnnealingLR(opt, T_max=max(1, epochs))
    else:
        warm = max(1, epochs // 10)
        warmup = LinearLR(opt, 0.1, 1.0, total_iters=warm)
        cos = CosineAnnealingLR(opt, T_max=epochs - warm)
        sched = SequentialLR(opt, [warmup, cos], milestones=[warm])
    return opt, sched


# ─────────────────────── train loops ───────────────────────

def _step(student, teacher, images, dtype, device, teacher_device: str | None = None):
    td = (teacher_device or device).strip()
    use_cpu_teacher = td.lower() == "cpu"
    with autocast_context(device, dtype):
        student_out = student(images)
    teacher_ctx = nullcontext() if use_cpu_teacher else autocast_context(td, dtype)
    teacher_in = images.detach().cpu() if use_cpu_teacher else images
    with torch.no_grad(), teacher_ctx:
        teacher_out = teacher(teacher_in)
    if use_cpu_teacher:
        teacher_out = _move_tensors_to_device(teacher_out, device)
    loss_dict = distillation_loss(student_out, teacher_out)
    return loss_dict


def train_loop(
    student, teacher, dataloader, opt, sched, device, dtype, epoch: int,
    max_batches: int = 0, teacher_device: str | None = None,
    grad_accum_steps: int = 1,
):
    student.train()
    total = 0.0
    n_micro = 0
    accum = 0
    ga = max(1, int(grad_accum_steps))
    opt.zero_grad(set_to_none=True)
    for step, images in enumerate(dataloader):
        if max_batches and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        t0 = time.perf_counter()
        loss_dict = _step(student, teacher, images, dtype, device, teacher_device)
        loss = loss_dict["loss_total"]
        if not loss.requires_grad:
            continue
        raw = loss.item()
        total += raw
        n_micro += 1
        (loss / ga).backward()
        accum += 1
        if accum >= ga:
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
        if step % 10 == 0:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor))
            print(f"  e{epoch} s{step} {time.perf_counter()-t0:.1f}s {parts}", flush=True)
    if accum > 0:
        nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total / max(n_micro, 1), n_micro


def stage2_step(student, images, dtype, device):
    with autocast_context(device, dtype):
        out = student(images)
        loss_dict = consistency_loss(out, images)
    return loss_dict


def train_stage2_loop(
    student, dataloader, opt, sched, device, dtype, epoch: int, max_batches: int = 0,
    grad_accum_steps: int = 1,
):
    student.train()
    total = 0.0
    n_micro = 0
    accum = 0
    ga = max(1, int(grad_accum_steps))
    opt.zero_grad(set_to_none=True)
    for step, images in enumerate(dataloader):
        if max_batches and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        t0 = time.perf_counter()
        loss_dict = stage2_step(student, images, dtype, device)
        loss = loss_dict["loss_total"]
        if not loss.requires_grad:
            continue
        total += loss.item()
        n_micro += 1
        (loss / ga).backward()
        accum += 1
        if accum >= ga:
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
        if step % 10 == 0:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor))
            print(f"  e{epoch} s{step} {time.perf_counter()-t0:.1f}s {parts}", flush=True)
    if accum > 0:
        nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total / max(n_micro, 1), n_micro


# ─────────────────────── DL3DV streaming ───────────────────────

def acquire_scene(args, items, prefetch, cursor):
    if prefetch is not None:
        scene, root = prefetch.acquire()
        return root, root
    n = len(items)
    if cursor[0] % n == 0:
        random.shuffle(items)
    item = items[cursor[0] % n]; cursor[0] += 1
    guess = downloaded_scene_root(item, args.dl3dv_local_dir)
    try:
        scene, root = download_scene_input(item, args)
        return root, root
    except Exception as e:
        print(f"download failed {item.get('rel_path')}: {e}")
        if guess.exists():
            shutil.rmtree(guess, ignore_errors=True)
        return None, None


def stream_train(args, student, teacher, opt, sched, device, dtype, items, prefetcher):
    cursor = [0]
    nw = args.num_workers
    pf = args.prefetch_factor
    for epoch in range(args.epochs):
        ep_total = 0.0; ep_n = 0
        for _ in range(args.stream_scenes_per_epoch):
            root, cleanup = acquire_scene(args, items, prefetcher, cursor)
            if root is None:
                continue
            try:
                dl = make_dataloader(
                    str(root), args.seq_len, args.batch_size, nw, pf, target_size=args.target_size,
                )
            except RuntimeError as e:
                print(f"skip: {e}")
                if cleanup and cleanup.exists():
                    shutil.rmtree(cleanup, ignore_errors=True)
                continue
            try:
                # reset LaCT memory at scene boundary
                unwrap_compiled(student).reset_memory()
                if args.stage == 1:
                    avg, nb = train_loop(
                        student, teacher, dl, opt, sched, device, dtype,
                        epoch + 1, args.stream_max_batches_per_scene,
                        teacher_device=args.teacher_device or device,
                        grad_accum_steps=args.grad_accum_steps,
                    )
                else:
                    avg, nb = train_stage2_loop(
                        student, dl, opt, sched, device, dtype,
                        epoch + 1, args.stream_max_batches_per_scene,
                        grad_accum_steps=args.grad_accum_steps,
                    )
                ep_total += avg * nb; ep_n += nb
            finally:
                if cleanup and cleanup.exists():
                    shutil.rmtree(cleanup, ignore_errors=True)
        if ep_n == 0:
            print(f"== epoch {epoch+1}/{args.epochs} produced no batches; skipping sched.step() and ckpt")
            continue
        sched.step()
        print(f"== epoch {epoch+1}/{args.epochs} avg={ep_total/ep_n:.4f} batches={ep_n}")
        save_lact_ckpt(student, args.save_dir, args.stage)


def local_train(args, student, teacher, opt, sched, device, dtype):
    dl = make_dataloader(
        args.data_dir, args.seq_len, args.batch_size, args.num_workers, args.prefetch_factor,
        target_size=args.target_size,
    )
    for epoch in range(args.epochs):
        unwrap_compiled(student).reset_memory()
        if args.stage == 1:
            avg, n = train_loop(
                student, teacher, dl, opt, sched, device, dtype, epoch + 1,
                teacher_device=args.teacher_device or device,
                grad_accum_steps=args.grad_accum_steps,
            )
        else:
            avg, n = train_stage2_loop(
                student, dl, opt, sched, device, dtype, epoch + 1,
                grad_accum_steps=args.grad_accum_steps,
            )
        sched.step()
        print(f"== epoch {epoch+1}/{args.epochs} avg={avg:.4f} n={n}")
        save_lact_ckpt(student, args.save_dir, args.stage)


# ─────────────────────── main ───────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fast_weight_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument(
        "--target_size", type=int, default=518,
        help="Shortest-side resize for training images (default 518 = VGGT native). "
        "Lower values (e.g. 336, 392) cut activation memory sharply.",
    )
    p.add_argument("--chunk_size", type=int, default=1)
    p.add_argument("--lact_ttt_update_cap", type=int, default=256)
    p.add_argument("--lact_ttt_apply_cap", type=int, default=512)
    p.add_argument("--tbptt_window", type=int, default=0,
                   help=">0 keeps gradients flowing across that many chunks (more VRAM).")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Gradient-checkpoint frame_blocks. Off by default on Blackwell.")
    p.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help="Clip+optimizer step every N dataloader batches; each batch still runs a full forward/backward "
        "slice, so peak VRAM is unchanged vs 1 (OOM levers: --target_size, --seq_len, --teacher_device cpu, "
        "LaCT caps). Use >1 for a larger effective update / smoother grads only.",
    )
    p.add_argument("--save_dir", default="./checkpoints",
                   help="Use a Google Drive path on Colab; only LaCT weights are saved.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num_workers", type=int, default=-1,
                   help="-1 picks 8 for streaming, 16 for local.")
    p.add_argument("--prefetch_factor", type=int, default=4)
    # streaming
    p.add_argument("--stream_dl3dv", action="store_true")
    p.add_argument("--dl3dv_local_dir", default="/content/dl3dv_finetune_temp")
    p.add_argument("--dl3dv_subset", default="2K")
    p.add_argument("--dl3dv_resolution", choices=["480P", "960P", "2K", "4K"], default="480P")
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--scene_offset", type=int, default=0)
    p.add_argument("--stream_prefetch_scenes", type=int, default=1)
    p.add_argument("--stream_prefetch_workers", type=int, default=2)
    p.add_argument("--stream_scenes_per_epoch", type=int, default=8)
    p.add_argument("--stream_max_batches_per_scene", type=int, default=0)
    p.add_argument(
        "--teacher_device",
        default="",
        metavar="DEVICE",
        help="Where to run the frozen teacher (stage 1 only; stage 2 ignores this and "
        "always loads the teacher transiently on CPU before discarding it). "
        "Default: same as --device. Use ``cpu`` to free GPU VRAM during stage 1 "
        "(slower teacher forward); implies a separate copy of frame-wise blocks on "
        "the student GPU.",
    )
    args = p.parse_args()

    if not args.stream_dl3dv and not args.data_dir.strip():
        p.error("--data_dir required unless --stream_dl3dv")
    if args.num_workers < 0:
        args.num_workers = 8 if args.stream_dl3dv else 16

    os.makedirs(args.save_dir, exist_ok=True)
    set_inductor_cache()
    configure_cuda_backend()
    dtype = resolve_amp_dtype(args.device, args.precision)
    teacher_dev = (args.teacher_device or args.device).strip()
    print(
        f"device={args.device} teacher_device={teacher_dev!r} amp={dtype} "
        f"compile={args.compile} workers={args.num_workers}"
    )

    if args.stage == 2:
        student = build_stage2_student(args, args.device, dtype)
        teacher = None
    else:
        teacher, student = build_teacher_and_student(args, args.device, dtype, teacher_dev)
    load_lact_ckpt_if_present(student, args.save_dir, args.stage, args.device)
    opt, sched = build_optim(student, args.lr, args.epochs)
    student = maybe_compile(student, args.compile)
    if args.stage != 2:
        teacher_on_cuda = next(teacher.parameters()).is_cuda
        teacher = maybe_compile(teacher, args.compile and teacher_on_cuda)

    prefetch = None
    items = None
    if args.stream_dl3dv:
        os.makedirs(args.dl3dv_local_dir, exist_ok=True)
        items = build_dl3dv_download_items(args)
        if args.stream_prefetch_scenes > 1:
            prefetch = StreamingDl3dvPrefetcher(items, args, args.stream_prefetch_scenes, args.stream_prefetch_workers)
            prefetch.start()
    try:
        if args.stream_dl3dv:
            stream_train(args, student, teacher, opt, sched, args.device, dtype, items, prefetch)
        else:
            local_train(args, student, teacher, opt, sched, args.device, dtype)
    finally:
        if prefetch is not None:
            prefetch.shutdown()
        save_lact_ckpt(student, args.save_dir, args.stage, suffix="_final")


if __name__ == "__main__":
    main()
