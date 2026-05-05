"""Benchmark VGGT (teacher) vs VGGT-TTT (LaCT student, stage 1) across sequence lengths.

Headline story: LaCT is a linear-memory drop-in for VGGT's global attention. We measure
trunk-level outputs (camera/depth/point) plus efficiency (peak VRAM, latency) on
DL3DV-Evaluation. Teacher is expected to OOM at long N; student keeps going.

Usage (streaming from HF — recommended; per-scene tars are ~9 GB):
  python scripts/benchmark_vs_vggt.py \
      --hf_repo DL3DV/DL3DV-Evaluation \
      --hf_cache /content/_dl3dv_eval_cache \
      --lact_ckpt /content/drive/MyDrive/vggt_ttt_ckpts/vggt_ttt_lact_stage1.pt \
      --num_scenes 20 --seq_lens 16,32,48,64,96,128 \
      --json_out bench.json

Usage (already-extracted local dir):
  python scripts/benchmark_vs_vggt.py --data_dir /path/to/extracted --lact_ckpt ...

Metrics:
  pose:   rotation error (deg), translation error (relative), vs GT extrinsics
  depth:  AbsRel + delta<1.25 vs teacher depth (where teacher succeeds)
  point:  per-pixel L2 vs teacher world points (where teacher succeeds)
  perf:   peak VRAM (GB), wall time (s), tokens/s

Teacher-as-GT for depth/point is fair — student was distilled from teacher;
this measures distillation fidelity. Pose is vs ground-truth extrinsics.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import tarfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.io_utils import torch_load_checkpoint
from model.vggt_ttt import VGGT_TTT
from pipeline.input_pipeline import load_images


# ── inlined scene helpers (DL3DV nerfstudio/colmap layouts) ──

def _find_transforms(image_dir: Path) -> "Path | None":
    for cand in ("transforms.json", "transforms_train.json"):
        if (image_dir / cand).exists(): return image_dir / cand
        if (image_dir.parent / cand).exists(): return image_dir.parent / cand
    return None


def _scene_input_from_root(root: Path) -> "tuple[Path, Path] | None":
    """Return (image_dir, transforms_path) for a DL3DV-style root, or None."""
    candidates = [
        root / "nerfstudio" / "images_2", root / "nerfstudio" / "images",
        root / "colmap" / "images_2", root / "colmap" / "images_4",
        root / "colmap" / "images_8", root / "colmap" / "images",
        root / "images_2", root / "images_4", root / "images_8", root / "images",
        root,
    ]
    for image_dir in candidates:
        if not image_dir.is_dir(): continue
        tf = _find_transforms(image_dir)
        if tf is None: continue
        if not any(image_dir.glob("*.jpg")) and not any(image_dir.glob("*.png")):
            continue
        return image_dir, tf
    return None


def load_scene_meta(scene_dir: Path) -> tuple[dict, Path]:
    si = _scene_input_from_root(scene_dir)
    if si is not None:
        _, tf = si
        return json.loads(tf.read_text()), tf.parent
    for cand in ("transforms.json", "transforms_train.json"):
        p = scene_dir / cand
        if p.exists(): return json.loads(p.read_text()), p.parent
        for sub in sorted(scene_dir.iterdir()) if scene_dir.is_dir() else []:
            if sub.is_dir() and (sub / cand).exists():
                return json.loads((sub / cand).read_text()), sub
    raise FileNotFoundError(f"transforms.json not found under {scene_dir}")


def _resolve_frame_path(scene: Path, image_root: Path, fp: str) -> str:
    p = Path(fp)
    if p.is_absolute(): return str(p)
    primary = image_root / p
    if primary.exists(): return str(primary)
    if p.parts and p.parts[0] == "images" and len(p.parts) > 1:
        tail = Path(*p.parts[1:])
        for base in (image_root, scene):
            for folder in ("images_8", "images_2", "images_4", "images_1", "images"):
                alt = base / folder / tail
                if alt.exists(): return str(alt)
    return str(primary)


def _scene_usable(scene_dir: Path) -> bool:
    try:
        meta, _ = load_scene_meta(scene_dir)
    except Exception:
        return False
    return bool(meta.get("frames"))


def find_eval_scenes(root: Path, n: int, max_depth: int = 12) -> list[Path]:
    """Collect up to ``n`` scene roots under ``root`` (BFS through nested layouts)."""
    from collections import deque
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {root}")
    hits: list[Path] = []
    seen: set[Path] = set()
    def push(d: Path):
        if d in seen or len(hits) >= n: return
        if not _scene_usable(d): return
        seen.add(d); hits.append(d)
    push(root)
    q = deque([(root, 0)])
    while q and len(hits) < n:
        d, depth = q.popleft()
        if depth >= max_depth: continue
        try:
            subs = sorted(p for p in d.iterdir()
                          if p.is_dir() and not p.name.startswith(".") and p.name != "_hf_cache")
        except OSError:
            continue
        for sub in subs:
            if sub in seen: continue
            if _scene_usable(sub): push(sub)
            else: seen.add(sub); q.append((sub, depth + 1))
    if not hits:
        raise RuntimeError(f"no scenes with usable transforms under {root}")
    return hits[:n]


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """tarfile.extractall is path-traversal unsafe. Reject members whose resolved
    path escapes ``dest`` (zip-slip)."""
    dest_resolved = dest.resolve()
    safe_members = []
    for m in tf.getmembers():
        if m.issym() or m.islnk():
            print(f"[stream] skipping link member: {m.name}")
            continue
        target = (dest / m.name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            print(f"[stream] skipping unsafe tar member: {m.name}")
            continue
        safe_members.append(m)
    tf.extractall(dest, members=safe_members)


def _cuda_sync(t: torch.Tensor) -> None:
    if t.is_cuda:
        torch.cuda.synchronize(t.device)


def _gt_extrinsic_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """transforms.json gives c2w; VGGT extrinsic is w2c (3x4). Invert + slice."""
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    return torch.cat([Rt, -Rt @ t], dim=-1)  # (..., 3, 4)


def _rot_err_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error per pair, in degrees."""
    M = R_pred @ R_gt.transpose(-1, -2)
    tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return cos.acos() * (180.0 / math.pi)


def _align_translation_scale(t_pred: torch.Tensor, t_gt: torch.Tensor) -> float:
    """Solve scalar s minimizing ||s*t_pred - t_gt||^2; robust to global scale gauge."""
    num = (t_pred * t_gt).sum().item()
    den = (t_pred * t_pred).sum().item() + 1e-12
    return num / den


def pose_metrics(extr_pred: torch.Tensor, c2w_gt: torch.Tensor) -> dict:
    """extr_pred: (V, 3, 4) w2c.   c2w_gt: (V, 4, 4)."""
    extr_gt = _gt_extrinsic_w2c(c2w_gt)
    R_p, t_p = extr_pred[..., :3], extr_pred[..., 3]
    R_g, t_g = extr_gt[..., :3], extr_gt[..., 3]
    rot = _rot_err_deg(R_p, R_g)
    s = _align_translation_scale(t_p, t_g)
    t_rel_err = ((s * t_p - t_g).norm(dim=-1) / (t_g.norm(dim=-1) + 1e-6))
    return {"rot_deg_mean": rot.mean().item(), "rot_deg_med": rot.median().item(),
            "trans_rel_mean": t_rel_err.mean().item(), "trans_rel_med": t_rel_err.median().item()}


def depth_metrics(d_pred: torch.Tensor, d_ref: torch.Tensor) -> dict:
    """AbsRel + delta<1.25 with median scale alignment per scene."""
    p = d_pred.flatten().float()
    r = d_ref.flatten().float()
    mask = (r > 1e-3) & torch.isfinite(p) & torch.isfinite(r)
    if mask.sum() < 100:
        return {"absrel": float("nan"), "delta125": float("nan")}
    p, r = p[mask], r[mask]
    s = (r.median() / p.median().clamp_min(1e-6)).item()
    p = p * s
    absrel = ((p - r).abs() / r).mean().item()
    ratio = torch.maximum(p / r, r / p)
    delta = (ratio < 1.25).float().mean().item()
    return {"absrel": absrel, "delta125": delta}


def point_metrics(wp_pred: torch.Tensor, wp_ref: torch.Tensor) -> dict:
    """Per-pixel world-point L2 (median-scaled)."""
    p = wp_pred.reshape(-1, 3).float()
    r = wp_ref.reshape(-1, 3).float()
    mask = torch.isfinite(p).all(-1) & torch.isfinite(r).all(-1)
    if mask.sum() < 100:
        return {"point_l2": float("nan")}
    p, r = p[mask], r[mask]
    s = (r.norm(dim=-1).median() / p.norm(dim=-1).median().clamp_min(1e-6)).item()
    return {"point_l2": (p * s - r).norm(dim=-1).mean().item()}


def _iter_local_scenes(data_dir: Path, max_scenes: int):
    """Yield (scene_path, cleanup_fn) for already-extracted scenes."""
    scenes = find_eval_scenes(data_dir, max_scenes)
    for s in scenes:
        yield s, (lambda: None)


def _list_hf_scene_tars(repo: str) -> list[str]:
    """Return repo-relative paths of per-scene tar files in DL3DV-Evaluation."""
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(
            f"failed to list {repo}: {e}. For gated datasets, run `huggingface-cli login` "
            f"and accept terms at https://huggingface.co/datasets/{repo}."
        ) from e
    tars = sorted(f for f in files if f.endswith((".tar", ".tar.gz", ".tgz")))
    if not tars:
        raise RuntimeError(
            f"no .tar files found in dataset repo {repo}. "
            "If this is a gated repo, you may be unauthenticated — run `huggingface-cli login`."
        )
    return tars


def _iter_hf_streaming_scenes(repo: str, cache_dir: Path, max_scenes: int):
    """Download → extract → yield → delete, one tar at a time. Bounded disk."""
    from huggingface_hub import hf_hub_download
    cache_dir.mkdir(parents=True, exist_ok=True)
    tars = _list_hf_scene_tars(repo)
    n_yielded = 0
    for rel in tars:
        if n_yielded >= max_scenes:
            return
        print(f"[stream] downloading {rel} ({n_yielded + 1}/{max_scenes})...")
        try:
            local = hf_hub_download(
                repo_id=repo, repo_type="dataset", filename=rel,
                local_dir=str(cache_dir / "_dl"),
            )
        except Exception as e:
            print(f"[stream] download failed for {rel}: {e}")
            continue

        extract_root = cache_dir / "_extract" / Path(rel).stem
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(local, "r:*") as tf:
                _safe_extract_tar(tf, extract_root)
        except Exception as e:
            print(f"[stream] extract failed for {rel}: {e}")
            shutil.rmtree(extract_root, ignore_errors=True)
            Path(local).unlink(missing_ok=True)
            continue

        try:
            scenes = find_eval_scenes(extract_root, n=1)
        except Exception as e:
            print(f"[stream] no usable scene in {rel}: {e}")
            scenes = []

        if not scenes:
            shutil.rmtree(extract_root, ignore_errors=True)
            Path(local).unlink(missing_ok=True)
            continue

        scene = scenes[0]

        def _cleanup(extract_root=extract_root, local=local):
            shutil.rmtree(extract_root, ignore_errors=True)
            Path(local).unlink(missing_ok=True)

        try:
            yield scene, _cleanup
        finally:
            _cleanup()
        n_yielded += 1


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_teacher(teacher, images: torch.Tensor) -> tuple[dict, float, float]:
    """Returns (outputs, secs, peak_gb). May raise OOM. Uses teacher.forward
    (the package's official entry point) instead of poking aggregator+heads."""
    _free_cuda()
    _cuda_sync(images)
    t0 = time.time()
    dev_type = images.device.type
    with torch.no_grad(), torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
        out = teacher(images)
    _cuda_sync(images)
    secs = time.time() - t0
    return out, secs, _peak_vram_gb()


def run_student(student, images: torch.Tensor) -> tuple[dict, float, float]:
    _free_cuda()
    _cuda_sync(images)
    t0 = time.time()
    dev_type = images.device.type
    with torch.no_grad(), torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
        student.reset_memory()
        out = student(images, strict_heads=False)
    _cuda_sync(images)
    secs = time.time() - t0
    return out, secs, _peak_vram_gb()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="", help="Local extracted scenes dir (alternative to --hf_repo).")
    p.add_argument("--hf_repo", default="", help="HF dataset repo to stream (e.g. DL3DV/DL3DV-Evaluation).")
    p.add_argument("--hf_cache", default="/content/_dl3dv_eval_cache",
                   help="Local cache for streamed tars (deleted per-scene after benchmark).")
    p.add_argument("--lact_ckpt", required=True)
    p.add_argument("--num_scenes", type=int, default=20)
    p.add_argument("--seq_lens", default="16,32,48,64,96,128")
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json_out", default="benchmark_results.json")
    p.add_argument("--skip_teacher_above", type=int, default=10**9,
                   help="Skip teacher run when seq_len > this (after a known OOM, save time).")
    p.add_argument("--teacher_oom_streak", type=int, default=2,
                   help="Drop the teacher OOM floor only after this many consecutive OOMs at the same N "
                        "(guards against single-scene flakes).")
    p.add_argument("--share_frame_blocks", action=argparse.BooleanOptionalAction, default=True,
                   help="Match the stage-1 training config (default True). Recorded in JSON output.")
    args = p.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    device = args.device

    print("loading VGGT teacher...")
    from vggt.models.vggt import VGGT
    teacher = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    for q in teacher.parameters(): q.requires_grad = False

    print("loading VGGT-TTT student (LaCT stage 1)...")
    student = VGGT_TTT.from_pretrained(
        share_frame_blocks=args.share_frame_blocks,
        chunk_size=args.chunk_size, gradient_checkpoint=True,
    ).to(device).eval()
    student.load_lact_state_dict(torch_load_checkpoint(args.lact_ckpt, map_location=device), strict=True)
    student.freeze_pretrained()

    if not args.hf_repo and not args.data_dir:
        raise SystemExit("must pass either --hf_repo or --data_dir")
    if args.hf_repo:
        scene_iter = _iter_hf_streaming_scenes(args.hf_repo, Path(args.hf_cache), args.num_scenes)
    else:
        scene_iter = _iter_local_scenes(Path(args.data_dir), args.num_scenes)
    print(f"benchmarking up to {args.num_scenes} scenes at seq_lens={seq_lens}")

    teacher_oom_floor = args.skip_teacher_above
    teacher_oom_streak: dict[int, int] = {}
    # collect rows keyed by N so we run all seq_lens per scene (download tar once)
    rows_by_N: dict[int, list[dict]] = {N: [] for N in seq_lens}

    for scene, cleanup in scene_iter:
      try:
        try:
            meta, image_root = load_scene_meta(scene)
        except Exception as e:
            print(f"  skip {scene.name[:16]}: {e}")
            continue
        frames = meta["frames"]
        print(f"\n--- scene {scene.name[:24]} ({len(frames)} frames) ---")

        for N in seq_lens:
            if len(frames) < N:
                print(f"  N={N}: scene only has {len(frames)} frames, skip")
                continue
            try:
                paths = [_resolve_frame_path(scene, image_root, f["file_path"]) for f in frames[:N]]
                imgs = load_images(paths).unsqueeze(0).to(device)
                c2w = torch.stack([torch.tensor(f["transform_matrix"], dtype=torch.float32)
                                    for f in frames[:N]]).to(device)
            except Exception as e:
                print(f"  N={N}: load failed: {e}")
                continue

            row = {"scene": scene.name, "N": N}
            s_out = None  # cleared per-N so a non-OOM raise can't leak across iterations

            try:
                s_out, s_secs, s_vram = run_student(student, imgs)
                stats = {
                    "secs": s_secs, "peak_vram_gb": s_vram,
                    "tok_per_s": (N * (imgs.shape[-2] // 14) * (imgs.shape[-1] // 14)) / max(s_secs, 1e-6),
                }
                if "extrinsic" in s_out:
                    stats.update(pose_metrics(s_out["extrinsic"][0], c2w))
                row["student"] = stats
            except torch.OutOfMemoryError:
                row["student"] = {"oom": True}
                _free_cuda()
            except Exception as e:
                row["student"] = {"error": repr(e)}
                _free_cuda()

            run_teacher_now = N <= teacher_oom_floor
            if run_teacher_now:
                try:
                    t_out, t_secs, t_vram = run_teacher(teacher, imgs)
                    t_stats = {"secs": t_secs, "peak_vram_gb": t_vram}
                    if "extrinsic" in t_out:
                        t_stats.update(pose_metrics(t_out["extrinsic"][0], c2w))
                    row["teacher"] = t_stats
                    teacher_oom_streak[N] = 0  # reset streak on successful teacher run
                    if s_out is not None:
                        sv = {}
                        if "depth" in s_out and "depth" in t_out:
                            if s_out["depth"].shape == t_out["depth"].shape:
                                sv.update(depth_metrics(s_out["depth"][0], t_out["depth"][0]))
                            else:
                                print(f"  depth shape mismatch s={tuple(s_out['depth'].shape)} t={tuple(t_out['depth'].shape)}")
                        if "world_points" in s_out and "world_points" in t_out:
                            if s_out["world_points"].shape == t_out["world_points"].shape:
                                sv.update(point_metrics(s_out["world_points"][0], t_out["world_points"][0]))
                            else:
                                print(f"  point shape mismatch")
                        if sv:
                            row["student_vs_teacher"] = sv
                    del t_out
                except torch.OutOfMemoryError:
                    # Per-scene flag (record); only ratchet the global floor down after
                    # ``--teacher_oom_streak`` consecutive OOMs at this N, since OOM can be
                    # scene-dependent (frame resolution, frame count beyond N, etc.).
                    print(f"  teacher OOM at N={N}")
                    row["teacher"] = {"oom": True}
                    teacher_oom_streak[N] = teacher_oom_streak.get(N, 0) + 1
                    if teacher_oom_streak[N] >= args.teacher_oom_streak:
                        teacher_oom_floor = min(teacher_oom_floor, N - 1)
                        print(f"  → {teacher_oom_streak[N]} consecutive OOMs at N={N}, "
                              f"skipping teacher for N>{teacher_oom_floor}")
                    _free_cuda()
            else:
                row["teacher"] = {"skipped_after_oom": True}

            rows_by_N[N].append(row)

            def _tag(d: dict) -> str:
                if not d:
                    return "?"
                if d.get("oom"):
                    return "OOM"
                if d.get("skipped_after_oom"):
                    return "skip"
                if "error" in d:
                    return f"err:{d['error'][:40]}"
                if "secs" in d and "peak_vram_gb" in d:
                    return f"{d['secs']:.1f}s/{d['peak_vram_gb']:.1f}GB"
                return "?"

            tag_s = _tag(row.get("student", {}))
            tag_t = _tag(row.get("teacher", {}))
            print(f"  N={N:<4} student={tag_s:<22} teacher={tag_t}")
            del imgs, c2w
            s_out = None
            _free_cuda()
      finally:
        cleanup()

    # aggregate
    summary = []
    for N in seq_lens:
        rows = rows_by_N[N]
        def agg(side: str, key: str):
            xs = [r[side][key] for r in rows if side in r and key in r[side] and isinstance(r[side][key], (int, float))]
            return sum(xs) / len(xs) if xs else None
        summary.append({
            "N": N,
            "n_scenes": len(rows),
            "student": {k: agg("student", k) for k in
                        ["secs", "peak_vram_gb", "tok_per_s", "rot_deg_med", "trans_rel_med"]},
            "teacher": {k: agg("teacher", k) for k in
                        ["secs", "peak_vram_gb", "rot_deg_med", "trans_rel_med"]},
            "student_vs_teacher": {k: agg("student_vs_teacher", k) for k in
                                    ["absrel", "delta125", "point_l2"]},
        })

    print("\n=== SUMMARY ===")
    print(f"{'N':>4} {'S_VRAM':>8} {'T_VRAM':>8} {'S_secs':>7} {'T_secs':>7} {'S_rot°':>7} {'T_rot°':>7} {'AbsRel':>7} {'δ<1.25':>7}")
    for s in summary:
        def f(x, fmt=".2f"): return ("-" if x is None else format(x, fmt))
        print(f"{s['N']:>4} {f(s['student']['peak_vram_gb']):>8} {f(s['teacher']['peak_vram_gb']):>8} "
              f"{f(s['student']['secs']):>7} {f(s['teacher']['secs']):>7} "
              f"{f(s['student']['rot_deg_med']):>7} {f(s['teacher']['rot_deg_med']):>7} "
              f"{f(s['student_vs_teacher']['absrel'], '.3f'):>7} {f(s['student_vs_teacher']['delta125'], '.3f'):>7}")

    Path(args.json_out).write_text(json.dumps({
        "config": {
            "seq_lens": seq_lens, "chunk_size": args.chunk_size,
            "share_frame_blocks": args.share_frame_blocks,
            "lact_ckpt": args.lact_ckpt, "hf_repo": args.hf_repo or None,
            "data_dir": args.data_dir or None, "num_scenes": args.num_scenes,
        },
        "summary": summary,
        "raw": [{"N": N, "rows": rows_by_N[N]} for N in seq_lens],
    }, indent=2))
    print(f"\nsaved {args.json_out}")


if __name__ == "__main__":
    main()
