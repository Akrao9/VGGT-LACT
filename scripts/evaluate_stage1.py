"""
Evaluate VGGT-TTT against the frozen VGGT teacher on held-out scenes.

Typical post-stage-1 check:

    python scripts/evaluate_stage1.py \
        --data_dir /content/DL3DV-Evaluation \
        --checkpoint /content/drive/MyDrive/vggt_lact_checkpoints/vggt_ttt_stage1_scene200.pt \
        --compare_fresh \
        --seq_lens 16 \
        --num_scenes 5 \
        --batches_per_scene 20 \
        --json_out /content/eval_stage1_scene200.json

Lower checkpoint losses than the fresh LaCT baseline mean the trained LaCT
blocks are matching VGGT better on camera, depth, and world points.
"""

import argparse
import fnmatch
import gc
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.io_utils import torch_load_checkpoint
from model.losses import distillation_loss
from model.vggt_ttt import VGGT_TTT
from scripts.finetune import VideoFrameDataset
from scripts.script_common import autocast_context, configure_cuda_backend, resolve_amp_dtype


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".zip")


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def tensor_item(value):
    return value.detach().float().item() if isinstance(value, torch.Tensor) else float(value)


def has_direct_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    for ext in IMAGE_EXTS:
        if next(path.glob(f"*{ext}"), None) is not None:
            return True
        if next(path.glob(f"*{ext.upper()}"), None) is not None:
            return True
    return False


def has_any_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    for ext in IMAGE_EXTS:
        if next(path.rglob(f"*{ext}"), None) is not None:
            return True
        if next(path.rglob(f"*{ext.upper()}"), None) is not None:
            return True
    return False


def looks_like_scene_dir(path: Path) -> bool:
    """Return True for one scene dir, including common nested image layouts."""
    if has_direct_images(path):
        return True

    for child in path.iterdir() if path.is_dir() else []:
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "image" in name or name in {"rgb", "color", "frames"}:
            if has_direct_images(child):
                return True
    return False


def find_scene_dirs(data_dir: str, num_scenes: Optional[int], scene_offset: int) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    if looks_like_scene_dir(root):
        scenes = [root]
    else:
        scenes = []
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if has_any_images(child):
                scenes.append(child)
        if not scenes and has_any_images(root):
            scenes = [root]

    if scene_offset:
        scenes = scenes[scene_offset:]
    if num_scenes is not None:
        scenes = scenes[:num_scenes]
    if not scenes:
        raise RuntimeError(f"No scene directories with images found under {data_dir}")
    return scenes


def is_archive_file(path: str) -> bool:
    path = path.lower()
    return any(path.endswith(ext) for ext in ARCHIVE_EXTS)


def matches_any_pattern(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def safe_extract_tar(archive_path: Path, dst_dir: Path):
    """Extract a tar archive while preventing path traversal."""
    dst_dir = dst_dir.resolve()
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            member_path = (dst_dir / member.name).resolve()
            if member_path != dst_dir and not str(member_path).startswith(str(dst_dir) + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Unsafe tar member type: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported tar member type: {member.name}")
        try:
            tar.extractall(dst_dir, filter="data")
        except TypeError:
            tar.extractall(dst_dir)


def safe_extract_zip(archive_path: Path, dst_dir: Path):
    """Extract a zip archive while preventing path traversal."""
    dst_dir = dst_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            member = info.filename
            member_path = (dst_dir / member).resolve()
            if member_path != dst_dir and not str(member_path).startswith(str(dst_dir) + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member}")
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise RuntimeError(f"Unsafe zip symlink blocked: {member}")
        zf.extractall(dst_dir)


def extract_archive(archive_path: Path, dst_dir: Path):
    print(f"Extracting {archive_path.name}...")
    if tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, dst_dir)
    elif zipfile.is_zipfile(archive_path):
        safe_extract_zip(archive_path, dst_dir)
    else:
        raise RuntimeError(f"Unsupported archive: {archive_path}")


def ensure_hf_dataset_available(args):
    """Download a small subset of a gated Hugging Face dataset if data_dir is missing."""
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        return
    if not args.hf_repo:
        if "DL3DV-Evaluation" in data_dir.name or "DL3DV-Evaluation" in str(data_dir):
            args.hf_repo = "DL3DV/DL3DV-Evaluation"
            print(f"Inferring --hf_repo {args.hf_repo} from missing data_dir {args.data_dir}")
        else:
            raise FileNotFoundError(
                f"data_dir does not exist: {args.data_dir}. "
                "Pass --hf_repo to download a small HF subset automatically."
            )

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face download deps first: "
            "pip install huggingface_hub tqdm"
        ) from exc

    print(f"{args.data_dir} does not exist; downloading from {args.hf_repo}")
    print("This requires accepting the dataset terms on Hugging Face and running huggingface-cli login.")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else data_dir / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=args.hf_token)
    repo_files = api.list_repo_files(
        repo_id=args.hf_repo,
        repo_type="dataset",
        revision=args.hf_revision,
    )

    if args.hf_allow_patterns:
        patterns = [part.strip() for part in args.hf_allow_patterns.split(",") if part.strip()]
        archive_files = [path for path in repo_files if matches_any_pattern(path, patterns)]
    else:
        archive_files = [path for path in repo_files if is_archive_file(path)]

    archive_files = sorted(path for path in archive_files if not Path(path).name.startswith("."))
    if not archive_files:
        raise RuntimeError(f"No downloadable archive files found in {args.hf_repo}")

    max_archives = args.hf_max_archives
    if max_archives is None:
        max_archives = args.num_scenes
    if max_archives is not None:
        archive_files = archive_files[:max_archives]

    print(f"Downloading {len(archive_files)} archive(s) into {data_dir}")
    for filename in tqdm(archive_files, desc="HF archives"):
        archive_path = Path(
            hf_hub_download(
                repo_id=args.hf_repo,
                filename=filename,
                repo_type="dataset",
                revision=args.hf_revision,
                cache_dir=str(cache_dir),
                token=args.hf_token,
            )
        )
        extract_archive(archive_path, data_dir)

    print(f"Dataset subset ready at {data_dir}")


def move_batch_to_device(images: torch.Tensor, device: str) -> torch.Tensor:
    return images.to(device, non_blocking=(torch.device(device).type == "cuda"))


def average_metrics(totals: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def load_student(checkpoint: Optional[str], args) -> torch.nn.Module:
    model = VGGT_TTT.from_pretrained(
        chunk_size=args.chunk_size,
        fast_weight_lr=args.fast_weight_lr,
    ).to(args.device)
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        state = torch_load_checkpoint(checkpoint, map_location="cpu")
        if any(k.startswith("aggregator.lact_blocks.") for k in state) and not any(
            k == "camera_head.weight" or k.startswith("camera_head.") for k in state
        ):
            model.load_lact_state_dict(state, strict=True)
        else:
            model.load_state_dict(state)
    else:
        print("No checkpoint provided; evaluating fresh untrained LaCT weights.")
    model.eval()
    return model


def evaluate_scene(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    scene_dir: Path,
    seq_len: int,
    args,
) -> Dict:
    device_type = torch.device(args.device).type
    dataset = VideoFrameDataset(str(scene_dir), seq_len=seq_len, target_size=args.target_size)
    if len(dataset) == 0:
        return {
            "scene": scene_dir.name,
            "seq_len": seq_len,
            "batches": 0,
            "metrics": {},
            "error": "empty_dataset",
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device_type == "cuda",
    )

    totals: Dict[str, float] = {}
    count = 0

    try:
        with torch.no_grad():
            for batch_idx, images in enumerate(dataloader):
                if batch_idx >= args.batches_per_scene:
                    break

                images = move_batch_to_device(images, args.device)
                with autocast_context(args.device, args.dtype):
                    student_out = model(images, chunk_size=args.chunk_size)
                    teacher_out = teacher(images)

                loss_dict = distillation_loss(student_out, teacher_out)
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        totals[key] = totals.get(key, 0.0) + tensor_item(value)
                count += 1

                if args.verbose_batches:
                    parts = " | ".join(
                        f"{key}={tensor_item(value):.4f}"
                        for key, value in loss_dict.items()
                        if isinstance(value, torch.Tensor)
                    )
                    print(f"    batch {batch_idx}: {parts}")

                del images, student_out, teacher_out, loss_dict

    finally:
        if hasattr(dataloader, "_iterator"):
            dataloader._iterator = None
        del dataloader
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    return {
        "scene": scene_dir.name,
        "seq_len": seq_len,
        "batches": count,
        "metrics": average_metrics(totals, count),
    }


def weighted_summary(records: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for record in records:
        if record.get("error") or record.get("batches", 0) == 0:
            continue
        key = f'{record["label"]}/seq{record["seq_len"]}'
        grouped.setdefault(key, {})
        counts[key] = counts.get(key, 0) + record["batches"]
        for metric_name, metric_value in record["metrics"].items():
            grouped[key][metric_name] = grouped[key].get(metric_name, 0.0) + (
                metric_value * record["batches"]
            )

    summary = {}
    for key, metric_totals in grouped.items():
        denom = max(counts[key], 1)
        summary[key] = {
            metric_name: metric_value / denom
            for metric_name, metric_value in sorted(metric_totals.items())
        }
        summary[key]["batches"] = counts[key]
    return summary


def print_summary(summary: Dict[str, Dict[str, float]]):
    print("\nSummary:")
    for group, metrics in sorted(summary.items()):
        batch_count = int(metrics.get("batches", 0))
        metric_text = " | ".join(
            f"{key}={value:.4f}"
            for key, value in metrics.items()
            if key != "batches"
        )
        print(f"  {group} ({batch_count} batches): {metric_text}")

    fresh_keys = [key for key in summary if key.startswith("fresh/")]
    for fresh_key in sorted(fresh_keys):
        ckpt_key = fresh_key.replace("fresh/", "checkpoint/", 1)
        if ckpt_key not in summary:
            continue
        fresh_total = summary[fresh_key].get("loss_total")
        ckpt_total = summary[ckpt_key].get("loss_total")
        if fresh_total is None or ckpt_total is None:
            continue
        delta = ckpt_total - fresh_total
        pct = 100.0 * delta / max(abs(fresh_total), 1e-8)
        print(
            f"  delta {ckpt_key} - {fresh_key}: "
            f"loss_total {delta:+.4f} ({pct:+.1f}%)"
        )


def evaluate_model(label: str, checkpoint: Optional[str], scenes: List[Path], seq_lens: List[int], args) -> List[Dict]:
    print(f"\n=== Evaluating {label} ===")
    model = load_student(checkpoint, args)
    records: List[Dict] = []

    try:
        for seq_len in seq_lens:
            print(f"\nSequence length: {seq_len}")
            for scene_idx, scene_dir in enumerate(scenes, start=1):
                print(f"  [{scene_idx}/{len(scenes)}] {scene_dir.name}")
                try:
                    record = evaluate_scene(model, args.teacher, scene_dir, seq_len, args)
                except torch.OutOfMemoryError as exc:
                    print(f"    OOM at seq_len={seq_len}; skipping scene: {exc}")
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
                    record = {
                        "scene": scene_dir.name,
                        "seq_len": seq_len,
                        "batches": 0,
                        "metrics": {},
                        "error": "cuda_oom",
                    }

                record["label"] = label
                records.append(record)
                if record.get("metrics"):
                    metric_text = " | ".join(
                        f"{key}={value:.4f}"
                        for key, value in sorted(record["metrics"].items())
                    )
                    print(f"    avg: {metric_text}")
                elif record.get("error"):
                    print(f"    skipped: {record['error']}")
    finally:
        del model
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate VGGT-TTT stage1 distillation on held-out scenes.")
    parser.add_argument("--data_dir", default=None, help="held-out scene dir or parent dir")
    parser.add_argument("--checkpoint", default=None, help="optional path to vggt_ttt_stage1.pt")
    parser.add_argument("--compare_fresh", action="store_true", help="also evaluate a fresh untrained VGGT-TTT baseline")
    parser.add_argument("--seq_len", type=int, default=16, help="single seq length when --seq_lens is omitted")
    parser.add_argument("--seq_lens", default=None, help="comma-separated seq lengths, e.g. 16 or 16,32")
    parser.add_argument("--num_scenes", type=int, default=None, help="limit number of held-out scenes")
    parser.add_argument("--scene_offset", type=int, default=0, help="skip this many scene dirs before evaluating")
    parser.add_argument("--batches_per_scene", type=int, default=20)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--fast_weight_lr", type=float, default=1e-3,
                        help="Internal LaCT scene-memory update step size.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", default=None, help="optional path for machine-readable results")
    parser.add_argument("--verbose_batches", action="store_true")
    parser.add_argument("--hf_repo", default=None, help="optional gated HF dataset repo, e.g. DL3DV/DL3DV-Evaluation")
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--hf_token", default=None, help="optional HF token; otherwise uses huggingface-cli login")
    parser.add_argument("--hf_max_archives", type=int, default=None, help="limit archives downloaded; defaults to --num_scenes")
    parser.add_argument("--hf_allow_patterns", default=None, help="comma-separated HF filename patterns, e.g. '*.tar,*.tar.gz'")
    args = parser.parse_args()

    if args.data_dir is None:
        if args.hf_repo:
            args.data_dir = "./DL3DV-Evaluation"
        else:
            parser.error("--data_dir is required unless --hf_repo is provided")

    device_type = torch.device(args.device).type
    configure_cuda_backend(allow_tf32=True, cudnn_benchmark=True)
    args.dtype = resolve_amp_dtype(args.device, args.precision)

    seq_lens = parse_int_list(args.seq_lens) if args.seq_lens else [args.seq_len]
    ensure_hf_dataset_available(args)
    scenes = find_scene_dirs(args.data_dir, args.num_scenes, args.scene_offset)
    print(f"Found {len(scenes)} held-out scene(s)")
    print("Scenes:")
    for scene in scenes:
        print(f"  {scene}")

    print("\nLoading VGGT teacher once...")
    from vggt.models.vggt import VGGT

    teacher = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device)
    teacher.eval()
    args.teacher = teacher

    records: List[Dict] = []
    if args.compare_fresh and args.checkpoint:
        records.extend(evaluate_model("fresh", None, scenes, seq_lens, args))
    elif args.compare_fresh:
        print("\n--compare_fresh was requested without --checkpoint; evaluating fresh once.")

    label = "checkpoint" if args.checkpoint else "fresh"
    records.extend(evaluate_model(label, args.checkpoint, scenes, seq_lens, args))

    summary = weighted_summary(records)
    print_summary(summary)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_dir": args.data_dir,
            "checkpoint": args.checkpoint,
            "seq_lens": seq_lens,
            "num_scenes": len(scenes),
            "records": records,
            "summary": summary,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {out_path}")

    del teacher
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
