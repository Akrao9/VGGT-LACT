"""
run_inference.py — run VGGT-TTT on images or video.

Examples:
    # On a video
    python scripts/run_inference.py --input my_video.mp4 --fps 2

    # On a folder of images
    python scripts/run_inference.py --input ./frames/ --ext jpg

    # On a single image
    python scripts/run_inference.py --input photo.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.io_utils import torch_load_checkpoint
from model.vggt_ttt import VGGT_TTT
from pipeline.input_pipeline import prepare_input


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def save_point_cloud(
    point_map: torch.Tensor,
    conf: torch.Tensor,
    out_path: str,
    colors: torch.Tensor = None,
    conf_threshold: float = 0.5,
    max_points: int = 2_000_000,
):
    """Save point cloud as a colored .ply file when RGB frames are available."""
    # point_map: (B, N, H, W, 3)
    # conf: (B, N, H, W)
    pts = point_map[0].reshape(-1, 3).cpu().float().numpy()
    c   = conf[0].reshape(-1).cpu().float().numpy()

    mask = c > conf_threshold
    pts = pts[mask]
    rgb = None
    if colors is not None:
        # colors: (B, N, 3, H, W), in [0, 1]
        rgb = colors[0].permute(0, 2, 3, 1).reshape(-1, 3).cpu().float().numpy()
        rgb = np.clip(rgb[mask] * 255.0, 0, 255).astype(np.uint8)

    if max_points and len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
        pts = pts[keep]
        if rgb is not None:
            rgb = rgb[keep]

    # Write PLY
    with open(out_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, color in zip(pts, rgb):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

    print(f"Saved {len(pts)} points to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="VGGT-TTT inference")
    parser.add_argument("--input",   required=True, help="video, image, or image folder")
    parser.add_argument("--fps",     type=float, default=2.0,  help="fps for video extraction")
    parser.add_argument("--max_frames", type=int, default=300, help="max frames to use")
    parser.add_argument("--chunk_size", type=int, default=1,   help="frames per LaCT chunk")
    parser.add_argument("--ext",     default="jpg", help="image extension if folder given")
    parser.add_argument("--recursive", action="store_true", help="recursively find images in folder input")
    parser.add_argument("--out",     default="./output", help="output directory")
    parser.add_argument("--checkpoint", default=None, help="optional VGGT-TTT checkpoint to load")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=2_000_000, help="cap exported PLY size; 0 disables cap")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Resolve input ---
    source = args.input
    if os.path.isdir(source):
        base = Path(source)
        candidates = base.rglob("*") if args.recursive else base.glob("*")
        image_paths = sorted(p for p in candidates if p.suffix in IMAGE_EXTS)
        image_paths = [str(p) for p in image_paths][: args.max_frames]
        print(f"Found {len(image_paths)} images in {source}")
        if not image_paths:
            raise RuntimeError(
                f"No images found in {source}. "
                "Use --recursive for nested scene folders, or point --input at the image folder."
            )
        source = image_paths

    # --- Load model ---
    print("Loading VGGT-TTT...")
    device_type = torch.device(args.device).type
    if args.precision == "fp32" or device_type != "cuda":
        dtype = None
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        major, _ = torch.cuda.get_device_capability(torch.device(args.device))
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    model = VGGT_TTT.from_pretrained(
        chunk_size=args.chunk_size,
    )
    model = model.to(args.device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch_load_checkpoint(args.checkpoint, map_location="cpu")
        if all(k.startswith("aggregator.lact_blocks.") for k in state):
            model.load_lact_state_dict(state, strict=True)
        else:
            model.load_state_dict(state)
    model.eval()

    # --- Prepare input ---
    images = prepare_input(
        source,
        fps=args.fps,
        max_frames=args.max_frames,
        device=args.device,
    )
    print(f"Input shape: {images.shape}")  # (1, N, 3, H, W)

    # --- Run inference ---
    print("Running inference...")
    with torch.no_grad():
        amp_context = (
            torch.amp.autocast(device_type=device_type, dtype=dtype)
            if device_type == "cuda" and dtype is not None
            else torch.no_grad()
        )
        with amp_context:
            outputs = model(images, chunk_size=args.chunk_size)

    # --- Print results ---
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")

    # --- Save outputs ---
    if "world_points" in outputs and "world_points_conf" in outputs:
        ply_path = os.path.join(args.out, "reconstruction.ply")
        save_point_cloud(
            outputs["world_points"],
            outputs["world_points_conf"],
            ply_path,
            colors=images,
            conf_threshold=args.conf_threshold,
            max_points=args.max_points,
        )

    if "extrinsic" in outputs:
        poses = outputs["extrinsic"][0].cpu().float().numpy()
        np.save(os.path.join(args.out, "camera_extrinsics.npy"), poses)
        print(f"Saved {len(poses)} camera poses")

    if "depth" in outputs:
        depth = outputs["depth"][0].cpu().float().numpy()
        np.save(os.path.join(args.out, "depth.npy"), depth)
        print(f"Saved depth: {depth.shape}")

    if "world_points_conf" in outputs:
        conf = outputs["world_points_conf"][0].cpu().float().numpy()
        np.save(os.path.join(args.out, "world_points_conf.npy"), conf)
        print(f"Saved world point confidence: {conf.shape}")

    print(f"\nDone. Outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
