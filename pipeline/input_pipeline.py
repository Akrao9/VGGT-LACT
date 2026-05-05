"""
Input pipeline — video and image → preprocessed tensors → model → 3D outputs.

Handles:
  - Single image or list of images
  - Video file (extracts frames with ffmpeg)
  - Automatic resizing and normalization (matches VGGT's preprocessing)
  - Memory-efficient chunked loading for long videos
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


# VGGT uses these normalization stats (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# VGGT patch size and expected input resolution
VGGT_PATCH_SIZE = 14
VGGT_DEFAULT_RES = 518  # shortest side


def extract_frames_from_video(
    video_path: str,
    fps: float = 2.0,
    max_frames: int = 500,
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Extract frames from a video using ffmpeg.

    fps: frames per second to sample (2.0 = 1 frame every 0.5s)
    max_frames: hard cap on extracted frames
    output_dir: where to save frames (temp dir if None)

    Returns list of frame file paths in order.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vggt_ttt_frames_")
    os.makedirs(output_dir, exist_ok=True)

    frame_pattern = os.path.join(output_dir, "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",           # JPEG quality (2 = high)
        "-frames:v", str(max_frames),
        frame_pattern,
        "-y",                  # overwrite
        "-loglevel", "error",
    ]

    print(f"Extracting frames from {video_path} at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    print(f"Extracted {len(frames)} frames")
    return [str(f) for f in frames]


def preprocess_image(
    img: Union[str, Image.Image, np.ndarray],
    target_size: int = VGGT_DEFAULT_RES,
) -> torch.Tensor:
    """
    Preprocess one image for VGGT-TTT.

    Resizes so shortest side = target_size, keeps aspect ratio.
    Crops to make both dims divisible by patch_size.
    Normalizes with ImageNet stats.

    Returns: (3, H, W) float tensor
    """
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    # Resize shortest side to target_size
    W, H = img.size
    scale = target_size / min(W, H)
    new_W, new_H = int(W * scale), int(H * scale)
    img = img.resize((new_W, new_H), Image.LANCZOS)

    # Crop to be divisible by patch_size
    crop_W = (new_W // VGGT_PATCH_SIZE) * VGGT_PATCH_SIZE
    crop_H = (new_H // VGGT_PATCH_SIZE) * VGGT_PATCH_SIZE
    img = img.crop((0, 0, crop_W, crop_H))

    # To tensor [0,1] — do NOT normalize here.
    # The aggregator normalizes with ImageNet mean/std internally,
    # just like VGGT's original aggregator.forward() does.
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1)  # (3, H, W)

    return t


def load_images(
    image_paths: List[str],
    target_size: int = VGGT_DEFAULT_RES,
) -> torch.Tensor:
    """
    Load and preprocess a list of image paths.

    Returns: (N, 3, H, W) — all images same size (resized to first image's output size)
    """
    frames = []
    ref_size = None

    for path in image_paths:
        t = preprocess_image(path, target_size)
        if ref_size is None:
            ref_size = t.shape[1:]  # (H, W)
        elif t.shape[1:] != ref_size:
            # Resize to match first frame (needed for batching)
            t = F.interpolate(
                t.unsqueeze(0), size=ref_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        frames.append(t)

    return torch.stack(frames, dim=0)  # (N, 3, H, W)


def prepare_input(
    source: Union[str, List[str]],
    fps: float = 2.0,
    max_frames: int = 300,
    target_size: int = VGGT_DEFAULT_RES,
    device: str = "cuda",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Main entry point. Accepts:
      - A video file path  (str ending in .mp4 / .avi / .mov etc.)
      - A single image path (str)
      - A list of image paths (List[str])

    Returns: (B, N, 3, H, W) ready for model.forward()
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    temp_frame_dir: Optional[str] = None
    if isinstance(source, str):
        ext = Path(source).suffix.lower()
        if ext in video_extensions:
            # Extract frames into a temp dir we own and will clean up after load.
            temp_frame_dir = tempfile.mkdtemp(prefix="vggt_ttt_frames_")
            image_paths = extract_frames_from_video(
                source, fps=fps, max_frames=max_frames, output_dir=temp_frame_dir,
            )
        else:
            # Single image
            image_paths = [source]
    else:
        image_paths = source

    print(f"Loading {len(image_paths)} frames...")
    try:
        frames = load_images(image_paths, target_size=target_size)  # (N, 3, H, W)
    finally:
        if temp_frame_dir is not None:
            shutil.rmtree(temp_frame_dir, ignore_errors=True)

    # Add batch dimension
    frames = frames.unsqueeze(0)  # (1, N, 3, H, W)
    frames = frames.to(device)

    return frames
