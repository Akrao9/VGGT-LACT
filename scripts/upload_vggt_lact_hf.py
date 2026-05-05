#!/usr/bin/env python3
"""Upload LaCT stage-1 checkpoint to Hugging Face (optional Hub README).

Requires: pip install huggingface_hub && huggingface-cli login
   (or set HF_TOKEN with write access.)

Usage from repo root:
  python scripts/upload_vggt_lact_hf.py
  python scripts/upload_vggt_lact_hf.py --repo-id akrao9/VGGT-LACT --ckpt ./custom.pt
  python scripts/upload_vggt_lact_hf.py --readme ./MY_HF_README.md   # also refresh Hub model card
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="akrao9/VGGT-LACT")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=root / "vggt_ttt_lact_stage1.pt",
        help="LaCT-only checkpoint path",
    )
    p.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="If set, upload this file as the Hub README.md (model card)",
    )
    args = p.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if args.readme is not None and not args.readme.is_file():
        raise SystemExit(f"README not found: {args.readme}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.ckpt),
        path_in_repo="vggt_ttt_lact_stage1.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    print("  - vggt_ttt_lact_stage1.pt")
    if args.readme is not None:
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("  - README.md")


if __name__ == "__main__":
    main()
