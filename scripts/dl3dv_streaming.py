"""
DL3DV-ALL Hugging Face download + bounded-disk streaming.

Shared by ``train_tttlrm_decoder_dataset.py`` and ``finetune.py`` so LaCT training
can stream 480P (or other) scenes without a pre-downloaded ``--data_dir``.
"""

from __future__ import annotations

import queue
import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def find_nerfstudio_transforms(image_dir: str) -> "Path | None":
    p = Path(image_dir)
    for cand in ("transforms.json", "transforms_train.json"):
        if (p / cand).exists():
            return p / cand
        if (p.parent / cand).exists():
            return p.parent / cand
    return None


def discover_images(image_dir: str, recursive: bool = False, max_frames: int = 0) -> list[str]:
    p = Path(image_dir)
    pattern_jpg = "**/*.jpg" if recursive else "*.jpg"
    pattern_png = "**/*.png" if recursive else "*.png"
    found = sorted([str(x) for x in p.glob(pattern_jpg)] + [str(x) for x in p.glob(pattern_png)])
    if not found:
        raise RuntimeError(f"no images under {image_dir}")
    return found if max_frames <= 0 else found[:max_frames]


@dataclass
class SceneInput:
    name: str
    image_dir: str
    transforms_path: str


def scene_input_from_root(root: Path) -> SceneInput | None:
    candidates = [
        root / "nerfstudio" / "images_2",
        root / "nerfstudio" / "images",
        root / "colmap" / "images_2",
        root / "colmap" / "images_4",
        root / "colmap" / "images_8",
        root / "colmap" / "images",
        root / "images_2",
        root / "images_4",
        root / "images_8",
        root / "images",
        root,
    ]
    for image_dir in candidates:
        if not image_dir.exists() or not image_dir.is_dir():
            continue
        tf = find_nerfstudio_transforms(str(image_dir))
        if tf is None:
            continue
        try:
            discover_images(str(image_dir), recursive=False, max_frames=1)
        except RuntimeError:
            continue
        return SceneInput(root.name, str(image_dir), str(tf))
    return None


def build_dl3dv_download_items(args: Any) -> list[dict]:
    from scripts.download import get_download_list, resolution2repo, verify_access

    repo = resolution2repo[args.dl3dv_resolution]
    if not verify_access(repo):
        raise RuntimeError(
            f"No access to {repo}. Open https://huggingface.co/datasets/{repo}, "
            "accept the terms, then run huggingface-cli login."
        )
    items = get_download_list(
        args.dl3dv_subset,
        "",
        args.dl3dv_resolution,
        "images+poses",
        args.dl3dv_local_dir,
    )
    if args.scene_offset:
        items = items[args.scene_offset :]
    if args.max_scenes > 0:
        items = items[: args.max_scenes]
    if not items:
        raise RuntimeError("No DL3DV download items selected")
    return items


def downloaded_scene_root(item: dict, local_dir: str) -> Path:
    rel_path = Path(item["rel_path"])
    return Path(local_dir) / rel_path.parent / rel_path.stem


def download_scene_input(item: dict, args: Any) -> tuple[SceneInput, Path]:
    from scripts.download import download

    ok = download([item], args.dl3dv_local_dir, is_clean_cache=False)
    if not ok:
        raise RuntimeError(f"Download failed for {item['rel_path']}")
    scene_root = downloaded_scene_root(item, args.dl3dv_local_dir)
    scene = scene_input_from_root(scene_root)
    if scene is None:
        image_count = len(list(scene_root.rglob("*.jpg"))) + len(list(scene_root.rglob("*.png")))
        raise RuntimeError(
            f"Downloaded {scene_root}, found {image_count} images, but no supported camera metadata. "
            "Expected transforms.json under nerfstudio/images_* or colmap/images_*."
        )
    return scene, scene_root


class StreamingDl3dvPrefetcher:
    """
    Keep N DL3DV scenes on disk while the GPU consumes the oldest ready scene.
    ``buffer_size`` must be >= 2.
    """

    def __init__(self, download_items: list[dict], args: Any, buffer_size: int, num_workers: int):
        if buffer_size < 2:
            raise ValueError("buffer_size must be >= 2")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.download_items = download_items
        self.args = args
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._cursor = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _take_next_item(self) -> dict:
        with self._lock:
            n = len(self.download_items)
            if self._cursor % n == 0:
                random.shuffle(self.download_items)
            item = self.download_items[self._cursor % n]
            self._cursor += 1
            return item

    def _put_ok(self, scene: SceneInput, root: Path) -> None:
        while True:
            if self._stop.is_set():
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
                return
            try:
                self._q.put(("ok", scene, root), timeout=0.5)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._take_next_item()
            guess = downloaded_scene_root(item, self.args.dl3dv_local_dir)
            try:
                scene, root = download_scene_input(item, self.args)
            except Exception as exc:
                print(f"WARNING: prefetch download failed {item.get('rel_path')}: {exc}")
                if guess.exists():
                    shutil.rmtree(guess, ignore_errors=True)
                continue
            if self._stop.is_set():
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
                return
            self._put_ok(scene, root)

    def start(self) -> None:
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker, name="dl3dv-prefetch", daemon=True)
            self._threads.append(t)
            t.start()

    def acquire(self) -> tuple[SceneInput, Path]:
        while True:
            kind, scene, root = self._q.get()
            if kind == "ok":
                return scene, root

    def shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=60.0)
        while True:
            try:
                kind, scene, root = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == "ok" and root is not None and Path(root).exists():
                print(f"Prefetch shutdown: removing {root}")
                shutil.rmtree(root, ignore_errors=True)
