"""Shared helpers for training/eval scripts (Blackwell-friendly defaults)."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Optional

import torch


def configure_cuda_memory_allocator() -> None:
    """Reduce VRAM fragmentation from many alloc/free cycles (training, streaming scenes).

    Uses PyTorch's recommended default when unset: ``expandable_segments:True``.
    See https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def prefer_memory_efficient_sdp(
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
    enable_math: bool = True,
) -> None:
    """Prefer FlashAttention-style and memory-efficient SDP for ``scaled_dot_product_attention``.

    VGGT frame-wise blocks use PyTorch SDPA; enabling these backends cuts attention
    workspace (often the difference between OOM and runnable) without changing numerics
    versus the math fallback when a kernel applies.
    """
    if not torch.cuda.is_available():
        return
    cuda = getattr(torch.backends, "cuda", None)
    if cuda is None:
        return
    for attr, val in (
        ("enable_flash_sdp", enable_flash),
        ("enable_mem_efficient_sdp", enable_mem_efficient),
        ("enable_math_sdp", enable_math),
    ):
        fn = getattr(cuda, attr, None)
        if callable(fn):
            try:
                fn(val)
            except (RuntimeError, TypeError):
                pass


def configure_cuda_backend(allow_tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """Enable TF32 + cuDNN autotune. Safe on SM>=80 (Ampere/Hopper/Blackwell).

    Also enables memory-efficient SDP backends for attention in dependencies (e.g. VGGT)
    and sets a fragmentation-friendly CUDA allocator default when unset.
    """
    configure_cuda_memory_allocator()
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    prefer_memory_efficient_sdp()


def resolve_amp_dtype(device: str, precision: str) -> Optional[torch.dtype]:
    if precision == "fp32" or torch.device(device).type != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast_context(device: str, dtype: Optional[torch.dtype]):
    if torch.device(device).type == "cuda" and dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def maybe_compile(module: torch.nn.Module, enabled: bool, mode: str = "default") -> torch.nn.Module:
    """torch.compile guard. ``default`` mode avoids Inductor's max-autotune path
    which currently crashes on Blackwell (NoValidChoicesError on bias_addmm).
    cudagraphs are implicitly off because LaCT mutates buffers between calls."""
    if not enabled:
        return module
    try:
        return torch.compile(module, mode=mode, dynamic=True)
    except Exception as e:
        print(f"torch.compile failed ({e}); running eager.")
        return module


def unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_orig_mod", module)


def fused_adamw(params, lr: float, weight_decay: float = 0.05, betas=(0.9, 0.95)) -> torch.optim.AdamW:
    """AdamW with fused=True (Blackwell). Falls back to foreach if fused unsupported."""
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, foreach=True)


def set_inductor_cache(path: str = "/tmp/torchinductor_cache") -> None:
    os.makedirs(path, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", path)
