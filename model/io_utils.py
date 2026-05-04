"""Small helpers for safe checkpoint I/O."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def torch_load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor] | Any:
    """Load a torch checkpoint, preferring ``weights_only=True``.

    Falls back to a full unpickle only for two known-safe reasons:
    - ``TypeError``: older PyTorch that doesn't accept the ``weights_only`` kwarg.
    - ``pickle.UnpicklingError`` / ``RuntimeError`` containing "allowlist": the file
      holds legitimate non-tensor metadata (e.g. ints in a resume dict).

    All other exceptions propagate so that a blocked malicious pickle is never
    silently retried as an unsafe load.
    """
    path = str(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch < 1.13 has no weights_only kwarg.
        return torch.load(path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if "allowlist" not in str(exc) and not isinstance(exc, pickle.UnpicklingError):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)
