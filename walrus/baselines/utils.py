"""Shared helpers for Walrus baseline adapters."""

from __future__ import annotations

import torch


def squeeze_inflated_spatial(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Drop trailing size-1 spatial dims from inflated Well tensors.

    ``BatchInflatedWellDataset`` defaults to ``pad_cartesian_data_to_d=3``, so
    native 2D fields arrive as ``(T, B, C, H, W, 1)``. 2D baselines need the
    singleton stripped; callers should restore it on the prediction with
    :func:`restore_inflated_spatial` so shapes match ``y_ref``.
    """
    n_squeezed = 0
    while x.ndim > 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
        n_squeezed += 1
    return x, n_squeezed


def restore_inflated_spatial(x: torch.Tensor, n_squeezed: int) -> torch.Tensor:
    """Undo :func:`squeeze_inflated_spatial` on model outputs."""
    for _ in range(n_squeezed):
        x = x.unsqueeze(-1)
    return x
