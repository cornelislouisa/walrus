"""Semi-Lagrangian self-advection of a 2D velocity field."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def advect_velocity(
    u: torch.Tensor,
    dt: float = 1.0,
    *,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """One semi-Lagrangian step of self-advection: ``u(x) ← u(x - dt · u(x))``.

    Args:
        u: Velocity ``(B, C, H, W)`` with ``C >= 2``. Channels ``0`` and ``1`` are
            treated as ``(v_x, v_y)`` in pixel units per unit ``dt`` (along ``W`` / ``H``).
            Extra channels (if any) are passively advected by the same flow.
        dt: Integration step in grid-pixel units. With RevIN-normalized velocities
            of O(1), values near ``1`` move by roughly one pixel per step.
        padding_mode: Passed to ``grid_sample`` (``border``, ``zeros``, or ``reflection``).
        align_corners: ``grid_sample`` align_corners flag.

    Returns:
        Advected field with the same shape as ``u``.
    """
    if u.ndim != 4:
        raise ValueError(f"advect_velocity expects (B, C, H, W); got {tuple(u.shape)}")
    if u.shape[1] < 2:
        raise ValueError(f"Need at least 2 channels for (vx, vy); got C={u.shape[1]}")

    b, _, h, w = u.shape
    device = u.device
    dtype = u.dtype

    # Pixel-coordinate mesh (x along W, y along H).
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    vx = u[:, 0]
    vy = u[:, 1]
    sample_x = grid_x - dt * vx
    sample_y = grid_y - dt * vy

    # Map pixel coords -> [-1, 1] for grid_sample.
    if w > 1:
        sample_x = 2.0 * sample_x / (w - 1) - 1.0
    else:
        sample_x = torch.zeros_like(sample_x)
    if h > 1:
        sample_y = 2.0 * sample_y / (h - 1) - 1.0
    else:
        sample_y = torch.zeros_like(sample_y)

    grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(
        u,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
