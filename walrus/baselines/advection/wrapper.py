"""Walrus trainer adapter for velocity self-advection baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from walrus.baselines.utils import restore_inflated_spatial, squeeze_inflated_spatial

from .model import advect_velocity


class AdvectionWrapper(nn.Module):
    """Evolve the last context velocity one step by self-advection.

    Predicts absolute next fields (use ``trainer.prediction_type=full``).
    Includes a unused dummy parameter so Adam still constructs cleanly.
    """

    def __init__(
        self,
        dt: float = 1.0,
        n_steps: int = 1,
        time_history: int = 10,
        time_future: int = 1,
        in_channels: int = 2,
        out_channels: int = 2,
        padding_mode: str = "border",
        n_states=None,
        **kwargs,
    ):
        super().__init__()
        del n_states, kwargs
        self.causal_in_time = False
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.time_history = time_history
        self.time_future = time_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Optimizer requires at least one parameter (unused in forward).
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
        **kwargs,
    ):
        """Args:
            x: ``(T, B, C, H, W)`` or inflated ``(..., 1)`` Walrus input.

        Returns:
            ``(T_out, B, C_out, H, W[, 1])`` with ``T_out == time_future``.
        """
        del state_labels, bcs, metadata, proj_axes, return_att, train, kwargs
        x, n_squeezed = squeeze_inflated_spatial(x)
        if x.ndim != 5:
            raise ValueError(
                f"AdvectionWrapper expects (T, B, C, H, W); got {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("AdvectionWrapper needs at least one input frame")

        # Last context frame; keep configured velocity channels.
        u = x[-1, :, : self.in_channels].contiguous()  # (B, C, H, W)
        for _ in range(self.n_steps):
            u = advect_velocity(u, dt=self.dt, padding_mode=self.padding_mode)
        u = u[:, : self.out_channels]

        # Repeat the one-step field for each requested future frame (autoregressive
        # rollouts re-call forward each step with updated context).
        out = u.unsqueeze(0).expand(self.time_future, -1, -1, -1, -1).clone()
        return restore_inflated_spatial(out, n_squeezed)
