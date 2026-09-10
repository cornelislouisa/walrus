"""Constant mean-field prediction baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from walrus.baselines.utils import restore_inflated_spatial, squeeze_inflated_spatial


class MeanFieldWrapper(nn.Module):
    """Predict a constant time- and ensemble-averaged flow field.

    During training, maintains an EMA of the spatial mean of context frames
    (in RevIN-normalized space). At inference, always returns that field.

    Use ``trainer.prediction_type=full``. Includes a dummy parameter so Adam
    still constructs cleanly.
    """

    def __init__(
        self,
        time_history: int = 10,
        time_future: int = 1,
        in_channels: int = 2,
        out_channels: int = 2,
        momentum: float = 0.05,
        spatial: bool = True,
        use_all_context_frames: bool = True,
        n_states=None,
        **kwargs,
    ):
        super().__init__()
        del n_states, kwargs
        self.causal_in_time = False
        self.time_history = time_history
        self.time_future = time_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.momentum = float(momentum)
        self.spatial = bool(spatial)
        self.use_all_context_frames = bool(use_all_context_frames)
        self._dummy = nn.Parameter(torch.zeros(1))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        # ``mean_field`` is registered on the first training update.

    @property
    def _mean_initialized(self) -> bool:
        return (
            "mean_field" in self._buffers
            and self.mean_field is not None
            and self.mean_field.numel() > 0
        )

    def _batch_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``(T, B, C, H, W)`` to a mean field broadcastable as ``(1,1,C,*,*)``."""
        fields = x[:, :, : self.in_channels]
        if not self.use_all_context_frames:
            fields = fields[-1:]
        if self.spatial:
            return fields.mean(dim=(0, 1), keepdim=True)  # (1, 1, C, H, W)
        return fields.mean(dim=(0, 1, 3, 4), keepdim=True)  # (1, 1, C, 1, 1)

    def _update_mean(self, batch_mean: torch.Tensor) -> None:
        if not self._mean_initialized:
            self.register_buffer("mean_field", batch_mean.detach().clone())
            self.n_updates.fill_(1)
            return
        if self.mean_field.shape != batch_mean.shape:
            raise ValueError(
                f"Mean-field shape changed: had {tuple(self.mean_field.shape)}, "
                f"got {tuple(batch_mean.shape)}"
            )
        self.mean_field.lerp_(batch_mean.detach(), self.momentum)
        self.n_updates += 1

    def set_mean_field(self, mean_field: torch.Tensor, *, n_updates: int = 1) -> None:
        """Install a precomputed normalized mean for frozen evaluation."""
        value = mean_field.detach().clone()
        if self._mean_initialized:
            self.mean_field = value
        else:
            self.register_buffer("mean_field", value)
        self.n_updates.fill_(int(n_updates))

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
            ``(T_out, B, C_out, H, W[, 1])`` constant mean field.
        """
        del state_labels, bcs, metadata, proj_axes, return_att, kwargs
        x, n_squeezed = squeeze_inflated_spatial(x)
        if x.ndim != 5:
            raise ValueError(
                f"MeanFieldWrapper expects (T, B, C, H, W); got {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("MeanFieldWrapper needs at least one input frame")

        batch_mean = self._batch_mean(x)
        if train and self.training:
            self._update_mean(batch_mean)

        field = self.mean_field if self._mean_initialized else batch_mean
        b = x.shape[1]
        c = self.out_channels
        pred = field[:, :, :c].expand(self.time_future, b, c, -1, -1).clone()
        return restore_inflated_spatial(pred, n_squeezed)
