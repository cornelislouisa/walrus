"""Walrus trainer adapter for SineNet."""

from __future__ import annotations

import torch.nn as nn
from einops import rearrange

from walrus.baselines.utils import restore_inflated_spatial, squeeze_inflated_spatial

from .model import SineNet


class SineNetWrapper(nn.Module):
    """Wraps :class:`SineNet` to the Walrus ``forward(x, state_labels, bcs, metadata)`` API.

    Walrus tensors are ``(T, B, C, H, W)`` (optionally inflated with a trailing
    singleton spatial dim) and already RevIN-normalized by the trainer.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_waves: int = 2,
        num_layers: int = 4,
        num_blocks: int = 1,
        mult: float = 2,
        time_history: int = 10,
        time_future: int = 1,
        in_channels: int = 2,
        out_channels: int = 2,
        padding_mode: str = "zeros",
        activation: str = "gelu",
        norm: bool = True,
        residual: bool = True,
        wave_residual: bool = True,
        disentangle: bool = True,
        down_pool: bool = True,
        avg_pool: bool = True,
        up_interpolation: bool = True,
        interpolation_mode: str = "bicubic",
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

        self.inner_model = SineNet(
            n_input_scalar_components=in_channels,
            n_input_vector_components=0,
            n_output_scalar_components=out_channels,
            n_output_vector_components=0,
            time_history=time_history,
            time_future=time_future,
            hidden_channels=hidden_channels,
            padding_mode=padding_mode,
            activation=activation,
            num_layers=num_layers,
            num_waves=num_waves,
            num_blocks=num_blocks,
            norm=norm,
            mult=mult,
            residual=residual,
            wave_residual=wave_residual,
            disentangle=disentangle,
            down_pool=down_pool,
            avg_pool=avg_pool,
            up_interpolation=up_interpolation,
            interpolation_mode=interpolation_mode,
        )

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
        del state_labels, bcs, metadata, proj_axes, return_att, train, kwargs
        x, n_squeezed = squeeze_inflated_spatial(x)
        if x.ndim != 5:
            raise ValueError(
                f"SineNetWrapper expects 2D inputs (T, B, C, H, W); got shape {tuple(x.shape)}"
            )
        if x.shape[0] > self.time_history:
            x = x[-self.time_history :]
        elif x.shape[0] < self.time_history:
            raise ValueError(
                f"SineNetWrapper expected time_history={self.time_history}, got T={x.shape[0]}"
            )

        # (T, B, C, H, W) -> (B, T, C, H, W); keep configured field channels.
        x_in = rearrange(x, "t b c h w -> b t c h w")[:, :, : self.in_channels]
        preds = self.inner_model(x_in)  # (B, T_out, C_out, H, W)
        out = rearrange(preds, "b t c h w -> t b c h w")
        return restore_inflated_spatial(out, n_squeezed)
