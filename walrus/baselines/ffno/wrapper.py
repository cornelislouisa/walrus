"""Walrus trainer adapter for Factorized FNO."""

from __future__ import annotations

import torch.nn as nn
from einops import rearrange

from walrus.baselines.utils import restore_inflated_spatial, squeeze_inflated_spatial

from .model import FFNO


class FFNOWrapper(nn.Module):
    """Wraps :class:`FFNO` to the Walrus ``forward(x, state_labels, bcs, metadata)`` API.

    Walrus tensors are ``(T, B, C, H, W)`` (optionally with a trailing inflated
    singleton spatial dim) and already RevIN-normalized by the trainer.
    Internal FFNO normalization is therefore disabled by default.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        time_history: int = 10,
        time_future: int = 1,
        in_channels: int = 2,
        out_channels: int = 2,
        dropout: float = 0.0,
        in_dropout: float = 0.0,
        share_weight: bool = False,
        share_fork: bool = False,
        factor: int = 2,
        ff_weight_norm: bool = False,
        n_ff_layers: int = 2,
        gain: float = 1.0,
        layer_norm: bool = False,
        mode: str = "full",
        use_position: bool = True,
        use_fourier_position: bool = False,
        k_max: float | None = None,
        should_normalize: bool = False,
        noise_std: float = 0.0,
        n_states=None,
        **kwargs,
    ):
        super().__init__()
        del n_states, kwargs  # Hydra may inject unused keys
        self.causal_in_time = False
        self.time_history = time_history
        self.time_future = time_future
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inner_model = FFNO(
            n_input_scalar_components=in_channels,
            n_input_vector_components=0,
            n_output_scalar_components=out_channels,
            n_output_vector_components=0,
            time_history=time_history,
            time_future=time_future,
            modes=modes,
            width=width,
            dropout=dropout,
            in_dropout=in_dropout,
            n_layers=n_layers,
            share_weight=share_weight,
            share_fork=share_fork,
            factor=factor,
            ff_weight_norm=ff_weight_norm,
            n_ff_layers=n_ff_layers,
            gain=gain,
            layer_norm=layer_norm,
            mode=mode,
            use_position=use_position,
            use_fourier_position=use_fourier_position,
            k_max=k_max,
            should_normalize=should_normalize,
            noise_std=noise_std,
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
        """Args:
            x: ``(T, B, C, H, W)`` or inflated ``(..., 1)`` Walrus input.

        Returns:
            ``(T_out, B, C_out, H, W[, 1])`` with ``T_out == time_future``.
        """
        del state_labels, bcs, metadata, proj_axes, return_att, train, kwargs
        x, n_squeezed = squeeze_inflated_spatial(x)
        if x.ndim != 5:
            raise ValueError(
                f"FFNOWrapper expects 2D inputs (T, B, C, H, W); got shape {tuple(x.shape)}"
            )
        # Use the trailing time_history frames if a longer context is provided.
        if x.shape[0] > self.time_history:
            x = x[-self.time_history :]
        elif x.shape[0] < self.time_history:
            raise ValueError(
                f"FFNOWrapper expected time_history={self.time_history}, got T={x.shape[0]}"
            )

        # (T, B, C, H, W) -> (B, T, H, W, C)
        x_ffno = rearrange(x, "t b c h w -> b t h w c")
        # Keep only the configured field channels if constants were concatenated.
        x_ffno = x_ffno[..., : self.in_channels]
        preds = self.inner_model(x_ffno)  # (B, T_out, H, W, C_out)
        out = rearrange(preds, "b t h w c -> t b c h w")
        return restore_inflated_spatial(out, n_squeezed)
