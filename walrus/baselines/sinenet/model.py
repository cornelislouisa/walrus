"""SineNet (dual / disentangled) — vendored from AIRS OpenPDE and cleaned for Walrus.

Source: https://github.com/divelab/AIRS/blob/main/OpenPDE/SineNet/pdearena/pdearena/modules/sinenet_dual.py
Paper: Zhang, Helwig, Ji — SineNet (ICLR 2024), arXiv:2403.19507
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    registry = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in registry:
        raise NotImplementedError(f"Activation {name} not implemented")
    return registry[name]()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding_mode,
        num_groups=1,
        norm: bool = True,
        activation="gelu",
    ) -> None:
        super().__init__()
        self.activation = _get_activation(activation)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        if norm:
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in0_channels,
        padding_mode,
        num_blocks,
        residual,
        pool,
        avg,
        num_groups=1,
        norm: bool = True,
        activation="gelu",
        first=False,
        disentangle=True,
    ) -> None:
        super().__init__()
        self.channels = in_channels, out_channels
        self.residual = residual
        self.num_blocks = num_blocks
        self.first = first
        self.disentangle = disentangle
        if not pool:
            raise NotImplementedError(
                "SineNet Down currently only supports pooling downsampling "
                "(down_pool=True)."
            )
        self.down = nn.AvgPool2d(2) if avg else nn.MaxPool2d(2)
        self.conv = nn.ModuleList()
        for block in range(num_blocks):
            in_c = (
                (in_channels + in0_channels * (not first and disentangle))
                if block == 0
                else out_channels
            )
            self.conv.append(
                ConvBlock(in_c, out_channels, padding_mode, num_groups, norm, activation)
            )
        if residual:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, dx: torch.Tensor, h: torch.Tensor, down=True):
        if down:
            if self.disentangle:
                dx = self.down(dx)
            h = self.down(h)
            h0 = h.clone()
            if not self.first and self.disentangle:
                h = torch.cat([h, dx], dim=1)
        else:
            h0 = h.clone()
        for block in range(self.num_blocks):
            h = self.conv[block](h)
            if self.residual:
                h = h + (self.shortcut(h0) if block == 0 else h0)
                h0 = h.clone()
        return dx, h


class CircularInterpolate(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="circular")
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
        return x[..., 4:-4, 4:-4]


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult,
        padding_mode,
        num_blocks,
        residual,
        interp,
        interp_mode,
        num_groups=1,
        norm: bool = True,
        activation="gelu",
    ) -> None:
        super().__init__()
        del mult  # kept for API parity with upstream
        self.channels = in_channels, out_channels
        self.residual = residual
        self.num_blocks = num_blocks
        if not interp:
            raise NotImplementedError(
                "SineNet Up currently only supports interpolation upsampling "
                "(up_interpolation=True)."
            )
        in0_conv = in_channels + out_channels
        in0_resid = in_channels
        self.up = (
            CircularInterpolate(mode=interp_mode)
            if padding_mode == "circular"
            else partial(
                torch.nn.functional.interpolate, scale_factor=2, mode=interp_mode
            )
        )
        self.conv = nn.ModuleList()
        for block in range(num_blocks):
            self.conv.append(
                ConvBlock(
                    in0_conv if block == 0 else out_channels,
                    out_channels,
                    padding_mode,
                    num_groups,
                    norm,
                    activation,
                )
            )
        if residual:
            self.shortcut = nn.Conv2d(in0_resid, out_channels, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, up=True):
        if up:
            h = self.up(x1)
            x = h.clone()
            h = torch.cat([x2, h], dim=1)
        else:
            h = x1
            x = h.clone()
        for block in range(self.num_blocks):
            h = self.conv[block](h)
            if self.residual:
                h = h + (self.shortcut(x) if block == 0 else x)
                x = h.clone()
        return h


class SineNet(nn.Module):
    """Multi-wave U-Net with disentangled dual-branch downsampling (SineNet).

    Args follow the AIRS OpenPDE ``sinenet_dual`` API. Input/output tensors are
    ``(B, T, C, H, W)``. Spatial height/width must be divisible by ``2**num_layers``.
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        padding_mode: str,
        activation="gelu",
        num_layers=4,
        num_waves=2,
        num_blocks=1,
        norm=True,
        mult=2,
        residual=True,
        wave_residual=True,
        disentangle=True,
        down_pool=True,
        avg_pool=True,
        up_interpolation=True,
        interpolation_mode="bicubic",
        par1=None,
    ) -> None:
        super().__init__()
        del par1
        if num_layers != 4:
            raise ValueError(
                "This SineNet port hard-codes a 4-level U-Net; num_layers must be 4."
            )
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.n_input_channels = (
            n_input_scalar_components + n_input_vector_components * 2
        )
        self.n_output_channels = (
            n_output_scalar_components + n_output_vector_components * 2
        )
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.wave_residual = wave_residual
        self.num_layers = num_layers
        self.num_waves = num_waves

        insize = time_history * self.n_input_channels
        n_channels = hidden_channels
        self.image_proj = nn.Conv2d(
            insize, n_channels, kernel_size=3, padding=1, padding_mode=padding_mode
        )

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        down_args = dict(
            norm=norm,
            activation=activation,
            residual=residual,
            padding_mode=padding_mode,
            num_blocks=num_blocks,
            disentangle=disentangle,
            pool=down_pool,
            avg=avg_pool,
            in0_channels=n_channels,
        )
        up_args = dict(
            norm=norm,
            activation=activation,
            mult=mult,
            residual=residual,
            padding_mode=padding_mode,
            num_blocks=num_blocks,
            interp=up_interpolation,
            interp_mode=interpolation_mode,
        )
        for _ in range(self.num_waves):
            self.down.append(
                nn.ModuleList(
                    [
                        Down(n_channels, int(n_channels * mult), **down_args, first=True),
                        Down(
                            int(n_channels * mult),
                            int(n_channels * mult**2),
                            **down_args,
                        ),
                        Down(
                            int(n_channels * mult**2),
                            int(n_channels * mult**3),
                            **down_args,
                        ),
                        Down(
                            int(n_channels * mult**3),
                            int(n_channels * mult**4),
                            **down_args,
                        ),
                    ]
                )
            )
            self.up.append(
                nn.ModuleList(
                    [
                        Up(
                            int(n_channels * mult**4),
                            int(n_channels * mult**3),
                            **up_args,
                        ),
                        Up(
                            int(n_channels * mult**3),
                            int(n_channels * mult**2),
                            **up_args,
                        ),
                        Up(
                            int(n_channels * mult**2),
                            int(n_channels * mult),
                            **up_args,
                        ),
                        Up(int(n_channels * mult), n_channels, **up_args),
                    ]
                )
            )

        out_channels = time_future * self.n_output_channels
        self.final = nn.Conv2d(
            n_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(B, T, C, H, W)``.

        Returns:
            ``(B, T_out, C_out, H, W)`` with ``T_out == time_future``.
        """
        if x.dim() != 5:
            raise ValueError(f"SineNet expects (B, T, C, H, W); got {tuple(x.shape)}")
        _, _, _, h, w = x.shape
        scale = 2**self.num_layers
        if h % scale != 0 or w % scale != 0:
            raise ValueError(
                f"Spatial size {(h, w)} must be divisible by {scale} "
                f"(2**num_layers with num_layers={self.num_layers})."
            )

        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        x = self.image_proj(x)

        for stack in range(self.num_waves):
            x0 = x.clone()
            xs = [x]
            dx = x
            for i in range(self.num_layers):
                dx, h_feat = self.down[stack][i](dx, xs[-1])
                xs.append(h_feat)
            x = xs.pop(-1)
            for i in range(self.num_layers):
                x = self.up[stack][i](x, xs.pop(-1))
            if self.wave_residual:
                x = x0 + x

        x = self.final(x)
        return x.reshape(
            orig_shape[0],
            self.time_future,
            self.n_output_channels,
            *orig_shape[3:],
        )
