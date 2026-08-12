from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath

from ..shared_utils.mlps import MLP
from ..shared_utils.normalization import RMSGroupNorm
from ..shared_utils.position_biases import (
    RelativePositionBias,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


class AxialAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        max_d=3,
        weight_tied_axes=True,
        gradient_checkpointing=False,
        noise_cond_dim: int = 0,
        norm_cond_dim: int = 0,  # accepted for API parity; AdaLN uses noise_cond_dim
        norm_layer: Callable = RMSGroupNorm,
    ):
        del norm_cond_dim  # unused in axial spatial block (AdaLN-only)
        super().__init__()
        self.num_heads = num_heads
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        self.norm1 = norm_layer(num_heads, hidden_dim, affine=True)
        self.norm2 = norm_layer(num_heads, hidden_dim, affine=True)
        self.gamma_att = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.gamma_mlp = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        self.input_heads = nn.ModuleList(
            [nn.Conv3d(hidden_dim, 3 * hidden_dim, 1) for _ in range(max_d)]
        )
        self.output_heads = nn.ModuleList(
            [nn.Conv3d(hidden_dim, hidden_dim, 1) for _ in range(max_d)]
        )
        self.qnorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim // num_heads) for _ in range(max_d)]
        )
        self.knorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim // num_heads) for _ in range(max_d)]
        )
        if False and bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif False and bias_type == "continuous":
            raise NotImplementedError("continuous position bias is not implemented")
        elif True or bias_type == "rotary":
            self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads)
        else:
            self.rel_pos_biases = nn.ModuleList(
                [RelativePositionBias(n_heads=num_heads) for _ in range(3)]
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = MLP(hidden_dim)
        self.mlp_norm = norm_layer(num_heads, hidden_dim, affine=True)

        if noise_cond_dim != 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(noise_cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3 * hidden_dim),
                Rearrange("... (n C) -> n ... 1 C", n=3),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)

    def get_rotary_embedding(self, n, device):
        pos_emb = self.rotary_emb(n, device=device)
        return pos_emb

    def make_rope_learnable(self, per_axis=False):
        if hasattr(self, "rotary_emb"):
            self.rotary_emb.make_learnable(per_axis)

    def spatial_forward(self, x, bcs, axis_index, model_index, return_att=False):
        B, C, H, W, D = x.shape
        shapes = [H, W, D]
        if self.weight_tied_axes:
            model_index = 0
        all_inds = ["h", "w", "d"]
        remainder_inds = list(filter(lambda x: x != all_inds[axis_index], all_inds))
        forward_string = f'b he h w d c -> (b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c'
        backward_string = f'(b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c -> b (he c) h w d'
        x = self.input_heads[model_index](x)
        x = rearrange(x, "b (he c) h w d ->  b he h w d c", he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorms[model_index](q), self.knorms[model_index](k)
        qx, kx, vx = map(lambda x: rearrange(x, forward_string), [q, k, v])
        positions = self.get_rotary_embedding(
            shapes[axis_index], self.norm1.weight.device
        )
        qx, kx = map(lambda t: apply_rotary_pos_emb(positions, t), (qx, kx))
        xx = F.scaled_dot_product_attention(
            qx.contiguous(), kx.contiguous(), vx.contiguous()
        )
        if axis_index == 0:
            xx = rearrange(xx, backward_string, w=W, d=D)
        elif axis_index == 1:
            xx = rearrange(xx, backward_string, h=H, d=D)
        else:
            xx = rearrange(xx, backward_string, h=H, w=W)
        xx = self.output_heads[model_index](xx)
        return xx

    def forward(self, x, bcs, return_att=False, cond=None, axis_order=None):
        B, C, H, W, D = x.shape
        if W == 1:
            ndims = 1
        elif D == 1:
            ndims = 2
        else:
            ndims = 3

        if axis_order is None:
            axis_order = torch.arange(ndims, device=x.device)

        if cond is not None and hasattr(self, "ada_zero"):
            a, b, c = self.ada_zero(cond)
        else:
            a, b, c = 0, 0, 1

        input = x.clone()
        x = self.norm1(x)

        x = (a + 1) * x + b

        out = torch.zeros_like(x)
        for axis_index, model_index in enumerate(axis_order):
            out = out + self.spatial_forward(
                x, bcs, axis_index, model_index, return_att=return_att
            )

        x = out / ndims
        x = self.drop_path(x * self.gamma_att[None, :, None, None, None]) + input

        input = x.clone()
        x = self.mlp_norm(x)

        x = rearrange(x, "b c h w d -> b h w d c")
        x = self.mlp(x)
        x = rearrange(x, "b h w d c -> b c h w d")

        if cond is not None and hasattr(self, "ada_zero"):
            output = (
                c * self.drop_path(self.gamma_mlp[None, :, None, None, None] * x)
                + input
            ) * torch.rsqrt(1 + c * c)
        else:
            output = input + self.drop_path(
                self.gamma_mlp[None, :, None, None, None] * x
            )
        if return_att:
            return output, []
        else:
            return output, []
