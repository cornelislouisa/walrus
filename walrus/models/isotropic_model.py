from dataclasses import replace
from functools import reduce
from operator import mul
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from the_well.data.datasets import BoundaryCondition

from walrus.models.shared_utils.flexi_utils import (
    choose_kernel_size_deterministic,
    choose_kernel_size_random,
)
from walrus.models.shared_utils.normalization import RMSGroupNorm
from walrus.models.shared_utils.patch_jitterers import (
    FixedPatchJittererBoundaryPad,
    PatchJittererBoundaryPad,
)


def dim_pad(x, max_d):
    """
    Assume T B C are first channels, then see how many spatial dims we need to append/
    """
    squeeze = 0
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    return x, squeeze


class IsotropicModel(nn.Module):
    """
    Naive model that operates at a single dimension with a repeating block.

    Args:
        patch_size (tuple): Size of the input patch
        hidden_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """

    def __init__(
        self,
        encoder,
        decoder,
        processor,
        projection_dim: int = 96,
        intermediate_dim: int = 192,
        hidden_dim: int = 768,
        processor_blocks: int = 8,
        n_states: int = 4,
        drop_path: float = 0.2,
        input_field_drop: float = 0.1,
        groups: int = 12,
        max_d: int = 3,
        jitter_patches: bool = True,
        use_periodic_fixed_jitter: bool = False,
        gradient_checkpointing_freq: int = 0,
        causal_in_time: bool = False,
        include_d: List[int] = [2, 3],  # Temporary due to FSDP resume issue
        override_dimensionality: Optional[
            int
        ] = 0,  # Temporary due to FSDP resume issue
        dim_key_override: Optional[int] = None,  # Temporary due to FSDP resume issue
        norm_layer: Callable = RMSGroupNorm,
        encoder_norm_layer: Optional[Callable] = None,
        processor_norm_layer: Optional[Callable] = None,
        decoder_norm_layer: Optional[Callable] = None,
        num_samples: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.drop_path = drop_path
        self.max_d = max_d
        self.jitter_patches = jitter_patches
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.causal_in_time = causal_in_time
        self.gradient_checkpointing_freq = gradient_checkpointing_freq
        self.override_dimensionality = override_dimensionality
        self.dim_key_override = dim_key_override
        self.encoder_dummy = nn.Parameter(
            torch.ones(1)
        )  # for grad checkpointing, see: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.hidden_dim = hidden_dim
        if (
            self.override_dimensionality is not None
            and self.override_dimensionality > 0
        ):
            include_d = [self.override_dimensionality]
        self.input_field_drop = input_field_drop
        if use_periodic_fixed_jitter:
            self.patch_jitterer = FixedPatchJittererBoundaryPad(
                stage_dim=projection_dim,
                patch_size=None,
                max_d=self.max_d,
                jitter_patches=jitter_patches,
            )
        else:
            self.patch_jitterer = PatchJittererBoundaryPad(
                stage_dim=projection_dim,
                patch_size=None,
                max_d=self.max_d,
                jitter_patches=jitter_patches,
            )
        self.num_samples = num_samples
        self.embed = nn.ModuleDict(
            {
                str(i): encoder(
                    spatial_dims=3,
                    input_dim=n_states,
                    inner_dim=intermediate_dim,
                    output_dim=hidden_dim,
                    groups=groups,
                    norm_layer=(
                        encoder_norm_layer
                        if encoder_norm_layer is not None
                        else norm_layer
                    ),
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

        # Subclasses (e.g. IsotropicModelWithNoise) set self.noise_dim/self.noise_blocks
        # before calling super().__init__ so that the processor blocks can be
        # conditioned on a noise embedding in the appropriate blocks.
        if hasattr(self, "noise_dim"):
            noise_dim_block = [
                self.noise_dim if i in self.noise_blocks else 0
                for i in range(processor_blocks)
            ]
        else:
            noise_dim_block = [0 for i in range(processor_blocks)]

        self.blocks = nn.ModuleList(
            [
                processor(
                    hidden_dim=hidden_dim,
                    drop_path=self.dp[i],
                    causal_in_time=causal_in_time,
                    gradient_checkpointing=(
                        i % gradient_checkpointing_freq == 0
                        if gradient_checkpointing_freq > 0
                        else False
                    ),
                    norm_layer=(
                        processor_norm_layer
                        if processor_norm_layer is not None
                        else norm_layer
                    ),
                    noise_cond_dim=noise_dim_block[i],
                )
                for i in range(processor_blocks)
            ]
        )
        self.debed = nn.ModuleDict(
            {
                str(i): decoder(
                    input_dim=hidden_dim,
                    inner_dim=intermediate_dim,
                    output_dim=n_states,
                    spatial_dims=3,
                    groups=groups,
                    norm_layer=(
                        decoder_norm_layer
                        if decoder_norm_layer is not None
                        else norm_layer
                    ),
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

    def add_ft_options(
        self,
        ft_param_dict: dict,
    ):
        """
        Add options for fine-tuning the model.
        """
        # APE
        if "ape_shape" in ft_param_dict:
            print("activated ape")
            shape = ft_param_dict["ape_shape"]
            self.ape = nn.Parameter(5e-3 * torch.randn(1, 1, self.hidden_dim, *shape))

        if "learnable_rope" in ft_param_dict and ft_param_dict["learnable_rope"]:
            print("activated learnable rope", ft_param_dict.get("rope_per_axis", False))
            for blk in self.blocks:
                if hasattr(blk, "make_rope_learnable"):
                    blk.make_rope_learnable(
                        per_axis=ft_param_dict.get("rope_per_axis", False)
                    )

        if "freeze" in ft_param_dict:
            raise NotImplementedError("Freezing not implemented yet")

    def _create_ensemble(self, x, num_samples):
        """Create an ensemble of samples by adding noise if model is IsotropicModelWithNoise."""
        if type(self) is not IsotropicModelWithNoise:
            return x

        # x is of size (t, b, c, ...)
        t, b, c = x.shape[:3]
        spatial_shape = x.shape[3:]
        # Repeat x along batch dimension num_samples times
        x = x.unsqueeze(2)  # (T, B, 1, C, ...)
        x = x.expand(
            -1, -1, num_samples, -1, *spatial_shape
        )  # (T, B, num_samples, C, ...)

        # Collapse back into batch dimension for the forward pass
        x = x.reshape(t, b * num_samples, c, *spatial_shape)
        if self.noise_type == "channel":
            # Generate noise tensor of shape (t, b * num_samples, 1, ...)
            noise = torch.randn(
                (t, b * num_samples, 1, *spatial_shape), device=x.device, dtype=x.dtype
            )
            # Concatenate noise channel to x along channel dimension
            x = torch.cat([x, noise], dim=2)
        return x

    def _encoder_forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        patch_size,
        dynamic_ks=None,
        encoder_dummy=None,
        num_samples=None,
    ):
        num_samples = num_samples if num_samples is not None else self.num_samples

        if self.override_dimensionality > 0:
            n_spatial_dims = metadata.n_spatial_dims
        else:
            n_spatial_dims = sum([int(dim != 1) for dim in x.shape[3:]])
        if self.dim_key_override is None:
            dim_key = str(n_spatial_dims)
        else:
            dim_key = str(self.dim_key_override)
        T, B = x.shape[:2]

        # Create an ensemble by appending noise (no-op unless model is IsotropicModelWithNoise)
        x = self._create_ensemble(x, num_samples)

        # Project into higher dim
        x = rearrange(
            x, "t b c h ... -> b c (t h) ..."
        )  # Field dropout is intended to drop out the entire field. We could either implement our own mask or reshape to use existing function and this was slightly faster
        x = F.dropout3d(
            x, training=self.training, p=self.input_field_drop / x.shape[1]
        )  # Bonferonni correction for variable fields - all
        x = rearrange(x, "b c (t h) ... -> t b c h ...", t=T)
        x = (
            x * encoder_dummy
        )  # NOTE - this is just a single scalar to work around a bug in PyTorch's grad checkpointing - if this moves away from zero, we can add it to the space bag weights in postprocessing
        # x = self.space_bag(x, state_labels)
        # x = rearrange(x, "t b ... c -> t b c ...")
        # Now encoder
        if (
            hasattr(self.embed[dim_key], "learned_pad")
            and self.embed[dim_key].learned_pad
        ):
            x, jitter_info = self.patch_jitterer(
                x,
                bcs[0],
                metadata,
                patch_size=patch_size,
                learned_pad=self.embed[dim_key].learned_pad,
                random_kernel=dynamic_ks,
                base_kernel=self.embed[dim_key].base_kernel_size,
            )
        else:
            x, jitter_info = self.patch_jitterer(
                x, bcs[0], metadata, patch_size=patch_size
            )

        # Sparse proj
        state_labels = torch.cat(
            [
                state_labels,
                torch.tensor(
                    [2, 0, 1], device=state_labels.device, dtype=state_labels.dtype
                ),
            ],
            dim=0,
        )
        x, stage_info = self.embed[dim_key](
            x, state_labels, bcs[0], metadata, random_kernel=dynamic_ks
        )
        if hasattr(self, "ape"):
            x = x + self.ape
        return x, stage_info, jitter_info

    def _decoder_forward(self, x, state_labels, bcs, stage_info, jitter_info, metadata):
        """Run the decoder and invert the jitter"""
        if self.override_dimensionality > 0:
            n_spatial_dims = metadata.n_spatial_dims
        else:
            n_spatial_dims = sum([int(dim != 1) for dim in x.shape[3:]])
        if self.dim_key_override is None:
            dim_key = str(n_spatial_dims)
        else:
            dim_key = str(self.dim_key_override)
        x = self.debed[dim_key](x, state_labels, bcs[0], stage_info, metadata)
        if (
            hasattr(self.embed[dim_key], "learned_pad")
            and self.embed[dim_key].learned_pad
        ):
            x = self.patch_jitterer.unjitter(
                x, jitter_info, learned_pad=self.embed[dim_key].learned_pad
            )
        else:
            x = self.patch_jitterer.unjitter(x, jitter_info)
        return x

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
        num_samples=None,
        cond_noise=None,
    ):
        # x - T B C H [W D]
        # state_labels - C
        # bcs - #dims, 2
        # proj axes - #dims - Permutes axes to discourage learning axes - dependent relationships
        # NOTE: Everything gets padded to max_d below, so we want the metadata to reflect this
        metadata = replace(metadata, n_spatial_dims=self.max_d)
        n_spatial_dims = metadata.n_spatial_dims
        dim_key = str(n_spatial_dims)
        # Use provided value or fall back to instance default
        num_samples = num_samples if num_samples is not None else self.num_samples
        # Pad to max dims so we can just use 3D convs - same flops, but empirically would be faster
        # to dynamically adjust which conv is used, but more verbose for compiler-friendly version
        x, squeeze_out = dim_pad(x, self.max_d)
        T, B, C = x.shape[:3]
        x_shape = x.shape[3:]

        dynamic_ks = []
        patch_size = []

        if metadata.dataset_name == "post_neutron_star_merger":
            # Neutron star has a size 66 dimension - just interpolate to 64 for simplicity
            x = rearrange(x, "t b c h w d -> (t b) c h w d")
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 64),
                mode="trilinear",
                align_corners=False,
            )
            x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
            x_shape = x.shape[3:]
        # Choose the variable patches if applicable
        if (
            hasattr(self.embed[dim_key], "variable_downsample")
            and (self.embed[dim_key].variable_downsample)
            and self.embed[dim_key].variable_deterministic_ds
        ):
            # support for variable but deterministic downsampling
            dynamic_ks = choose_kernel_size_deterministic(x_shape)
            patch_size = [reduce(mul, k) for k in dynamic_ks]
            # patch_size doesn't matter for the dimension that is higher than the number of spatial dims
            patch_size.extend([0] * (self.max_d - len(patch_size)))

        # support for variable and random downsampling.
        # this will probably not be used in Walrus but a needed feature for dedicated paper
        elif hasattr(self.embed[dim_key], "variable_downsample") and (
            self.embed[dim_key].variable_downsample
        ):
            for _ in range(self.max_d):
                ks = (
                    choose_kernel_size_random(self.embed[dim_key].kernel_scales_seq)
                    if train
                    else (2, 2)
                )
                patch_size.append(ks[0] * ks[1])
                dynamic_ks.append(ks)
            dynamic_ks = tuple(dynamic_ks)
        # constant downsampling as with hmlp
        else:
            patch_size = [self.embed[dim_key].patch_size] * self.max_d

        # We either pass some noise into the model (created outside of the model)
        # or create one inside the model
        if cond_noise is None:
            # For stochastic models with latent noise, create noise tensor
            if (
                hasattr(self, "noise_type")
                and self.noise_type == "latent"
                and self.noise_dim is not None
            ):
                if self.noise_mode == "global":
                    cond_noise = torch.randn(
                        (T, B * num_samples, self.noise_dim),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    cond_noise = self.noise_mlp(cond_noise)
                elif self.noise_mode == "spatial":
                    cond_noise = None  # We will create this after we get the shape from the encoder
                else:
                    raise ValueError(
                        f"Invalid noise mode {self.noise_mode}, choices are ['global', 'spatial']"
                    )
            else:
                cond_noise = None
        else:
            assert (
                self.noise_mode != "spatial"
            ), "For spatial noise we need to get the spatial dimensions from the encoder."
            if self.noise_mode == "global":
                cond_noise = self.noise_mlp(cond_noise)

        # Always assume we need to checkpoint the encoder if any checkpointing is on
        if self.gradient_checkpointing_freq > 0:
            x, stage_info, jitter_info = torch.utils.checkpoint.checkpoint(
                self._encoder_forward,
                x,
                state_labels,
                bcs,
                metadata,
                patch_size,
                dynamic_ks,
                self.encoder_dummy,
                num_samples=num_samples,
                use_reentrant=False,
            )
        else:
            x, stage_info, jitter_info = self._encoder_forward(
                x,
                state_labels,
                bcs,
                metadata,
                patch_size,
                dynamic_ks,
                self.encoder_dummy,
                num_samples=num_samples,
            )

        # For spatial noise we only inject the noise in the processor
        # and create it after we get the output shape of the encoder
        if (
            hasattr(self, "noise_type")
            and self.noise_type == "latent"
            and self.noise_dim is not None
            and self.noise_mode == "spatial"
        ):
            cond_noise = torch.randn(
                (T, B * num_samples, self.noise_dim, *x.shape[3:]),
                device=x.device,
                dtype=x.dtype,
            )
            # Move channels to last dim
            cond_noise = cond_noise.movedim(2, -1)
            cond_noise = self.noise_mlp(cond_noise)
            # Bring channels back
            cond_noise = cond_noise.movedim(-1, 2)

        # Process
        all_att_maps = []
        # Compute a periodic roll
        # Blk inputs are T, B, C, H, W, D
        periodic_dims = []
        for dim in range(len(bcs[0])):
            if bcs[0][dim][0] == BoundaryCondition["PERIODIC"].value:
                periodic_dims.append(dim + 3)
        periodic_dim_shapes = [x.shape[dim] for dim in periodic_dims]
        roll_total = [0] * len(periodic_dims)
        for ii, blk in enumerate(self.blocks):
            # Randomly roll dimensions of x corresponding to periodic BCs
            if len(periodic_dims) > 0 and self.jitter_patches:
                roll_quantities = [
                    np.random.randint(0, periodic_dim_shapes[dim])
                    for dim in range(len(periodic_dims))
                ]
                roll_total = [
                    roll_quantities[dim] + r for dim, r in enumerate(roll_total)
                ]
                x = torch.roll(
                    x,
                    shifts=roll_quantities,
                    dims=periodic_dims,
                )
            if hasattr(self, "noise_blocks") and ii in self.noise_blocks:
                cond_noise_block = cond_noise
            else:
                cond_noise_block = None
            x, att_maps = blk(x, bcs, return_att=return_att, cond=cond_noise_block)
            all_att_maps += att_maps
        # If we randomly rolled, we need to roll back
        if sum(roll_total) > 0 and self.jitter_patches:
            x = torch.roll(
                x,
                shifts=[-r for r in roll_total],
                dims=periodic_dims,
            )
        # Decode
        # If not causal, no need to debed all time steps so just take the last one
        if not self.causal_in_time:
            x = x[-1:]

        if self.gradient_checkpointing_freq > 0:
            x = torch.utils.checkpoint.checkpoint(
                self._decoder_forward,
                x,
                state_labels,
                bcs,
                stage_info,
                jitter_info,
                metadata,
                use_reentrant=False,
            )
        else:
            x = self._decoder_forward(
                x, state_labels, bcs, stage_info, jitter_info, metadata
            )

        # If neutron star, interpolate back to original size
        if metadata.dataset_name == "post_neutron_star_merger":
            T = x.shape[0]
            x = rearrange(x, "t b c h w d -> (t b) c h w d")
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 66),
                mode="trilinear",
                align_corners=False,
            )
            x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)

        # De-inflate the extra channels if they were added:
        for _ in range(squeeze_out):
            x = x.squeeze(-1)
        # Return T, (num_samples,) B, C, H, [W], [D]
        return x  # TODO - Return attention maps for debugging


class IsotropicModelWithNoise(IsotropicModel):
    """
    IsotropicModel variant that supports CRPS-style stochastic ensembles.

    Draws `num_samples` stochastic realizations per input by either concatenating
    a noise channel to the input (`noise_type="channel"`) or by conditioning the
    processor blocks on a learned noise embedding (`noise_type="latent"`).
    """

    def __init__(
        self,
        *args,
        num_samples: int = 4,
        noise_field_idx: int = 0,
        noise_type: str = "channel",
        noise_dim: int = 64,
        noise_mode: str = "global",
        mlp_layers: int = 0,
        noise_layernorm: bool = True,
        noise_blocks: list = [],
        **kwargs,
    ):
        # These need to be set before super().__init__ so that IsotropicModel can
        # size the processor blocks' noise conditioning dimension appropriately.
        self.noise_dim = noise_dim
        self.noise_blocks = noise_blocks
        if noise_blocks == []:
            assert (
                "processor_blocks" in kwargs
            ), "Must specify processor_blocks if noise_blocks not specified"
            self.noise_blocks = list(range(kwargs["processor_blocks"]))
        super().__init__(*args, num_samples=num_samples, **kwargs)
        assert (
            num_samples > 1
        ), "Number of samples must be greater than 1 for model with stochasticity"
        assert noise_type in [
            "channel",
            "latent",
        ], "Invalid noise type, choices are ['channel', 'latent']"
        self.noise_type = noise_type
        self.noise_mlp = nn.Identity()
        if noise_type == "latent":
            self.noise_field_idx = None
            self.noise_dim = noise_dim
            self.noise_mode = noise_mode
            if mlp_layers > 0:
                self.noise_mlp = nn.Sequential(
                    nn.Linear(self.noise_dim, 4 * self.noise_dim),
                    nn.SiLU(),
                    nn.Linear(4 * self.noise_dim, self.noise_dim),
                    nn.LayerNorm(self.noise_dim) if noise_layernorm else nn.Identity(),
                )
            else:
                self.noise_mlp = nn.Linear(self.noise_dim, self.noise_dim)
        elif noise_type == "channel":
            self.noise_field_idx = noise_field_idx
            self.noise_dim = None
        else:
            raise ValueError("Invalid noise type, choices are ['channel', 'latent']")
