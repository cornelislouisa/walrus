import unittest
from dataclasses import dataclass

import torch

from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.models.shared_utils.patch_jitterers import (
    PatchJitterer,
    PatchJittererBoundaryPad,
)


@dataclass
class DummyMetadata:
    n_spatial_dims: int


class TestJitterer(unittest.TestCase):
    """Right now these just test that forward jitters at least some of the time and that
    unjitter(jitter(x)) is identity"""

    def setUp(self):
        self.jitterer = PatchJitterer(
            3, (16, 16, 16), num_bcs=3, max_d=3, jitter_patches=True
        )

    def test_inverse_1d_boundary(self):
        metadata = DummyMetadata(1)
        x = torch.randn(1, 1, 3, 16, 1, 1)  # T B C H W D
        mid, jitter_info = self.jitterer(x, torch.tensor([[0, 0]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "1D inverse jitter failed for nonperiodic BC"

    def test_inverse_2d_boundary(self):
        metadata = DummyMetadata(2)
        x = torch.randn(1, 1, 3, 16, 16, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[0, 0], [0, 0]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "2D inverse jitter failed for nonperiodic BC"

    def test_inverse_3d_boundary(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[0, 0], [0, 0], [0, 0]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for nonperiodic BC"

    def test_inverse_1d_periodic(self):
        metadata = DummyMetadata(1)
        x = torch.randn(1, 1, 3, 16, 1, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[2, 2]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "1D inverse jitter failed for periodic BC"

    def test_inverse_2d_periodic(self):
        metadata = DummyMetadata(2)
        x = torch.randn(1, 1, 3, 16, 16, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[2, 2], [2, 2]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "2D inverse jitter failed for periodic BC"

    def test_inverse_3d_periodic(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[2, 2], [2, 2], [2, 2]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for periodic BC"

    def test_inverse_3d_mixed(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[0, 0], [2, 2], [0, 0]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for mixed BC"

    def test_3d_turned_off(self):
        jitterer = PatchJitterer(
            3, (16, 16, 16), num_bcs=3, max_d=3, jitter_patches=False
        )
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = jitterer(x, torch.tensor([[0, 0], [2, 2], [0, 0]]), metadata)
        assert torch.allclose(x, mid), "3D jitter failed for turned off jitterer"

    def test_3d_jittering_nonperiodic(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        counter = 0
        # Jitter can randomly return true... so we'll just check a few times - should be p=(1/patch_size)^d
        for i in range(3):
            mid, jitter_info = self.jitterer(
                x, torch.tensor([[2, 2], [2, 2], [2, 2]]), metadata
            )
            if not torch.allclose(x, mid):
                counter += 1
        assert counter > 0, "3D jitter failed"


class TestBoundaryPadTiling(unittest.TestCase):
    """The padded axis must be tiled exactly by the encoder's two strided convs,
    otherwise the decoder's transposed convs return fewer pixels than they were given.
    """

    BASE_KERNEL = (8, 4)
    SIZES = [32, 64, 96, 128, 256, 384, 512, 1024]

    def _padded_length(self, jitter_patches, size, bc):
        jitterer = PatchJittererBoundaryPad(
            3, patch_size=None, max_d=3, jitter_patches=jitter_patches
        )
        shape = (size, size, 1)
        kernel = choose_kernel_size_deterministic(shape[:2]) + (self.BASE_KERNEL,)
        bcs = torch.full((3, 2), bc)
        constant, periodic, _, _ = jitterer.get_paddings(
            shape,
            bcs,
            2,
            None,
            {
                "base_kernel": (self.BASE_KERNEL,) * 3,
                "random_kernel": kernel,
            },
        )
        # Paddings are ordered last-axis-first, so the leading axis sits at the end.
        return size + sum(constant[-2:]) + sum(periodic[-2:]), kernel[0]

    def test_padded_axis_tiles_exactly(self):
        kernel1, kernel2 = self.BASE_KERNEL
        for jitter_patches in (True, False):
            for bc in (0, 2):  # non-periodic, periodic
                for size in self.SIZES:
                    padded, (stride1, stride2) = self._padded_length(
                        jitter_patches, size, bc
                    )
                    tokens1, rem1 = divmod(padded - kernel1, stride1)
                    tokens2, rem2 = divmod(tokens1 + 1 - kernel2, stride2)
                    msg = f"{size=} {bc=} {jitter_patches=} {stride1=} {stride2=}"
                    assert rem1 == 0, msg
                    assert rem2 == 0, msg
                    unpatched = (tokens2 * stride2 + kernel2 - 1) * stride1 + kernel1
                    assert unpatched == padded, msg


class TestMorphogenesisLikeRoundTrip(unittest.TestCase):
    """WT myosin is 64x96x1 with OPEN on AP (64) and PERIODIC on DV (96).
    Periodic decode used to over-crop when the stride did not divide (k-s).
    """

    def test_open_by_periodic_64x96(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from the_well.data.datasets import WellMetadata

        from walrus.train import CONFIG_DIR

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name=None,
                overrides=[
                    "+model=isotropic_model",
                    "model/processor/space_mixing=full_spatial_attention",
                    "model.hidden_dim=64",
                    "model.projection_dim=16",
                    "model.intermediate_dim=32",
                    "model.processor_blocks=1",
                    "model.groups=4",
                    "model.processor.space_mixing.num_heads=2",
                    "model.processor.time_mixing.num_heads=2",
                    "model.causal_in_time=True",
                    "model.override_dimensionality=0",
                    "model.jitter_patches=True",
                    "++model.use_periodic_fixed_jitter=True",
                    "++model.input_field_drop=0",
                ],
            )
        model = instantiate(cfg.model, n_states=16).eval()
        T, B, C, H, W = 2, 1, 13, 64, 96
        x = torch.randn(T, B, C, H, W, 1)
        bcs = torch.zeros(1, 3, 2, dtype=torch.long)
        bcs[0, 0] = 1  # OPEN on AP / height
        bcs[0, 1] = 2  # PERIODIC on DV / width
        meta = WellMetadata(
            dataset_name="smoke",
            n_spatial_dims=3,
            spatial_resolution=(H, W, 1),
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: [f"f{i}" for i in range(C)], 1: []},
            constant_field_names={},
            boundary_condition_types=["DIRICHLET"],
            n_files=1,
            n_trajectories_per_file=[1],
            n_steps_per_trajectory=[T],
        )
        with torch.no_grad():
            y = model(x, torch.arange(3, 3 + C), bcs, meta, train=False)
        y = y[0] if isinstance(y, (tuple, list)) else y
        assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"
