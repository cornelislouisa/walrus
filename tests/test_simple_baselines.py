"""Tests for advection and mean-field baselines."""

import torch

from walrus.baselines.advection.model import advect_velocity
from walrus.baselines.advection.wrapper import AdvectionWrapper
from walrus.baselines.mean_field.wrapper import MeanFieldWrapper


def test_advect_velocity_shape_and_uniform_field():
    # Uniform velocity should translate rigidly; value at interior stays the same.
    u = torch.ones(2, 2, 16, 24)
    u[:, 0] = 0.5
    u[:, 1] = -0.25
    out = advect_velocity(u, dt=1.0, padding_mode="border")
    assert out.shape == u.shape
    assert torch.allclose(out[:, :, 2:-2, 2:-2], u[:, :, 2:-2, 2:-2], atol=1e-5)


def test_advection_wrapper_walrus_contract():
    model = AdvectionWrapper(dt=1.0, n_steps=1, time_history=10, time_future=1)
    assert model.causal_in_time is False
    x = torch.randn(10, 2, 5, 64, 96)
    y = model(x, None, None, None)
    assert y.shape == (1, 2, 2, 64, 96)


def test_mean_field_accumulates_and_predicts_constant():
    model = MeanFieldWrapper(
        momentum=1.0,  # replace with latest batch mean
        spatial=True,
        time_history=10,
        time_future=1,
        in_channels=2,
        out_channels=2,
    )
    model.train()
    x1 = torch.ones(10, 2, 2, 8, 12)
    y1 = model(x1, None, None, None)
    assert y1.shape == (1, 2, 2, 8, 12)
    assert torch.allclose(y1, torch.ones_like(y1))

    x2 = torch.full((10, 2, 2, 8, 12), 3.0)
    y2 = model(x2, None, None, None)
    # momentum=1 → mean becomes 3
    assert torch.allclose(y2, torch.full_like(y2, 3.0))

    model.eval()
    x3 = torch.zeros(10, 4, 2, 8, 12)
    y3 = model(x3, None, None, None, train=False)
    # Eval must keep the stored mean (3), ignore current batch.
    assert y3.shape == (1, 4, 2, 8, 12)
    assert torch.allclose(y3, torch.full_like(y3, 3.0))


def test_mean_field_can_install_frozen_source_mean():
    model = MeanFieldWrapper(
        spatial=True,
        time_history=10,
        time_future=1,
        in_channels=2,
        out_channels=2,
    )
    source_mean = torch.full((1, 1, 2, 8, 12), 2.5)
    model.set_mean_field(source_mean, n_updates=7)
    model.eval()

    target_context = torch.full((10, 3, 2, 8, 12), 99.0)
    pred = model(target_context, None, None, None, train=False)

    assert model.n_updates.item() == 7
    assert pred.shape == (1, 3, 2, 8, 12)
    assert torch.allclose(pred, torch.full_like(pred, 2.5))


def test_hydra_advection_and_mean_field():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = str(Path(__file__).resolve().parents[1] / "walrus" / "configs" / "model")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        adv = instantiate(compose(config_name="advection"), n_states=64)
        mf = instantiate(compose(config_name="mean_field"), n_states=64)
    assert isinstance(adv, AdvectionWrapper)
    assert isinstance(mf, MeanFieldWrapper)
    x = torch.randn(10, 1, 2, 32, 48)
    assert adv(x, None, None, None).shape == (1, 1, 2, 32, 48)
    mf.train()
    assert mf(x, None, None, None).shape == (1, 1, 2, 32, 48)
