"""Tests for the SineNet baseline package."""

import torch

from walrus.baselines.sinenet.model import SineNet
from walrus.baselines.sinenet.wrapper import SineNetWrapper


def _make_model(**overrides):
    kwargs = dict(
        n_input_scalar_components=2,
        n_input_vector_components=0,
        n_output_scalar_components=2,
        n_output_vector_components=0,
        time_history=4,
        time_future=1,
        hidden_channels=16,
        padding_mode="zeros",
        num_layers=4,
        num_waves=2,
        num_blocks=1,
        mult=2,
        residual=True,
        wave_residual=True,
        disentangle=True,
    )
    kwargs.update(overrides)
    return SineNet(**kwargs)


def test_sinenet_forward_nonsquare():
    model = _make_model()
    x = torch.randn(2, 4, 2, 64, 96)
    y = model(x)
    assert y.shape == (2, 1, 2, 64, 96)


def test_sinenet_forward_square():
    model = _make_model(num_waves=1)
    x = torch.randn(1, 4, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 2, 64, 64)


def test_sinenet_rejects_bad_spatial_size():
    model = _make_model(num_waves=1)
    x = torch.randn(1, 4, 2, 60, 96)
    try:
        model(x)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_wrapper_walrus_contract():
    wrapper = SineNetWrapper(
        hidden_channels=16,
        num_waves=2,
        num_layers=4,
        time_history=10,
        time_future=1,
        in_channels=2,
        out_channels=2,
        padding_mode="zeros",
    )
    assert wrapper.causal_in_time is False
    x = torch.randn(10, 2, 5, 64, 96)
    y = wrapper(x, state_labels=None, bcs=None, metadata=None)
    assert y.shape == (1, 2, 2, 64, 96)


def test_hydra_sinenet_config_target():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = str(Path(__file__).resolve().parents[1] / "walrus" / "configs" / "model")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="sinenet")
    model = instantiate(cfg, n_states=64)
    assert isinstance(model, SineNetWrapper)
    x = torch.randn(10, 1, 2, 32, 48)
    y = model(x, None, None, None)
    assert y.shape == (1, 1, 2, 32, 48)
