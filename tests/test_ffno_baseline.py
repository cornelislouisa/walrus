"""Tests for the Factorized FNO baseline package."""

import torch

from walrus.baselines.ffno.model import FFNO, FNOFactorized2DBlock
from walrus.baselines.ffno.wrapper import FFNOWrapper


def test_factorized_block_nonsquare():
    block = FNOFactorized2DBlock(
        modes=8,
        width=16,
        input_dim=4,
        output_dim=2,
        n_layers=2,
        mode="full",
    )
    x = torch.randn(2, 64, 96, 4)
    y = block(x)
    assert y.shape == (2, 64, 96, 2)


def test_factorized_block_uses_residual_stack():
    """Head should see residual-accumulated features, not only the last residual."""
    torch.manual_seed(0)
    block = FNOFactorized2DBlock(
        modes=4,
        width=8,
        input_dim=3,
        output_dim=3,
        n_layers=2,
        mode="full",
        dropout=0.0,
        in_dropout=0.0,
    )
    x = torch.randn(1, 16, 16, 3)
    with torch.no_grad():
        y = block(x)
        # Recompute with head on last residual only (the old buggy path).
        h = block.in_proj(x)
        last_b = None
        for layer in block.spectral_layers:
            b, _ = layer(h)
            h = h + b
            last_b = b
        y_bug = block.out(last_b)
    assert not torch.allclose(y, y_bug), "out(x) and out(b) should differ"


def test_ffno_forward_nonsquare():
    model = FFNO(
        n_input_scalar_components=2,
        n_input_vector_components=0,
        n_output_scalar_components=2,
        n_output_vector_components=0,
        time_history=4,
        time_future=1,
        modes=8,
        width=16,
        n_layers=2,
        should_normalize=False,
        use_position=True,
        use_fourier_position=False,
    )
    x = torch.randn(2, 4, 64, 96, 2)
    y = model(x)
    assert y.shape == (2, 1, 64, 96, 2)


def test_ffno_fourier_position_requires_k_max():
    model = FFNO(
        n_input_scalar_components=2,
        n_input_vector_components=0,
        n_output_scalar_components=2,
        n_output_vector_components=0,
        time_history=2,
        time_future=1,
        modes=4,
        width=8,
        n_layers=1,
        should_normalize=False,
        use_position=True,
        use_fourier_position=True,
        k_max=None,
    )
    x = torch.randn(1, 2, 16, 16, 2)
    try:
        model(x)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_wrapper_walrus_contract():
    wrapper = FFNOWrapper(
        modes=8,
        width=16,
        n_layers=2,
        time_history=10,
        time_future=1,
        in_channels=2,
        out_channels=2,
        should_normalize=False,
    )
    assert wrapper.causal_in_time is False
    # T, B, C, H, W — C may include trailing constants; wrapper keeps first in_channels.
    x = torch.randn(10, 2, 5, 64, 96)
    y = wrapper(x, state_labels=None, bcs=None, metadata=None)
    assert y.shape == (1, 2, 2, 64, 96)


def test_wrapper_squeezes_inflated_spatial_dim():
    """BatchInflatedWellDataset pads 2D morph data to (T,B,C,H,W,1)."""
    wrapper = FFNOWrapper(
        modes=8,
        width=16,
        n_layers=2,
        time_history=10,
        time_future=1,
        in_channels=2,
        out_channels=2,
        should_normalize=False,
    )
    x = torch.randn(10, 1, 3, 64, 96, 1)
    y = wrapper(x, None, None, None)
    assert y.shape == (1, 1, 2, 64, 96, 1)


def test_hydra_ffno_config_target():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from pathlib import Path

    cfg_dir = str(Path(__file__).resolve().parents[1] / "walrus" / "configs" / "model")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="ffno")
    model = instantiate(cfg, n_states=64)
    assert isinstance(model, FFNOWrapper)
    x = torch.randn(10, 1, 2, 32, 48)
    y = model(x, None, None, None)
    assert y.shape == (1, 1, 2, 32, 48)
