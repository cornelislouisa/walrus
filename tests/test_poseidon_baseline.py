"""Smoke tests for Poseidon / ScOTWrapper (no HuggingFace download)."""

import torch

from walrus.models.baseline_wrappers import ScOTWrapper


def _tiny_scot(**overrides):
    kwargs = dict(
        image_size=[64, 96],
        patch_size=4,
        num_channels=2,
        num_out_channels=2,
        embed_dim=24,
        depths=[1, 1, 1, 1],
        num_heads=[3, 3, 3, 3],
        skip_connections=[2, 2, 2, 0],
        window_size=4,
        use_conditioning=True,
        residual_model="convnext",
        from_pretrained="",
    )
    kwargs.update(overrides)
    return ScOTWrapper(**kwargs)


def test_scot_wrapper_walrus_contract_nonsquare():
    model = _tiny_scot()
    assert model.causal_in_time is False
    x = torch.randn(1, 2, 2, 64, 96)
    y = model(x, None, None, None)
    assert y.shape == (1, 2, 2, 64, 96)


def test_scot_wrapper_uses_last_frame_only():
    model = _tiny_scot()
    x = torch.randn(5, 1, 2, 64, 96)
    y = model(x, None, None, None)
    assert y.shape == (1, 1, 2, 64, 96)


def test_from_pretrained_rejects_bad_path():
    try:
        ScOTWrapper(
            image_size=32,
            patch_size=4,
            num_channels=2,
            num_out_channels=2,
            embed_dim=24,
            depths=[1, 1, 1, 1],
            num_heads=[3, 3, 3, 3],
            window_size=4,
            from_pretrained="/nonexistent/Poseidon-L",
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_hydra_poseidon_morph_config_target():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    cfg_dir = str(Path(__file__).resolve().parents[1] / "walrus" / "configs" / "model")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="poseidon_morph")
    assert cfg._target_ == "walrus.models.ScOTWrapper"
    assert list(cfg.image_size) == [64, 96]
    assert cfg.num_channels == 2
    assert cfg.embed_dim == 192
