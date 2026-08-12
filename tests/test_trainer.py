import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize

from walrus.train import CONFIG_DIR, CONFIG_NAME, main
from walrus.trainer.training import Trainer

from .utils import generate_parameters

# Set the different options to test
conf_options = {
    "trainer.prediction_type": ["delta", "full"],
    "trainer.enable_amp": ["False"],
    "model.causal_in_time": ["True", "False"],
}


def test_temporal_split_losses_logs_cumulative_vrmse_horizons():
    loss_values = torch.arange(50, dtype=torch.float32).reshape(2, 25)

    losses = Trainer.__new__(Trainer).temporal_split_losses(
        loss_values,
        temporal_loss_intervals=[0, 25],
        loss_name="VRMSE",
        dset_name="dummy",
    )

    assert torch.equal(
        losses["dummy/full_VRMSE_T=0:10"], loss_values[:, :10].mean(dim=1)
    )
    assert torch.equal(
        losses["dummy/full_VRMSE_T=0:20"], loss_values[:, :20].mean(dim=1)
    )


@pytest.mark.parametrize("conf", generate_parameters(conf_options), indirect=True)
def test_train(conf):
    """Test training terminates normally for different sets of config."""
    overrides = conf
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        with torch.autograd.detect_anomaly():
            main(cfg)
        assert True
