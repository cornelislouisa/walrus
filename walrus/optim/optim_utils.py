"""Optimizer helpers including staged learning for CRPS finetuning."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

from walrus.optim.staged_lr_scheduler import StagedLRScheduler

logger = logging.getLogger(__name__)


def build_param_groups(model: torch.nn.Module, param_groups_cfg):
    """Build parameter groups for the optimizer with different learning rates.

    Currently very hardcoded, but factored into separate function to make this easier to change down the line.
    """
    param_groups = []
    for group_cfg in param_groups_cfg:
        layer_params = getattr(model, group_cfg["params"])  # .parameters()
        if isinstance(layer_params, torch.nn.Parameter):
            layer_params = [layer_params]
        elif isinstance(layer_params, torch.nn.Module):
            layer_params = layer_params.parameters()
        elif isinstance(layer_params, (list, tuple)):
            # Assume list of parameters
            pass
        else:
            raise ValueError(
                f"Unknown type {type(layer_params)} for param group {group_cfg.params}"
            )
        options = {k: v for k, v in group_cfg.items() if k != "params"}
        param_groups.append(
            {"params": layer_params, "name": group_cfg["params"], **options}
        )
    return param_groups


def create_crps_parameter_groups(
    model: torch.nn.Module,
    model_info: Optional[Dict] = None,
    new_params_lr: float = 1e-3,
    common_params_lr: float = 1e-4,
    new_params_kwargs: Optional[Dict] = None,
    common_params_kwargs: Optional[Dict] = None,
) -> Tuple[List[Dict], bool]:
    """
    Split parameters into new (missing from deterministic ckpt) vs common groups.
    """
    new_params = []
    common_params = []

    has_missing_params = (
        model_info and "missing" in model_info and len(model_info["missing"]) > 0
    )

    if has_missing_params:
        missing_param_names = set(model_info["missing"])
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in missing_param_names:
                    new_params.append(param)
                else:
                    common_params.append(param)
        if len(new_params) == 0:
            logger.warning("No new parameters found! Disabling staged learning")
            new_params = common_params
            common_params = []
    else:
        new_params = [p for p in model.parameters() if p.requires_grad]

    if common_params:
        logger.info(
            f"Staged learning: {len(new_params)} new params, {len(common_params)} common params"
        )
    else:
        logger.info(
            f"No new layers detected: {len(new_params)} params (normal training)"
        )

    param_groups = []
    if new_params:
        group = {"params": new_params, "lr": new_params_lr}
        if new_params_kwargs:
            group.update(new_params_kwargs)
        param_groups.append(group)
    if common_params:
        group = {"params": common_params, "lr": common_params_lr}
        if common_params_kwargs:
            group.update(common_params_kwargs)
        param_groups.append(group)

    return param_groups, len(common_params) > 0


def setup_crps_optimizer_and_scheduler(
    cfg: DictConfig,
    model: torch.nn.Module,
    model_info: Optional[Dict] = None,
    last_epoch: int = -1,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Setup optimizer/scheduler with optional staged learning for CRPS finetuning."""
    enable_staged_learning = getattr(cfg.trainer, "enable_staged_learning", False)
    new_params_lr = cfg.optimizer.get("new_params_lr", cfg.optimizer.lr)
    common_params_lr = cfg.optimizer.get("common_params_lr", cfg.optimizer.lr)
    new_params_kwargs = dict(cfg.optimizer.get("new_params_kwargs", {}))
    common_params_kwargs = dict(cfg.optimizer.get("common_params_kwargs", {}))

    for d in (new_params_kwargs, common_params_kwargs):
        for k in ["_target_", "lr", "params"]:
            d.pop(k, None)

    param_groups, has_common_params = create_crps_parameter_groups(
        model,
        model_info,
        new_params_lr=new_params_lr,
        common_params_lr=common_params_lr,
        new_params_kwargs=new_params_kwargs,
        common_params_kwargs=common_params_kwargs,
    )
    if not param_groups:
        raise ValueError("No parameter groups created!")

    optimizer_class = get_class(cfg.optimizer._target_)
    optimizer_kwargs = {}
    for key, value in cfg.optimizer.items():
        if key not in [
            "_target_",
            "lr",
            "params",
            "new_params_lr",
            "common_params_lr",
            "new_params_kwargs",
            "common_params_kwargs",
            "param_groups",
        ]:
            optimizer_kwargs[key] = value

    optimizer = optimizer_class(param_groups, **optimizer_kwargs)

    lr_scheduler = None
    if hasattr(cfg, "lr_scheduler"):
        if cfg.trainer.lr_scheduler_per_step:
            step_mult_factor = (
                cfg.data.module_parameters.max_samples / cfg.trainer.grad_acc_steps
            )
        else:
            step_mult_factor = 1

        main_lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.max_epoch,
            step_mult_factor=step_mult_factor,
            last_epoch=max(-1, last_epoch - 1),
        )

        if has_common_params and enable_staged_learning:
            warmup_epochs = getattr(cfg.trainer, "common_params_warmup_epochs", 5)
            logger.info(
                f"Setting up staged learning: common params start at epoch {warmup_epochs}"
            )
            lr_scheduler = StagedLRScheduler(
                optimizer=optimizer,
                main_scheduler=main_lr_scheduler,
                warmup_epochs=warmup_epochs,
                last_epoch=max(-1, last_epoch - 1),
            )
        else:
            lr_scheduler = main_lr_scheduler

    return optimizer, lr_scheduler
