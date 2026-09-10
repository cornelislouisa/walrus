"""Analyze Walrus checkpoints from Weights & Biases runs.

Given a wandb project and run, locate the checkpoint that minimizes a validation
metric, reload the model, and run the same test / rollout-test evaluation used in
training. See ``demo_notebooks/analyze_checkpoints.ipynb`` for usage, and
``demo_notebooks/zero_shot_eval.ipynb`` for evaluating a run on a different dataset.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.analysis.rollout_cache import (
    DEFAULT_CACHE_DIR,
    RolloutCache,
    RolloutCacheKey,
)
from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.training import Trainer

# Selection splits (lower is better). Scores are VRMSE, not the train ``loss_fn``
# scalar — that scalar is MAE for det runs and CRPS for CRPS runs, so it is not
# cross-run comparable. Wandb keys look like:
#   rollout_valid_<dataset>/full_VRMSE_T=all_mean
ROLLOUT = "rollout_valid"
SINGLE_STEP = "valid"
COMPARE_SCORE = "VRMSE"

# Optimizer config keys consumed by our own setup code rather than the torch optimizer.
_NON_OPTIMIZER_KEYS = {
    "param_groups",
    "new_params_lr",
    "common_params_lr",
    "new_params_kwargs",
    "common_params_kwargs",
}

_DATA_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs" / "data"

# Old on-disk names → current locations (datasets renamed / moved).
# 128×128 atlas runs lived under processed_morphodynamic_atlas; that tree is now WT_old.
# Do NOT alias .../WT → WT_old: current 64×96 configs also use .../WT.
_DATA_PATH_ALIASES = {
    "/data/lcornelis/morphogenesis_data/processed_morphodynamic_atlas": (
        "/data/lcornelis/morphogenesis_data/WT_old"
    ),
    "/data/lcornelis/morphogenesis_data/even-skipped_r13": (
        "/data/lcornelis/morphogenesis_data/even_skipped_r13"
    ),
}

# Fallback stats when WT_old/stats.yaml is missing (regenerated excluding NaNs).
_WT_OLD_STATS_FALLBACK = (
    pathlib.Path(__file__).resolve().parents[1] / "configs" / "data" / "WT_old_stats.yaml"
)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _remap_data_paths(cfg: DictConfig) -> DictConfig:
    """Rewrite known renamed dataset roots inside ``cfg.data``."""
    info = OmegaConf.select(cfg, "data.module_parameters.well_dataset_info")
    if info is None:
        return cfg
    for _name, meta in info.items():
        path = meta.get("path") if isinstance(meta, dict) or OmegaConf.is_dict(meta) else None
        if path in _DATA_PATH_ALIASES:
            new_path = _DATA_PATH_ALIASES[path]
            OmegaConf.update(
                cfg,
                f"data.module_parameters.well_dataset_info.{_name}.path",
                new_path,
                merge=False,
            )
            # GlobalRevNormalization needs stats.yaml next to the Well root (or an
            # absolute normalization_path). WT_old often lacks an on-disk stats file.
            if new_path.rstrip("/").endswith("WT_old") and _WT_OLD_STATS_FALLBACK.is_file():
                on_disk = pathlib.Path(new_path) / "stats.yaml"
                if not on_disk.is_file():
                    OmegaConf.update(
                        cfg,
                        f"data.module_parameters.well_dataset_info.{_name}.normalization_path",
                        str(_WT_OLD_STATS_FALLBACK),
                        merge=False,
                    )
    return cfg


# --------------------------------------------------------------------------- #
# data-config helpers (for zero-shot on a different dataset)
# --------------------------------------------------------------------------- #
def list_data_configs() -> list[str]:
    """Hydra data config names available under ``walrus/configs/data/``."""
    return sorted(p.stem for p in _DATA_CONFIG_DIR.glob("*.yaml") if p.stem != "README")


def load_data_config(data: str) -> DictConfig:
    """Compose a Hydra data config by name (e.g. ``morphogenesis_WT``)."""
    if data not in list_data_configs():
        raise ValueError(
            f"Unknown data config '{data}'. Available: {', '.join(list_data_configs())}"
        )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_DATA_CONFIG_DIR), version_base=None):
        return compose(config_name=data)


def apply_data_config(
    run_cfg: DictConfig,
    data: Union[str, DictConfig],
    *,
    keep_n_steps_input: bool = True,
    keep_field_index_map: bool = True,
    keep_dataset_kws: bool = True,
    **data_overrides,
) -> DictConfig:
    """Return a copy of ``run_cfg`` whose ``data`` block is replaced by ``data``.

    By default keeps the run's ``n_steps_input``, ``field_index_map_override`` and
    ``dataset_kws`` so the model architecture (context length + embed/debed width)
    and the channel layout still match the checkpoint. Pass
    ``keep_field_index_map=False`` only if you intentionally want to rebuild/align
    to the new dataset's smaller field set.

    Extra kwargs are applied under ``data.module_parameters`` (e.g. ``batch_size=1``).
    """
    cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
    data_cfg = load_data_config(data) if isinstance(data, str) else data
    trained_n_steps = OmegaConf.select(
        run_cfg, "data.module_parameters.n_steps_input"
    )
    trained_field_map = OmegaConf.select(run_cfg, "data.field_index_map_override")
    trained_dataset_kws = OmegaConf.select(
        run_cfg, "data.module_parameters.dataset_kws"
    )
    cfg.data = data_cfg
    if keep_n_steps_input and trained_n_steps is not None and "n_steps_input" not in data_overrides:
        cfg.data.module_parameters.n_steps_input = trained_n_steps
    # Critical for zero-shot: checkpoint embed/debed widths follow the trained field
    # map (often the full Well index). Swapping in bc_only_override would shrink the
    # model and break load_state_dict.
    if keep_field_index_map and trained_field_map is not None:
        cfg.data.field_index_map_override = OmegaConf.create(
            OmegaConf.to_container(trained_field_map, resolve=True)
        )
    # 2D baselines train with pad_cartesian_data_to_d=2; dropping it here would
    # inflate velocity to 3 components and mismatch the model's channel count.
    if (
        keep_dataset_kws
        and trained_dataset_kws is not None
        and "dataset_kws" not in data_overrides
    ):
        OmegaConf.update(
            cfg,
            "data.module_parameters.dataset_kws",
            OmegaConf.to_container(trained_dataset_kws, resolve=True),
            merge=False,
            force_add=True,
        )
    for key, value in data_overrides.items():
        if value is None:
            continue
        OmegaConf.update(cfg, f"data.module_parameters.{key}", value, merge=False)
    return cfg


# --------------------------------------------------------------------------- #
# wandb access
# --------------------------------------------------------------------------- #
def get_run(project: str, run: str, entity: Optional[str] = None):
    """Fetch a wandb run by id or display name within ``entity/project``."""
    api = wandb.Api()
    entity = entity or api.default_entity
    try:
        return api.run(f"{entity}/{project}/{run}")
    except Exception:
        matches = list(api.runs(f"{entity}/{project}", filters={"display_name": run}))
        if not matches:
            raise ValueError(f"No run '{run}' found in {entity}/{project}")
        return sorted(matches, key=lambda r: r.created_at, reverse=True)[0]


def _score_columns(df: pd.DataFrame, split: str, score: str = COMPARE_SCORE) -> list[str]:
    """Wandb columns ``{split}_*/full_{score}_T=all_mean``."""
    prefix, suffix = f"{split}_", f"/full_{score}_T=all_mean"
    return [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]


def _mean_over_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(float("nan"), index=df.index)
    return df[columns].astype(float).mean(axis=1)


def _aggregate_dataset_scores(per_dataset: dict[str, float]) -> float:
    if not per_dataset:
        return float("nan")
    return float(sum(per_dataset.values()) / len(per_dataset))


def run_history(run, score: str = COMPARE_SCORE) -> pd.DataFrame:
    """One row per epoch with comparable ``valid`` / ``rollout_valid`` scores.

    Prefers ``{split}_*/full_{score}_T=all_mean`` (default ``VRMSE``), averaged
    across datasets when several are present. Falls back to the train-loss
    scalars ``valid`` / ``rollout_valid`` only if no score columns exist.
    """
    df = run.history(samples=100_000, pandas=True)
    if "epoch" not in df.columns:
        return pd.DataFrame(columns=["epoch", SINGLE_STEP, ROLLOUT])
    out = pd.DataFrame({"epoch": df["epoch"]})
    for split in (SINGLE_STEP, ROLLOUT, "test", "rollout_test"):
        cols = _score_columns(df, split, score)
        if cols:
            out[split] = _mean_over_columns(df, cols)
        elif split in df.columns:
            out[split] = df[split]
    return (
        out.dropna(subset=["epoch"])
        .groupby("epoch", as_index=False)
        .first()
    )


def experiment_dir(run) -> pathlib.Path:
    return pathlib.Path(run.config["checkpoint"]["save_dir"]).parent


def load_config(run) -> DictConfig:
    """Prefer the on-disk resolved config, fall back to the logged wandb config."""
    cfg_path = experiment_dir(run) / "extended_config.yaml"
    if cfg_path.is_file():
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = OmegaConf.create(run.config)
    return _remap_data_paths(cfg)


# --------------------------------------------------------------------------- #
# checkpoint selection
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    metric: str
    epoch: int
    value: float
    path: pathlib.Path


@dataclass(frozen=True)
class VFAlignedCohort:
    """Common test embryos that support a native context ending at VF onset."""

    files: tuple[pathlib.Path, ...]
    excluded: Mapping[str, str]
    max_context: int
    horizon_minutes: int


_OPTIONAL_CHECKPOINT_TARGETS = (
    "walrus.baselines.advection.AdvectionWrapper",
    "walrus.baselines.mean_field.MeanFieldWrapper",
)


def _is_checkpoint_dir(path: pathlib.Path) -> bool:
    """True if ``path`` looks like a saved Trainer checkpoint folder."""
    if not path.is_dir():
        return False
    return (path / "full_checkpoint.pt").exists() or (path / "metadata.pt").exists()


def allows_missing_checkpoint(run_or_cfg) -> bool:
    """Advection needs no weights; mean-field can run from the current context mean."""
    cfg = run_or_cfg if isinstance(run_or_cfg, DictConfig) else OmegaConf.create(
        getattr(run_or_cfg, "config", run_or_cfg)
    )
    target = str(OmegaConf.select(cfg, "model._target_") or "")
    return target in _OPTIONAL_CHECKPOINT_TARGETS


def available_checkpoints(ckpt_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    """Map epoch -> checkpoint dir for every saved checkpoint (step_*, best, last)."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    epochs: dict[int, pathlib.Path] = {}
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            epochs[int(d.name.split("_")[1])] = d
        except ValueError:
            continue
    for name, fallback_epoch in (("best", -1), ("last", 0)):
        folder = ckpt_dir / name
        if not _is_checkpoint_dir(folder):
            continue
        meta = folder / "metadata.pt"
        epoch = fallback_epoch
        if meta.exists():
            stored = torch.load(meta, weights_only=False).get("epoch")
            if stored is not None:
                epoch = int(stored)
        epochs.setdefault(epoch, folder)
    return epochs


def select_checkpoint(
    run, metric: str = ROLLOUT, checkpoints_dir: Optional[str] = None
) -> Selection:
    """Checkpoint minimizing ``metric`` among epochs that were actually saved.

    ``metric`` is typically ``rollout_valid`` or ``valid``. Values come from
    ``run_history`` (default: mean ``full_VRMSE_T=all_mean`` under that split).

    ``checkpoints_dir`` overrides the config ``save_dir`` when checkpoints were
    moved (e.g. the run was trained on a different machine).
    """
    ckpt_dir = pathlib.Path(checkpoints_dir or run.config["checkpoint"]["save_dir"])
    available = available_checkpoints(ckpt_dir)
    if not available:
        if allows_missing_checkpoint(run):
            return Selection(metric, 0, float("nan"), ckpt_dir / "_untrained")
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    history = run_history(run)
    if metric in history.columns:
        scored = history[history.epoch.isin(available) & history[metric].notna()]
        if len(scored):
            row = scored.loc[scored[metric].idxmin()]
            epoch = int(row.epoch)
            return Selection(metric, epoch, float(row[metric]), available[epoch])
    # Fall back to the on-disk "best" checkpoint (best one-step val_loss).
    best = ckpt_dir / "best"
    epoch = next((e for e, p in available.items() if p == best), max(available))
    return Selection(metric, epoch, float("nan"), available[epoch])


# --------------------------------------------------------------------------- #
# model / trainer construction and evaluation
# --------------------------------------------------------------------------- #
def _instantiate_analysis_datamodule(
    cfg: DictConfig,
    well_base_path: Optional[str],
    *,
    data_workers: Optional[int] = None,
):
    """Instantiate one local datamodule from a resolved analysis config."""
    return instantiate(
        cfg.data.module_parameters,
        world_size=1,
        rank=0,
        data_workers=cfg.get("data_workers", 1) if data_workers is None else data_workers,
        well_base_path=well_base_path or cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )


def _is_mean_field(cfg: DictConfig) -> bool:
    target = str(OmegaConf.select(cfg, "model._target_") or "")
    return target.endswith(".MeanFieldWrapper")


@torch.no_grad()
def _fit_original_training_mean(
    model,
    source_cfg: DictConfig,
    target_cfg: DictConfig,
    target_datamodule,
    device: torch.device,
    well_base_path: Optional[str],
) -> None:
    """Fit MeanField on the run's training split, then freeze it for target eval.

    The source average is accumulated in physical units. It is subsequently
    expressed in the target dataset's normalization space so Trainer's usual
    denormalization returns the same source-domain field during zero-shot rollout.
    """
    from walrus.baselines.utils import squeeze_inflated_spatial

    source_datamodule = _instantiate_analysis_datamodule(
        source_cfg, well_base_path, data_workers=0
    )
    total = None
    count = None
    n_batches = 0
    loaders = source_datamodule.build_loaders_from_dset_list(
        source_datamodule.train_dataset.sub_dsets,
        batch_size=int(
            OmegaConf.select(source_cfg, "data.module_parameters.batch_size") or 1
        ),
        replicas=1,
        rank=0,
        full=True,
    )
    for loader in loaders:
        for batch in loader:
            # Well batches are B,T,...,C. MeanField only models dynamic fields;
            # constants appended by the Trainer formatter are intentionally absent.
            fields = batch["input_fields"].movedim(-1, 2).transpose(0, 1)
            fields, _ = squeeze_inflated_spatial(fields)
            fields = fields[:, :, : model.in_channels].to(dtype=torch.float64)
            finite = torch.isfinite(fields)
            batch_total = torch.where(finite, fields, 0.0).sum(dim=(0, 1))
            batch_count = finite.sum(dim=(0, 1))
            if total is None:
                total = batch_total
                count = batch_count
            else:
                if total.shape != batch_total.shape:
                    raise ValueError(
                        "Mean-field source shape changed from "
                        f"{tuple(total.shape)} to {tuple(batch_total.shape)}"
                    )
                total += batch_total
                count += batch_count
            n_batches += 1
    if total is None or count is None or n_batches == 0:
        raise ValueError("Original training split contains no mean-field samples")
    if torch.any(count == 0):
        raise ValueError("Original training split has pixels with no finite values")

    physical_mean = (total / count).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if not model.spatial:
        spatial_dims = tuple(range(3, physical_mean.ndim))
        physical_mean = physical_mean.mean(dim=spatial_dims, keepdim=True)

    target_metadata = list(target_datamodule.train_dataset.dset_to_metadata.values())
    unique_names = {metadata.dataset_name for metadata in target_metadata}
    if len(target_metadata) != 1 or len(unique_names) != 1:
        raise ValueError(
            "Frozen mean-field zero-shot evaluation currently requires exactly "
            "one target dataset"
        )
    metadata = target_metadata[0]
    expected_spatial = tuple(metadata.spatial_resolution)
    actual_spatial = tuple(physical_mean.shape[3:])
    if model.spatial and actual_spatial != expected_spatial:
        raise ValueError(
            f"Original mean-field grid {actual_spatial} does not match target "
            f"grid {expected_spatial}"
        )

    revin_factory = instantiate(target_cfg.trainer.revin)
    target_revin = revin_factory(target_datamodule.train_dataset, device)
    physical_mean = physical_mean.to(device)
    target_stats = target_revin.compute_stats(physical_mean, metadata)
    normalized_mean = target_revin.normalize_stdmean(physical_mean, target_stats)
    model.set_mean_field(normalized_mean, n_updates=n_batches)
    print(
        "Mean-field fitted on the run's original training split "
        f"({n_batches} batches) and frozen for target evaluation."
    )


def build_trainer(
    cfg: DictConfig,
    ckpt_path: pathlib.Path,
    device: torch.device,
    viz_folder: pathlib.Path,
    well_base_path: Optional[str] = None,
    source_cfg: Optional[DictConfig] = None,
    **trainer_overrides,
) -> Trainer:
    """Rebuild the model + Trainer from a config and load ``ckpt_path`` weights."""
    datamodule = _instantiate_analysis_datamodule(cfg, well_base_path)
    field_to_index_map = datamodule.train_dataset.field_to_index_map
    model = instantiate(cfg.model, n_states=max(field_to_index_map.values()) + 1)
    # Reapply finetuning structural changes (e.g. learnable per-axis RoPE) so the
    # architecture matches the checkpoint, exactly as train.py does before loading.
    if "finetuning_mods" in cfg and hasattr(model, "add_ft_options"):
        model.add_ft_options(cfg.finetuning_mods)
    ckpt_path = pathlib.Path(ckpt_path)
    if _is_checkpoint_dir(ckpt_path) or ckpt_path.is_file():
        loader = CheckPointLoader(
            save_dir=ckpt_path.parent,
            load_checkpoint_path=ckpt_path,
            prioritize_resume=False,
        )
        loader.load(model, local=True)
    elif allows_missing_checkpoint(cfg):
        target = str(OmegaConf.select(cfg, "model._target_") or "")
        if "mean_field" in target:
            print(
                f"No mean-field checkpoint under {ckpt_path}; "
                "fitting the frozen mean from the original training split."
            )
        else:
            print(f"No checkpoint under {ckpt_path}; evaluating the stateless baseline.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    if _is_mean_field(cfg):
        _fit_original_training_mean(
            model,
            source_cfg or cfg,
            cfg,
            datamodule,
            device,
            well_base_path,
        )
    model = model.to(device).eval()

    # CRPS runs carry staged-learning keys that the raw optimizer class doesn't accept.
    opt_cfg = OmegaConf.masked_copy(
        cfg.optimizer, [k for k in cfg.optimizer if k not in _NON_OPTIMIZER_KEYS]
    )
    optimizer = instantiate(opt_cfg, params=model.parameters(), _convert_="all")

    overrides = dict(
        video_validation=False,
        image_validation=False,
        skip_spectral_metrics=True,
        debug_mode=False,
    )
    overrides.update(trainer_overrides)
    return instantiate(
        cfg.trainer,
        experiment_name=cfg.get("name", "analysis"),
        viz_folder=str(viz_folder),
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=None,
        checkpointer=instantiate(cfg.checkpoint, rank=0),
        device=device,
        device_mesh=None,
        distribution_type="local",
        rank=0,
        world_size=1,
        formatter=ChannelsFirstWithTimeFormatter,
        batch_aggregation_fns=[
            get_method(n) for n in cfg.trainer.get("batch_aggregation_fns", ["torch.mean"])
        ],
        wandb_logging=False,
        start_epoch=1,
        **overrides,
    )


def _dataset_scores(metrics: dict, split: str, name: str = "VRMSE") -> dict[str, float]:
    """Extract per-dataset field-time-averaged scores from a validation loss dict."""
    prefix, suffix = f"{split}_", f"/full_{name}_T=all_mean"
    return {
        k[len(prefix) : -len(suffix)]: float(v)
        for k, v in metrics.items()
        if k.startswith(prefix) and k.endswith(suffix)
    }


def _file_signature(paths) -> list[tuple[str, int, int]]:
    """Cheap invalidation signature for local data/checkpoint files."""
    signature = []
    for path in sorted({pathlib.Path(p) for p in paths}, key=str):
        if not path.is_file():
            continue
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return signature


def _cache_source_signatures(
    cfg: DictConfig,
    checkpoint_path: pathlib.Path,
    *,
    data_splits: Sequence[str] = ("test",),
) -> dict[str, list[tuple[str, int, int]]]:
    checkpoint_files = [
        checkpoint_path / name
        for name in ("full_checkpoint.pt", "metadata.pt", ".metadata")
    ]
    data_files = []
    info = OmegaConf.select(cfg, "data.module_parameters.well_dataset_info") or {}
    for metadata in info.values():
        root_value = (
            metadata.get("path")
            if isinstance(metadata, dict) or OmegaConf.is_dict(metadata)
            else None
        )
        if not root_value:
            continue
        root = pathlib.Path(root_value)
        data_files.append(root / "stats.yaml")
        # Names, sizes and mtimes catch regenerated/moved trajectories without
        # hashing multi-GB array contents.
        for split in data_splits:
            data_files.extend((root / "data" / split).glob("*.hdf5"))
        normalization_path = metadata.get("normalization_path")
        if normalization_path:
            data_files.append(pathlib.Path(normalization_path))
    return {
        "checkpoint_files": _file_signature(checkpoint_files),
        "data_files": _file_signature(data_files),
    }


def _model_cache_label(cfg: DictConfig) -> str:
    """Human-readable model name used in cache paths, e.g. ``FFNOWrapper``."""
    target = OmegaConf.select(cfg, "model._target_") or "unknown_model"
    return str(target).rsplit(".", 1)[-1]


def _rollout_cache(
    run,
    selection: Selection,
    cfg: DictConfig,
    data_tag: str,
    split: str,
    *,
    full: bool,
    cache_dir: Union[str, pathlib.Path],
    trainer_overrides: dict,
    extra_settings: Optional[Mapping[str, Any]] = None,
    source_cfg: Optional[DictConfig] = None,
) -> RolloutCache:
    """Build a cache key from every setting that can change inference output."""
    model_cfg = OmegaConf.select(cfg, "model")
    settings = {
        "full": full,
        "model": OmegaConf.to_container(model_cfg, resolve=True) if model_cfg else None,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "source_files": _cache_source_signatures(cfg, selection.path),
        "trainer_overrides": {
            key: value
            for key, value in trainer_overrides.items()
            if key
            not in {
                "num_detailed_logs",
                "video_size_multiplier",
                "video_validation",
                "image_validation",
                "rollout_artifact_callback",
            }
        },
        "max_rollout_steps": trainer_overrides.get(
            "max_rollout_steps",
            OmegaConf.select(cfg, "trainer.max_rollout_steps"),
        ),
        "validation_ensemble_size": trainer_overrides.get(
            "validation_ensemble_size",
            OmegaConf.select(cfg, "trainer.validation_ensemble_size"),
        ),
        "validation_full_trajectory_ensemble_size": trainer_overrides.get(
            "validation_full_trajectory_ensemble_size",
            OmegaConf.select(cfg, "trainer.validation_full_trajectory_ensemble_size"),
        ),
        "validation_one_step_ensemble_size": trainer_overrides.get(
            "validation_one_step_ensemble_size",
            OmegaConf.select(cfg, "trainer.validation_one_step_ensemble_size"),
        ),
    }
    if extra_settings:
        settings["analysis_protocol"] = dict(extra_settings)
    if _is_mean_field(cfg):
        original_cfg = source_cfg or cfg
        settings["mean_field_source"] = {
            "protocol": "original_training_physical_mean_v1",
            "data": OmegaConf.to_container(original_cfg.data, resolve=True),
            "source_files": _cache_source_signatures(
                original_cfg,
                selection.path,
                data_splits=("train",),
            ),
        }
    key = RolloutCacheKey(
        entity=str(getattr(run, "entity", None) or "unknown_entity"),
        project=str(getattr(run, "project", None) or "unknown_project"),
        model=_model_cache_label(cfg),
        run_id=str(getattr(run, "id", "unknown")),
        run_name=str(getattr(run, "name", getattr(run, "id", "unknown"))),
        data=data_tag,
        split=split,
        checkpoint_epoch=selection.epoch,
        checkpoint_path=str(selection.path.resolve()),
        settings=settings,
    )
    return RolloutCache(key, cache_dir)


def _render_cached_videos(
    cache: RolloutCache,
    _viz: pathlib.Path,
    split: str,
    *,
    num_detailed_logs: int,
    video_style: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Render videos from cached arrays; never load or execute the model.

    Uses the shared diverging, auto-centered style from
    ``walrus.analysis.rollout_video`` rather than the_well's viridis ``make_video``,
    so every rollout video (zero-shot, in-distribution, any dataset) looks the same.
    ``video_style`` overrides individual keys of ``DEFAULT_VIDEO_STYLE``.
    """
    from walrus.analysis.rollout_video import (
        DEFAULT_VIDEO_STYLE,
        cache_artifacts,
        render_styled_video,
        style_tag,
    )

    style = {**DEFAULT_VIDEO_STYLE, **(video_style or {})}
    # Videos are cache products too: if they already exist, merely return them.
    # Keeping them beside their source arrays makes them independent of notebook cwd,
    # and tagging by style keeps a colormap change from mixing with older renders.
    viz = cache.video_dir / style_tag(style)
    viz.mkdir(parents=True, exist_ok=True)
    # Preserve Trainer's historical ``count < num_detailed_logs`` behavior.
    limit = max(0, int(num_detailed_logs) - 1)
    written = []
    for i, artifact in enumerate(cache_artifacts(cache.path)[:limit]):
        out = viz / f"{split}_{i:02d}.mp4"
        if not out.is_file():
            render_styled_video(artifact, out, **style)
        written.append(str(out))
    return written


@torch.no_grad()
def evaluate_checkpoint(
    run,
    metric: str = ROLLOUT,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = True,
    viz_folder: Optional[str] = None,
    make_videos: bool = False,
    checkpoints_dir: Optional[str] = None,
    data: Optional[Union[str, DictConfig]] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    splits: tuple[str, ...] = ("test", "rollout_test"),
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    video_style: Optional[Mapping[str, Any]] = None,
    **trainer_overrides,
) -> dict:
    """Load the checkpoint best by ``metric`` and run test / rollout-test evaluation.

    Pass ``data`` (Hydra data config name or DictConfig) to evaluate on a different
    dataset than the run was trained on (zero-shot). ``data_overrides`` are applied
    under ``data.module_parameters`` (e.g. ``{"batch_size": 1}``).

    ``splits`` defaults to both ``test`` and ``rollout_test``. Pass
    ``splits=("rollout_test",)`` to skip the one-step test loop.

    ``video_style`` overrides keys of
    ``walrus.analysis.rollout_video.DEFAULT_VIDEO_STYLE``. It only affects rendering,
    never the cache key, so restyling videos never triggers inference.
    """
    device = device or default_device()
    # One shared default across tables, distributions, RMS and videos is essential:
    # otherwise identical-looking analyses produce different cache keys/horizons.
    trainer_overrides.setdefault("max_rollout_steps", 200)
    selection = select_checkpoint(run, metric, checkpoints_dir)
    data_tag = (
        data
        if isinstance(data, str)
        else ("custom" if data is not None else "training_data")
    )
    default_viz = (
        f"./_analysis_viz/{run.name}/zero_shot_{data_tag}"
        if data is not None
        else f"./_analysis_viz/{run.name}/{metric}"
    )
    viz = pathlib.Path(viz_folder or default_viz)
    viz.mkdir(parents=True, exist_ok=True)
    source_cfg = load_config(run)
    cfg = source_cfg
    if data is not None:
        cfg = apply_data_config(
            source_cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )

    wanted = set(splits)
    unknown = wanted - {"test", "rollout_test"}
    if unknown:
        raise ValueError(f"splits must be 'test' and/or 'rollout_test'; got {splits}")

    caches = {
        split: _rollout_cache(
            run,
            selection,
            cfg,
            str(data_tag),
            split,
            full=full,
            cache_dir=cache_dir,
            trainer_overrides=trainer_overrides,
            source_cfg=source_cfg,
        )
        for split in wanted
    }
    if refresh_cache:
        for cache in caches.values():
            cache.reset()
    missing = [
        split
        for split, cache in caches.items()
        if not (use_cache and cache.complete)
    ]

    computed: dict[str, dict] = {}
    if missing:
        active_cache: Optional[RolloutCache] = None

        def save_artifact(**artifact) -> None:
            if active_cache is None:
                raise RuntimeError("No active rollout cache")
            active_cache.save_batch(**artifact)

        trainer = build_trainer(
            cfg,
            selection.path,
            device,
            viz,
            well_base_path=well_base_path,
            source_cfg=source_cfg,
            # Videos are rendered from cache after inference, so a future style/edit
            # change never requires another model rollout.
            video_validation=False,
            image_validation=False,
            rollout_artifact_callback=save_artifact,
            **trainer_overrides,
        )
        dm = trainer.datamodule
        for split in ("test", "rollout_test"):
            if split not in missing:
                continue
            active_cache = caches[split]
            active_cache.reset()
            active_cache.begin()
            if split == "test":
                loaders = dm.test_dataloaders(replicas=1, rank=0, full=full)
            else:
                loaders = dm.rollout_test_dataloaders(replicas=1, rank=0, full=full)
            loss, metrics = trainer.validation_loop(loaders, split, full=full)
            per_dataset = _dataset_scores(metrics, split, COMPARE_SCORE)
            summary = {
                "loss": float(loss),
                "per_dataset": per_dataset,
                "score": _aggregate_dataset_scores(per_dataset),
                "n_artifacts": len(list(active_cache.artifact_dir.glob("*.npz"))),
            }
            active_cache.finish(summary)
            computed[split] = summary
        active_cache = None

    summaries = {
        split: computed.get(split, cache.summary())
        for split, cache in caches.items()
    }
    test_summary = summaries.get(
        "test", {"loss": float("nan"), "per_dataset": {}, "score": float("nan")}
    )
    rollout_summary = summaries.get(
        "rollout_test",
        {"loss": float("nan"), "per_dataset": {}, "score": float("nan")},
    )
    test_loss = float(test_summary["loss"])
    rollout_loss = float(rollout_summary["loss"])
    test_per_dataset = {
        str(k): float(v) for k, v in test_summary["per_dataset"].items()
    }
    rollout_test_per_dataset = {
        str(k): float(v) for k, v in rollout_summary["per_dataset"].items()
    }

    videos = []
    if make_videos and "rollout_test" in caches:
        videos = _render_cached_videos(
            caches["rollout_test"],
            viz,
            "rollout_test",
            num_detailed_logs=int(trainer_overrides.get("num_detailed_logs", 3)),
            video_style=video_style,
        )
    return {
        "selection": selection,
        "data": data_tag if data is not None else None,
        # Cross-run comparable: mean full_VRMSE over datasets (not train loss_fn).
        "test": _aggregate_dataset_scores(test_per_dataset),
        "rollout_test": _aggregate_dataset_scores(rollout_test_per_dataset),
        "test_loss": test_loss,
        "rollout_test_loss": rollout_loss,
        "test_per_dataset": test_per_dataset,
        "rollout_test_per_dataset": rollout_test_per_dataset,
        # Only show files written by this evaluation. The default viz directory is
        # reused across notebook runs and can contain videos from an older dataset.
        "videos": sorted(videos),
        "viz_folder": str(viz),
        "cache": {split: str(cache.path) for split, cache in caches.items()},
        "cache_hit": not missing,
    }


def _squeeze_to_spatial(arr: np.ndarray, n_spatial_dims: int) -> np.ndarray:
    """``(T, *spatial, C)``, dropping extra singleton spatial axes from 3D loaders."""
    x = np.asarray(arr)
    expected = n_spatial_dims + 2
    while x.ndim > expected:
        squeezed = False
        for axis in range(1, x.ndim - 1):
            if x.shape[axis] == 1:
                x = np.squeeze(x, axis=axis)
                squeezed = True
                break
        if not squeezed:
            break
    if x.ndim != expected:
        raise ValueError(
            f"expected (T, *spatial[{n_spatial_dims}], C); got {x.shape}"
        )
    return x


def vrmse_from_rollouts(
    rollouts: Sequence,
    n_frames: Optional[int] = None,
    eps: float = 1e-5,
) -> float:
    """Mean field-and-time VRMSE matching ``compare_zero_shot``.

    ``n_frames`` keeps only the first N predicted frames (``None`` = the full
    cached trajectory). Samples are averaged within each dataset, then datasets
    are averaged equally. Reads cached physical-unit arrays; no model call.
    """
    from the_well.benchmark.metrics import VRMSE

    per_dataset: dict[str, list[float]] = defaultdict(list)
    for item in rollouts:
        n_spatial = item.metadata.n_spatial_dims
        pred = _squeeze_to_spatial(item.pred, n_spatial)
        ref = _squeeze_to_spatial(item.ref, n_spatial)
        horizon = pred.shape[0] if n_frames is None else min(pred.shape[0], int(n_frames))
        loss = VRMSE.eval(
            torch.as_tensor(pred[:horizon][None]),
            torch.as_tensor(ref[:horizon][None]),
            item.metadata,
            eps=eps,
        )
        per_dataset[item.dataset].append(float(loss.mean().item()))
    return _aggregate_dataset_scores(
        {name: float(np.mean(values)) for name, values in per_dataset.items()}
    )


def spectral_metrics_from_rollouts(
    rollouts: Sequence,
    n_frames: Optional[int] = None,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Mean Well spectral errors for low, medium, and high frequency bins.

    Returns both raw MSE and normalized MSE keys using the names emitted by
    :class:`the_well.benchmark.metrics.binned_spectral_mse`. ``n_frames`` keeps
    only the first N predicted frames. All calculations use cached physical-unit
    trajectories and never execute the model.
    """
    from the_well.benchmark.metrics import binned_spectral_mse

    per_dataset: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in rollouts:
        n_spatial = item.metadata.n_spatial_dims
        pred = _squeeze_to_spatial(item.pred, n_spatial)
        ref = _squeeze_to_spatial(item.ref, n_spatial)
        metadata = item.metadata
        # Walrus inflates 2D data to (H, W, 1), whereas native 2D baselines keep
        # (H, W). The Well's default bins depend on n_spatial_dims, so retaining
        # that artificial singleton would give Walrus different bins.
        while metadata.n_spatial_dims > 2:
            singleton = next(
                (axis for axis, size in enumerate(pred.shape[1:-1], start=1) if size == 1),
                None,
            )
            if singleton is None:
                break
            pred = np.squeeze(pred, axis=singleton)
            ref = np.squeeze(ref, axis=singleton)
            metadata = replace(
                metadata,
                n_spatial_dims=metadata.n_spatial_dims - 1,
                spatial_resolution=tuple(pred.shape[1:-1]),
            )
        horizon = pred.shape[0] if n_frames is None else min(
            pred.shape[0], int(n_frames)
        )
        losses = binned_spectral_mse.eval(
            torch.as_tensor(pred[:horizon][None]),
            torch.as_tensor(ref[:horizon][None]),
            metadata,
            eps=eps,
        )
        for name, loss in losses.items():
            per_dataset[item.dataset][name].append(float(loss.mean().item()))

    dataset_means = {
        dataset: {
            metric: float(np.mean(values)) for metric, values in metrics.items()
        }
        for dataset, metrics in per_dataset.items()
    }
    metric_names = sorted(
        {metric for metrics in dataset_means.values() for metric in metrics}
    )
    return {
        metric: float(
            np.mean(
                [
                    metrics[metric]
                    for metrics in dataset_means.values()
                    if metric in metrics
                ]
            )
        )
        for metric in metric_names
    }


def cached_rollouts(
    run,
    data: Optional[Union[str, DictConfig]] = None,
    *,
    metric: str = ROLLOUT,
    split: str = "rollout_test",
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    full: bool = True,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    **trainer_overrides,
):
    """Return physical-unit rollout artifacts, running inference only on cache miss."""
    trainer_overrides.setdefault("max_rollout_steps", 200)
    if split not in {"test", "rollout_test"}:
        raise ValueError("split must be 'test' or 'rollout_test'")
    selection = select_checkpoint(run, metric, checkpoints_dir)
    source_cfg = load_config(run)
    cfg = source_cfg
    data_tag = (
        data
        if isinstance(data, str)
        else ("custom" if data is not None else "training_data")
    )
    if data is not None:
        cfg = apply_data_config(
            source_cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )
    cache = _rollout_cache(
        run,
        selection,
        cfg,
        str(data_tag),
        split,
        full=full,
        cache_dir=cache_dir,
        trainer_overrides=trainer_overrides,
        source_cfg=source_cfg,
    )
    was_hit = use_cache and cache.complete and not refresh_cache
    if not was_hit:
        evaluate_checkpoint(
            run,
            metric=metric,
            device=device,
            well_base_path=well_base_path,
            full=full,
            checkpoints_dir=checkpoints_dir,
            data=data,
            keep_n_steps_input=keep_n_steps_input,
            data_overrides=data_overrides,
            splits=(split,),
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            **trainer_overrides,
        )
    print(f"{'cache hit' if was_hit else 'cache saved'}: {cache.path}")
    return cache.rollouts()


def _target_test_files(cfg: DictConfig) -> list[tuple[str, pathlib.Path]]:
    """Local test HDF5 files selected by a composed data config."""
    selected: list[tuple[str, pathlib.Path]] = []
    info = OmegaConf.select(cfg, "data.module_parameters.well_dataset_info") or {}
    for dataset_name, metadata in info.items():
        root_value = metadata.get("path")
        if not root_value:
            continue
        include = [str(x) for x in (metadata.get("include_filters") or [])]
        exclude = [str(x) for x in (metadata.get("exclude_filters") or [])]
        for path in sorted((pathlib.Path(root_value) / "data" / "test").glob("*.hdf5")):
            name = path.name
            if include and not any(token in name for token in include):
                continue
            if any(token in name for token in exclude):
                continue
            selected.append((str(dataset_name), path.resolve()))
    if not selected:
        raise FileNotFoundError("No local test HDF5 files selected by the data config")
    return selected


def select_vf_aligned_cohort(
    runs: Sequence,
    data: str,
    *,
    horizon_minutes: int = 20,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
) -> VFAlignedCohort:
    """Common held-out cohort supporting context through VF and a fixed horizon.

    The Well scalar ``t0_frame`` is the first frame *after* VF time zero in these
    morphogenesis files: ``time[t0_frame - 1] == 0`` and
    ``time[t0_frame] == +1``. It can therefore be passed directly as
    ``start_output_steps_at_t``.
    """
    import h5py

    if not runs:
        raise ValueError("at least one run is required")
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    configs = []
    for run in runs:
        cfg = load_config(run)
        cfg = apply_data_config(
            cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )
        configs.append(cfg)
    max_context = max(
        int(OmegaConf.select(cfg, "data.module_parameters.n_steps_input"))
        for cfg in configs
    )
    files = _target_test_files(configs[0])
    eligible: list[pathlib.Path] = []
    excluded: dict[str, str] = {}
    for _dataset, path in files:
        with h5py.File(path, "r") as handle:
            if "scalars/t0_frame" not in handle:
                excluded[path.name] = "missing scalars/t0_frame"
                continue
            time = np.asarray(handle["dimensions/time"], dtype=float)
            output_start = int(round(float(np.asarray(handle["scalars/t0_frame"]))))
        if output_start < 1 or output_start >= len(time):
            excluded[path.name] = f"invalid t0_frame={output_start}"
            continue
        if not np.isclose(time[output_start - 1], 0.0):
            excluded[path.name] = (
                f"time[t0_frame-1]={time[output_start - 1]:g}, expected VF t=0"
            )
            continue
        if output_start < max_context:
            excluded[path.name] = (
                f"only {output_start} frames through VF; needs {max_context}"
            )
            continue
        available = len(time) - output_start
        if available < horizon_minutes:
            excluded[path.name] = (
                f"only {available} post-VF frames; needs {horizon_minutes}"
            )
            continue
        eligible.append(path)
    if not eligible:
        raise ValueError(
            f"No test embryos support {max_context} context frames through VF "
            f"and {horizon_minutes} post-VF frames"
        )
    return VFAlignedCohort(
        files=tuple(eligible),
        excluded=excluded,
        max_context=max_context,
        horizon_minutes=int(horizon_minutes),
    )


def _single_file_rollout_datamodule(
    cfg: DictConfig,
    path: pathlib.Path,
    *,
    output_start: int,
    horizon_minutes: int,
    well_base_path: Optional[str],
):
    """Instantiate an aligned datamodule and return the target file's sample index."""
    info = OmegaConf.select(cfg, "data.module_parameters.well_dataset_info") or {}
    chosen_name = None
    chosen_metadata = None
    for dataset_name, metadata in info.items():
        root = metadata.get("path")
        if root and path.parent.parent.parent.resolve() == pathlib.Path(root).resolve():
            chosen_name = str(dataset_name)
            chosen_metadata = OmegaConf.to_container(metadata, resolve=True)
            break
    if chosen_name is None or not isinstance(chosen_metadata, dict):
        raise ValueError(f"{path} does not belong to any configured dataset root")

    module_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.module_parameters, resolve=True)
    )
    OmegaConf.update(
        module_cfg,
        "well_dataset_info",
        {chosen_name: chosen_metadata},
        merge=False,
    )
    # Construct with the default start to avoid an upstream off-by-one guard that
    # rejects the valid case output_start == n_steps_input. Set the selected
    # rollout-test dataset to the exact per-file index after construction.
    OmegaConf.update(
        module_cfg, "start_rollout_valid_output_at_t", -1, merge=False
    )
    OmegaConf.update(
        module_cfg, "max_rollout_steps", int(horizon_minutes), merge=False
    )
    datamodule = instantiate(
        module_cfg,
        world_size=1,
        rank=0,
        data_workers=0,
        well_base_path=well_base_path or cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    mixed = datamodule.rollout_test_datasets[0]
    inner = mixed.sub_dsets[0]
    inner.start_output_steps_at_t = int(output_start)
    resolved_files = [pathlib.Path(value).resolve() for value in inner.files_paths]
    try:
        file_index = resolved_files.index(path.resolve())
    except ValueError as error:
        raise ValueError(f"{path} was not found in the rollout-test dataset") from error
    sample_index = max(int(inner.file_index_offsets[file_index]), 0)
    return datamodule, sample_index


@torch.no_grad()
def cached_vf_aligned_rollouts(
    run,
    data: str,
    cohort: VFAlignedCohort,
    *,
    metric: str = ROLLOUT,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    **trainer_overrides,
):
    """Return native-context rollouts whose first prediction is VF ``t=+1``."""
    import h5py

    device = device or default_device()
    selection = select_checkpoint(run, metric, checkpoints_dir)
    source_cfg = load_config(run)
    cfg = apply_data_config(
        source_cfg,
        data,
        keep_n_steps_input=keep_n_steps_input,
        keep_field_index_map=True,
        **(data_overrides or {}),
    )
    native_context = int(
        OmegaConf.select(cfg, "data.module_parameters.n_steps_input")
    )
    cohort_names = tuple(path.name for path in cohort.files)
    protocol = {
        "name": "paper_residual_vf_aligned_v1",
        "alignment_scalar": "t0_frame",
        "first_scored_minute": 1,
        "horizon_minutes": cohort.horizon_minutes,
        "native_context": native_context,
        "common_max_context": cohort.max_context,
        "cohort_files": cohort_names,
    }
    trainer_overrides = {
        "max_rollout_steps": cohort.horizon_minutes,
        **trainer_overrides,
    }
    cache = _rollout_cache(
        run,
        selection,
        cfg,
        data,
        "rollout_test_vf_aligned",
        full=True,
        cache_dir=cache_dir,
        trainer_overrides=trainer_overrides,
        extra_settings=protocol,
        source_cfg=source_cfg,
    )
    was_hit = use_cache and cache.complete and not refresh_cache
    if was_hit:
        print(f"cache hit: {cache.path}")
        return cache.rollouts()
    if refresh_cache:
        cache.reset()

    active_cache = cache
    current_source_path: Optional[pathlib.Path] = None

    def save_aligned_artifact(**artifact) -> None:
        if current_source_path is None:
            raise RuntimeError("VF-aligned callback has no active source file")
        with h5py.File(current_source_path, "r") as handle:
            absolute_time = np.asarray(handle["dimensions/time"], dtype=np.float32)
            output_start = int(
                round(float(np.asarray(handle["scalars/t0_frame"])))
            )
        n_input = int(artifact["context"].shape[1])
        n_output = int(artifact["pred"].shape[1])
        artifact["input_time"] = absolute_time[
            output_start - n_input : output_start
        ][None]
        artifact["output_time"] = absolute_time[
            output_start : output_start + n_output
        ][None]
        artifact["file_paths"] = [str(current_source_path)]
        active_cache.save_batch(**artifact)

    viz = pathlib.Path(
        f"./_analysis_viz/{run.name}/zero_shot_{data}_vf_aligned"
    )
    trainer = build_trainer(
        cfg,
        selection.path,
        device,
        viz,
        well_base_path=well_base_path,
        source_cfg=source_cfg,
        video_validation=False,
        image_validation=False,
        rollout_artifact_callback=save_aligned_artifact,
        **trainer_overrides,
    )
    cache.begin()
    try:
        for path in cohort.files:
            current_source_path = path
            with h5py.File(path, "r") as handle:
                output_start = int(
                    round(float(np.asarray(handle["scalars/t0_frame"])))
                )
            datamodule, sample_index = _single_file_rollout_datamodule(
                cfg,
                path,
                output_start=output_start,
                horizon_minutes=cohort.horizon_minutes,
                well_base_path=well_base_path,
            )
            loaders = datamodule.rollout_test_dataloaders(
                replicas=1, rank=0, full=True
            )
            # MixedWellDataset accepts a list of indices as one already-batched sample.
            # Restrict to this embryo and disable pinning: aligned evaluation may run
            # deliberately on CPU while another process occupies the GPU.
            from torch.utils.data import DataLoader, default_convert

            loader = loaders[0]
            aligned_loader = DataLoader(
                loader.dataset,
                batch_size=None,
                sampler=[[sample_index]],
                collate_fn=default_convert,
                num_workers=0,
                pin_memory=False,
            )
            trainer.validation_loop(
                [aligned_loader], "rollout_test_vf_aligned", full=True
            )
        cache.finish(
            {
                "protocol": protocol,
                "n_artifacts": len(list(cache.artifact_dir.glob("*.npz"))),
                "excluded": dict(cohort.excluded),
            }
        )
    except Exception:
        cache.reset()
        raise
    print(f"cache saved: {cache.path}")
    return cache.rollouts()


@torch.no_grad()
def collect_prediction_values(
    run,
    data: Optional[Union[str, DictConfig]] = None,
    *,
    metric: str = ROLLOUT,
    split: str = "test",
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    max_batches: Optional[int] = None,
    fields: Optional[list[str]] = None,
    max_frames: Optional[int] = None,
    horizons: Optional[list] = None,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> dict:
    """Collect flattened prediction / reference values per field for histogram plots.

    ``split`` is ``\"test\"`` (one-step) or ``\"rollout_test\"``.

    ``max_frames`` truncates the rollout time axis before pooling (first N frames).
    ``horizons`` (e.g. ``[5, 10, 20, 30, None]``) runs the model once and returns
    ``{horizon_key: {\"pred\": ..., \"ref\": ..., ...}}`` where ``None`` / ``\"full\"``
    means the full rollout. If ``horizons`` is None, returns a single
    ``{\"pred\": {field: array}, \"ref\": {field: array}, ...}`` dict.
    """
    import numpy as np
    selection = select_checkpoint(run, metric, checkpoints_dir)
    artifacts = cached_rollouts(
        run,
        data,
        metric=metric,
        split=split,
        device=device,
        well_base_path=well_base_path,
        checkpoints_dir=checkpoints_dir,
        keep_n_steps_input=keep_n_steps_input,
        data_overrides=data_overrides,
        full=True,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )

    # Collect per-sample time series from disk, then slice by horizon.
    timed: dict[str, dict[str, list]] = {"pred": {}, "ref": {}}
    for j, artifact in enumerate(artifacts):
        if max_batches is not None and j >= max_batches:
            break
        pred = artifact.pred if max_frames is None else artifact.pred[:max_frames]
        ref = artifact.ref if max_frames is None else artifact.ref[:max_frames]
        for ci, fname in enumerate(artifact.field_names):
            if fields is not None and fname not in fields:
                continue
            # Restore a batch dimension expected by the pooling helper.
            timed["pred"].setdefault(fname, []).append(pred[None, ..., ci])
            timed["ref"].setdefault(fname, []).append(ref[None, ..., ci])

    def _pool(chunks: list, n_frames: Optional[int]) -> "np.ndarray":
        parts = []
        for arr in chunks:
            # arr: (B, T, ...)
            sl = arr if n_frames is None else arr[:, :n_frames]
            parts.append(sl.reshape(-1))
        return np.concatenate(parts) if parts else np.array([])

    meta = {
        "selection": selection,
        "data": data if isinstance(data, str) else None,
        "split": split,
    }

    if horizons is None:
        return {
            "pred": {k: _pool(v, max_frames) for k, v in timed["pred"].items()},
            "ref": {k: _pool(v, max_frames) for k, v in timed["ref"].items()},
            **meta,
        }

    out = {}
    for h in horizons:
        key = "full" if h is None else int(h)
        n = None if h is None else int(h)
        out[key] = {
            "pred": {k: _pool(v, n) for k, v in timed["pred"].items()},
            "ref": {k: _pool(v, n) for k, v in timed["ref"].items()},
            **meta,
            "horizon": key,
        }
    return out


# --------------------------------------------------------------------------- #
# top-level notebook helpers
# --------------------------------------------------------------------------- #
def zero_shot(
    project: str,
    run: str,
    data: str,
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = True,
    make_videos: bool = True,
    checkpoints_dir: Optional[str] = None,
    keep_n_steps_input: bool = True,
    batch_size: int = 1,
    n_steps_input: Optional[int] = None,
    viz_folder: Optional[str] = None,
    compare_to_original: bool = True,
    original_data: Optional[str] = None,
    num_detailed_logs: Optional[int] = None,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    video_style: Optional[Mapping[str, Any]] = None,
    **data_overrides,
) -> dict:
    """Zero-shot: best-by-rollout ``VRMSE`` checkpoint of ``run``, evaluated on ``data``.

    ``data`` is a Hydra data config name under ``walrus/configs/data/`` (e.g.
    ``morphogenesis_WT``). Selects the checkpoint minimizing mean
    ``rollout_valid_*/full_VRMSE_T=all_mean`` among saved epochs, then runs
    **test** and **rollout_test** on that dataset (with videos when
    ``make_videos=True``). Comparison tables report ``VRMSE``, not the train
    ``loss_fn`` scalar.

    When ``compare_to_original=True`` (default), also re-evaluates the same checkpoint
    on an "original" test set and displays a side-by-side comparison table.
    ``original_data`` selects that set (Hydra data config name). If omitted, uses the
    run's training data config — which fails if that path is no longer on disk
    (pass e.g. ``original_data="morphogenesis_WT"``).

    By default keeps the run's ``n_steps_input`` and uses ``batch_size=1``. Pass
    ``n_steps_input=...`` or other kwargs to override ``data.module_parameters``.
    ``num_detailed_logs`` controls how many rollout videos are written (trainer default 3).
    ``video_style`` overrides keys of
    ``walrus.analysis.rollout_video.DEFAULT_VIDEO_STYLE`` (colormap, layout, rows).
    """
    from IPython.display import Video, display

    r = get_run(project, run, entity)
    overrides = dict(data_overrides)
    overrides["batch_size"] = batch_size
    if n_steps_input is not None:
        overrides["n_steps_input"] = n_steps_input
    trainer_overrides = {}
    if num_detailed_logs is not None:
        trainer_overrides["num_detailed_logs"] = num_detailed_logs

    # Same checkpoint for both legs (best by rollout_valid).
    common = dict(
        metric=ROLLOUT,
        device=device,
        well_base_path=well_base_path,
        full=full,
        checkpoints_dir=checkpoints_dir,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        video_style=video_style,
        **trainer_overrides,
    )

    original = None
    original_label = "(run training data)"
    if compare_to_original:
        try:
            if original_data is not None:
                original_label = original_data
                original = evaluate_checkpoint(
                    r,
                    viz_folder=f"./_analysis_viz/{r.name}/original_{original_data}",
                    make_videos=False,
                    data=original_data,
                    keep_n_steps_input=keep_n_steps_input,
                    data_overrides=overrides,
                    **common,
                )
            else:
                original = evaluate_checkpoint(
                    r,
                    viz_folder=f"./_analysis_viz/{r.name}/original_test",
                    make_videos=False,
                    data=None,
                    **common,
                )
        except Exception as e:
            hint = (
                f"Original-data eval failed ({e}). "
                "The run's training path may be missing on disk. "
                'Pass original_data="morphogenesis_WT" (or another available '
                "Hydra data config) to compare against a dataset that exists."
            )
            raise FileNotFoundError(hint) from e

    res = evaluate_checkpoint(
        r,
        viz_folder=viz_folder,
        make_videos=make_videos,
        data=data,
        keep_n_steps_input=keep_n_steps_input,
        data_overrides=overrides,
        **common,
    )
    sel = res["selection"]

    # Side-by-side: original test set vs zero-shot dataset.
    rows = []
    if original is not None:
        rows.append(
            {
                "eval": "original_test",
                "data": original_label,
                "test_VRMSE": original["test"],
                "rollout_test_VRMSE": original["rollout_test"],
            }
        )
    rows.append(
        {
            "eval": "zero_shot",
            "data": data,
            "test_VRMSE": res["test"],
            "rollout_test_VRMSE": res["rollout_test"],
        }
    )
    comparison = pd.DataFrame(rows)
    comparison.insert(0, "ckpt_epoch", sel.epoch)
    comparison.insert(1, "rollout_valid_VRMSE_at_ckpt", sel.value)
    display(comparison)

    if res["test_per_dataset"] or res["rollout_test_per_dataset"]:
        print(f"zero-shot per-dataset VRMSE ({data}):")
        display(
            pd.DataFrame(
                {
                    "test": res["test_per_dataset"],
                    "rollout_test": res["rollout_test_per_dataset"],
                }
            )
        )
    if original is not None and (
        original["test_per_dataset"] or original["rollout_test_per_dataset"]
    ):
        print(f"original per-dataset VRMSE ({original_label}):")
        display(
            pd.DataFrame(
                {
                    "test": original["test_per_dataset"],
                    "rollout_test": original["rollout_test_per_dataset"],
                }
            )
        )

    print(f"checkpoint: {sel.path}")
    print(f"viz_folder: {res['viz_folder']}")
    for path in res["videos"]:
        print(path)
        display(Video(path, embed=True, width=640))
    out = {"zero_shot": res, "comparison": comparison}
    if original is not None:
        out["original"] = original
    return out


def compare_zero_shot(
    project: str,
    runs: list[str],
    data: str,
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = True,
    make_videos: bool = False,
    checkpoints_dirs: Optional[dict[str, str]] = None,
    keep_n_steps_input: bool = True,
    batch_size: int = 1,
    n_steps_input: Optional[int] = None,
    skip_missing_checkpoints: bool = True,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    video_style: Optional[Mapping[str, Any]] = None,
    **data_overrides,
) -> pd.DataFrame:
    """Zero-shot **rollout_test VRMSE** for the best-by-rollout checkpoint of each run.

    Does **not** re-evaluate the original training test set. Skips one-step ``test``.
    Checkpoint selection is mean ``rollout_valid_*/full_VRMSE_T=all_mean`` among
    saved epochs. Duplicate display names are resolved once (newest wandb run).

    ``checkpoints_dirs`` optionally maps a run name/id to an override checkpoint dir.
    Runs with no on-disk checkpoints are skipped when ``skip_missing_checkpoints``.
    """
    device = device or default_device()
    checkpoints_dirs = checkpoints_dirs or {}
    overrides = dict(data_overrides)
    overrides["batch_size"] = batch_size
    if n_steps_input is not None:
        overrides["n_steps_input"] = n_steps_input

    seen_ids: set[str] = set()
    rows = []
    for name in runs:
        r = get_run(project, name, entity)
        if r.id in seen_ids:
            continue
        seen_ids.add(r.id)
        ckpt_dir = pathlib.Path(
            checkpoints_dirs.get(name)
            or checkpoints_dirs.get(r.id)
            or r.config["checkpoint"]["save_dir"]
        )
        if not available_checkpoints(ckpt_dir) and not allows_missing_checkpoint(r):
            msg = f"No checkpoints found under {ckpt_dir}"
            if skip_missing_checkpoints:
                print(f"skip {r.name} ({r.id}): {msg}")
                continue
            raise FileNotFoundError(msg)
        res = evaluate_checkpoint(
            r,
            metric=ROLLOUT,
            device=device,
            well_base_path=well_base_path,
            full=full,
            make_videos=make_videos,
            checkpoints_dir=str(ckpt_dir),
            data=data,
            keep_n_steps_input=keep_n_steps_input,
            data_overrides=overrides,
            splits=("rollout_test",),
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            video_style=video_style,
        )
        cache_path = res["cache"].get("rollout_test")
        print(
            f"{'cache hit' if res['cache_hit'] else 'cache saved'}: "
            f"{r.name} -> {cache_path}"
        )
        sel = res["selection"]
        rows.append(
            {
                "run": r.name,
                "id": r.id,
                "ckpt_epoch": sel.epoch,
                "rollout_valid_VRMSE": sel.value,
                "zero_shot_rollout_test_VRMSE": res["rollout_test"],
            }
        )
    table = pd.DataFrame(rows)
    if len(table):
        table = table.set_index("run")
    return table


def compare_paper_residual(
    project: str,
    runs: Sequence[str],
    data: str,
    *,
    entity: Optional[str] = None,
    horizon_minutes: int = 20,
    horizons: Sequence[int] = (15, 20),
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dirs: Optional[Mapping[str, str]] = None,
    keep_n_steps_input: bool = True,
    batch_size: int = 1,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    include_paper_mean_field: bool = True,
    **data_overrides,
) -> dict[str, Any]:
    """VF-aligned, velocity-only SI Eq. (6) benchmark for several runs.

    Native contexts end at VF ``t=0`` and only genuine predictions from ``t=+1``
    onward are scored. All models use the intersection cohort eligible for the
    largest native context. The returned ``table`` reports fractions; multiply by
    100 for the paper's percentage convention.
    """
    from walrus.analysis.paper_residual import (
        mean_field_residual_from_rollouts,
        paper_mean_velocity,
        residual_metrics_from_rollouts,
    )

    device = device or default_device()
    checkpoints_dirs = dict(checkpoints_dirs or {})
    overrides = dict(data_overrides)
    overrides["batch_size"] = batch_size
    resolved = []
    seen_ids: set[str] = set()
    for name in runs:
        run = get_run(project, name, entity)
        if run.id not in seen_ids:
            resolved.append(run)
            seen_ids.add(run.id)
    cohort = select_vf_aligned_cohort(
        resolved,
        data,
        horizon_minutes=horizon_minutes,
        keep_n_steps_input=keep_n_steps_input,
        data_overrides=overrides,
    )

    aggregates = {}
    rollout_sets = {}
    rows = []
    for run in resolved:
        ckpt_override = checkpoints_dirs.get(run.name) or checkpoints_dirs.get(run.id)
        rollouts = cached_vf_aligned_rollouts(
            run,
            data,
            cohort,
            device=device,
            well_base_path=well_base_path,
            checkpoints_dir=ckpt_override,
            keep_n_steps_input=keep_n_steps_input,
            data_overrides=overrides,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        aggregate = residual_metrics_from_rollouts(
            rollouts, horizons=horizons
        )
        aggregates[run.name] = aggregate
        rollout_sets[run.name] = rollouts
        selection = select_checkpoint(run, ROLLOUT, ckpt_override)
        row = {
            "run": run.name,
            "id": run.id,
            "ckpt_epoch": selection.epoch,
        }
        for horizon in horizons:
            horizon = int(horizon)
            row[f"residual_t{horizon}"] = aggregate.horizon_scores[horizon]
            row[f"residual_t{horizon}_std"] = aggregate.horizon_std[horizon]
            row[f"residual_t{horizon}_n"] = aggregate.horizon_n[horizon]
        rows.append(row)

    paper_mean = None
    if include_paper_mean_field and rollout_sets:
        first_run = next(iter(rollout_sets))
        reference_rollouts = rollout_sets[first_run]
        cfg = apply_data_config(
            load_config(resolved[0]),
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **overrides,
        )
        selected = _target_test_files(cfg)
        roots = sorted(
            {
                path.parent.parent.parent
                for _dataset, path in selected
            },
            key=str,
        )
        all_paths = [
            path
            for root in roots
            for split in ("train", "valid", "test")
            for path in sorted((root / "data" / split).glob("*.hdf5"))
        ]
        mean_velocity = paper_mean_velocity(all_paths)
        paper_mean = mean_field_residual_from_rollouts(
            reference_rollouts, mean_velocity, horizons=horizons
        )
        aggregate_key = "Paper mean-field (entire dataset)"
        aggregates[aggregate_key] = paper_mean
        row = {"run": aggregate_key, "id": "paper_definition", "ckpt_epoch": np.nan}
        for horizon in horizons:
            horizon = int(horizon)
            row[f"residual_t{horizon}"] = paper_mean.horizon_scores[horizon]
            row[f"residual_t{horizon}_std"] = paper_mean.horizon_std[horizon]
            row[f"residual_t{horizon}_n"] = paper_mean.horizon_n[horizon]
        rows.append(row)

    table = pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()
    return {
        "table": table,
        "aggregates": aggregates,
        "rollouts": rollout_sets,
        "cohort": cohort,
        "paper_mean_field": paper_mean,
    }


def show_run(
    project: str,
    run: str,
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = False,
    make_videos: bool = False,
    checkpoints_dir: Optional[str] = None,
    metric: str = ROLLOUT,
    **trainer_overrides,
) -> dict[str, dict]:
    """Evaluate one (or both) selected checkpoints of a run.

    ``metric`` chooses which saved checkpoint to load:

    - ``\"rollout_valid\"`` (default) — best by rollout validation VRMSE
    - ``\"valid\"`` — best by one-step validation VRMSE
    - ``\"both\"`` — evaluate both (previous behavior)

    Runs test + rollout-test for each selected checkpoint, prints a summary
    table of ``VRMSE`` scores, and displays any rollout videos. Returns the
    raw results keyed by selection metric.
    """
    from IPython.display import Video, display

    aliases = {
        "rollout_valid": ROLLOUT,
        "rollout_val": ROLLOUT,
        "rollout": ROLLOUT,
        "valid": SINGLE_STEP,
        "val": SINGLE_STEP,
        "single_step": SINGLE_STEP,
        "both": "both",
    }
    key = aliases.get(metric, metric)
    if key == "both":
        metrics = (ROLLOUT, SINGLE_STEP)
    elif key in (ROLLOUT, SINGLE_STEP):
        metrics = (key,)
    else:
        raise ValueError(
            f"metric must be 'rollout_valid', 'valid', or 'both'; got {metric!r}"
        )

    r = get_run(project, run, entity)
    results = {}
    for m in metrics:
        results[m] = evaluate_checkpoint(
            r,
            metric=m,
            device=device,
            well_base_path=well_base_path,
            full=full,
            make_videos=make_videos,
            checkpoints_dir=checkpoints_dir,
            **trainer_overrides,
        )

    summary = pd.DataFrame(
        {
            m: {
                "epoch": res["selection"].epoch,
                "test_VRMSE": res["test"],
                "rollout_test_VRMSE": res["rollout_test"],
            }
            for m, res in results.items()
        }
    ).T
    summary.index.name = f"{run} — selected by (VRMSE)"
    display(summary)

    for m, res in results.items():
        for path in res["videos"]:
            print(f"[{m}] {path}")
            display(Video(path, embed=True, width=640))
    return results


def compare_runs(
    project: str,
    runs: list[str],
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = True,
    checkpoints_dirs: Optional[dict[str, str]] = None,
    **trainer_overrides,
) -> pd.DataFrame:
    """Table of test / rollout_test **VRMSE** for the best-by-rollout and
    best-by-single-step checkpoint of each run.

    Checkpoint selection and table scores use mean
    ``{split}_*/full_VRMSE_T=all_mean`` (not the train ``loss_fn`` scalar).

    ``checkpoints_dirs`` optionally maps a run name to an override checkpoint dir.
    """
    device = device or default_device()
    checkpoints_dirs = checkpoints_dirs or {}
    labels = {ROLLOUT: "best_by_rollout", SINGLE_STEP: "best_by_single_step"}
    rows = {}
    for name in runs:
        r = get_run(project, name, entity)
        ckpt_dir = checkpoints_dirs.get(name)
        cache: dict[pathlib.Path, dict] = {}
        row = {}
        for metric, label in labels.items():
            sel = select_checkpoint(r, metric, ckpt_dir)
            if sel.path not in cache:
                cache[sel.path] = evaluate_checkpoint(
                    r,
                    metric=metric,
                    device=device,
                    well_base_path=well_base_path,
                    full=full,
                    checkpoints_dir=ckpt_dir,
                    **trainer_overrides,
                )
            res = cache[sel.path]
            row[(label, "epoch")] = sel.epoch
            row[(label, "test_VRMSE")] = res["test"]
            row[(label, "rollout_test_VRMSE")] = res["rollout_test"]
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def _squeeze_spatial(x: "np.ndarray") -> "np.ndarray":
    """Drop singleton depth so fields are (T, H, W) or (T, H, W, C)."""
    import numpy as np

    x = np.asarray(x)
    # Common Walrus layout after formatter: (T, H, W, 1) or (T, H, W, 1, C)
    while x.ndim >= 3 and x.shape[-2] == 1:
        x = np.squeeze(x, axis=-2)
    if x.ndim >= 3 and x.shape[-1] == 1 and x.shape[-2] != 1:
        # already channel-less spatial; leave alone
        pass
    return x


def _velocity_from_fields(
    fields: "np.ndarray",
    field_names: list[str],
    mask: "np.ndarray",
) -> "np.ndarray":
    """Stack velocity_x/y (or _ap/_dv) into (T, H, W, 2) from a (T, ..., C) tensor."""
    import numpy as np

    used = [f for i, f in enumerate(field_names) if bool(mask[i])]
    name_to_idx = {f: i for i, f in enumerate(used)}
    pairs = [
        ("velocity_x", "velocity_y"),
        ("velocity_ap", "velocity_dv"),
    ]
    for a, b in pairs:
        if a in name_to_idx and b in name_to_idx:
            vx = fields[..., name_to_idx[a]]
            vy = fields[..., name_to_idx[b]]
            v = np.stack([vx, vy], axis=-1)
            return _squeeze_spatial(v)
    raise KeyError(
        f"Could not find velocity components in {used}. "
        "Expected velocity_x/y or velocity_ap/dv."
    )


def _valid_from_fields(
    fields: "np.ndarray",
    field_names: list[str],
    mask: "np.ndarray",
    threshold: float = 0.5,
) -> "np.ndarray":
    """Optional valid_velocity channel → bool mask; else all-True."""
    import numpy as np

    used = [f for i, f in enumerate(field_names) if bool(mask[i])]
    if "valid_velocity" not in used:
        spatial = _squeeze_spatial(fields[..., 0])
        return np.ones(spatial.shape, dtype=bool)
    idx = used.index("valid_velocity")
    return _squeeze_spatial(fields[..., idx]) >= threshold


@torch.no_grad()
def evaluate_t0_omega(
    run,
    metric: str = ROLLOUT,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    data: Optional[Union[str, DictConfig]] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    split: str = "rollout_test",
    peak_rule: str = "first",
    peak_fraction: float = 0.5,
    scale_velocity_to_um: bool = True,
    max_batches: Optional[int] = None,
    **trainer_overrides,
) -> pd.DataFrame:
    """Compute ``t0_omega`` on GT and open-loop predictions for each rollout trajectory.

    For each full-trajectory sample we stitch ``input_fields`` (context) with the
    rollout outputs so the VF→GBE landmark is covered even when it falls inside
    the context window:

    - GT:   concat(context, y_ref)
    - pred: concat(context, y_pred)   # teacher-forced context, then open-loop

    Absolute developmental time ``t`` and grid spacings are read from the Well
    file (batch time grids are normalized to start at 0). Well fields are assumed
    already AP-cropped.
    """
    import numpy as np
    from the_well.data.utils import flatten_field_names

    from walrus.analysis.t0_omega import PIX_UM_ORIG, load_well, t0_omega

    device = device or default_device()
    selection = select_checkpoint(run, metric, checkpoints_dir)
    source_cfg = load_config(run)
    cfg = source_cfg
    if data is not None:
        cfg = apply_data_config(
            source_cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )
    viz = pathlib.Path("/tmp/walrus_t0_omega")
    viz.mkdir(parents=True, exist_ok=True)
    # Rollouts can be long; keep max_rollout_steps large enough for full trajs.
    trainer_overrides = {
        "max_rollout_steps": trainer_overrides.pop("max_rollout_steps", 200),
        **trainer_overrides,
    }
    trainer = build_trainer(
        cfg,
        selection.path,
        device,
        viz,
        well_base_path=well_base_path,
        source_cfg=source_cfg,
        video_validation=False,
        image_validation=False,
        **trainer_overrides,
    )
    dm = trainer.datamodule
    if split == "rollout_test":
        loaders = dm.rollout_test_dataloaders(replicas=1, rank=0, full=True)
    elif split == "rollout_valid":
        loaders = dm.rollout_val_dataloaders(replicas=1, rank=0, full=True)
    else:
        raise ValueError("split must be 'rollout_test' or 'rollout_valid'")

    rows = []
    for loader in loaders:
        dataset = loader.dataset.sub_dsets[0]
        dset_name = dataset.metadata.dataset_name
        field_names = flatten_field_names(dataset.metadata, include_constants=False)
        formatter = trainer.formatter_dict[dset_name]
        file_paths = list(getattr(dataset, "files_paths", []) or [])

        for j, batch in enumerate(loader):
            if max_batches is not None and j >= max_batches:
                break
            y_pred, y_ref = trainer.rollout_model(
                trainer.model, batch, formatter, train=False
            )
            mask = batch["padded_field_mask"]
            if mask.shape[0] != y_pred.shape[-1]:
                mask = mask[: y_pred.shape[-1]]
            y_pred = y_pred[..., mask]
            y_ref = y_ref[..., mask]
            # B=1 for morph rollouts; take first sample.
            context = batch["input_fields"][0].detach().float().cpu().numpy()
            context = context[..., mask.cpu().numpy()]
            pred = y_pred[0].detach().float().cpu().numpy()
            ref = y_ref[0].detach().float().cpu().numpy()

            gt_traj = np.concatenate([context, ref], axis=0)
            pred_traj = np.concatenate([context, pred], axis=0)
            n = min(gt_traj.shape[0], pred_traj.shape[0])
            gt_traj, pred_traj = gt_traj[:n], pred_traj[:n]

            v_gt = _velocity_from_fields(gt_traj, field_names, mask.cpu().numpy())
            v_pred = _velocity_from_fields(pred_traj, field_names, mask.cpu().numpy())

            scale = PIX_UM_ORIG if scale_velocity_to_um else 1.0
            v_gt_scaled = v_gt * scale
            v_pred_scaled = v_pred * scale

            # Absolute time / spacing from the Well file when available.
            well_path = file_paths[j] if j < len(file_paths) else None
            if well_path is not None:
                well = load_well(well_path)
                t = well["t"][:n]
                dx, dy = well["dx"], well["dy"]
                t0v_frame = well["t0v_frame"]
                well_v = (
                    well["v"]
                    if scale_velocity_to_um
                    else well["v"] / PIX_UM_ORIG
                )
                # This must be the same trajectory as concat(context, y_ref).
                # Fail loudly rather than reporting two contradictory GT values.
                if not np.allclose(
                    v_gt_scaled,
                    well_v[:n],
                    rtol=1e-5,
                    atol=1e-6,
                    equal_nan=True,
                ):
                    max_diff = float(
                        np.nanmax(np.abs(v_gt_scaled - well_v[:n]))
                    )
                    raise RuntimeError(
                        "Rollout GT does not match its Well file "
                        f"{well_path} (max |Δv|={max_diff:.3g}). "
                        "Check evaluation sampler/file ordering."
                    )
                t0_gt = t0_omega(
                    well_v,
                    well["t"],
                    dx,
                    dy,
                    peak_rule=peak_rule,
                    peak_fraction=peak_fraction,
                )
            else:
                tin = batch["input_time_grid"][0].detach().cpu().numpy()
                tout = batch["output_time_grid"][0].detach().cpu().numpy()
                t = np.concatenate([tin, tout])[:n]
                grid = batch["space_grid"][0].detach().cpu().numpy()
                dx = float(np.mean(np.diff(grid[:, 0, 0, 0])))
                dy = float(np.mean(np.diff(grid[0, :, 0, 1])))
                t0v_frame = float("nan")
                t0_gt = t0_omega(
                    v_gt_scaled,
                    t,
                    dx,
                    dy,
                    peak_rule=peak_rule,
                    peak_fraction=peak_fraction,
                )
            t0_pred = t0_omega(
                v_pred_scaled,
                t,
                dx,
                dy,
                peak_rule=peak_rule,
                peak_fraction=peak_fraction,
            )
            rows.append(
                {
                    "dataset": dset_name,
                    "batch": j,
                    "file": pathlib.Path(well_path).name if well_path else None,
                    "n_frames": n,
                    "t0v_frame_index": t0v_frame,
                    "t0_omega_gt": t0_gt,
                    "t0_omega_pred": t0_pred,
                    "abs_err": abs(t0_pred - t0_gt)
                    if np.isfinite(t0_pred) and np.isfinite(t0_gt)
                    else float("nan"),
                    "selection_epoch": selection.epoch,
                    "selection_metric": metric,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_flow_metrics(
    run,
    metric: str = ROLLOUT,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    data: Optional[Union[str, DictConfig]] = None,
    keep_n_steps_input: bool = True,
    data_overrides: Optional[dict] = None,
    split: str = "rollout_test",
    peak_rule: str = "first",
    peak_fraction: float = 0.5,
    scale_velocity_to_um: bool = True,
    rms_smooth_window: int = 0,
    max_batches: Optional[int] = None,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    **trainer_overrides,
) -> list:
    """Compute Fig. 3 flow metrics on GT vs open-loop predictions per embryo.

    For each full-trajectory sample we stitch ``input_fields`` (context) with the
    rollout outputs:

    - GT:   ``concat(context, y_ref)``
    - pred: ``concat(context, y_pred)``  (teacher-forced context, then open-loop)

    Returns a list of ``EmbryoFlowResult`` (see ``walrus.analysis.flow_metrics``)
    with:

    - vorticity Pearson autocorrelation matrices (SI Note 10 Eq. 8; Fig. 3c)
    - r.m.s. tissue velocity curves (SI Eq. 9 / Note 13; Fig. 3i)

    Time is absolute developmental time from the Well file. ``t_rel`` is relative to
    the GT onset of GBE, defined as in the paper by the maximum of the r.m.s. velocity
    derivative, so ``t = 0`` matches the axes of Figs. 3 and 5. The ``t0_omega``
    vorticity landmark is still reported per embryo for reference.
    """
    import numpy as np
    from walrus.analysis.flow_metrics import (
        EmbryoFlowResult,
        align_time_to_t0,
        flow_autocorrelation,
        gbe_onset_from_rms,
        rms_velocity,
    )
    from walrus.analysis.t0_omega import PIX_UM_ORIG, load_well, t0_omega

    trainer_overrides = {
        "max_rollout_steps": trainer_overrides.pop("max_rollout_steps", 200),
        **trainer_overrides,
    }
    if split != "rollout_test":
        raise ValueError(
            "Cached flow analysis currently supports split='rollout_test' only"
        )
    artifacts = cached_rollouts(
        run,
        data,
        metric=metric,
        split=split,
        device=device,
        well_base_path=well_base_path,
        checkpoints_dir=checkpoints_dir,
        keep_n_steps_input=keep_n_steps_input,
        data_overrides=data_overrides,
        full=True,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        **trainer_overrides,
    )

    results = []
    for j, artifact in enumerate(artifacts):
        if max_batches is not None and j >= max_batches:
            break
        gt_traj = np.concatenate([artifact.context, artifact.ref], axis=0)
        pred_traj = np.concatenate([artifact.context, artifact.pred], axis=0)
        n = min(gt_traj.shape[0], pred_traj.shape[0])
        gt_traj, pred_traj = gt_traj[:n], pred_traj[:n]
        channel_mask = np.ones(len(artifact.field_names), dtype=bool)
        v_gt = _velocity_from_fields(
            gt_traj, list(artifact.field_names), channel_mask
        )
        v_pred = _velocity_from_fields(
            pred_traj, list(artifact.field_names), channel_mask
        )

        well_path = artifact.file
        if well_path is not None and pathlib.Path(well_path).is_file():
            well = load_well(well_path)
            t = well["t"][:n]
            dx, dy = well["dx"], well["dy"]
        else:
            t = artifact.time[:n]
            grid = artifact.space_grid
            # Walrus represents a 2D grid as H×W×1×3; 2D baselines use H×W×2.
            if grid.ndim == 4 and grid.shape[-2] == 1:
                grid = np.squeeze(grid, axis=-2)
            dx = float(np.mean(np.diff(grid[:, 0, 0])))
            dy = float(np.mean(np.diff(grid[0, :, 1])))

        scale = PIX_UM_ORIG if scale_velocity_to_um else 1.0
        v_gt_s = v_gt * scale
        v_pred_s = v_pred * scale
        t0_gt = t0_omega(
            v_gt_s,
            t,
            dx,
            dy,
            peak_rule=peak_rule,
            peak_fraction=peak_fraction,
        )
        rms_gt = rms_velocity(v_gt_s)
        rms_pred = rms_velocity(v_pred_s)
        onset_gt = gbe_onset_from_rms(t, rms_gt, smooth_window=rms_smooth_window)
        onset_pred = gbe_onset_from_rms(
            t, rms_pred, smooth_window=rms_smooth_window
        )
        origin = onset_gt if np.isfinite(onset_gt) else 0.0
        results.append(
            EmbryoFlowResult(
                file=pathlib.Path(well_path).name if well_path else None,
                dataset=artifact.dataset,
                batch=artifact.batch,
                t=t,
                t_rel=align_time_to_t0(t, origin),
                gbe_onset_gt=float(onset_gt),
                gbe_onset_pred=float(onset_pred),
                t0_omega_gt=float(t0_gt),
                rms_gt=rms_gt,
                rms_pred=rms_pred,
                autocorr_gt=flow_autocorrelation(v_gt_s, dx, dy),
                autocorr_pred=flow_autocorrelation(v_pred_s, dx, dy),
            )
        )
    return results


def compare_flow_metrics(
    project: str,
    runs: list[str],
    data: Optional[Union[str, DictConfig]] = None,
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    checkpoints_dirs: Optional[dict[str, str]] = None,
    metric: str = ROLLOUT,
    split: str = "rollout_test",
    keep_n_steps_input: bool = True,
    batch_size: int = 1,
    n_steps_input: Optional[int] = None,
    max_rollout_steps: int = 200,
    skip_missing_checkpoints: bool = True,
    cache_dir: Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    refresh_cache: bool = False,
    **flow_kwargs,
) -> dict[str, list]:
    """Flow metrics for the best-by-``metric`` checkpoint of several runs.

    Returns ``{run_name: [EmbryoFlowResult, ...]}`` in the order given, ready to hand
    to ``plot_rms_velocity_multi_run`` for a single overlaid figure. Pass ``data`` to
    evaluate zero-shot on another dataset.

    Duplicate display names are resolved once (newest wandb run) and runs with no
    on-disk checkpoints are skipped when ``skip_missing_checkpoints``.
    """
    device = device or default_device()
    checkpoints_dirs = checkpoints_dirs or {}
    overrides = {"batch_size": batch_size}
    if n_steps_input is not None:
        overrides["n_steps_input"] = n_steps_input

    seen_ids: set[str] = set()
    results_by_run: dict[str, list] = {}
    for name in runs:
        r = get_run(project, name, entity)
        if r.id in seen_ids:
            continue
        seen_ids.add(r.id)
        ckpt_dir = pathlib.Path(
            checkpoints_dirs.get(name)
            or checkpoints_dirs.get(r.id)
            or r.config["checkpoint"]["save_dir"]
        )
        if not available_checkpoints(ckpt_dir) and not allows_missing_checkpoint(r):
            msg = f"No checkpoints found under {ckpt_dir}"
            if skip_missing_checkpoints:
                print(f"skip {r.name} ({r.id}): {msg}")
                continue
            raise FileNotFoundError(msg)
        results_by_run[r.name] = evaluate_flow_metrics(
            r,
            metric=metric,
            device=device,
            well_base_path=well_base_path,
            checkpoints_dir=str(ckpt_dir),
            data=data,
            keep_n_steps_input=keep_n_steps_input,
            data_overrides=overrides,
            split=split,
            max_rollout_steps=max_rollout_steps,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            **flow_kwargs,
        )
        print(f"{r.name}: {len(results_by_run[r.name])} embryos")
    return results_by_run
