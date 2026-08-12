"""Analyze Walrus checkpoints from Weights & Biases runs.

Given a wandb project and run, locate the checkpoint that minimizes a validation
metric, reload the model, and run the same test / rollout-test evaluation used in
training. See ``demo_notebooks/analyze_checkpoints.ipynb`` for usage, and
``demo_notebooks/zero_shot_eval.ipynb`` for evaluating a run on a different dataset.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
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

# Old on-disk names → current locations (datasets renamed in place).
_DATA_PATH_ALIASES = {
    "/data/lcornelis/morphogenesis_data/processed_morphodynamic_atlas": (
        "/data/lcornelis/morphogenesis_data/WT"
    ),
}


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
    **data_overrides,
) -> DictConfig:
    """Return a copy of ``run_cfg`` whose ``data`` block is replaced by ``data``.

    By default keeps the run's ``n_steps_input`` and ``field_index_map_override`` so
    the model architecture (context length + embed/debed width) still matches the
    checkpoint. Pass ``keep_field_index_map=False`` only if you intentionally want
    to rebuild/align to the new dataset's smaller field set.

    Extra kwargs are applied under ``data.module_parameters`` (e.g. ``batch_size=1``).
    """
    cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
    data_cfg = load_data_config(data) if isinstance(data, str) else data
    trained_n_steps = OmegaConf.select(
        run_cfg, "data.module_parameters.n_steps_input"
    )
    trained_field_map = OmegaConf.select(run_cfg, "data.field_index_map_override")
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


def available_checkpoints(ckpt_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    """Map epoch -> checkpoint dir for every saved checkpoint (step_*, best, last)."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    epochs: dict[int, pathlib.Path] = {}
    for d in sorted(ckpt_dir.glob("step_*")):
        try:
            epochs[int(d.name.split("_")[1])] = d
        except ValueError:
            continue
    for name in ("best", "last"):
        meta = ckpt_dir / name / "metadata.pt"
        if meta.exists():
            epoch = torch.load(meta, weights_only=False).get("epoch")
            if epoch is not None:
                epochs.setdefault(int(epoch), ckpt_dir / name)
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
def build_trainer(
    cfg: DictConfig,
    ckpt_path: pathlib.Path,
    device: torch.device,
    viz_folder: pathlib.Path,
    well_base_path: Optional[str] = None,
    **trainer_overrides,
) -> Trainer:
    """Rebuild the model + Trainer from a config and load ``ckpt_path`` weights."""
    datamodule = instantiate(
        cfg.data.module_parameters,
        world_size=1,
        rank=0,
        data_workers=cfg.get("data_workers", 1),
        well_base_path=well_base_path or cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    field_to_index_map = datamodule.train_dataset.field_to_index_map
    model = instantiate(cfg.model, n_states=max(field_to_index_map.values()) + 1)
    # Reapply finetuning structural changes (e.g. learnable per-axis RoPE) so the
    # architecture matches the checkpoint, exactly as train.py does before loading.
    if "finetuning_mods" in cfg and hasattr(model, "add_ft_options"):
        model.add_ft_options(cfg.finetuning_mods)
    loader = CheckPointLoader(
        save_dir=ckpt_path.parent, load_checkpoint_path=ckpt_path, prioritize_resume=False
    )
    loader.load(model, local=True)
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
    **trainer_overrides,
) -> dict:
    """Load the checkpoint best by ``metric`` and run test + rollout-test evaluation.

    Pass ``data`` (Hydra data config name or DictConfig) to evaluate on a different
    dataset than the run was trained on (zero-shot). ``data_overrides`` are applied
    under ``data.module_parameters`` (e.g. ``{"batch_size": 1}``).
    """
    device = device or default_device()
    selection = select_checkpoint(run, metric, checkpoints_dir)
    data_tag = data if isinstance(data, str) else "custom"
    default_viz = (
        f"./_analysis_viz/{run.name}/zero_shot_{data_tag}"
        if data is not None
        else f"./_analysis_viz/{run.name}/{metric}"
    )
    viz = pathlib.Path(viz_folder or default_viz)
    viz.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run)
    if data is not None:
        cfg = apply_data_config(
            cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )

    trainer = build_trainer(
        cfg,
        selection.path,
        device,
        viz,
        well_base_path=well_base_path,
        video_validation=make_videos,
        image_validation=make_videos,
        **trainer_overrides,
    )
    dm = trainer.datamodule
    test_loss, test_metrics = trainer.validation_loop(
        dm.test_dataloaders(replicas=1, rank=0, full=full), "test", full=full
    )
    rollout_loss, rollout_metrics = trainer.validation_loop(
        dm.rollout_test_dataloaders(replicas=1, rank=0, full=full),
        "rollout_test",
        full=full,
    )
    test_per_dataset = _dataset_scores(test_metrics, "test", COMPARE_SCORE)
    rollout_test_per_dataset = _dataset_scores(
        rollout_metrics, "rollout_test", COMPARE_SCORE
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
        "videos": sorted(str(p) for p in viz.rglob("*.mp4")),
        "viz_folder": str(viz),
    }


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
    from the_well.data.utils import flatten_field_names

    device = device or default_device()
    selection = select_checkpoint(run, metric, checkpoints_dir)
    cfg = load_config(run)
    if data is not None:
        cfg = apply_data_config(
            cfg,
            data,
            keep_n_steps_input=keep_n_steps_input,
            keep_field_index_map=True,
            **(data_overrides or {}),
        )
    viz = pathlib.Path("/tmp/walrus_pred_dist")
    viz.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        cfg,
        selection.path,
        device,
        viz,
        well_base_path=well_base_path,
        video_validation=False,
        image_validation=False,
    )
    dm = trainer.datamodule
    if split == "test":
        loaders = dm.test_dataloaders(replicas=1, rank=0, full=True)
    elif split == "rollout_test":
        loaders = dm.rollout_test_dataloaders(replicas=1, rank=0, full=True)
    else:
        raise ValueError("split must be 'test' or 'rollout_test'")

    # Collect per-sample time series, then slice by horizon (one model pass).
    timed: dict[str, dict[str, list]] = {"pred": {}, "ref": {}}
    for loader in loaders:
        dataset = loader.dataset.sub_dsets[0]
        dset_name = dataset.metadata.dataset_name
        field_names = flatten_field_names(dataset.metadata, include_constants=False)
        formatter = trainer.formatter_dict[dset_name]
        for j, batch in enumerate(loader):
            if max_batches is not None and j >= max_batches:
                break
            y_pred, y_ref = trainer.rollout_model(
                trainer.model, batch, formatter, train=False
            )
            # y_* : B, T, ..., C
            mask = batch["padded_field_mask"]
            if mask.shape[0] != y_pred.shape[-1]:
                mask = mask[: y_pred.shape[-1]]
            y_pred = y_pred[..., mask]
            y_ref = y_ref[..., mask]
            if max_frames is not None:
                y_pred = y_pred[:, :max_frames]
                y_ref = y_ref[:, :max_frames]
            used = [f for i, f in enumerate(field_names) if mask[i]]
            for ci, fname in enumerate(used):
                if fields is not None and fname not in fields:
                    continue
                # Keep (B, T, spatial...) so horizons can truncate T.
                timed["pred"].setdefault(fname, []).append(
                    y_pred[..., ci].detach().float().cpu().numpy()
                )
                timed["ref"].setdefault(fname, []).append(
                    y_ref[..., ci].detach().float().cpu().numpy()
                )

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


def show_run(
    project: str,
    run: str,
    entity: Optional[str] = None,
    device: Optional[torch.device] = None,
    well_base_path: Optional[str] = None,
    full: bool = False,
    make_videos: bool = False,
    checkpoints_dir: Optional[str] = None,
    **trainer_overrides,
) -> dict[str, dict]:
    """Evaluate the best-by-rollout and best-by-single-step VRMSE checkpoints of one run.

    Runs test + rollout-test for both checkpoints, prints a summary table of
    ``VRMSE`` scores, and displays any rollout videos. Returns the raw results
    keyed by selection metric.
    """
    from IPython.display import Video, display

    r = get_run(project, run, entity)
    results = {}
    for metric in (ROLLOUT, SINGLE_STEP):
        results[metric] = evaluate_checkpoint(
            r,
            metric=metric,
            device=device,
            well_base_path=well_base_path,
            full=full,
            make_videos=make_videos,
            checkpoints_dir=checkpoints_dir,
            **trainer_overrides,
        )

    summary = pd.DataFrame(
        {
            metric: {
                "epoch": res["selection"].epoch,
                "test_VRMSE": res["test"],
                "rollout_test_VRMSE": res["rollout_test"],
            }
            for metric, res in results.items()
        }
    ).T
    summary.index.name = f"{run} — selected by (VRMSE)"
    display(summary)

    for metric, res in results.items():
        for path in res["videos"]:
            print(f"[{metric}] {path}")
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
    cfg = load_config(run)
    if data is not None:
        cfg = apply_data_config(
            cfg,
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

            # Absolute time / spacing from the Well file when available.
            well_path = file_paths[j] if j < len(file_paths) else None
            if well_path is not None:
                well = load_well(well_path)
                t = well["t"][:n]
                dx, dy = well["dx"], well["dy"]
                t0v_frame = well["t0v_frame"]
                t0_file = t0_omega(
                    well["v"],
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
                t0_file = float("nan")

            scale = PIX_UM_ORIG if scale_velocity_to_um else 1.0
            t0_gt = t0_omega(
                v_gt * scale,
                t,
                dx,
                dy,
                peak_rule=peak_rule,
                peak_fraction=peak_fraction,
            )
            t0_pred = t0_omega(
                v_pred * scale,
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
                    "t0v_frame": t0v_frame,
                    "t0_omega_file_gt": t0_file,
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
