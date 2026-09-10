"""Check that WT_old reproduces the validation metrics logged by an old morphogenesis run.

Rebuilds a saved checkpoint, re-runs the exact validation the trainer ran at that epoch
(one-step ``valid`` and ``rollout_valid``, ``full=False`` as during training), and diffs
the resulting ``full_VRMSE_T=all_mean`` against the values stored in the run's
``viz/loss_dicts``.
"""

import argparse
import pathlib
import pickle

import torch
from omegaconf import OmegaConf

from walrus.analysis.checkpoint_analysis import _remap_data_paths, build_trainer

SCORE_SUFFIX = "full_VRMSE_T=all_mean"


def logged_scores(run_dir: pathlib.Path, split: str, epoch: int) -> dict[str, float]:
    path = run_dir / "viz" / "loss_dicts" / f"{split}_loss_dict_epoch{epoch}_rank0.pkl"
    if not path.is_file():
        return {}
    original_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = cpu_load
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    finally:
        torch.load = original_load
    return {str(k): float(v) for k, v in raw.items() if str(k).endswith(SCORE_SUFFIX)}


def rerun_scores(trainer, split: str) -> dict[str, float]:
    dm = trainer.datamodule
    if split == "valid":
        loaders = dm.val_dataloaders(replicas=1, rank=0, full=False)
    else:
        loaders = dm.rollout_val_dataloaders(replicas=1, rank=0, full=False)
    _, metrics = trainer.validation_loop(loaders, valid_or_test=split, full=False)
    return {str(k): float(v) for k, v in metrics.items() if str(k).endswith(SCORE_SUFFIX)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="/scr/louisa/walrus/runs/morphogenesis/Walrus_ft_morpho_lr_scheduler-inv",
    )
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    ckpt = run_dir / "checkpoints" / f"step_{args.epoch}"
    cfg = _remap_data_paths(OmegaConf.load(run_dir / "extended_config.yaml"))

    info = cfg.data.module_parameters.well_dataset_info
    meta = next(iter(info.values()))
    print(f"run       : {run_dir.name}")
    print(f"checkpoint: {ckpt}")
    print(f"data path : {meta.path}")
    print(f"stats     : {meta.get('normalization_path', 'stats.yaml (in data root)')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz = pathlib.Path("./_verify_viz") / run_dir.name
    viz.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, ckpt, device, viz)

    for split in ("valid", "rollout_valid"):
        logged = logged_scores(run_dir, split, args.epoch)
        fresh = rerun_scores(trainer, split)
        print(f"\n=== {split} (epoch {args.epoch}) ===")
        for key in sorted(set(logged) | set(fresh)):
            old = logged.get(key)
            new = fresh.get(key)
            if old is None or new is None:
                print(f"  {key}: logged={old} rerun={new}")
                continue
            rel = abs(new - old) / max(abs(old), 1e-12)
            print(f"  {key}\n    logged={old:.6f}  rerun={new:.6f}  rel_diff={rel:.3%}")


if __name__ == "__main__":
    main()
