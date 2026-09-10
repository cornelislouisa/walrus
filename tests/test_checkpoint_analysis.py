"""Tests for zero-shot config surgery in :mod:`walrus.analysis.checkpoint_analysis`."""

import numpy as np
from omegaconf import OmegaConf

import walrus.analysis.checkpoint_analysis as checkpoint_analysis
from walrus.analysis.checkpoint_analysis import (
    allows_missing_checkpoint,
    apply_data_config,
    available_checkpoints,
    select_checkpoint,
    select_vf_aligned_cohort,
    spectral_metrics_from_rollouts,
    vrmse_from_rollouts,
)


def _run_cfg(**module_parameters):
    """Minimal stand-in for a run's saved ``extended_config.yaml``."""
    return OmegaConf.create(
        {
            "data": {
                "well_base_path": "/tmp",
                "field_index_map_override": {"velocity_x": 0, "velocity_y": 1},
                "module_parameters": {
                    "n_steps_input": 10,
                    "batch_size": 8,
                    **module_parameters,
                },
            }
        }
    )


def _target_cfg():
    return OmegaConf.create(
        {
            "well_base_path": "/tmp",
            "module_parameters": {"n_steps_input": 4, "batch_size": 2},
        }
    )


def test_keeps_trained_dataset_kws():
    """2D baselines train with pad_cartesian_data_to_d=2; losing it inflates velocity to 3D."""
    cfg = apply_data_config(
        _run_cfg(dataset_kws={"pad_cartesian_data_to_d": 2}), _target_cfg()
    )
    assert OmegaConf.select(cfg, "data.module_parameters.dataset_kws") == {
        "pad_cartesian_data_to_d": 2
    }


def test_keeps_n_steps_input_and_field_index_map():
    cfg = apply_data_config(_run_cfg(), _target_cfg())
    assert cfg.data.module_parameters.n_steps_input == 10
    assert cfg.data.field_index_map_override == {"velocity_x": 0, "velocity_y": 1}


def test_no_dataset_kws_when_run_had_none():
    cfg = apply_data_config(_run_cfg(), _target_cfg())
    assert OmegaConf.select(cfg, "data.module_parameters.dataset_kws") is None


def test_overrides_win_over_kept_values():
    cfg = apply_data_config(
        _run_cfg(dataset_kws={"pad_cartesian_data_to_d": 2}),
        _target_cfg(),
        batch_size=1,
        n_steps_input=6,
        dataset_kws={"pad_cartesian_data_to_d": 3},
    )
    mp = cfg.data.module_parameters
    assert mp.batch_size == 1
    assert mp.n_steps_input == 6
    assert mp.dataset_kws == {"pad_cartesian_data_to_d": 3}


def test_can_opt_out_of_keeping():
    cfg = apply_data_config(
        _run_cfg(dataset_kws={"pad_cartesian_data_to_d": 2}),
        _target_cfg(),
        keep_dataset_kws=False,
        keep_n_steps_input=False,
        keep_field_index_map=False,
    )
    assert OmegaConf.select(cfg, "data.module_parameters.dataset_kws") is None
    assert cfg.data.module_parameters.n_steps_input == 4
    assert OmegaConf.select(cfg, "data.field_index_map_override") is None


def test_vrmse_from_rollouts_truncates_to_horizon():
    """First-N VRMSE ignores later frames; identical prefixes score ~0."""
    from types import SimpleNamespace

    from the_well.data.datasets import WellMetadata

    metadata = WellMetadata(
        dataset_name="example",
        n_spatial_dims=2,
        spatial_resolution=(4, 5),
        scalar_names=[],
        constant_scalar_names=[],
        field_names={0: [], 1: ["u"], 2: []},
        constant_field_names={0: [], 1: [], 2: []},
        boundary_condition_types=["PERIODIC", "PERIODIC"],
        n_files=1,
        n_trajectories_per_file=[1],
        n_steps_per_trajectory=[8],
        grid_type="cartesian",
    )
    t, h, w, c = 8, 4, 5, 1
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    ref = np.broadcast_to((xx + yy)[None, ..., None], (t, h, w, c)).copy()
    pred = ref.copy()
    pred[5:] += 10.0  # only late frames disagree
    item = SimpleNamespace(dataset="example", pred=pred, ref=ref, metadata=metadata)
    early = vrmse_from_rollouts([item], n_frames=5)
    full = vrmse_from_rollouts([item], n_frames=None)
    assert early < 1e-5
    assert full > 1.0


def test_spectral_metrics_from_rollouts_returns_three_bins_and_horizon():
    from dataclasses import replace
    from types import SimpleNamespace

    from the_well.data.datasets import WellMetadata

    metadata = WellMetadata(
        dataset_name="example",
        n_spatial_dims=2,
        spatial_resolution=(8, 8),
        scalar_names=[],
        constant_scalar_names=[],
        field_names={0: [], 1: ["u"], 2: []},
        constant_field_names={0: [], 1: [], 2: []},
        boundary_condition_types=["PERIODIC", "PERIODIC"],
        n_files=1,
        n_trajectories_per_file=[1],
        n_steps_per_trajectory=[8],
        grid_type="cartesian",
    )
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(8, 8, 8, 1)).astype(np.float32)
    pred = ref.copy()
    pred[5:] += rng.normal(size=pred[5:].shape).astype(np.float32)
    item = SimpleNamespace(dataset="example", pred=pred, ref=ref, metadata=metadata)

    early = spectral_metrics_from_rollouts([item], n_frames=5)
    full = spectral_metrics_from_rollouts([item])
    assert len(early) == 6
    for bin_index in range(3):
        key = f"spectral_error_nmse_per_bin_{bin_index}"
        assert early[key] < 1e-6
        assert full[key] > 0.0

    # Inflating the same 2D fields to (H, W, 1), as Walrus does, must not
    # change the bins or scores relative to native-2D baselines.
    inflated = SimpleNamespace(
        dataset="example",
        pred=pred[..., None, :],
        ref=ref[..., None, :],
        metadata=replace(
            metadata, n_spatial_dims=3, spatial_resolution=(8, 8, 1)
        ),
    )
    inflated_full = spectral_metrics_from_rollouts([inflated])
    for key in full:
        assert np.isclose(inflated_full[key], full[key])


def test_available_checkpoints_finds_last_without_metadata(tmp_path):
    last = tmp_path / "last"
    last.mkdir()
    (last / "full_checkpoint.pt").write_bytes(b"not-a-real-ckpt")
    found = available_checkpoints(tmp_path)
    assert found[0] == last


def test_allows_missing_checkpoint_for_advection_and_mean_field():
    adv = OmegaConf.create(
        {"model": {"_target_": "walrus.baselines.advection.AdvectionWrapper"}}
    )
    mf = OmegaConf.create(
        {"model": {"_target_": "walrus.baselines.mean_field.MeanFieldWrapper"}}
    )
    ffno = OmegaConf.create({"model": {"_target_": "walrus.baselines.ffno.FFNOWrapper"}})
    assert allows_missing_checkpoint(adv)
    assert allows_missing_checkpoint(mf)
    assert not allows_missing_checkpoint(ffno)


def test_select_checkpoint_allows_stateless_baseline_without_files(tmp_path):
    from types import SimpleNamespace

    run = SimpleNamespace(
        config={
            "checkpoint": {"save_dir": str(tmp_path)},
            "model": {"_target_": "walrus.baselines.advection.AdvectionWrapper"},
        }
    )
    sel = select_checkpoint(run)
    assert sel.path.name == "_untrained"
    assert sel.epoch == 0


def test_vf_aligned_cohort_uses_common_context_and_excludes_short_history(
    tmp_path, monkeypatch
):
    import h5py
    from types import SimpleNamespace

    root = tmp_path / "wt"
    test = root / "data" / "test"
    test.mkdir(parents=True)

    def make_file(name, start, n):
        path = test / name
        with h5py.File(path, "w") as handle:
            handle.create_dataset("dimensions/time", data=np.arange(start, start + n))
            scalars = handle.create_group("scalars")
            # First output is +1, so t0_frame is the index immediately after t=0.
            scalars.create_dataset("t0_frame", data=float(-start + 1))
        return path

    eligible_a = make_file("a.hdf5", -9, 40)  # 10 frames through VF
    eligible_b = make_file("b.hdf5", -11, 40)  # 12 frames through VF
    make_file("short_context.hdf5", -3, 40)  # only 4 frames through VF
    make_file("short_future.hdf5", -9, 20)  # only 10 post-VF frames

    def cfg(context):
        return OmegaConf.create(
            {
                "data": {
                    "well_base_path": str(tmp_path),
                    "module_parameters": {
                        "n_steps_input": context,
                        "well_dataset_info": {
                            "atlas": {
                                "path": str(root),
                                "include_filters": [],
                                "exclude_filters": [],
                            }
                        },
                    },
                }
            }
        )

    runs = [SimpleNamespace(cfg=cfg(1)), SimpleNamespace(cfg=cfg(10))]
    monkeypatch.setattr(checkpoint_analysis, "load_config", lambda run: run.cfg)
    monkeypatch.setattr(
        checkpoint_analysis,
        "apply_data_config",
        lambda run_cfg, *_args, **_kwargs: run_cfg,
    )
    cohort = select_vf_aligned_cohort(runs, "unused", horizon_minutes=20)
    assert cohort.max_context == 10
    assert cohort.files == (eligible_a.resolve(), eligible_b.resolve())
    assert "needs 10" in cohort.excluded["short_context.hdf5"]
    assert "post-VF frames" in cohort.excluded["short_future.hdf5"]
