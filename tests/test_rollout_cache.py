"""Tests for persistent rollout artifacts used by zero-shot analysis."""

import json

import numpy as np
from the_well.data.datasets import WellMetadata

from walrus.analysis.rollout_cache import (
    RolloutCache,
    RolloutCacheKey,
    cache_fingerprint,
)


def _metadata():
    return WellMetadata(
        dataset_name="example",
        n_spatial_dims=2,
        spatial_resolution=(4, 5),
        scalar_names=[],
        constant_scalar_names=["dt"],
        field_names={0: [], 1: ["velocity_ap", "velocity_dv"], 2: []},
        constant_field_names={0: [], 1: [], 2: []},
        boundary_condition_types=["OPEN", "PERIODIC"],
        n_files=2,
        n_trajectories_per_file=[1, 1],
        n_steps_per_trajectory=[6, 6],
        grid_type="cartesian",
    )


def _key(**overrides):
    settings = overrides.pop("settings", None)
    payload = {
        "entity": "cgl",
        "project": "morphogenesis_no_myosin",
        "model": "FFNOWrapper",
        "run_id": "abc123",
        "run_name": "run [with spaces]",
        "data": "morphogenesis_test",
        "split": "rollout_test",
        "checkpoint_epoch": 20,
        "checkpoint_path": "/tmp/checkpoints/step_20",
        "settings": {"full": True} if settings is None else settings,
    }
    payload.update(overrides)
    return RolloutCacheKey(**payload)


def test_fingerprint_is_stable_to_mapping_order():
    assert cache_fingerprint({"a": 1, "b": 2}) == cache_fingerprint({"b": 2, "a": 1})


def test_inference_setting_changes_cache_path(tmp_path):
    a = RolloutCache(_key(settings={"full": True, "max_rollout_steps": 50}), tmp_path)
    b = RolloutCache(_key(settings={"full": True, "max_rollout_steps": 100}), tmp_path)
    assert a.path != b.path
    assert "abc123" in str(a.path)
    assert "morphogenesis_no_myosin" in str(a.path)
    assert "FFNOWrapper" in str(a.path)


def test_round_trip_artifacts_and_summary(tmp_path):
    cache = RolloutCache(_key(), tmp_path)
    cache.begin()
    batch, context_t, pred_t, h, w, channels = 2, 3, 4, 4, 5, 2
    cache.save_batch(
        dataset="example",
        batch_index=0,
        pred=np.arange(batch * pred_t * h * w * channels, dtype=np.float32).reshape(
            batch, pred_t, h, w, channels
        ),
        ref=np.ones((batch, pred_t, h, w, channels), dtype=np.float32),
        context=np.zeros((batch, context_t, h, w, channels), dtype=np.float32),
        input_time=np.tile(np.arange(context_t), (batch, 1)),
        output_time=np.tile(np.arange(context_t, context_t + pred_t), (batch, 1)),
        space_grid=np.zeros((batch, h, w, 2), dtype=np.float32),
        field_names=["velocity_ap", "velocity_dv"],
        metadata=_metadata(),
        file_paths=["/data/embryo_a.hdf5", "/data/embryo_b.hdf5"],
    )
    assert not cache.complete
    cache.finish({"score": 0.25, "per_dataset": {"example": 0.25}, "loss": 0.25})
    assert cache.complete
    assert cache.summary()["score"] == 0.25

    artifacts = cache.rollouts()
    assert len(artifacts) == 2
    assert artifacts[0].file == "/data/embryo_a.hdf5"
    assert artifacts[1].file == "/data/embryo_b.hdf5"
    assert artifacts[0].pred.shape == (pred_t, h, w, channels)
    assert artifacts[0].context.shape == (context_t, h, w, channels)
    assert artifacts[0].field_names == ("velocity_ap", "velocity_dv")
    assert artifacts[0].metadata == _metadata()
    np.testing.assert_array_equal(
        artifacts[0].time, np.arange(context_t + pred_t, dtype=np.float32)
    )


def test_manifest_is_human_readable(tmp_path):
    cache = RolloutCache(_key(), tmp_path)
    cache.begin()
    manifest = json.loads(cache.manifest_path.read_text())
    assert manifest["run_id"] == "abc123"
    assert manifest["checkpoint_epoch"] == 20
    assert manifest["schema_version"] == 2
    assert manifest["entity"] == "cgl"
    assert manifest["project"] == "morphogenesis_no_myosin"
    assert manifest["model"] == "FFNOWrapper"
    assert manifest["data"] == "morphogenesis_test"


def test_project_dataset_and_model_are_isolated(tmp_path):
    base = RolloutCache(_key(), tmp_path)
    other_project = RolloutCache(_key(project="morphogenesis_with_myosin"), tmp_path)
    other_dataset = RolloutCache(_key(data="morphogenesis_WT_27_degrees"), tmp_path)
    other_model = RolloutCache(_key(model="ScOTWrapper"), tmp_path)
    same = RolloutCache(_key(), tmp_path)

    assert same.path == base.path
    assert other_project.path != base.path
    assert other_dataset.path != base.path
    assert other_model.path != base.path
    assert "morphogenesis_with_myosin" in str(other_project.path)
    assert "morphogenesis_WT_27_degrees" in str(other_dataset.path)
    assert "ScOTWrapper" in str(other_model.path)
    # Different identities never share a parent folder beyond the cache root.
    assert other_project.path.parent != base.path.parent
    assert other_dataset.path.parent != base.path.parent
    assert other_model.path.parent != base.path.parent


def test_complete_cache_is_not_overwritten(tmp_path):
    cache = RolloutCache(_key(), tmp_path)
    cache.begin()
    cache.finish({"score": 0.1})
    marker = cache.summary_path.read_text()
    try:
        cache.begin()
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    assert cache.summary_path.read_text() == marker
    cache.begin(overwrite=True)
    assert not cache.complete


def test_mismatched_manifest_is_not_treated_as_complete(tmp_path):
    cache = RolloutCache(_key(), tmp_path)
    cache.begin()
    cache.finish({"score": 0.1})
    assert cache.complete
    manifest = json.loads(cache.manifest_path.read_text())
    manifest["project"] = "someone_elses_project"
    cache.manifest_path.write_text(json.dumps(manifest))
    assert not cache.complete
    try:
        cache.summary()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
