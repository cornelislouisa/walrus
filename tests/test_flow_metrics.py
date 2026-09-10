"""Tests for the multi-run r.m.s. velocity overlay."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from walrus.analysis.flow_metrics import (  # noqa: E402
    MODEL_PALETTE,
    EmbryoFlowResult,
    plot_rms_velocity_by_group,
    plot_rms_velocity_multi_run,
)


def _embryo(seed: int, *, gt_gain: float = 1.0, pred_gain: float = 1.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-30.0, 30.0, 60)
    gt = 2.0 + 1.5 * np.exp(-(((t - 5.0) / 12.0) ** 2)) + 0.01 * rng.standard_normal(60)
    return EmbryoFlowResult(
        file=f"embryo_{seed}.hdf5",
        dataset="wt",
        batch=seed,
        t=t,
        t_rel=t,
        gbe_onset_gt=0.0,
        gbe_onset_pred=1.0,
        t0_omega_gt=0.0,
        rms_gt=gt * gt_gain,
        rms_pred=gt * pred_gain,
        autocorr_gt=np.eye(3),
        autocorr_pred=np.eye(3),
    )


def _results(n_runs: int = 3, n_embryos: int = 2):
    return {
        f"run{k}": [
            _embryo(10 * k + i, pred_gain=0.8 + 0.1 * k) for i in range(n_embryos)
        ]
        for k in range(n_runs)
    }


def _gt_warnings(recorded):
    return [w for w in recorded if "Ground-truth RMS curves differ" in str(w.message)]


def test_draws_one_line_per_run_plus_ground_truth():
    ax = plot_rms_velocity_multi_run(_results(n_runs=3))
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels[:4] == ["ground truth", "run0", "run1", "run2"]
    assert ax.get_ylim() == (0.0, 7.0)


def test_ylim_can_be_overridden():
    ax = plot_rms_velocity_multi_run(_results(n_runs=1), ylim=(0.0, 3.5))
    assert ax.get_ylim() == (0.0, 3.5)


def test_labels_and_mapping_colors_are_applied():
    ax = plot_rms_velocity_multi_run(
        _results(n_runs=2),
        labels={"run0": "FFNO", "run1": "SineNet"},
        colors={"FFNO": "#4C78A8", "SineNet": "#54A24B"},
    )
    by_label = {line.get_label(): line for line in ax.get_lines()}
    assert set(by_label) >= {"ground truth", "FFNO", "SineNet"}
    assert by_label["FFNO"].get_color() == "#4C78A8"
    assert by_label["SineNet"].get_color() == "#54A24B"


def test_empty_runs_are_dropped():
    results = _results(n_runs=2)
    results["run_empty"] = []
    ax = plot_rms_velocity_multi_run(results)
    assert "run_empty" not in [line.get_label() for line in ax.get_lines()]


def test_consistent_ground_truth_does_not_warn(recwarn):
    plot_rms_velocity_multi_run(_results(n_runs=3))
    assert _gt_warnings(recwarn) == []


def test_warns_when_ground_truth_disagrees_between_runs():
    """A run whose GT differs means the pooled reference is not one curve."""
    results = _results(n_runs=2)
    results["run_odd"] = [_embryo(999, gt_gain=1.5)]
    with pytest.warns(UserWarning, match="Ground-truth RMS curves differ"):
        plot_rms_velocity_multi_run(results)


def test_gt_from_selects_a_single_reference_and_skips_the_check(recwarn):
    results = _results(n_runs=2)
    results["run_odd"] = [_embryo(999, gt_gain=1.5)]
    ax = plot_rms_velocity_multi_run(results, gt_from="run0")
    assert _gt_warnings(recwarn) == []
    gt = next(line for line in ax.get_lines() if line.get_label() == "ground truth")
    assert np.nanmax(gt.get_ydata()) == pytest.approx(_embryo(0).rms_gt.max(), rel=0.05)


def test_unknown_gt_from_raises():
    with pytest.raises(KeyError):
        plot_rms_velocity_multi_run(_results(), gt_from="missing")


def test_all_empty_raises():
    with pytest.raises(ValueError):
        plot_rms_velocity_multi_run({"a": [], "b": []})


def _by_group():
    return {
        "17° C": [_embryo(1, gt_gain=0.7, pred_gain=0.5)],
        "27° C": [_embryo(2, gt_gain=1.3, pred_gain=0.9)],
    }


def test_by_group_draws_one_line_per_group_and_no_reference_curve():
    ax = plot_rms_velocity_by_group(_by_group(), colors={"17° C": "#3C6DA8"})
    labels = [line.get_label() for line in ax.get_lines()]
    assert [x for x in labels if not x.startswith("_")] == ["17° C", "27° C"]
    assert "ground truth" not in labels
    assert ax.get_ylim() == (0.0, 7.0)


def test_by_group_mapping_colors_and_palette_fallback():
    ax = plot_rms_velocity_by_group(_by_group(), colors={"17° C": "#3C6DA8"})
    by_label = {line.get_label(): line for line in ax.get_lines()}
    assert by_label["17° C"].get_color() == "#3C6DA8"
    # Unmapped group falls back to the shared palette rather than erroring.
    assert by_label["27° C"].get_color() in MODEL_PALETTE


def test_by_group_gt_and_pred_select_different_curves():
    groups = _by_group()
    ax_gt = plot_rms_velocity_by_group(groups, which="gt")
    ax_pred = plot_rms_velocity_by_group(groups, which="pred")
    peak_gt = max(np.nanmax(line.get_ydata()) for line in ax_gt.get_lines()[:2])
    peak_pred = max(np.nanmax(line.get_ydata()) for line in ax_pred.get_lines()[:2])
    assert peak_pred < peak_gt


def test_by_group_rejects_unknown_which():
    with pytest.raises(ValueError, match="which must be"):
        plot_rms_velocity_by_group(_by_group(), which="truth")


def test_by_group_all_empty_raises():
    with pytest.raises(ValueError):
        plot_rms_velocity_by_group({"17° C": []})
