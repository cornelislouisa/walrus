"""Tests for the normalized residual used in arXiv:2405.18382 SI Eq. (6)."""

from types import SimpleNamespace

import numpy as np

from walrus.analysis.paper_residual import (
    aggregate_residual_curves,
    extract_velocity,
    mean_field_residual_from_rollouts,
    paper_mean_velocity,
    paper_residual_cosine,
    paper_residual_curve,
    paper_residual_map,
    residual_metrics_from_rollouts,
)


def _field(x=(1.0, 0.0), *, t=3, h=2, w=4):
    return np.broadcast_to(np.asarray(x), (t, h, w, 2)).astype(float).copy()


def test_perfect_and_positive_scaled_fields_score_zero():
    reference = _field()
    assert np.allclose(paper_residual_curve(reference, reference), 0.0)
    assert np.allclose(paper_residual_curve(3.0 * reference, reference), 0.0)


def test_orthogonal_and_anticorrelated_fields_score_one_and_two():
    reference = _field((1.0, 0.0))
    orthogonal = _field((0.0, 1.0))
    opposite = _field((-1.0, 0.0))
    assert np.allclose(paper_residual_curve(orthogonal, reference), 1.0)
    assert np.allclose(paper_residual_curve(opposite, reference), 2.0)


def test_direct_equation_matches_cosine_identity():
    rng = np.random.default_rng(42)
    prediction = rng.normal(size=(5, 4, 3, 2))
    reference = rng.normal(size=(5, 4, 3, 2))
    valid = rng.random((5, 4, 3)) > 0.2
    direct = paper_residual_curve(prediction, reference, valid)
    cosine = paper_residual_cosine(prediction, reference, valid)
    assert np.allclose(direct, cosine)


def test_mask_excludes_pixels_and_zero_energy_is_undefined():
    reference = _field(t=1, h=1, w=2)
    prediction = reference.copy()
    prediction[:, :, 1] *= -1
    valid = np.array([[True, False]])
    assert paper_residual_curve(prediction, reference, valid)[0] == 0.0
    assert np.isnan(paper_residual_curve(np.zeros_like(reference), reference)[0])
    assert np.isnan(paper_residual_map(np.zeros_like(reference), reference)).all()


def test_velocity_aliases_and_singleton_spatial_axes():
    fields = np.zeros((2, 3, 4, 1, 6))
    fields[..., 0] = 1.0
    fields[..., 1] = 2.0
    apdv = extract_velocity(
        fields,
        [
            "velocity_ap",
            "velocity_dv",
            "myosin_tensor_apap",
            "myosin_tensor_apdv",
            "myosin_tensor_dvap",
            "myosin_tensor_dvdv",
        ],
    )
    assert apdv.shape == (2, 3, 4, 2)
    assert np.all(apdv[..., 0] == 1.0)
    assert np.all(apdv[..., 1] == 2.0)
    xy = extract_velocity(fields[..., :2], ["velocity_x", "velocity_y"])
    assert np.array_equal(xy, apdv)


def test_macro_aggregation_uses_first_15_and_20_frames():
    first = np.arange(1, 21, dtype=float) / 100.0
    second = first + 0.1
    result = aggregate_residual_curves([first, second])
    assert result.horizon_scores[15] == np.mean(
        [np.mean(first[:15]), np.mean(second[:15])]
    )
    assert result.horizon_scores[20] == np.mean(
        [np.mean(first), np.mean(second)]
    )
    assert result.horizon_n == {15: 2, 20: 2}
    assert np.array_equal(result.times, np.arange(1, 21))


def test_rollout_driver_extracts_only_velocity():
    reference = np.concatenate([_field(t=20), np.ones((20, 2, 4, 4))], axis=-1)
    prediction = reference.copy()
    prediction[..., 2:] *= -100.0
    item = SimpleNamespace(
        pred=prediction,
        ref=reference,
        field_names=(
            "velocity_x",
            "velocity_y",
            "myosin_tensor_xx",
            "myosin_tensor_xy",
            "myosin_tensor_yx",
            "myosin_tensor_yy",
        ),
    )
    result = residual_metrics_from_rollouts([item])
    assert result.horizon_scores[15] == 0.0
    assert result.horizon_scores[20] == 0.0


def test_paper_mean_field_averages_samples_times_and_embryos(tmp_path):
    import h5py

    paths = []
    for index, value in enumerate((1.0, 3.0)):
        path = tmp_path / f"{index}.hdf5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("t1_fields")
            group.create_dataset(
                "velocity", data=np.full((1, 2, 3, 4, 2), value)
            )
        paths.append(path)
    mean = paper_mean_velocity(paths)
    assert mean.shape == (3, 4, 2)
    assert np.all(mean == 2.0)


def test_mean_field_residual_uses_rollout_reference():
    reference_velocity = _field(t=20)
    fields = np.concatenate(
        [reference_velocity, np.ones((20, 2, 4, 4))], axis=-1
    )
    item = SimpleNamespace(
        pred=np.zeros_like(fields),
        ref=fields,
        field_names=(
            "velocity_x",
            "velocity_y",
            "myosin_tensor_xx",
            "myosin_tensor_xy",
            "myosin_tensor_yx",
            "myosin_tensor_yy",
        ),
    )
    result = mean_field_residual_from_rollouts(
        [item], np.ones((2, 4, 2)), horizons=(20,)
    )
    expected = paper_residual_curve(
        np.ones_like(reference_velocity), reference_velocity
    ).mean()
    assert result.horizon_scores[20] == expected
