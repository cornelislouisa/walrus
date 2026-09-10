"""Paper-style normalized residual from SI Eq. (6) of arXiv:2405.18382.

After spatial averaging, the residual is one minus the cosine similarity between
two vector fields.  It measures pattern and direction, not velocity magnitude:
positive rescaling of either field leaves the score unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

VELOCITY_NAME_PAIRS = (
    ("velocity_x", "velocity_y"),
    ("velocity_ap", "velocity_dv"),
)


def _drop_singleton_spatial(array: np.ndarray) -> np.ndarray:
    """Drop artificial singleton spatial axes while preserving T and C."""
    out = np.asarray(array)
    while out.ndim > 4:
        singleton = next(
            (axis for axis in range(1, out.ndim - 1) if out.shape[axis] == 1),
            None,
        )
        if singleton is None:
            break
        out = np.squeeze(out, axis=singleton)
    return out


def extract_velocity(array: np.ndarray, field_names: Sequence[str]) -> np.ndarray:
    """Return ``(T, H, W, 2)`` velocity from either x/y or AP/DV field names."""
    fields = _drop_singleton_spatial(np.asarray(array))
    if fields.ndim != 4:
        raise ValueError(
            f"expected fields shaped (T,H,W,C), allowing singleton spatial axes; "
            f"got {np.asarray(array).shape}"
        )
    names = [str(name) for name in field_names]
    for first, second in VELOCITY_NAME_PAIRS:
        if first in names and second in names:
            return np.stack(
                [fields[..., names.index(first)], fields[..., names.index(second)]],
                axis=-1,
            )
    raise KeyError(
        f"Could not find velocity components in {names}; expected velocity_x/y "
        "or velocity_ap/dv"
    )


def _valid_mask(valid: Optional[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast an optional spatial/time mask to ``(T,H,W)``."""
    if valid is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == shape[1:]:
        mask = np.broadcast_to(mask, shape)
    if mask.shape != shape:
        raise ValueError(f"valid mask {mask.shape} is incompatible with {shape}")
    return mask


def paper_residual_map(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    energy_epsilon: float = 1e-12,
) -> np.ndarray:
    """Evaluate SI Eq. (6), returning a residual map shaped ``(T,H,W)``.

    Spatial RMS energies are computed independently for each frame. Frames where
    either field has energy at or below ``energy_epsilon`` are undefined and return
    all-NaN rather than being assigned a misleading perfect or maximal score.
    """
    u = np.asarray(prediction, dtype=np.float64)
    v = np.asarray(reference, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 4 or u.shape[-1] != 2:
        raise ValueError(
            "prediction and reference must have identical (T,H,W,2) shapes; "
            f"got {u.shape} and {v.shape}"
        )
    mask = _valid_mask(valid, u.shape[:-1])
    finite = np.isfinite(u).all(axis=-1) & np.isfinite(v).all(axis=-1)
    mask = mask & finite

    u2 = np.sum(u * u, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    uv = np.sum(u * v, axis=-1)
    masked_u2 = np.where(mask, u2, np.nan)
    masked_v2 = np.where(mask, v2, np.nan)
    with np.errstate(invalid="ignore"):
        mean_u2 = np.nanmean(masked_u2, axis=(1, 2))
        mean_v2 = np.nanmean(masked_v2, axis=(1, 2))
    scale = np.sqrt(mean_u2 * mean_v2)
    defined = (
        np.isfinite(scale)
        & (mean_u2 > energy_epsilon)
        & (mean_v2 > energy_epsilon)
    )

    numerator = (
        mean_u2[:, None, None] * v2
        + mean_v2[:, None, None] * u2
        - 2.0 * scale[:, None, None] * uv
    )
    denominator = 2.0 * mean_u2 * mean_v2
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = numerator / denominator[:, None, None]
    residual[~mask] = np.nan
    residual[~defined] = np.nan
    # Roundoff can produce tiny negative values for otherwise identical fields.
    return np.maximum(residual, 0.0)


def paper_residual_curve(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    energy_epsilon: float = 1e-12,
) -> np.ndarray:
    """Spatial mean of SI Eq. (6), one score per frame."""
    residual = paper_residual_map(
        prediction, reference, valid=valid, energy_epsilon=energy_epsilon
    )
    counts = np.isfinite(residual).sum(axis=(1, 2))
    totals = np.nansum(residual, axis=(1, 2))
    return np.divide(
        totals,
        counts,
        out=np.full(residual.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def paper_residual_cosine(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    energy_epsilon: float = 1e-12,
) -> np.ndarray:
    """Equivalent scalar identity ``1 - cosine_similarity``, per frame."""
    u = np.asarray(prediction, dtype=np.float64)
    v = np.asarray(reference, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 4 or u.shape[-1] != 2:
        raise ValueError(
            "prediction and reference must have identical (T,H,W,2) shapes; "
            f"got {u.shape} and {v.shape}"
        )
    mask = _valid_mask(valid, u.shape[:-1])
    mask &= np.isfinite(u).all(axis=-1) & np.isfinite(v).all(axis=-1)
    dot = np.where(mask[..., None], u * v, 0.0).sum(axis=(1, 2, 3))
    u2 = np.where(mask[..., None], u * u, 0.0).sum(axis=(1, 2, 3))
    v2 = np.where(mask[..., None], v * v, 0.0).sum(axis=(1, 2, 3))
    denom = np.sqrt(u2 * v2)
    return np.divide(
        denom - dot,
        denom,
        out=np.full(u.shape[0], np.nan, dtype=np.float64),
        where=denom > energy_epsilon,
    )


@dataclass(frozen=True)
class ResidualAggregate:
    """Macro-average of per-embryo residual curves."""

    times: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray
    horizon_scores: Mapping[int, float]
    horizon_std: Mapping[int, float]
    horizon_n: Mapping[int, int]


def aggregate_residual_curves(
    curves: Sequence[np.ndarray],
    *,
    times: Optional[np.ndarray] = None,
    horizons: Sequence[int] = (15, 20),
) -> ResidualAggregate:
    """Average frames within embryos, then embryos, for fixed minute horizons."""
    if not curves:
        raise ValueError("at least one residual curve is required")
    arrays = [np.asarray(curve, dtype=np.float64) for curve in curves]
    width = max(len(curve) for curve in arrays)
    stacked = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for row, curve in enumerate(arrays):
        stacked[row, : len(curve)] = curve
    counts = np.isfinite(stacked).sum(axis=0)
    mean = np.divide(
        np.nansum(stacked, axis=0),
        counts,
        out=np.full(width, np.nan),
        where=counts > 0,
    )
    std = np.array(
        [
            np.nanstd(stacked[:, i]) if counts[i] else np.nan
            for i in range(width)
        ]
    )
    if times is None:
        times = np.arange(1, width + 1, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape != (width,):
        raise ValueError(f"times must have shape {(width,)}, got {times.shape}")

    scores: dict[int, float] = {}
    score_std: dict[int, float] = {}
    score_n: dict[int, int] = {}
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        per_embryo = np.array(
            [
                np.nanmean(row[:horizon])
                if np.isfinite(row[:horizon]).any()
                else np.nan
                for row in stacked
            ]
        )
        finite = per_embryo[np.isfinite(per_embryo)]
        scores[int(horizon)] = float(np.mean(finite)) if len(finite) else float("nan")
        score_std[int(horizon)] = (
            float(np.std(finite)) if len(finite) else float("nan")
        )
        score_n[int(horizon)] = int(len(finite))
    return ResidualAggregate(
        times=times,
        mean=mean,
        std=std,
        n=counts,
        horizon_scores=scores,
        horizon_std=score_std,
        horizon_n=score_n,
    )


def residual_metrics_from_rollouts(
    rollouts: Sequence,
    *,
    horizons: Sequence[int] = (15, 20),
) -> ResidualAggregate:
    """Compute velocity-only paper residuals from aligned cached rollouts."""
    curves = []
    for item in rollouts:
        prediction = extract_velocity(item.pred, item.field_names)
        reference = extract_velocity(item.ref, item.field_names)
        curves.append(paper_residual_curve(prediction, reference))
    return aggregate_residual_curves(curves, horizons=horizons)


def paper_mean_velocity(paths: Sequence) -> np.ndarray:
    """Time- and embryo-averaged spatial velocity field from Well HDF5 files.

    This intentionally matches the paper's stated mean-field definition over the
    entire supplied dataset. Passing train, valid, and test paths therefore creates
    a descriptive paper-style comparator, not a leakage-free learned baseline.
    """
    import h5py

    total = None
    count = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            velocity = np.asarray(handle["t1_fields/velocity"], dtype=np.float64)
        # Collapse sample and time while retaining H, W, components.
        if velocity.ndim != 5 or velocity.shape[-1] != 2:
            raise ValueError(
                f"{path}: expected velocity shaped (sample,time,H,W,2), "
                f"got {velocity.shape}"
            )
        finite = np.isfinite(velocity)
        file_total = np.where(finite, velocity, 0.0).sum(axis=(0, 1))
        file_count = finite.sum(axis=(0, 1))
        if total is None:
            total = file_total
            count = file_count
        else:
            if total.shape != file_total.shape:
                raise ValueError(
                    f"mean-field spatial shape changed from {total.shape} "
                    f"to {file_total.shape} at {path}"
                )
            total += file_total
            count += file_count
    if total is None or count is None:
        raise ValueError("at least one Well HDF5 file is required")
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    )


def mean_field_residual_from_rollouts(
    rollouts: Sequence,
    mean_velocity: np.ndarray,
    *,
    horizons: Sequence[int] = (15, 20),
) -> ResidualAggregate:
    """Score one constant paper-style mean field against aligned rollout GT."""
    mean_velocity = np.asarray(mean_velocity, dtype=np.float64)
    curves = []
    for item in rollouts:
        reference = extract_velocity(item.ref, item.field_names)
        if mean_velocity.shape != reference.shape[1:]:
            raise ValueError(
                f"mean velocity {mean_velocity.shape} does not match rollout "
                f"spatial shape {reference.shape[1:]}"
            )
        prediction = np.broadcast_to(mean_velocity, reference.shape)
        curves.append(paper_residual_curve(prediction, reference))
    return aggregate_residual_curves(curves, horizons=horizons)
