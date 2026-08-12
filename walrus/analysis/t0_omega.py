#!/usr/bin/env python3
"""t0^ω from the vorticity autocorrelation, Mitchell et al. Nat Methods 2026.

Public entry point: t0_omega(v, t, dx, dy, peak_rule="first").

UNITS. Well-file velocity is original pullback px/min. load_well converts to
um/min. ρ_ω is scale-free, so either unit system is fine for t0.

GRID. The paper excludes about 8% of the AP coordinate at each pole. Crop those
rows before calling t0_omega. No interpolation/rejection mask is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

PIX_UM_ORIG = 0.2619


def load_well(path, valid_threshold=0.5):
    """Read one Well file. Velocity is returned in um/min."""
    with h5py.File(path, "r") as f:
        v = np.asarray(f["t1_fields/velocity"][0], float)
        if "t0_fields/valid_velocity" in f:
            valid = np.asarray(f["t0_fields/valid_velocity"][0], float)
        else:
            valid = np.ones(v.shape[:-1], float)
        t = np.asarray(f["dimensions/time"][:], float)
        ap = np.asarray(f["dimensions/ap"][:], float)
        dv = np.asarray(f["dimensions/dv"][:], float)
        t0v = float(np.asarray(f["scalars/t0v_frame"]))
    return dict(
        v=v * PIX_UM_ORIG,
        valid=valid >= valid_threshold,
        t=t,
        dx=float(np.mean(np.diff(ap))),
        dy=float(np.mean(np.diff(dv))),
        t0v_frame=t0v,
        path=str(path),
    )


def _d_ap(f, dx):
    return np.gradient(f, dx, axis=-2)


def _d_dv(f, dy):
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * dy)


def vorticity(v, dx, dy):
    """ω = ∂v_dv/∂ap − ∂v_ap/∂dv. v: (..., n_ap, n_dv, 2)."""
    v_ap, v_dv = v[..., 0], v[..., 1]
    return _d_ap(v_dv, dx) - _d_dv(v_ap, dy)


def rho_omega_matrix(omega, valid=None):
    """SI Note 10 Eq. 8: Pearson correlation of vorticity maps."""
    T = omega.shape[0]
    X = omega.reshape(T, -1).astype(float)
    if valid is not None:
        X = np.where(valid.reshape(T, -1), X, np.nan)
    X = X - np.nanmean(X, axis=1, keepdims=True)
    if np.isfinite(X).all():
        nrm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        X = X / np.maximum(nrm, 1e-12)
        return X @ X.T
    M = np.full((T, T), np.nan)
    for i in range(T):
        ai = X[i]
        for j in range(i, T):
            ok = np.isfinite(ai) & np.isfinite(X[j])
            if ok.sum() < 50:
                continue
            a, b = ai[ok], X[j, ok]
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a @ a) * (b @ b))
            if den > 0:
                M[i, j] = M[j, i] = float(a @ b / den)
    return M


def t0_omega(v, t, dx, dy, *, peak_rule="first", peak_fraction=0.5):
    """
    VF→GBE landmark from the vorticity-autocorrelation score.

        M(ti, tj) = ρ_ω(ω(ti), ω(tj))          SI Eq. 8
        s(t)      = Σ_{t'} ∂_t M(t, t')         Methods, rigid alignment

    ``peak_rule="argmax"`` is the published DynamicAtlas rule and is used to
    reproduce the atlas landmark on its coarse PIV. ``peak_rule="first"`` is
    used for our higher-resolution PIVlab fields: it selects the first local
    maximum reaching ``peak_fraction`` of the global maximum. This avoids
    switching to a later flow transition when the high-resolution score is
    multimodal.
    """
    if peak_rule not in {"first", "argmax"}:
        raise ValueError("peak_rule must be 'first' or 'argmax'")
    w = vorticity(v, dx, dy)
    M = rho_omega_matrix(w)
    score = np.sum(np.gradient(M, axis=0), axis=1)
    if not np.isfinite(score).any():
        return np.nan
    if peak_rule == "first":
        threshold = peak_fraction * np.nanmax(score)
        for i in range(1, len(score) - 1):
            if (
                score[i] >= score[i - 1]
                and score[i] > score[i + 1]
                and score[i] >= threshold
            ):
                return float(t[i])
    return float(t[int(np.nanargmax(score))])


def t0_omega_from_well(
    path: Union[str, Path],
    *,
    peak_rule: str = "first",
    peak_fraction: float = 0.5,
    valid_threshold: float = 0.5,
) -> dict:
    """Compute ``t0_omega`` for one Well file; returns landmark + file metadata.

    Well files are assumed already AP-cropped and ready for ``t0_omega``.
    """
    data = load_well(path, valid_threshold=valid_threshold)
    t0 = t0_omega(
        data["v"],
        data["t"],
        data["dx"],
        data["dy"],
        peak_rule=peak_rule,
        peak_fraction=peak_fraction,
    )
    return {
        "path": data.get("path", str(path)),
        "t0_omega": t0,
        "t0v_frame": data["t0v_frame"],
        "t_min": float(data["t"][0]),
        "t_max": float(data["t"][-1]),
        "n_frames": int(data["t"].shape[0]),
    }
