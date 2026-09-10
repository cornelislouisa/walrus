"""Flow autocorrelation and r.m.s. velocity (Mitchell et al. Nat Methods 2026 Fig. 3).

Implements the paper / SI definitions used for Fig. 3c (autocorrelation) and Fig. 3i
(r.m.s. tissue velocity):

- Autocorrelation (Fig. 3 caption: "Flow correlations: vorticity method"):
  Pearson correlation of vorticity maps, SI Note 10 Eq. 8
  (same as ``rho_omega_matrix`` in ``t0_omega``).

- r.m.s. surface velocity (SI Note 10 Eq. 9 / Note 13):
  ``v_RMS = sqrt( (1/A) ∬ |v|^2 dA )``.
  Well pullbacks are already AP-cropped; without an ImSAnE metric tensor we use
  equal-area grid weights (uniform ``dx*dy`` cells), which matches Eq. 9 on a flat
  metric. Optional ``valid`` masks out rejected PIV cells.

Time ``t = 0`` in Figs. 3 and 5 is the "onset of GBE", which the paper defines as the
timepoint where the derivative of the r.m.s. velocity is maximal (Methods, 'Statistical
comparison of embryos at varying temperatures'; SI note on consistency with ref. 20).
That is ``gbe_onset_from_rms``, not the ``t0_omega`` vorticity landmark.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from walrus.analysis.plot_style import PANEL_SIZE, panel_grid_size
from walrus.analysis.t0_omega import rho_omega_matrix, t0_omega, vorticity

# Shared with the VRMSE bar charts so a model keeps its colour across figures.
MODEL_PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
)


def rms_velocity(
    v: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    area_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Spatially averaged r.m.s. speed over time (SI Eq. 9 / Note 13).

    Parameters
    ----------
    v :
        Velocity ``(T, H, W, 2)`` in physical units (e.g. µm/min).
    valid :
        Optional bool mask ``(T, H, W)`` or ``(H, W)``. False cells are excluded.
    area_weights :
        Optional positive weights ``(H, W)`` for ``sqrt(det g)`` (SI Eq. 12).
        If None, every grid cell contributes equally.

    Returns
    -------
    rms : ndarray, shape ``(T,)``
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 4 or v.shape[-1] != 2:
        raise ValueError(f"v must be (T, H, W, 2); got {v.shape}")
    speed2 = (v * v).sum(axis=-1)  # |v|^2

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 2:
            valid = np.broadcast_to(valid, speed2.shape)
        elif valid.shape != speed2.shape:
            raise ValueError(
                f"valid shape {valid.shape} incompatible with velocity {v.shape}"
            )
        speed2 = np.where(valid, speed2, np.nan)

    if area_weights is None:
        return np.sqrt(np.nanmean(speed2, axis=(-2, -1)))

    w = np.asarray(area_weights, dtype=float)
    if w.shape != speed2.shape[-2:]:
        raise ValueError(
            f"area_weights shape {w.shape} must match spatial {speed2.shape[-2:]}"
        )
    # Weighted mean of |v|^2, then sqrt. NaN cells get zero weight.
    w2 = np.broadcast_to(w, speed2.shape).copy()
    if valid is not None:
        w2 = np.where(np.isfinite(speed2), w2, 0.0)
    speed2 = np.where(np.isfinite(speed2), speed2, 0.0)
    denom = w2.sum(axis=(-2, -1))
    mean_s2 = (speed2 * w2).sum(axis=(-2, -1)) / np.maximum(denom, 1e-12)
    return np.sqrt(mean_s2)


def flow_autocorrelation(
    v: np.ndarray,
    dx: float,
    dy: float,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pairwise vorticity Pearson autocorrelation matrix (SI Note 10 Eq. 8).

    Returns ``M`` with ``M[i, j] = ρ_ω(ω(t_i), ω(t_j)) ∈ [-1, 1]``.
    """
    omega = vorticity(np.asarray(v, dtype=float), dx, dy)
    return rho_omega_matrix(omega, valid=valid)


def gbe_onset_from_rms(
    t: np.ndarray,
    rms: np.ndarray,
    *,
    smooth_window: int = 0,
) -> float:
    """Onset of GBE: time of maximum acceleration of the r.m.s. velocity.

    The paper labels ``t = 0`` in Figs. 3 and 5 with this definition — "the time
    when the derivative of the root-mean-squared velocity is maximal" — chosen for
    consistency with earlier publications. Aligning ensembles by max acceleration
    of ``|v(t)|`` is the same rule used in Methods.

    ``smooth_window`` optionally applies a centered moving average (in frames) to
    the r.m.s. curve before differentiating, for noisy short trajectories.
    """
    t = np.asarray(t, dtype=float)
    rms = np.asarray(rms, dtype=float)
    if t.shape != rms.shape:
        raise ValueError(f"t {t.shape} and rms {rms.shape} must match")
    if len(t) < 3:
        return float("nan")
    if smooth_window and smooth_window > 1:
        kernel = np.ones(int(smooth_window)) / float(int(smooth_window))
        rms = np.convolve(rms, kernel, mode="same")
    accel = np.gradient(rms, t)
    if not np.isfinite(accel).any():
        return float("nan")
    return float(t[int(np.nanargmax(accel))])


def align_time_to_t0(t: np.ndarray, t0: float) -> np.ndarray:
    """Shift absolute developmental time so ``t0`` (GBE onset) maps to 0."""
    return np.asarray(t, dtype=float) - float(t0)


def ensemble_mean_curve(
    times: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
    *,
    t_grid: Optional[np.ndarray] = None,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate per-embryo curves onto a common grid; return mean ± s.d.

    Embryos with non-overlapping coverage contribute NaN outside their range and
    are ignored in the mean/std at those times (nanmean / nanstd).
    """
    times = [np.asarray(t, dtype=float) for t in times]
    values = [np.asarray(y, dtype=float) for y in values]
    if not times:
        raise ValueError("need at least one curve")
    if t_grid is None:
        t_min = max(t.min() for t in times)
        t_max = min(t.max() for t in times)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            t_min = min(t.min() for t in times)
            t_max = max(t.max() for t in times)
        t_grid = np.linspace(t_min, t_max, n_grid)
    stacked = []
    for t, y in zip(times, values):
        if len(t) < 2:
            continue
        yi = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
        # Only keep interpolation inside each embryo's observed interval.
        yi = np.where((t_grid >= t.min()) & (t_grid <= t.max()), yi, np.nan)
        stacked.append(yi)
    if not stacked:
        raise ValueError("no curves long enough to interpolate")
    arr = np.vstack(stacked)
    with warnings.catch_warnings():
        # Grid points outside every embryo's coverage are all-NaN columns.
        warnings.simplefilter("ignore", RuntimeWarning)
        return t_grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


@dataclass
class EmbryoFlowResult:
    """Per-embryo GT / prediction flow metrics for Fig. 3-style plots."""

    file: Optional[str]
    dataset: str
    batch: int
    t: np.ndarray  # absolute developmental time (min)
    t_rel: np.ndarray  # time relative to gbe_onset_gt (GBE onset = 0)
    gbe_onset_gt: float  # max of d(rms)/dt on the GT curve — paper's t = 0
    gbe_onset_pred: float  # same rule applied to the predicted curve
    t0_omega_gt: float  # vorticity landmark, kept for reference
    rms_gt: np.ndarray
    rms_pred: np.ndarray
    autocorr_gt: np.ndarray
    autocorr_pred: np.ndarray


def _paper_corr_cmap():
    """Colormap used for the Fig. 3 correlation matrices.

    ``twilight_shifted`` reproduces the published scale: dark purple at −1,
    through blue to white at 0, through orange back to dark purple at +1.
    """
    import matplotlib as mpl

    return mpl.colormaps["twilight_shifted"]


def plot_rms_velocity_overlay(
    results: Sequence[EmbryoFlowResult],
    *,
    ax=None,
    show_individuals: bool = False,
    title: str = "r.m.s. tissue velocity",
    ylim: tuple[float, float] = (0.0, 7.0),
):
    """Overlay ensemble mean ± s.d. RMS curves for GT vs predictions (Fig. 3i)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=PANEL_SIZE)

    t_gt = [r.t_rel for r in results]
    y_gt = [r.rms_gt for r in results]
    t_pr = [r.t_rel for r in results]
    y_pr = [r.rms_pred for r in results]

    if show_individuals:
        for t, y in zip(t_gt, y_gt):
            ax.plot(t, y, color="0.7", lw=0.8, alpha=0.7)
        for t, y in zip(t_pr, y_pr):
            ax.plot(t, y, color="#f4a582", lw=0.8, alpha=0.7)

    tg, mg, sg = ensemble_mean_curve(t_gt, y_gt)
    tp, mp, sp = ensemble_mean_curve(t_pr, y_pr, t_grid=tg)

    ax.fill_between(tg, mg - sg, mg + sg, color="0.75", alpha=0.5, linewidth=0)
    ax.plot(tg, mg, color="k", lw=2.0, label="true (test)")
    ax.fill_between(tp, mp - sp, mp + sp, color="#f4a582", alpha=0.35, linewidth=0)
    ax.plot(tp, mp, color="#d6604d", lw=2.0, label="predicted")

    ax.axvline(0.0, color="0.5", ls=":", lw=1)

    # Where the same max-acceleration rule lands on the predictions, relative to GT.
    pred_offsets = [
        r.gbe_onset_pred - r.gbe_onset_gt
        for r in results
        if np.isfinite(r.gbe_onset_pred) and np.isfinite(r.gbe_onset_gt)
    ]
    if pred_offsets:
        ax.axvline(
            float(np.mean(pred_offsets)),
            color="#d6604d",
            ls=":",
            lw=1,
            label="predicted GBE onset (mean)",
        )

    ax.set_xlabel("Time from GBE onset (min)")
    ax.set_ylabel(r"Velocity (µm min$^{-1}$)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper right")
    return ax


def plot_rms_velocity_by_group(
    results_by_group: Mapping[str, Sequence[EmbryoFlowResult]],
    *,
    which: str = "gt",
    ax=None,
    colors=None,
    show_bands: bool = True,
    show_individuals: bool = False,
    title: str = "r.m.s. tissue velocity",
    legend_loc: str = "upper right",
    legend_fontsize: Optional[float] = None,
    ylim: tuple[float, float] = (0.0, 7.0),
    xlim: Optional[tuple[float, float]] = None,
    time_coverage: str = "intersection",
    paper_style: bool = True,
):
    """One RMS curve per group — e.g. one per rearing temperature (Fig. 3i style).

    Unlike ``plot_rms_velocity_multi_run``, which compares models on a single dataset,
    this compares *datasets*: every group contributes one ensemble-mean curve and there
    is no shared reference curve. ``which`` selects ``"gt"`` (ground truth, independent
    of the model) or ``"pred"`` (that group's open-loop predictions).

    Each group keeps its own time grid, since datasets at different temperatures cover
    different windows around GBE onset. Time is still ``t_rel``, i.e. relative to the
    ground-truth GBE onset, so the ``"gt"`` and ``"pred"`` versions share an x axis.

    ``time_coverage`` controls that per-group grid. ``"intersection"`` (default) keeps only
    the window every embryo of the group covers. ``"union"`` spans the widest window any
    embryo covers, averaging over whichever embryos are present at each time — use it for
    genotypes whose onsets are spread out (e.g. halo twist, where most movies start ~30 min
    before GBE and the intersection collapses to a stub).

    ``colors`` may be a mapping keyed by group label or a sequence; unknown labels fall
    back to ``MODEL_PALETTE`` by position.
    """
    import matplotlib.pyplot as plt

    if which not in ("gt", "pred"):
        raise ValueError(f"which must be 'gt' or 'pred'; got {which!r}")
    attr = "rms_gt" if which == "gt" else "rms_pred"
    if time_coverage not in ("intersection", "union"):
        raise ValueError(
            f"time_coverage must be 'intersection' or 'union'; got {time_coverage!r}"
        )

    groups = [k for k, v in results_by_group.items() if v]
    if not groups:
        raise ValueError("results_by_group contains no non-empty results")

    if ax is None:
        _, ax = plt.subplots(figsize=PANEL_SIZE)

    def color_for(i: int, group: str) -> str:
        if isinstance(colors, Mapping):
            return colors.get(group, MODEL_PALETTE[i % len(MODEL_PALETTE)])
        if colors is None:
            return MODEL_PALETTE[i % len(MODEL_PALETTE)]
        return colors[i % len(colors)]

    for i, group in enumerate(groups):
        results = results_by_group[group]
        color = color_for(i, group)
        times = [r.t_rel for r in results]
        values = [getattr(r, attr) for r in results]
        if show_individuals:
            for t, y in zip(times, values):
                ax.plot(t, y, color=color, lw=0.7, alpha=0.4, zorder=2)
        # Own grid per group: temperatures do not share a developmental window.
        t_grid = None
        if time_coverage == "union":
            t_grid = np.linspace(
                min(t.min() for t in times), max(t.max() for t in times), 200
            )
        t, mean, sd = ensemble_mean_curve(times, values, t_grid=t_grid)
        if show_bands:
            ax.fill_between(
                t, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0, zorder=3
            )
        ax.plot(t, mean, color=color, lw=2.0, label=group, zorder=4)

    ax.axvline(0.0, color="0.5", ls=":", lw=1, zorder=0)
    ax.set_xlabel("Time from GBE onset (min)")
    ax.set_ylabel(r"r.m.s. velocity (µm min$^{-1}$)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if paper_style:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc, fontsize=legend_fontsize)
    return ax


def plot_rms_velocity_multi_run(
    results_by_run: Mapping[str, Sequence[EmbryoFlowResult]],
    *,
    ax=None,
    colors=None,
    labels: Optional[Mapping[str, str]] = None,
    gt_from: Optional[str] = None,
    show_gt_band: bool = True,
    show_run_bands: bool = False,
    show_individuals: bool = False,
    gt_rel_tolerance: float = 0.05,
    title: str = "r.m.s. tissue velocity",
    legend_loc: str = "upper right",
    legend_fontsize: Optional[float] = None,
    ylim: tuple[float, float] = (0.0, 7.0),
):
    """Overlay predicted RMS curves from several runs against one ground truth.

    ``results_by_run`` maps a run name to that run's ``EmbryoFlowResult`` list, as
    returned by ``evaluate_flow_metrics`` (or ``compare_flow_metrics``). Each run
    contributes one ensemble-mean curve over its embryos; the ground truth is drawn
    once in black.

    Every run sees the same test embryos, so their GT curves should coincide. By
    default they are pooled into a single reference curve and a warning is emitted if
    any run deviates by more than ``gt_rel_tolerance`` (a sign of mismatched
    ``n_steps_input`` or a different dataset). Pass ``gt_from=<run>`` to use one run's
    GT instead.

    ``colors`` may be a sequence or a mapping keyed by display label; it defaults to
    ``MODEL_PALETTE`` so models keep the colours used in the VRMSE bar charts.
    """
    import matplotlib.pyplot as plt

    runs = [k for k, v in results_by_run.items() if v]
    if not runs:
        raise ValueError("results_by_run contains no non-empty results")
    if gt_from is not None and gt_from not in results_by_run:
        raise KeyError(f"gt_from={gt_from!r} not in results_by_run")

    if ax is None:
        _, ax = plt.subplots(figsize=PANEL_SIZE)

    labels = dict(labels or {})

    def label_for(run: str) -> str:
        return labels.get(run, run)

    palette = MODEL_PALETTE if colors is None else colors

    def color_for(i: int, run: str) -> str:
        if isinstance(palette, Mapping):
            return palette.get(label_for(run), palette.get(run, "0.5"))
        return palette[i % len(palette)]

    # Ground truth defines the time axis; pool it unless a reference run was given.
    gt_runs = [gt_from] if gt_from is not None else runs
    t_gt = [r.t_rel for run in gt_runs for r in results_by_run[run]]
    y_gt = [r.rms_gt for run in gt_runs for r in results_by_run[run]]
    tg, mg, sg = ensemble_mean_curve(t_gt, y_gt)

    if gt_from is None and len(runs) > 1:
        scale = np.nanmax(np.abs(mg))
        drifted = []
        for run in runs:
            _, m_run, _ = ensemble_mean_curve(
                [r.t_rel for r in results_by_run[run]],
                [r.rms_gt for r in results_by_run[run]],
                t_grid=tg,
            )
            dev = np.nanmax(np.abs(m_run - mg))
            if scale > 0 and np.isfinite(dev) and dev / scale > gt_rel_tolerance:
                drifted.append((run, dev / scale))
        if drifted:
            detail = ", ".join(f"{run} ({dev:.0%})" for run, dev in drifted)
            warnings.warn(
                "Ground-truth RMS curves differ between runs, so the pooled 'ground "
                f"truth' is not a single consistent reference: {detail}. Check that "
                "the runs share n_steps_input and the same dataset, or pass gt_from=.",
                stacklevel=2,
            )

    if show_individuals:
        for t, y in zip(t_gt, y_gt):
            ax.plot(t, y, color="0.75", lw=0.7, alpha=0.7, zorder=1)

    if show_gt_band:
        ax.fill_between(tg, mg - sg, mg + sg, color="0.75", alpha=0.45, linewidth=0, zorder=2)
    ax.plot(tg, mg, color="k", lw=2.4, label="ground truth", zorder=6)

    for i, run in enumerate(runs):
        results = results_by_run[run]
        color = color_for(i, run)
        if show_individuals:
            for r in results:
                ax.plot(r.t_rel, r.rms_pred, color=color, lw=0.7, alpha=0.45, zorder=3)
        tp, mp, sp = ensemble_mean_curve(
            [r.t_rel for r in results], [r.rms_pred for r in results], t_grid=tg
        )
        if show_run_bands:
            ax.fill_between(
                tp, mp - sp, mp + sp, color=color, alpha=0.18, linewidth=0, zorder=3
            )
        ax.plot(tp, mp, color=color, lw=1.9, label=label_for(run), zorder=5)

    ax.axvline(0.0, color="0.5", ls=":", lw=1, zorder=0)
    ax.set_xlabel("Time from GBE onset (min)")
    ax.set_ylabel(r"Velocity (µm min$^{-1}$)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc=legend_loc, fontsize=legend_fontsize)
    return ax


def plot_autocorrelation_pair(
    result: EmbryoFlowResult,
    *,
    figsize=panel_grid_size(2),
    cmap=None,
):
    """Side-by-side true vs predicted vorticity autocorrelation (Fig. 3c style)."""
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = _paper_corr_cmap()

    t = result.t_rel
    t_lo, t_hi = float(t[0]), float(t[-1])
    # Paper convention: identical ranges on both axes, time increasing to the
    # right on x and downward on y, so the diagonal runs from the top-left.
    extent = [t_lo, t_hi, t_hi, t_lo]
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, M, label in zip(
        axes,
        (result.autocorr_gt, result.autocorr_pred),
        ("true (test)", "predicted"),
    ):
        im = ax.imshow(
            M,
            origin="upper",
            extent=extent,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(t_hi, t_lo)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Time (min)")
        name = result.file or f"batch {result.batch}"
        ax.set_title(f"{label}\n{name}")
        ax.axhline(0.0, color="0.4", ls=":", lw=0.8)
        ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
    fig.colorbar(
        im,
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Correlation",
        ticks=[-1, 0, 1],
    )
    return fig, axes


def plot_all_autocorrelations(
    results: Sequence[EmbryoFlowResult],
    *,
    max_embryos: Optional[int] = None,
):
    """Plot true vs predicted autocorrelation for each embryo."""
    import matplotlib.pyplot as plt

    figs = []
    for i, r in enumerate(results):
        if max_embryos is not None and i >= max_embryos:
            break
        fig, _ = plot_autocorrelation_pair(r)
        figs.append(fig)
        plt.show()
    return figs
