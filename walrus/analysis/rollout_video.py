"""Render rollout videos from cached artifacts with a configurable colormap.

``the_well.benchmark.metrics.make_video`` hardcodes ``viridis``/``inferno``. Velocity
is signed, so a sequential map hides the sign structure and puts the zero crossing at
an arbitrary color. These helpers read the same cached ``.npz`` rollouts and default to
a diverging map centered on zero.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

# Diverging maps center on zero; sequential maps are used for |error|.
DIVERGING_COLORMAPS = (
    "RdBu_r",
    "coolwarm",
    "bwr",
    "seismic",
    "PuOr_r",
    "BrBG_r",
    "Spectral_r",
    "PRGn_r",
)
SEQUENTIAL_COLORMAPS = ("magma", "inferno", "plasma", "cividis", "turbo", "viridis")

# One look for every rollout video, so the notebook figures and the videos written
# from the cache agree. ``layout="grid"`` is the gapless field grid; "panels" is the
# older per-panel-colorbar layout.
DEFAULT_CMAP = "RdBu_r"
DEFAULT_ERROR_CMAP = "magma"
DEFAULT_PERCENTILE = 97.0
DEFAULT_CENTER = "auto"
DEFAULT_LABELS = "outside"
DEFAULT_VIDEO_STYLE: dict[str, Any] = {
    "layout": "grid",
    "cmap": DEFAULT_CMAP,
    "error_cmap": DEFAULT_ERROR_CMAP,
    "percentile": DEFAULT_PERCENTILE,
    "center": DEFAULT_CENTER,
    "labels": DEFAULT_LABELS,
    "rows": ("true", "pred", "error"),
}

# Inches reserved around the panel grid for ``labels="outside"``. Sized in inches so
# the panels keep their exact data aspect ratio whatever the grid shape.
OUTSIDE_LABEL_MARGINS = {"left": 0.5, "top": 0.34, "bottom": 0.24}
ROW_LABELS = {
    "inside": {"true": "truth", "pred": "pred", "error": "|err|"},
    "outside": {"true": "ground truth", "pred": "predicted", "error": "|error|"},
}


def _label_mode(labels: Union[bool, str]) -> str:
    """Normalize ``labels`` to ``"inside"`` / ``"outside"`` / ``"none"``."""
    if labels is True:
        return "inside"
    if labels is False:
        return "none"
    mode = str(labels)
    if mode not in {"inside", "outside", "none"}:
        raise ValueError(
            "labels must be True, False, 'inside', 'outside' or 'none'; "
            f"got {labels!r}"
        )
    return mode


@dataclass
class RolloutFrames:
    """Ground-truth / predicted stacks for one embryo, as ``(T, H, W, C)``."""

    name: str
    dataset: str
    field_names: tuple[str, ...]
    true: np.ndarray
    pred: np.ndarray

    @property
    def error(self) -> np.ndarray:
        return np.abs(self.true - self.pred)


def _drop_singleton_spatial(array: np.ndarray) -> np.ndarray:
    """Cached arrays are ``(T, H, W, D, C)`` with ``D == 1`` for 2D datasets."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 5 and array.shape[-2] == 1:
        return array[..., 0, :]
    return array


def load_rollout_frames(path: Union[str, pathlib.Path]) -> RolloutFrames:
    """Read one cached ``.npz`` rollout artifact."""
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=False) as z:
        true = _drop_singleton_spatial(z["ref"])
        pred = _drop_singleton_spatial(z["pred"])
        fields = tuple(str(x) for x in z["field_names"].tolist())
        dataset = str(z["dataset"].item())
        source = str(z["source_path"].item())
    name = pathlib.Path(source).stem if source else path.stem
    return RolloutFrames(
        name=name, dataset=dataset, field_names=fields, true=true, pred=pred
    )


def cache_artifacts(cache_path: Union[str, pathlib.Path]) -> list[pathlib.Path]:
    """All artifact npz files under a cache entry directory."""
    cache_path = pathlib.Path(cache_path)
    artifacts = cache_path / "artifacts"
    root = artifacts if artifacts.is_dir() else cache_path
    return sorted(root.glob("*.npz"))


def _limits(
    frames: RolloutFrames,
    *,
    percentile: float,
    symmetric: bool,
    center: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-field color limits.

    ``center`` decides where a diverging colormap's neutral color sits. ``"zero"`` is
    right for signed velocity, but the myosin diagonal components carry a large offset,
    so centering them on zero pushes the whole field to one side of the map.
    ``"auto"`` keeps zero unless the field's median is a significant fraction of its
    spread, in which case it centers on the median.
    """
    n_fields = frames.true.shape[-1]
    flat_true = frames.true.reshape(-1, n_fields)
    flat_error = frames.error.reshape(-1, n_fields)
    emaxes = np.maximum(np.nanpercentile(flat_error, percentile, axis=0), 1e-12)

    if not symmetric:
        vmins = np.nanpercentile(flat_true, 100.0 - percentile, axis=0)
        vmaxes = np.nanpercentile(flat_true, percentile, axis=0)
        return vmins, vmaxes, emaxes

    medians = np.nanmedian(flat_true, axis=0)
    spans = np.nanpercentile(np.abs(flat_true - medians), percentile, axis=0)
    spans = np.maximum(spans, 1e-12)
    if center == "zero":
        centers = np.zeros_like(medians)
    elif center == "median":
        centers = medians
    elif center == "auto":
        centers = np.where(np.abs(medians) > 0.5 * spans, medians, 0.0)
    else:
        raise ValueError(f"center must be 'zero', 'median' or 'auto'; got {center!r}")
    return centers - spans, centers + spans, emaxes


def render_rollout_video(
    source: Union[str, pathlib.Path, RolloutFrames],
    out_path: Union[str, pathlib.Path],
    *,
    cmap: str = DEFAULT_CMAP,
    error_cmap: str = DEFAULT_ERROR_CMAP,
    symmetric: bool = True,
    center: str = DEFAULT_CENTER,
    percentile: float = DEFAULT_PERCENTILE,
    fps: Optional[int] = None,
    dpi: int = 130,
    dark: bool = True,
    title: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> pathlib.Path:
    """Write a true / predicted / |error| rollout video using ``cmap``.

    ``symmetric`` keeps the data rows centered on zero, which is what makes a
    diverging colormap meaningful for signed velocity components.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    frames = source if isinstance(source, RolloutFrames) else load_rollout_frames(source)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep = list(range(len(frames.field_names)))
    if fields is not None:
        keep = [i for i, name in enumerate(frames.field_names) if name in set(fields)]
        if not keep:
            raise ValueError(
                f"none of {list(fields)} in cached fields {list(frames.field_names)}"
            )
    names = [frames.field_names[i] for i in keep]
    true = frames.true[..., keep]
    pred = frames.pred[..., keep]
    frames = RolloutFrames(frames.name, frames.dataset, tuple(names), true, pred)

    vmins, vmaxes, emaxes = _limits(
        frames, percentile=percentile, symmetric=symmetric, center=center
    )
    n_fields = len(names)
    n_steps = frames.true.shape[0]
    fps = fps or max(5, min(16, n_steps // 8))

    style = "dark_background" if dark else "default"
    with plt.style.context(style):
        fig, axes = plt.subplots(
            3, n_fields, figsize=(3.6 * n_fields + 1.2, 7.6), dpi=dpi, squeeze=False
        )
        images = []
        stacks = (frames.true, frames.pred, frames.error)
        for col, name in enumerate(names):
            for row in range(3):
                is_error = row == 2
                im = axes[row][col].imshow(
                    stacks[row][0, ..., col],
                    cmap=error_cmap if is_error else cmap,
                    vmin=0.0 if is_error else vmins[col],
                    vmax=emaxes[col] if is_error else vmaxes[col],
                    origin="lower",
                    interpolation="nearest",
                    aspect="auto",
                )
                fig.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.02)
                axes[row][col].set_xticks([])
                axes[row][col].set_yticks([])
                images.append((im, row, col))
            axes[0][col].set_title(name, fontsize=11)
        for row, label in enumerate(("ground truth", "predicted", "|error|")):
            axes[row][0].set_ylabel(label, fontsize=10)

        heading = title or f"{frames.dataset} — {frames.name}"
        suptitle = fig.suptitle(f"{heading}\nframe 0/{n_steps - 1}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        writer = FFMpegWriter(
            fps=fps,
            bitrate=6000,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        with writer.saving(fig, str(out_path), dpi):
            for step in range(n_steps):
                for im, row, col in images:
                    im.set_array(stacks[row][step, ..., col])
                suptitle.set_text(f"{heading}\nframe {step}/{n_steps - 1}")
                writer.grab_frame()
        plt.close(fig)
    return out_path


def render_rollout_grid_video(
    source: Union[str, pathlib.Path, RolloutFrames],
    out_path: Union[str, pathlib.Path],
    *,
    cmap: str = DEFAULT_CMAP,
    error_cmap: str = DEFAULT_ERROR_CMAP,
    rows: Sequence[str] = ("true", "pred"),
    symmetric: bool = True,
    center: str = DEFAULT_CENTER,
    percentile: float = DEFAULT_PERCENTILE,
    fps: Optional[int] = None,
    panel_height: float = 1.9,
    dpi: int = 160,
    labels: Union[bool, str] = DEFAULT_LABELS,
    label_size: Optional[float] = None,
    fields: Optional[Sequence[str]] = None,
) -> pathlib.Path:
    """Gapless field grid: no colorbars, panels flush against each other.

    Every panel keeps the data aspect ratio and the figure is sized to match, so
    there is no letterboxing between fields. ``rows`` picks any of ``"true"``,
    ``"pred"``, ``"error"``.

    ``labels="outside"`` puts the field names, row names and frame counter in a black
    margin around the grid, leaving the data pixels untouched; ``"inside"`` draws them
    over the frames; ``"none"`` omits them.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    valid_rows = {"true", "pred", "error"}
    rows = list(rows)
    unknown = [r for r in rows if r not in valid_rows]
    if unknown:
        raise ValueError(f"rows must be from {sorted(valid_rows)}; got {unknown}")

    frames = source if isinstance(source, RolloutFrames) else load_rollout_frames(source)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep = list(range(len(frames.field_names)))
    if fields is not None:
        wanted = list(fields)
        keep = [
            i
            for name in wanted
            for i, cached in enumerate(frames.field_names)
            if cached == name
        ]
        if not keep:
            raise ValueError(
                f"none of {wanted} in cached fields {list(frames.field_names)}"
            )
    names = [frames.field_names[i] for i in keep]
    frames = RolloutFrames(
        frames.name,
        frames.dataset,
        tuple(names),
        frames.true[..., keep],
        frames.pred[..., keep],
    )

    vmins, vmaxes, emaxes = _limits(
        frames, percentile=percentile, symmetric=symmetric, center=center
    )
    stacks = {"true": frames.true, "pred": frames.pred, "error": frames.error}
    n_cols, n_rows = len(names), len(rows)
    n_steps = frames.true.shape[0]
    fps = fps or max(5, min(16, n_steps // 8))

    mode = _label_mode(labels)
    size = label_size if label_size is not None else (7.5 if mode == "inside" else 9.0)

    height, width = frames.true.shape[1:3]
    panel_width = panel_height * (width / height)
    margins = (
        OUTSIDE_LABEL_MARGINS
        if mode == "outside"
        else {"left": 0.0, "top": 0.0, "bottom": 0.0}
    )
    fig_width = panel_width * n_cols + margins["left"]
    fig_height = panel_height * n_rows + margins["top"] + margins["bottom"]
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor="black")
    # Zero spacing between panels; any figure margin exists only to hold outer labels.
    grid = fig.add_gridspec(
        n_rows,
        n_cols,
        wspace=0.0,
        hspace=0.0,
        left=margins["left"] / fig_width,
        right=1.0,
        bottom=margins["bottom"] / fig_height,
        top=1.0 - margins["top"] / fig_height,
    )

    images = []
    for r, row in enumerate(rows):
        for c, name in enumerate(names):
            ax = fig.add_subplot(grid[r, c])
            is_error = row == "error"
            im = ax.imshow(
                stacks[row][0, ..., c],
                cmap=error_cmap if is_error else cmap,
                vmin=0.0 if is_error else vmins[c],
                vmax=emaxes[c] if is_error else vmaxes[c],
                origin="lower",
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            images.append((im, row, c))
            if mode == "inside":
                if r == 0:
                    ax.text(
                        0.5,
                        0.975,
                        name,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=size,
                        color="white",
                        path_effects=_stroke(),
                    )
                if c == 0:
                    ax.text(
                        0.015,
                        0.03,
                        ROW_LABELS["inside"][row],
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=size,
                        color="white",
                        path_effects=_stroke(),
                    )
            elif mode == "outside":
                if r == 0:
                    ax.set_title(name, color="white", fontsize=size, pad=3)
                if c == 0:
                    ax.set_ylabel(
                        ROW_LABELS["outside"][row],
                        color="white",
                        fontsize=size,
                        labelpad=3,
                    )

    counter = None
    if mode != "none":
        counter = fig.text(
            0.995,
            0.008 if mode == "outside" else 0.01,
            "",
            ha="right",
            va="bottom",
            fontsize=size,
            color="white",
            path_effects=None if mode == "outside" else _stroke(),
        )

    writer = FFMpegWriter(
        fps=fps,
        bitrate=8000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )
    with writer.saving(fig, str(out_path), dpi):
        for step in range(n_steps):
            for im, row, col in images:
                im.set_array(stacks[row][step, ..., col])
            if counter is not None:
                counter.set_text(f"{step}/{n_steps - 1}")
            writer.grab_frame()
    plt.close(fig)
    return out_path


def style_tag(style: Optional[dict[str, Any]] = None) -> str:
    """Short folder-safe name for a style, so two looks never share a directory."""
    merged = {**DEFAULT_VIDEO_STYLE, **(style or {})}
    parts = [str(merged["layout"]), str(merged["cmap"])]
    if merged["layout"] == "grid":
        parts.append(_label_mode(merged["labels"]))
    return "_".join(parts)


def render_styled_video(
    source: Union[str, pathlib.Path, RolloutFrames],
    out_path: Union[str, pathlib.Path],
    **style: Any,
) -> pathlib.Path:
    """Render one rollout with ``DEFAULT_VIDEO_STYLE``, overridden by ``style``.

    ``layout="grid"`` is the gapless grid (no colorbars, labels inside the frames);
    ``layout="panels"`` keeps per-panel colorbars and titles and always draws
    truth / predicted / |error|.
    """
    merged = {**DEFAULT_VIDEO_STYLE, **style}
    layout = str(merged.pop("layout"))
    if layout == "grid":
        return render_rollout_grid_video(source, out_path, **merged)
    if layout == "panels":
        # This layout always draws all three rows and labels its own axes.
        merged.pop("rows", None)
        merged.pop("labels", None)
        merged.pop("label_size", None)
        return render_rollout_video(source, out_path, **merged)
    raise ValueError(f"layout must be 'grid' or 'panels'; got {layout!r}")


def _stroke():
    """Thin dark outline so in-panel labels stay legible over any colormap."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=1.6, foreground="black")]


def colormap_preview(
    source: Union[str, pathlib.Path, RolloutFrames],
    out_path: Union[str, pathlib.Path],
    *,
    cmaps: Iterable[str] = DIVERGING_COLORMAPS,
    frame: Optional[int] = None,
    field: int = 0,
    symmetric: bool = True,
    center: str = DEFAULT_CENTER,
    percentile: float = DEFAULT_PERCENTILE,
    dpi: int = 130,
    dark: bool = True,
) -> pathlib.Path:
    """Render one frame under several colormaps, for picking one before encoding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = source if isinstance(source, RolloutFrames) else load_rollout_frames(source)
    cmaps = list(cmaps)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmins, vmaxes, _ = _limits(
        frames, percentile=percentile, symmetric=symmetric, center=center
    )
    step = frame if frame is not None else frames.true.shape[0] // 2
    image = frames.true[step, ..., field]

    n_cols = min(4, len(cmaps))
    n_rows = int(np.ceil(len(cmaps) / n_cols))
    style = "dark_background" if dark else "default"
    with plt.style.context(style):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.4 * n_cols, 2.9 * n_rows), dpi=dpi, squeeze=False
        )
        for ax in axes.ravel():
            ax.axis("off")
        for i, name in enumerate(cmaps):
            ax = axes[i // n_cols][i % n_cols]
            ax.axis("on")
            im = ax.imshow(
                image,
                cmap=name,
                vmin=vmins[field],
                vmax=vmaxes[field],
                origin="lower",
                interpolation="nearest",
                aspect="auto",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{frames.dataset} — {frames.field_names[field]} @ frame {step}", fontsize=12
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def render_cache_videos(
    cache_path: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    *,
    cmap: str = DEFAULT_CMAP,
    limit: Optional[int] = None,
    **kwargs,
) -> list[pathlib.Path]:
    """Render every embryo in one cache entry with the same colormap."""
    artifacts = cache_artifacts(cache_path)
    if limit is not None:
        artifacts = artifacts[:limit]
    out_dir = pathlib.Path(out_dir)
    written = []
    for artifact in artifacts:
        frames = load_rollout_frames(artifact)
        written.append(
            render_rollout_video(
                frames, out_dir / f"{artifact.stem}__{cmap}.mp4", cmap=cmap, **kwargs
            )
        )
    return written


def cache_label(cache_path: Union[str, pathlib.Path]) -> str:
    """Human-readable ``model / run / data`` label from a cache manifest."""
    manifest = pathlib.Path(cache_path) / "manifest.json"
    if not manifest.is_file():
        return str(cache_path)
    info = json.loads(manifest.read_text())
    return f"{info.get('model')} / {info.get('run_name')} / {info.get('data')}"
