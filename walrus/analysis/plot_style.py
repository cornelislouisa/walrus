"""Shared figure style for the analysis notebooks.

Every notebook figure is built from one panel size so that bar charts, spectral
grids and RMS curves can be placed side by side in a paper figure without
rescaling (which would leave each panel with a different effective font size).
"""

from typing import Optional, Sequence

# One axes box. Multi-panel figures tile this via :func:`panel_grid_size`.
PANEL_SIZE = (4.6, 4.4)

# Point size for tick labels and body text; titles and labels scale off this.
BASE_FONT_SIZE = 13.0

# Axes-box margins, in inches rather than figure fractions so that a one-panel and
# a three-panel figure built by :func:`panel_figure` share the exact same axes box.
# ``bottom`` leaves room for rotated model labels, ``top`` for a per-panel title.
PANEL_MARGINS = {"left": 0.95, "right": 0.15, "bottom": 1.15, "top": 0.5}


def panel_grid_size(
    n_cols: int = 1,
    n_rows: int = 1,
    panel_size: Sequence[float] = PANEL_SIZE,
) -> tuple[float, float]:
    """``figsize`` for an ``n_rows`` x ``n_cols`` grid of equally sized panels."""
    width, height = panel_size
    return (width * n_cols, height * n_rows)


def apply_paper_style(font_size: Optional[float] = None) -> None:
    """Set Type 42 fonts and a single font scale for all notebook figures.

    Type 42 keeps text editable as fonts in Illustrator / Inkscape. Sizes are
    relative to ``font_size`` so one call changes every figure consistently.
    """
    import matplotlib as mpl

    size = BASE_FONT_SIZE if font_size is None else float(font_size)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": size,
            "axes.titlesize": size + 1,
            "axes.labelsize": size,
            "xtick.labelsize": size - 1,
            "ytick.labelsize": size - 1,
            "legend.fontsize": size - 1,
            "figure.titlesize": size + 2,
        }
    )


def panel_figure(
    n_cols: int = 1,
    panel_size: Sequence[float] = PANEL_SIZE,
    margins: Optional[dict] = None,
    **subplot_kw,
):
    """Row of ``n_cols`` panels whose axes boxes are the same size for any ``n_cols``.

    Returns ``(fig, axes)`` with ``axes`` always a 1-D array. Margins are fixed in
    inches instead of using ``tight_layout``, so a title that needs more room
    overflows the canvas rather than shrinking the axes; save and display with
    ``bbox_inches="tight"`` to keep the overflow. Figure-level titles are best
    placed at ``y=1.0, va="bottom"``, i.e. entirely above the reserved top margin.
    """
    import matplotlib.pyplot as plt

    m = dict(PANEL_MARGINS if margins is None else margins)
    width, height = panel_size
    fig_width = width * n_cols
    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_width, height), squeeze=False, **subplot_kw
    )
    # Inter-panel gap == left + right margin is what makes each panel exactly
    # ``width - left - right`` wide, independent of the panel count.
    gap = m["left"] + m["right"]
    fig.subplots_adjust(
        left=m["left"] / fig_width,
        right=1.0 - m["right"] / fig_width,
        bottom=m["bottom"] / height,
        top=1.0 - m["top"] / height,
        wspace=gap / (width - gap),
    )
    return fig, axes[0]


def annotation_font_size() -> float:
    """Slightly smaller than body text, for bar-value labels drawn inside axes."""
    import matplotlib as mpl

    return float(mpl.rcParams["font.size"]) - 2.0
