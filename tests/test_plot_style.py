import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from walrus.analysis import (  # noqa: E402
    PANEL_SIZE,
    annotation_font_size,
    apply_paper_style,
    panel_figure,
    panel_grid_size,
)


def _axes_size_inches(fig, ax):
    box = ax.get_position()
    return (box.width * fig.get_figwidth(), box.height * fig.get_figheight())


def test_panel_figure_axes_box_is_independent_of_panel_count():
    """A one-panel bar chart must be droppable next to a three-panel grid."""
    fig1, axes1 = panel_figure(1)
    fig3, axes3 = panel_figure(3)
    try:
        expected = _axes_size_inches(fig1, axes1[0])
        for ax in axes3:
            got = _axes_size_inches(fig3, ax)
            assert got[0] == pytest.approx(expected[0])
            assert got[1] == pytest.approx(expected[1])
        assert fig3.get_figwidth() == 3 * fig1.get_figwidth()
        assert fig3.get_figheight() == fig1.get_figheight()
    finally:
        plt.close(fig1)
        plt.close(fig3)


def test_panel_grid_size_tiles_the_panel():
    assert panel_grid_size(3, 2) == (PANEL_SIZE[0] * 3, PANEL_SIZE[1] * 2)


def test_apply_paper_style_scales_every_text_element():
    apply_paper_style(20.0)
    try:
        rc = matplotlib.rcParams
        assert rc["pdf.fonttype"] == 42
        assert rc["font.size"] == 20.0
        for key in (
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "figure.titlesize",
        ):
            assert float(rc[key]) >= 19.0, key
        assert annotation_font_size() == 18.0
    finally:
        apply_paper_style()
