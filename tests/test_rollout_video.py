"""Tests for the shared rollout-video style in :mod:`walrus.analysis.rollout_video`."""

import pytest

from walrus.analysis import rollout_video


def _capture(monkeypatch):
    """Record the kwargs each renderer would receive, without encoding a video."""
    calls = {}

    def fake(name):
        def render(source, out_path, **kwargs):
            calls[name] = kwargs
            return out_path

        return render

    monkeypatch.setattr(rollout_video, "render_rollout_grid_video", fake("grid"))
    monkeypatch.setattr(rollout_video, "render_rollout_video", fake("panels"))
    return calls


def test_default_style_is_diverging_and_auto_centered():
    style = rollout_video.DEFAULT_VIDEO_STYLE
    assert style["cmap"] in rollout_video.DIVERGING_COLORMAPS
    assert style["error_cmap"] in rollout_video.SEQUENTIAL_COLORMAPS
    assert style["center"] == "auto"
    assert style["layout"] == "grid"


def test_styled_video_passes_default_style_to_grid_renderer(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    rollout_video.render_styled_video(tmp_path / "in.npz", tmp_path / "out.mp4")
    assert "panels" not in calls
    assert calls["grid"]["cmap"] == rollout_video.DEFAULT_CMAP
    assert calls["grid"]["percentile"] == rollout_video.DEFAULT_PERCENTILE
    assert calls["grid"]["center"] == "auto"
    assert calls["grid"]["labels"] == "outside"
    assert calls["grid"]["rows"] == rollout_video.DEFAULT_VIDEO_STYLE["rows"]


def test_panels_layout_drops_rows_it_cannot_accept(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    rollout_video.render_styled_video(
        tmp_path / "in.npz", tmp_path / "out.mp4", layout="panels", cmap="PuOr_r"
    )
    assert "grid" not in calls
    assert "rows" not in calls["panels"]
    assert calls["panels"]["cmap"] == "PuOr_r"


def test_unknown_layout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="layout must be"):
        rollout_video.render_styled_video(
            tmp_path / "in.npz", tmp_path / "out.mp4", layout="mosaic"
        )


def test_style_tag_separates_layouts_colormaps_and_label_modes():
    assert rollout_video.style_tag() == "grid_RdBu_r_outside"
    assert rollout_video.style_tag({"cmap": "PuOr_r"}) == "grid_PuOr_r_outside"
    assert rollout_video.style_tag({"labels": True}) == "grid_RdBu_r_inside"
    assert rollout_video.style_tag({"labels": False}) == "grid_RdBu_r_none"
    # The panels layout labels its own axes, so the mode is not part of its tag.
    assert rollout_video.style_tag({"layout": "panels"}) == "panels_RdBu_r"


def test_label_mode_normalizes_booleans_and_rejects_unknown():
    assert rollout_video._label_mode(True) == "inside"
    assert rollout_video._label_mode(False) == "none"
    assert rollout_video._label_mode("outside") == "outside"
    with pytest.raises(ValueError, match="labels must be"):
        rollout_video._label_mode("beside")


def test_panels_layout_drops_label_keys_it_cannot_accept(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    rollout_video.render_styled_video(
        tmp_path / "in.npz",
        tmp_path / "out.mp4",
        layout="panels",
        label_size=11.0,
    )
    assert "labels" not in calls["panels"]
    assert "label_size" not in calls["panels"]
