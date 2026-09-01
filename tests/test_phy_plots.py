"""Expert-page figure builders: range policy, overlays, UTC axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bokeh.models import BoxAnnotation, Label, Span

from legenddashboard.geds.phy import phy_plots

DETS = ["A", "B"]


def _frames(n=48):
    idx = pd.date_range("2026-07-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    mean = pd.DataFrame(
        {"A": 14 + rng.normal(0, 0.3, n), "B": 30 + rng.normal(0, 0.3, n)}, index=idx
    )
    std = pd.DataFrame(1.0, index=idx, columns=DETS)
    std.iloc[10, 0] = 500.0  # one noise burst
    lo = mean - 1.5
    hi = mean + 1.5
    hi.iloc[10, 0] = 3000.0  # envelope spike
    return mean, std, lo, hi


def _meta():
    return phy_plots.PlotMeta(
        label="Noise", unit="ADC", abs_unit="ADC", flag_display="Pulser Events",
        param_display="Noise", string="1", run="r001", period="p19",
        experiment="l200",
    )  # fmt: skip


def _spans(p, dimension="width"):
    return [s for s in p.center if isinstance(s, Span) and s.dimension == dimension]


def test_auto_y_range_ignores_envelope_and_bursts():
    mean, std, lo, hi = _frames()
    p = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta())
    assert 5 < p.y_range.start < 14
    assert 30 < p.y_range.end < 45  # not the 3000 spike, not the 500 sigma


def test_threshold_spans_and_shading_from_limits():
    mean, std, lo, hi = _frames()
    p = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta(), limits=(0, 20))
    assert sorted(s.location for s in _spans(p)) == [0, 20]
    boxes = [b for b in p.center if isinstance(b, BoxAnnotation)]
    assert 20 in {b.bottom for b in boxes}
    assert 0 in {b.top for b in boxes}
    labels = [item.label["value"] for item in p.legend[0].items]
    assert "Threshold (limits)" in labels


def test_no_thresholds_without_limits():
    mean, std, lo, hi = _frames()
    p = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta())
    assert _spans(p) == []


def test_fwhm_spans_per_detector_and_missing_note():
    mean, std, lo, hi = _frames()
    p = phy_plots.phy_plot_binned_vsTime(
        mean, std, lo, hi, _meta(), fwhm={"A": 3.0, "B": 2.0}
    )
    assert sorted(s.location for s in _spans(p)) == [-1.5, -1.0, 1.0, 1.5]
    labels = [item.label["value"] for item in p.legend[0].items]
    assert any("FWHM" in label for label in labels)
    p2 = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta(), fwhm={})
    notes = [a.text for a in p2.center if isinstance(a, Label)]
    assert any("FWHM" in t for t in notes)
    # fwhm=None = not a calibrated-gain plot: no note
    p3 = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta())
    assert not [a for a in p3.center if isinstance(a, Label)]


def test_time_axis_is_utc():
    mean, std, lo, hi = _frames()
    p = phy_plots.phy_plot_binned_vsTime(mean, std, lo, hi, _meta())
    source = p.renderers[-1].data_source
    first = pd.Timestamp(source.data["datetime"][0])
    assert first.tz is None
    assert first == mean.index[0].tz_convert("UTC").tz_localize(None)
    assert "UTC" in p.xaxis.axis_label


def test_histogram_reports_flow_and_limits():
    edges = np.linspace(0, 10, 11)
    counts = np.ones(10)
    p = phy_plots.phy_plot_dist_histogram(
        (edges, counts, {"flow": (3.0, 4.0)}), _meta(), limits=(None, 8)
    )
    notes = [a.text for a in p.center if isinstance(a, Label)]
    assert any("underflow 3" in t and "overflow 4" in t for t in notes)
    assert [s.location for s in _spans(p, "height")] == [8]
    assert _spans(p, "width") == []  # limits are x-values: no horizontal lines


@pytest.mark.parametrize("n", [1, 2])
def test_degenerate_frames_do_not_crash(n):
    idx = pd.date_range("2026-07-01", periods=n, freq="1h", tz="UTC")
    const = pd.DataFrame(5.0, index=idx, columns=DETS)
    p = phy_plots.phy_plot_binned_vsTime(const, const * 0, const, const, _meta())
    assert p.y_range.start < 5 < p.y_range.end
