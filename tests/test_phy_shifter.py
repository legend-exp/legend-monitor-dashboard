"""Shifter figures from synthetic period-contract frames (no lmon import)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bokeh.models import BoxAnnotation, Label, Span

from legenddashboard.geds.phy import shifter_plots

DETS = ["V1", "V2", "V3", "B1", "B2", "P3"]


def summary_frame(**overrides):
    frame = pd.DataFrame(
        {
            "ged": DETS,
            "string": [2, 2, 2, 1, 1, 1],
            "pos": [3, 1, 2, 2, 1, 3],
            "mean": [0.1, -0.2, 0.0, 0.3, 2.5, -0.4],
            "std": [0.5, 0.4, 0.3, 0.2, 0.9, 0.1],
            "min": [-1.0, -1.5, -0.6, -0.2, 0.5, -0.9],
            "max": [1.2, 0.9, 0.7, 0.8, 4.1, 0.2],
            "fwhm": [3.0, 2.8, np.nan, 2.9, 3.3, 2.7],
            "usability": ["on", "ac", "on", "off", "on", "on"],
        }
    )
    for k, v in overrides.items():
        frame[k] = v
    return frame


def _spans(p):
    return [s for s in p.center if isinstance(s, Span) and s.dimension == "width"]


def test_detector_summary_order_and_elements():
    p = shifter_plots.detector_summary(summary_frame(), "pulser_stab", "p19", "r001")
    # string 1 first, by position within the string
    assert p.x_range.factors == ["B2", "B1", "P3", "V2", "V3", "V1"]
    labels = {item.label["value"] for item in p.legend[0].items}
    assert {"±FWHM (threshold)", "±1 sigma", "Min/Max", "Mean"} <= labels
    assert {"Usability: off", "Usability: ac"} <= labels
    assert (p.y_range.start, p.y_range.end) == (-6, 6)
    texts = [a.text for a in p.center if isinstance(a, Label)]
    assert texts.count("String 1") == 1
    assert texts.count("String 2") == 1


def test_detector_summary_thresholds_by_title():
    p = shifter_plots.detector_summary(summary_frame(), "baseln_stab", "p19", "r001")
    assert sorted(s.location for s in _spans(p)) == [-10, 0, 10]
    assert len([b for b in p.center if isinstance(b, BoxAnnotation)]) == 2
    assert (p.y_range.start, p.y_range.end) == (-20, 20)
    p2 = shifter_plots.detector_summary(summary_frame(), "baseln_spike", "p19", "r001")
    assert [s.location for s in _spans(p2)] == [50]
    assert (p2.y_range.start, p2.y_range.end) == (0, 100)


def test_detector_summary_without_fwhm_or_usability_columns():
    frame = summary_frame().drop(columns=["fwhm", "usability"])
    p = shifter_plots.detector_summary(frame, "pulser_stab_uncalib", "p19", "r001")
    labels = {item.label["value"] for item in p.legend[0].items}
    assert not any("FWHM" in label for label in labels)
    assert not any("Usability" in label for label in labels)
    assert p.y_range.start < -1.5  # auto range
    assert p.y_range.end > 4.1


def test_detector_summary_empty_frame():
    p = shifter_plots.detector_summary(
        summary_frame().iloc[:0], "pulser_stab", "p19", "r001"
    )
    assert "no detectors" in p.title.text


@pytest.fixture
def period_tree(tmp_path):
    root = tmp_path / "lmon"
    pdir = root / "generated/plt/hit/phy/p19"
    pdir.mkdir(parents=True)
    path = pdir / "l200-p19-phy-monitoring.hdf"
    with pd.HDFStore(path, "w") as store:
        store.put("detector_summary/pulser_stab/r001", summary_frame(), format="fixed")
        store.put("detector_summary/baseln_stab/r001", summary_frame(), format="fixed")
        store.put("detector_summary/baseln_stab/r000", summary_frame(), format="fixed")
    return root


def test_shifter_page_menus_follow_period_file(period_tree, monkeypatch):
    from legenddashboard.geds.phy.phy_shifter import PhyShifterMonitoring

    # bypass the production-tree discovery in Monitoring.__init__
    monkeypatch.setattr(
        "legenddashboard.base.get_sort_dets", lambda _p: None, raising=True
    )
    monkeypatch.setattr(
        "legenddashboard.base.get_dataflow_config", lambda _p: {}, raising=True
    )
    runs = {
        "r000": {"experiment": "l200", "timestamp": "20260101T000000Z"},
        "r001": {"experiment": "l200", "timestamp": "20260102T000000Z"},
    }
    mon = PhyShifterMonitoring.__new__(PhyShifterMonitoring)
    param_init = PhyShifterMonitoring.__mro__[-2]  # param.Parameterized
    param_init.__init__(
        mon,
        base_path=str(period_tree),
        phy_path=str(period_tree),
        periods={"p19": runs},
        run_dict=runs,
        period="p19",
        run="r001",
    )
    mon._update_menus()
    assert mon.param.shifter_metric.objects == ["pulser_stab", "baseln_stab"]
    assert mon.shifter_metric == "pulser_stab"
    mon.shifter_metric = "baseln_stab"
    mon.run = "r000"
    mon._update_menus()
    assert mon.param.shifter_metric.objects == ["baseln_stab"]
    assert mon.shifter_metric == "baseln_stab"
    assert mon.update_shifter_plot().x_range.factors[0] == "B2"
    mon.period = "p18"
    mon._update_menus()
    assert "No period contract" in mon.update_shifter_plot().title.text


def _trace(n=72, start="2026-07-01"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.Series(np.sin(np.arange(n) / 10), index=idx, name="V1")


def _cal_points(with_res=True):
    cal = pd.DataFrame(
        {
            "detector": ["V1", "V1"],
            "string": [1, 1],
            "position": [1, 1],
            "run_start": pd.to_datetime(["2026-06-24", "2026-07-01"]),
            "fep_diff": [0.2, -0.1],
            "cal_const_diff": [0.05, 0.02],
        }
    )
    if with_res:
        cal["res"] = [3.0, 2.6]
    return cal


def test_param_stability_fwhm_segments_and_thresholds():
    p = shifter_plots.param_stability(
        _trace(), None, None, pd.Timestamp("2026-07-01"), 2.6, "TrapemaxCtcCal",
        "p19", "V1", 1, 1,
    )  # fmt: skip
    labels = [item.label["value"] for item in p.legend[0].items]
    assert any("FWHM/2" in label for label in labels)
    annotations = [a.text for a in p.center if isinstance(a, Label)]
    assert "2.60" in annotations
    assert p.yaxis.axis_label == "Energy diff / keV"
    # baseline: fixed ±10 % thresholds from mtg-plot-settings
    p2 = shifter_plots.param_stability(
        _trace(), None, None, pd.Timestamp("2026-07-01"), None, "Baseline",
        "p19", "V1", 1, 1,
    )  # fmt: skip
    assert "Threshold" in [item.label["value"] for item in p2.legend[0].items]
    assert p2.yaxis.axis_label == "Baseline % variations"


def test_param_stability_without_res_notes_it():
    p = shifter_plots.param_stability(
        _trace(), None, None, pd.Timestamp("2026-07-01"), float("nan"),
        "TrapemaxCtcCal", "p19", "V1", 1, 1,
    )  # fmt: skip
    notes = [a.text for a in p.center if isinstance(a, Label)]
    assert any("FWHM" in t for t in notes)
    assert not any("FWHM/2" in i.label["value"] for i in p.legend[0].items)


def test_param_stability_pulser_trace_and_band():
    trace = _trace()
    std = trace * 0 + 0.1
    p = shifter_plots.param_stability(
        trace, std, trace * 0.5, pd.Timestamp("2026-07-01"), 2.0, "TrapemaxCtcCal",
        "p19", "V1", 1, 1,
    )  # fmt: skip
    labels = [item.label["value"] for item in p.legend[0].items]
    assert {"PULS01ANA", "GED corrected", "±1 sigma"} <= set(labels)


def test_gain_shift_cal_points_and_highlight():
    trace = _trace(n=24 * 14, start="2026-06-20")
    p = shifter_plots.gain_shift(
        trace, None, None, _cal_points(), "p19", "V1", 1, 1, corrected=False,
        highlight=(pd.Timestamp("2026-07-01"), trace.index.max()),
    )  # fmt: skip
    labels = [item.label["value"] for item in p.legend[0].items]
    assert {"GED uncorrected", "FEP gain", "cal. const. diff"} <= set(labels)
    assert any("FWHM/2" in label for label in labels)
    vlines = [s for s in p.center if isinstance(s, Span) and s.dimension == "height"]
    assert len(vlines) == 2  # one per cal run start
    assert any(isinstance(a, BoxAnnotation) for a in p.center)
    assert sorted(a.text for a in p.center if isinstance(a, Label)) == ["2.60", "3.00"]
    assert p.x_range.start < pd.Timestamp("2026-06-24")


def test_gain_shift_without_res_columns():
    trace = _trace(n=24 * 14, start="2026-06-20")
    p = shifter_plots.gain_shift(
        trace,
        None,
        None,
        _cal_points(with_res=False),
        "p19",
        "V1",
        1,
        1,
        corrected=False,
    )
    assert not any(isinstance(a, Label) for a in p.center)
    assert len(p.renderers) == 3  # trace + two marker sets


def _hourly(n=24, cols=("V1", "V2")):
    idx = pd.date_range("2026-07-01", periods=n, freq="1h")
    rng = np.random.default_rng(3)
    return pd.DataFrame(rng.random((n, len(cols))) * 0.3, index=idx, columns=list(cols))


def test_ft_per_string_has_percent_axis_and_shared_cycle():
    import itertools

    from legenddashboard.geds.phy import plot_style

    colors = itertools.cycle(plot_style.TAB20)
    next(colors)  # a previous string consumed one colour
    p = shifter_plots.ft_per_string(_hourly(), "p19", "r001", 1, 2.0, colors)
    assert len(p.yaxis) == 2
    assert p.yaxis[1].axis_label == "FT failure fraction (%)"
    assert p.renderers[0].glyph.line_color == plot_style.TAB20[1]
    assert p.renderers[0].glyph.mode == "center"
    assert p.legend[0].ncols == 2


def test_ft_per_string_without_total_forced_has_single_axis():
    import itertools

    from legenddashboard.geds.phy import plot_style

    p = shifter_plots.ft_per_string(
        _hourly(), "p19", "r001", 1, None, itertools.cycle(plot_style.TAB20)
    )
    assert len(p.yaxis) == 1


def test_ft_all_strings_and_survival():
    p = shifter_plots.ft_all_strings(_hourly(cols=("1", "4")), "p19", "r001")
    assert [i.label["value"] for i in p.legend[0].items] == ["String 1", "String 4"]
    p2 = shifter_plots.ft_survival(_hourly(cols=("survival_fraction",)), "p19")
    assert p2.renderers[0].glyph.line_color == "red"
    assert p2.yaxis.axis_label == "FT surviving events (%)"


def test_qc_rate_series_threshold_only_for_discharge_and_saturated():
    import itertools

    from legenddashboard.geds.phy import plot_style

    rates = _hourly()
    dets = [("V1", 1), ("V2", 2), ("V9", 3)]
    p = shifter_plots.qc_rate_series(
        rates, dets, {"V1": 1.234, "V2": 0.5}, "IsDischarge", "p19", "r001", 1,
        itertools.cycle(plot_style.TAB20),
    )  # fmt: skip
    labels = [i.label["value"] for i in p.legend[0].items]
    assert "V1 - pos 1 - 1.23 mHz" in labels
    assert "5 mHz upper threshold" in labels
    assert [s.location for s in _spans(p)] == [5]
    p2 = shifter_plots.qc_rate_series(
        rates, dets, {}, "IsValidBlSlope", "p19", "r001", 1,
        itertools.cycle(plot_style.TAB20),
    )  # fmt: skip
    assert _spans(p2) == []
    assert "V1 - pos 1 - nan mHz" in [i.label["value"] for i in p2.legend[0].items]


def test_qc_average_log_axis_keeps_empty_slots_and_dead_time_title():
    groups = {1: ["V1", "V2"], 4: ["B1"]}
    p = shifter_plots.qc_average(
        {"V1": 2.0, "B1": 0.0}, groups, "IsDischarge", 0.0173, "p19", "r001"
    )
    assert p.x_range.factors == ["V1", "V2", "B1"]
    assert p.renderers[0].data_source.data["det"] == ["V1"]  # zero dropped on log axis
    assert "tot dead time 0.017%" in p.title.text
    assert [s.location for s in _spans(p)] == [5]
    texts = [a.text for a in p.center if isinstance(a, Label)]
    assert sorted(texts) == ["String 1", "String 4"]


def test_classifier_grid_and_fraction_fallback():
    edges = np.linspace(-15, 15.4, 77)
    counts = {
        "All": {"V1": np.ones(76), "V2": np.ones(76)},
        "IsPulser": {"V1": np.ones(76)},
    }
    fracs = {("V1", "All"): 99.0, ("V1", "IsPulser"): 100.0}
    layout = shifter_plots.classifier_grid(
        edges, counts, [("V1", 1), ("V2", 2), ("V3", 3)], fracs,
        "IsValidBlSlopeClassifier", "p19", "r001", 1,
    )  # fmt: skip
    grid = layout[1].object  # bokeh GridPlot: children are (figure, row, col)
    figs = [child[0] for child in grid.children]
    assert len(figs) == 2  # V3 has no histogram -> skipped like the pipeline
    labels = [i.label["value"] for i in figs[0].legend[0].items]
    assert "All events - 99.0%" in labels
    assert "TP - 100.0%" in labels
    assert (figs[0].x_range.start, figs[0].x_range.end) == (-10, 10)

    frac = pd.DataFrame(
        {
            "classifier": ["IsValidBlSlopeClassifier"] * 4,
            "detector": ["V1", "V1", "V2", "V2"],
            "string": [1, 1, 1, 1],
            "event_type": ["All", "IsPulser", "All", "IsPulser"],
            "percent_in_range": [99.0, 100.0, 98.0, 97.0],
        }
    )
    p = shifter_plots.classifier_fraction_bars(
        frac, "IsValidBlSlopeClassifier", "p19", "r001", 1
    )
    assert p.x_range.factors == ["V1", "V2"]
    assert [i.label["value"] for i in p.legend[0].items] == ["All events", "TP"]


def test_event_rate_qc_series_and_mass_title():
    frame = _hourly(cols=("all_events", "failing_qc", "surviving_qc"))
    frame["on_mass_kg"] = 123.456
    p = shifter_plots.event_rate_qc(frame, "p19", "r001")
    labels = [i.label["value"] for i in p.legend[0].items]
    assert labels == ["All events", "Failing QC", "Surviving QC"]
    assert p.legend[0].title == "ON mass = 123.5 kg"
    assert len(p.renderers[0].data_source.data["x"]) == len(frame) + 1  # trailing edge
