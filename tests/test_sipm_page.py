"""SiPM page on a hand-written spms contract (plain h5py/pandas, no lmon)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest
from bokeh.models import Span

from legenddashboard.spms import sipm_monitoring as sm

SIPMS = ["S060", "S061", "S055", "S093"]  # IB top, IB bottom, OB top, OB bottom
DMAP = pd.DataFrame(
    {
        "name": SIPMS,
        "rawid": [1064000, 1064001, 1056002, 1056003],
        "barrel": ["IB", "IB", "OB", "OB"],
        "fiber": ["IB015016", "IB015016", "OB005006", "OB005006"],
        "position": ["top", "bottom", "top", "bottom"],
        "processable": [True] * 4,
        "usability": ["on"] * 4,
    }
)
N_BINS = 6


def _write_hist(f, key, n_det, values=None):
    g = f.create_group(key)
    a0 = g.create_group("ref_axes/axis_0")
    a0.attrs.update({"bins": N_BINS, "lower": 0.0, "upper": 360.0, "type": "regular"})
    a1 = g.create_group("ref_axes/axis_1")
    a1.create_dataset("categories", data=[s.encode() for s in SIPMS[:n_det]])
    shape = (N_BINS + 2, n_det + 1)
    g.create_dataset("storage/counts", data=np.full(shape, 5.0))
    vals = np.full(shape, 0.2) if values is None else values
    g.create_dataset("storage/values", data=vals)
    g.create_dataset("storage/variances", data=np.full(shape, 0.01))
    g.create_dataset("min", data=np.zeros((N_BINS, n_det)))
    g.create_dataset("max", data=np.ones((N_BINS, n_det)))
    g.attrs["unit"] = "a.u."
    g.attrs["label"] = "Pulses per event window"
    g.attrs["limits"] = json.dumps([None, None])
    g.attrs["event_type"] = key.split("/")[1].split("_")[0]


def _write_dist(f, key):
    g = f.create_group(key)
    a0 = g.create_group("ref_axes/axis_0")
    a0.attrs.update({"bins": 10, "lower": 0.0, "upper": 1.0, "type": "regular"})
    g.create_dataset("storage/values", data=np.arange(12.0))
    g.attrs["unit"] = "a.u."


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "lmon"
    run_dir = root / "generated/plt/hit/phy/p20/r001"
    run_dir.mkdir(parents=True)
    spms = run_dir / "l200-p20-r001-phy-spms-schema2.hdf"
    with h5py.File(spms, "w") as f:
        f.attrs["lmon_schema_version"] = 2
        for cad in ("1min", "10min", "60min"):
            _write_hist(f, f"hist/IsBsln_NPulses/{cad}", 4)
            _write_hist(f, f"hist/All_HasAnyNoise/{cad}", 4)
            _write_hist(f, f"hist/IsBsln_WfMode_var/{cad}", 4)
            _write_hist(f, f"hist/IsBsln_WfMode/{cad}", 4)
            _write_hist(f, f"hist/IsBsln_CurrFwhm/{cad}", 4)
        _write_dist(f, "hist/IsBsln_NPulses_dist")
    DMAP.to_hdf(spms, key="detector_map", format="fixed")
    keys = [
        f"hist/{k}/{c}"
        for k in (
            "IsBsln_NPulses",
            "All_HasAnyNoise",
            "IsBsln_WfMode_var",
            "IsBsln_WfMode",
            "IsBsln_CurrFwhm",
        )  # fmt: skip
        for c in ("1min", "10min", "60min")
    ] + ["hist/IsBsln_NPulses_dist"]
    manifest = {
        "schema_version": 2,
        "period": "p20",
        "run": "r001",
        "cadences": ["1min", "10min", "60min"],
        "files": {
            "l200-p20-r001-phy-geds-schema2.hdf": {
                "keys": ["hist/IsPulser_Trapemax/1min"]
            },
            spms.name: {"keys": keys, "cadences": ["1min", "10min", "60min"]},
        },
        "key_vocabulary": {
            "parameters": {
                "n_pulses": {"label": "Pulses per event window", "unit": "a.u."}
            }
        },
        "flagged_ranges": [],
    }
    (run_dir / "l200-p20-r001-manifest.json").write_text(json.dumps(manifest))
    (run_dir / "l200-p20-r001-qcp_summary.yaml").write_text(
        "S060:\n  phy: {spms_dark_rate: true, spms_noisy_frac: false}\n"
        "S061:\n  phy: {spms_dark_rate: true, spms_noisy_frac: true}\n"
    )
    period = root / "generated/plt/hit/phy/p20/l200-p20-phy-monitoring.hdf"
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC", name="datetime")
    with pd.HDFStore(period, "w") as store:
        store.put(
            "spms_noise/r001",
            pd.DataFrame(0.4, index=idx, columns=SIPMS),
            format="fixed",
        )
        store.put(
            "spms_calibration/r001",
            pd.DataFrame(
                {
                    "pe_a": [0.02] * 4,
                    "pe_m": [1.0] * 4,
                    "threshold_a": [3.0] * 4,
                    "source": ["lar/p17/r005/l200-par_hit-overwrite.yaml"] * 4,
                },
                index=pd.Index(SIPMS, name="name"),
            ),  # fmt: skip
            format="fixed",
        )
    return root


def _monitor(root):
    runs = {"r001": {"experiment": "l200", "timestamp": "20260101T000000Z"}}
    mon = sm.SiPMMonitoring.__new__(sm.SiPMMonitoring)
    sm.SiPMMonitoring.__mro__[-2].__init__(
        mon, base_path=str(root), phy_path=str(root), periods={"p20": runs},
        run_dict=runs, period="p20", run="r001",
    )  # fmt: skip
    mon._value_menu = mon._group_menu = None
    mon._update_menus()
    return mon


def test_group_labels_by_barrel_position_and_fiber():
    groups = sm.group_labels(DMAP, "Barrel x position")
    assert list(groups) == ["IB top", "IB bottom", "OB top", "OB bottom"]
    assert groups["IB top"] == ["S060"]
    fibers = sm.group_labels(DMAP, "Fiber")
    assert fibers["IB015016"] == ["S060", "S061"]  # top first
    assert sm.group_labels(None, "Fiber") == {}


def test_menus_follow_spms_file(tree):
    mon = _monitor(tree)
    assert mon.param.sipm_plots_types.objects == ["Forced trigger", "All events"]
    assert mon.param.sipm_plots.objects == [
        "Baseline (mode)",
        "Current noise FWHM",
        "Pulses per window (dark rate)",
    ]
    assert mon.param.sipm_group.objects == [
        "IB top",
        "IB bottom",
        "OB top",
        "OB bottom",
    ]
    mon.sipm_group_by = "Fiber"
    mon._update_menus()
    assert mon.param.sipm_group.objects == ["IB015016", "OB005006"]


def _spans(p):
    return sorted(
        s.location for s in p.center if isinstance(s, Span) and s.dimension == "width"
    )


def _n_areas(p):
    return sum(r.glyph.__class__.__name__ == "VArea" for r in p.renderers)


def _det_lines(p):
    return [r for r in p.renderers if r.glyph.__class__.__name__ == "Line" and r.name]


def test_explorer_bands_and_envelope(tree):
    mon = _monitor(tree)
    mon.sipm_plots = "Pulses per window (dark rate)"
    p = mon.update_sipm_plot()
    assert _spans(p) == [0.002, 1.0]  # spms_dark_rate band from mtg-plot-settings
    assert "IB top" in p.title.text
    assert _n_areas(p) == 2  # envelope + band for one SiPM
    mon.sipm_plots_types = "All events"
    mon._update_menus()  # the watcher is not registered in this bare harness
    mon.sipm_plots = "Noisy waveform fraction"
    p = mon.update_sipm_plot()
    assert _spans(p) == [0.05]
    assert _n_areas(p) == 1  # no envelope for the boolean rate
    mon.sipm_plots_types = "Forced trigger"
    mon._update_menus()
    mon.sipm_plots, mon.sipm_units = "Baseline (mode)", "Relative"
    p = mon.update_sipm_plot()
    assert _spans(p) == [-0.05, 0.05]


def test_fiber_view_dashes_bottom_and_histogram(tree):
    mon = _monitor(tree)
    mon.sipm_plots = "Pulses per window (dark rate)"
    mon.sipm_group_by = "Fiber"
    mon._update_menus()
    p = mon.update_sipm_plot()
    assert [r.glyph.line_dash for r in _det_lines(p)] == [[], [6]]
    mon.sipm_plot_style = "Histogram"
    p = mon.update_sipm_plot()
    assert p.renderers[0].glyph.__class__.__name__ == "Quad"


def test_summaries(tree):
    mon = _monitor(tree)
    mon.sipm_view = "Channel health"
    p = mon.update_sipm_plot()
    assert "failing: S060" in p.title.text
    assert list(p.y_range.factors) == ["spms_dark_rate", "spms_noisy_frac"]
    mon.sipm_view = "PE calibration"
    col = mon.update_sipm_plot()
    assert "p17/r005" in col[0].object
    assert "3 period(s) stale" in col[0].object
    assert list(col[1].value.columns) == [
        "name",
        "barrel",
        "fiber",
        "position",
        "pe_a",
        "pe_m",
        "threshold_a",
    ]
    mon.sipm_view = "Noise cross-check"
    p = mon.update_sipm_plot()
    assert "dotted" in p.title.text
    assert any(
        r.glyph.line_dash == [2, 4]
        for r in p.renderers
        if hasattr(r.glyph, "line_dash")
    )


def test_missing_spms_file_message(tree):
    mon = _monitor(tree)
    (
        tree / "generated/plt/hit/phy/p20/r001/l200-p20-r001-phy-spms-schema2.hdf"
    ).unlink()
    assert "No spms contract" in mon.update_sipm_plot().title.text


def test_calibration_staleness():
    assert sm.calibration_staleness("lar/p19/r005/x.yaml", "p22") == ("p19/r005", 3)
    assert sm.calibration_staleness("nothing", "p22") == (None, None)
