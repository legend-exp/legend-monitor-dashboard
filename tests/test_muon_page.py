"""Muon page on a hand-written pmts contract (plain h5py/pandas, no lmon)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import param
import pytest

from legenddashboard.muon import muon_monitoring as mm

PMTS = ["PMT101", "PMT201", "PMT301", "PMT302"]
DMAP = pd.DataFrame(
    {
        "name": PMTS,
        "rawid": [2001604, 2001610, 2001620, 2001621],
        "location": ["pillbox", "floor", "wall", "wall"],
        "processable": [True] * 4,
        "usability": ["on"] * 4,
    }
)
N_BINS = 6


def _write_hist(f, key, n_det=4):
    g = f.create_group(key)
    a0 = g.create_group("ref_axes/axis_0")
    a0.attrs.update({"bins": N_BINS, "lower": 0.0, "upper": 360.0, "type": "regular"})
    a1 = g.create_group("ref_axes/axis_1")
    a1.create_dataset("categories", data=[s.encode() for s in PMTS[:n_det]])
    shape = (N_BINS + 2, n_det + 1)
    g.create_dataset("storage/counts", data=np.full(shape, 5.0))
    g.create_dataset("storage/values", data=np.full(shape, 0.2))
    g.create_dataset("storage/variances", data=np.full(shape, 0.01))
    g.create_dataset("min", data=np.zeros((N_BINS, n_det)))
    g.create_dataset("max", data=np.ones((N_BINS, n_det)))
    g.attrs["unit"] = "Hz"
    g.attrs["label"] = "Event Rate"
    g.attrs["limits"] = json.dumps([None, None])


def _write_dist(f, key, n_bins=10):
    g = f.create_group(key)
    a0 = g.create_group("ref_axes/axis_0")
    a0.attrs.update({"bins": n_bins, "lower": 0.0, "upper": 40.0, "type": "regular"})
    g.create_dataset("storage/values", data=np.arange(float(n_bins + 2)))
    g.attrs["unit"] = "PMT"
    g.attrs["label"] = "Multiplicity"


def _write_dist2d(f, key, spp=25.0):
    g = f.create_group(key)
    n_bins = 500
    a0 = g.create_group("ref_axes/axis_0")
    a0.attrs.update({"bins": n_bins, "lower": 0.0, "upper": 100.0, "type": "regular"})
    a1 = g.create_group("ref_axes/axis_1")
    a1.create_dataset("categories", data=[s.encode() for s in PMTS])
    centres = np.linspace(0.1, 99.9, n_bins)
    spectrum = 200 * np.exp(-0.5 * ((centres - spp) / 4.0) ** 2) + 500 * np.exp(
        -centres / 1.5
    )
    values = np.zeros((n_bins + 2, len(PMTS) + 1))
    for i in range(len(PMTS)):
        values[1:-1, i] = spectrum
    values[1:-1, 1] = 0.0  # PMT201: empty spectrum -> no SPP found
    g.create_dataset("storage/values", data=values)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "lmon"
    run_dir = root / "generated/plt/hit/phy/p20/r001"
    run_dir.mkdir(parents=True)
    pmts = run_dir / "l200-p20-r001-phy-pmts-schema2.hdf"
    with h5py.File(pmts, "w") as f:
        f.attrs["lmon_schema_version"] = 2
        for cad in ("1min", "10min", "60min"):
            _write_hist(f, f"hist/All_EventRate/{cad}")
            _write_hist(f, f"hist/All_BlMean/{cad}")
        _write_dist(f, "hist/All_EventRate_dist")
        _write_dist(f, "hist/All_MuonMultiplicity_dist")  # dist-only parameter
        _write_dist2d(f, "hist/All_Pulseheight_dist2d")
    DMAP.to_hdf(pmts, key="detector_map", format="fixed")
    keys = (
        [f"hist/All_EventRate/{c}" for c in ("1min", "10min", "60min")]
        + [f"hist/All_BlMean/{c}" for c in ("1min", "10min", "60min")]
        + [
            "hist/All_EventRate_dist",
            "hist/All_MuonMultiplicity_dist",
            "hist/All_Pulseheight_dist2d",
        ]
    )
    manifest = {
        "schema_version": 2,
        "period": "p20",
        "run": "r001",
        "cadences": ["1min", "10min", "60min"],
        "files": {
            "l200-p20-r001-phy-geds-schema2.hdf": {"keys": []},
            pmts.name: {"keys": keys, "cadences": ["1min", "10min", "60min"]},
        },
        "key_vocabulary": {
            "parameters": {"event_rate": {"label": "Event Rate", "unit": "Hz"}}
        },
        "flagged_ranges": [],
    }
    (run_dir / "l200-p20-r001-manifest.json").write_text(json.dumps(manifest))
    period = root / "generated/plt/hit/phy/p20/l200-p20-phy-monitoring.hdf"
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC", name="datetime")
    with pd.HDFStore(period, "w") as store:
        store.put(
            "muon_veto/r001",
            pd.DataFrame(
                {
                    "muon_rate_hz": [0.02, 0.03, 0.025, 0.021],
                    "multiplicity_median": [12.0] * 4,
                    "light_sum_median": [300.0] * 4,
                    "ge_coincidence_frac": [0.9, 0.92, 0.91, 0.9],
                    "ge_coincidence_frac_offline": [0.95] * 4,
                },
                index=idx,
            ),
            format="fixed",
        )
    return root


def _monitor(root):
    runs = {"r001": {"experiment": "l200", "timestamp": "20260101T000000Z"}}
    mon = mm.MuonMonitoring.__new__(mm.MuonMonitoring)
    # bare param layer only: skip Monitoring.__init__'s tree discovery
    param.Parameterized.__init__(
        mon, base_path=str(root), phy_path=str(root), periods={"p20": runs},
        run_dict=runs, period="p20", run="r001",
    )  # fmt: skip
    mon._value_menu = None
    mon._update_menus()
    return mon


def test_menus_follow_contract_and_style(tree):
    mon = _monitor(tree)
    assert mon.param.muon_plots_types.objects == ["All events"]
    # Time style: only binned params
    assert mon.param.muon_plots.objects == ["Event rate", "Baseline mean"]
    mon.muon_plot_style = "Histogram"
    mon._update_menus()
    # Histogram style: dist keys, including the dist-only multiplicity
    assert mon.param.muon_plots.objects == ["Event rate", "Muon multiplicity"]


def test_explorer_time_and_histogram(tree):
    mon = _monitor(tree)
    mon.muon_group = "wall"
    p = mon.update_muon_plot()
    assert p.renderers
    assert "wall" in p.title.text
    mon.muon_plot_style = "Histogram"
    mon._update_menus()
    mon.muon_plots = "Muon multiplicity"
    p = mon.update_muon_plot()
    assert p.renderers[0].glyph.__class__.__name__ == "Quad"


def test_group_names_from_detector_map(tree):
    mon = _monitor(tree)
    assert mon._group_names("pillbox") == ["PMT101"]
    assert mon._group_names("wall") == ["PMT301", "PMT302"]


def test_spectra_reports_spp_position_without_grading(tree):
    mon = _monitor(tree)
    mon.muon_view = "PMT spectra"
    mon.muon_group = "pillbox"
    col = mon.update_muon_plot()
    assert "SPP located for 1/1" in col[0].object
    fig = col[1].object.children[0][0]
    import re

    assert re.search(r"SPP ≈ 2[45]\.\d LSB", fig.title.text)
    mon.muon_group = "floor"  # PMT201 has an empty spectrum
    col = mon.update_muon_plot()
    assert "SPP located for 0/1" in col[0].object
    fig = col[1].object.children[0][0]
    assert "no SPP found" in fig.title.text
    assert fig.title.text_color == "dimgray"


def test_veto_summary(tree):
    mon = _monitor(tree)
    mon.muon_view = "Veto summary"
    col = mon.update_muon_plot()
    p = col[0].object  # the Column wraps the figure in a Bokeh pane
    labels = [i.label["value"] for i in p.legend[0].items]
    assert "muon rate" in labels
    assert "ge_coincidence_frac" in labels
    assert len(p.yaxis) == 2  # coincidence fraction on the right axis
    # the dist row carries the dist-only multiplicity histogram
    assert len(col) == 2


def test_missing_pmts_file_message(tree):
    mon = _monitor(tree)
    (
        tree / "generated/plt/hit/phy/p20/r001/l200-p20-r001-phy-pmts-schema2.hdf"
    ).unlink()
    assert "No pmts contract" in mon.update_muon_plot().title.text
