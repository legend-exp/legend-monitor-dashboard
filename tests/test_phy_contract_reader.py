"""Tests for the dashboard's contract-v2 reader.

Fixtures are generated at test time by the vendored producer
(legend-data-monitor) — imported ONLY inside test fixtures, never by
dashboard code — plus one hand-written plain-h5py layout pin so the reader
contract survives even without the vendored tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from legenddashboard.geds.phy import contract_reader

REPO = Path(__file__).resolve().parents[1]
LMON_SRC = REPO / "legend-data-monitor" / "src"

DETS = ["V02160A", "V02160B", "P00574A"]


@pytest.fixture
def produced(tmp_path):
    """Write a v2 file + manifest with the vendored producer."""
    sys.path.insert(0, str(LMON_SRC))
    try:
        pytest.importorskip("legend_data_monitor.contract.writer")
        from legend_data_monitor.contract import writer
        from legend_data_monitor.processing import binning

        rng = np.random.default_rng(7)
        t0 = 1_766_257_200.0
        t = rng.uniform(t0, t0 + 2 * 3600, 4000)
        d = rng.choice(DETS, 4000)
        v = rng.normal(100.0, 5.0, 4000)
        binned = binning.fill_time_series(t, d, v, DETS, t0, t0 + 2 * 3600)

        run_dir = tmp_path / "generated/plt/hit/phy/p19/r001"
        run_dir.mkdir(parents=True)
        hdf = run_dir / "l200-p19-r001-phy-geds-schema2.hdf"
        keys = writer.write_binned_series(
            str(hdf),
            "IsPulser",
            "Trapemax",
            binned,
            attrs={"unit": "ADC", "label": "trapEmax", "limits": [None, None]},
        )
        keys.append(
            writer.write_distribution(
                str(hdf), "IsPulser", "Trapemax", binning.fill_distribution(v)
            )
        )
        means = pd.DataFrame([v.mean() * np.ones(len(DETS))], columns=DETS)
        keys.append(writer.write_frame(str(hdf), "IsPulser_Trapemax_mean", means))
        writer.write_manifest(
            str(run_dir),
            "p19",
            "r001",
            {hdf.name: {"keys": sorted(keys), "cadences": ["1min", "10min", "60min"]}},
            package_version="test",
        )
        return tmp_path, run_dir, hdf, binned
    finally:
        sys.path.remove(str(LMON_SRC))


def test_roundtrip_against_producer(produced):
    root, run_dir, hdf, binned = produced
    manifest = contract_reader.find_manifest(str(root), "p19", "r001")
    assert manifest is not None
    assert contract_reader.geds_file_from_manifest(manifest, run_dir) == hdf
    assert "IsPulser_Trapemax" in contract_reader.available_keys(manifest)
    assert "IsPulser_Trapemax_mean" in contract_reader.available_keys(manifest)

    for cadence in ("1min", "10min", "60min"):
        series = contract_reader.read_binned(hdf, "IsPulser", "Trapemax", cadence)
        assert series.detectors == tuple(DETS)
        expected = (
            binned
            if cadence == "1min"
            else binned.rebin({"10min": 10, "60min": 60}[cadence])
        )
        for stat in ("mean", "count", "min", "max"):
            got = series.to_frame(stat)
            want = expected.to_frame(stat)
            assert got.index.tz is not None
            np.testing.assert_allclose(
                # producer stores values/variances as float32
                got.to_numpy(),
                want.to_numpy(),
                equal_nan=True,
                rtol=1e-6,
            )
        # std matches sqrt of producer variance where count > 1
        got_std = series.to_frame("std").to_numpy()
        want_var = expected.to_frame("variance").to_numpy()
        mask = ~np.isnan(want_var)
        np.testing.assert_allclose(got_std[mask] ** 2, want_var[mask], rtol=1e-5)

    edges, counts, attrs = contract_reader.read_dist(hdf, "IsPulser", "Trapemax")
    assert len(edges) == len(counts) + 1
    assert 0 < counts.sum() <= 4000  # producer may clip the range
    assert attrs.get("schema") == 2

    means = contract_reader.read_mean_frame(hdf, "IsPulser", "Trapemax")
    assert list(means.columns) == DETS


def test_cache_immutability_firewall(produced):
    _, _, hdf, _ = produced
    frame = contract_reader.read_binned(hdf, "IsPulser", "Trapemax", "1min").to_frame(
        "mean"
    )
    original = frame.copy()
    # simulate phy_plot_vsTime's historical in-place mutations
    frame.index = frame.index + pd.Timedelta(hours=2)
    frame.iloc[:] = -1.0
    again = contract_reader.read_binned(hdf, "IsPulser", "Trapemax", "1min").to_frame(
        "mean"
    )
    pd.testing.assert_frame_equal(again, original)


def test_attrs_limits_json_decoded(produced):
    _, _, hdf, _ = produced
    series = contract_reader.read_binned(hdf, "IsPulser", "Trapemax", "1min")
    assert series.attrs["limits"] == [None, None]
    assert series.attrs["unit"] == "ADC"


def test_snap_cadence():
    cadences = ["1min", "10min", "60min"]
    assert contract_reader.snap_cadence(0, cadences) == "1min"
    assert contract_reader.snap_cadence(5, cadences) == "1min"
    assert contract_reader.snap_cadence(6, cadences) == "10min"
    assert contract_reader.snap_cadence(30, cadences) == "10min"
    assert contract_reader.snap_cadence(36, cadences) == "60min"
    assert contract_reader.snap_cadence(60, cadences) == "60min"
    assert contract_reader.snap_cadence(60, ["1min", "10min"]) == "10min"


def test_find_manifest_missing_and_wrong_schema(tmp_path):
    assert contract_reader.find_manifest(str(tmp_path), "p19", "r001") is None
    run_dir = tmp_path / "generated/plt/hit/phy/p19/r001"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "l200-p19-r001-manifest.json"
    manifest.write_text('{"schema_version": 3}')
    assert contract_reader.find_manifest(str(tmp_path), "p19", "r001") is None
    manifest.write_text('{"schema_version": 2, "files": {}}')
    assert contract_reader.find_manifest(str(tmp_path), "p19", "r001") is not None


def test_flagged_ranges_both_formats():
    manifest = {
        "flagged_ranges": [
            {"from": "20260701T000000Z", "to": "20260702T000000Z", "reason": "x"},
            {"from": "2026-07-03T00:00:00+00:00", "to": "2026-07-04T00:00:00+00:00"},
        ]
    }
    ranges = contract_reader.flagged_ranges(manifest)
    assert len(ranges) == 2
    assert ranges[0][0] == pd.Timestamp("2026-07-01", tz="UTC")
    assert ranges[1][1] == pd.Timestamp("2026-07-04", tz="UTC")
    assert ranges[1][2] == "flagged"


def test_label_unit_fallbacks():
    manifest = {
        "key_vocabulary": {
            "parameters": {"trapEmax_ctc_cal": {"label": "Cal. gain", "unit": "keV"}}
        }
    }
    label, unit = contract_reader.label_and_unit(
        manifest, {}, "TrapemaxCtcCal", relative=False
    )
    assert (label, unit) == ("Cal. gain", "keV")
    label, unit = contract_reader.label_and_unit(
        manifest, {"label": "L", "unit": "U"}, "TrapemaxCtcCal", relative=True
    )
    assert (label, unit) == ("L", "%")


def test_plain_h5py_layout_pin(tmp_path):
    """Hand-written file in the pinned layout — survives without the vendored tree."""
    path = tmp_path / "pin.hdf"
    n_bins, dets = 4, [b"A", b"B"]
    with h5py.File(path, "w") as f:
        f.attrs["lmon_schema_version"] = 2
        g = f.create_group("hist/IsPulser_X/1min")
        a0 = g.create_group("ref_axes/axis_0")
        a0.attrs.update(
            {"bins": n_bins, "lower": 0.0, "upper": 240.0, "type": "regular"}
        )
        g.create_group("ref_axes/axis_1").create_dataset("categories", data=dets)
        st = g.create_group("storage")
        st.create_dataset("counts", data=np.ones((n_bins + 2, len(dets) + 1)))
        st.create_dataset("values", data=np.full((n_bins + 2, len(dets) + 1), 5.0))
        st.create_dataset("variances", data=np.zeros((n_bins + 2, len(dets) + 1)))
        g.create_dataset("min", data=np.full((n_bins, len(dets)), 4.0))
        g.create_dataset("max", data=np.full((n_bins, len(dets)), 6.0))
        g.attrs["unit"] = "u"
    series = contract_reader.read_binned(path, "IsPulser", "X", "1min")
    assert series.detectors == ("A", "B")
    frame = series.to_frame("mean")
    assert frame.shape == (n_bins, 2)
    assert (frame.to_numpy() == 5.0).all()
    assert series.to_frame("max").iloc[0, 0] == 6.0


def test_no_lmon_import(tmp_path):
    script = (
        "import sys\n"
        "from legenddashboard.geds.phy import contract_reader, period_reader\n"
        "from legenddashboard.geds.phy import plot_style  # noqa: F401\n"
        f"assert contract_reader.find_manifest(r'{tmp_path}', 'p', 'r') is None\n"
        f"assert period_reader.list_keys(r'{tmp_path}/none.hdf') == ()\n"
        "assert not any(m.startswith('legend_data_monitor') for m in sys.modules)\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
