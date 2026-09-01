"""Period-contract reader and vocabulary helpers (plain pandas/h5py fixtures)."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from legenddashboard.geds.phy import contract_reader, period_reader

DETS = ["V1", "V2", "P3"]


@pytest.fixture
def period(tmp_path):
    root = tmp_path / "lmon"
    pdir = root / "generated/plt/hit/phy/p19"
    pdir.mkdir(parents=True)
    path = pdir / "l200-p19-phy-monitoring.hdf"
    summary = pd.DataFrame(
        {
            "ged": DETS,
            "string": [1, 1, 2],
            "pos": [1, 2, 1],
            "mean": [0.1, -0.2, 0.0],
            "std": [0.5, 0.4, 0.3],
            "min": [-1.0, -1.5, -0.6],
            "max": [1.2, 0.9, 0.7],
            "fwhm": [3.0, np.nan, 2.8],
            "usability": ["on", "ac", "off"],
        }
    )
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    series = pd.DataFrame(np.arange(12.0).reshape(4, 3), index=idx, columns=DETS)
    with pd.HDFStore(path, "w") as store:
        store.put("detector_summary/baseln_stab/r000", summary, format="fixed")
        store.put("detector_summary/baseln_stab/r001", summary, format="fixed")
        store.put("param_stability/BlStd/r001", series, format="fixed")
        store.put("qc_rate_series/IsDischarge/r001", series, format="fixed")
        store.put("qc_rate_series/IsSaturated/r000", series, format="fixed")
    with h5py.File(path, "a") as f:
        f.attrs["lmon_schema_version"] = 2
    return root, path


def test_list_keys_children_runs_flags(period):
    _, path = period
    keys = period_reader.list_keys(path)
    assert "detector_summary/baseln_stab/r000" in keys
    assert period_reader.children(path, "detector_summary") == ["baseln_stab"]
    assert period_reader.runs_for(path, "detector_summary/baseln_stab") == [
        "r000",
        "r001",
    ]
    assert period_reader.flags_for(path, "r001") == ["IsDischarge"]
    assert period_reader.has_key(path, "param_stability/BlStd/r001")
    assert not period_reader.has_key(path, "param_stability/BlStd/r000")


def test_missing_file_and_key_never_raise(tmp_path, period):
    _, path = period
    assert period_reader.list_keys(tmp_path / "nope.hdf") == ()
    assert period_reader.read_optional(path, "gain_shift/corr/r001") is None
    assert period_reader.read_optional(tmp_path / "nope.hdf", "x") is None
    assert period_reader.detector_map(tmp_path, "p19", "r001") is None


def test_read_returns_copies_and_keeps_tz(period):
    _, path = period
    frame = period_reader.read(path, "param_stability/BlStd/r001")
    assert str(frame.index.tz) == "UTC"
    frame.iloc[:] = -1
    again = period_reader.read(path, "param_stability/BlStd/r001")
    assert (again.to_numpy() >= 0).all()


def test_key_list_invalidates_on_rewrite(period):
    _, path = period
    assert "gain_shift/corr/r001" not in period_reader.list_keys(path)
    with pd.HDFStore(path, "a") as store:
        store.put(
            "gain_shift/corr/r001", pd.DataFrame({"a": [1.0, 2.0]}), format="fixed"
        )
    assert "gain_shift/corr/r001" in period_reader.list_keys(path)


MANIFEST = {
    "key_vocabulary": {
        "parameters": {
            "trapEmax_ctc_cal": {"label": "trapEmax_ctc_cal", "unit": "keV"},
            "AoE_Custom": {"label": "Custom A/E", "unit": "a.u."},
            "baseline": {"label": "FPGA baseline", "unit": "ADC"},
            "bl_std": {"label": "Noise", "unit": "ADC"},
        }
    }
}


@pytest.mark.parametrize(
    ("name", "label", "unit"),
    [
        ("TrapemaxCtcCal", "trapEmax_ctc_cal", "keV"),
        ("TrapemaxCtcCal_var", "trapEmax_ctc_cal", "keV"),
        ("AoeCustom", "Custom A/E", "a.u."),
        ("Baseline_pulser01anaRatio_var", "FPGA baseline / pulser01ana", "a.u."),
        ("BlStd_pulser01anaDiff", "Noise - pulser01ana", "ADC"),
        ("Unknown_mean", "Unknown", ""),
    ],
)
def test_vocab_entry_inverts_producer_naming(name, label, unit):
    assert contract_reader.vocab_entry(MANIFEST, name) == (label, unit)


def test_limits():
    assert contract_reader.limits({"limits": [None, 50]}) == (None, 50.0)
    assert contract_reader.limits({"limits": [-0.025, 0.025]}) == (-0.025, 0.025)
    assert contract_reader.limits({}) == (None, None)
    assert contract_reader.limits({"limits": "bad"}) == (None, None)


TWO_FILE_MANIFEST = {
    "schema_version": 2,
    "files": {
        "l200-p19-r001-phy-geds-schema2.hdf": {
            "keys": ["hist/IsPulser_Trapemax/1min", "IsPulser_Trapemax_mean"]
        },
        "l200-p19-r001-phy-spms-schema2.hdf": {
            "keys": ["hist/IsBsln_NPulses/1min", "hist/All_HasAnyNoise_dist"]
        },
    },
}


def test_manifest_helpers_by_subsystem(tmp_path):
    assert contract_reader.available_keys(TWO_FILE_MANIFEST) == {
        "IsPulser_Trapemax",
        "IsPulser_Trapemax_mean",
        "IsBsln_NPulses",
        "All_HasAnyNoise_dist",
    }
    assert contract_reader.available_keys(TWO_FILE_MANIFEST, "geds") == {
        "IsPulser_Trapemax",
        "IsPulser_Trapemax_mean",
    }
    assert contract_reader.available_keys(TWO_FILE_MANIFEST, "spms") == {
        "IsBsln_NPulses",
        "All_HasAnyNoise_dist",
    }
    geds = contract_reader.file_from_manifest(TWO_FILE_MANIFEST, tmp_path, "geds")
    spms = contract_reader.file_from_manifest(TWO_FILE_MANIFEST, tmp_path, "spms")
    assert geds.name.endswith("-geds-schema2.hdf")
    assert spms.name.endswith("-spms-schema2.hdf")
    assert contract_reader.geds_file_from_manifest(TWO_FILE_MANIFEST, tmp_path) == geds
    assert (
        contract_reader.file_from_manifest(TWO_FILE_MANIFEST, tmp_path, "muon") is None
    )
