"""Reader for legend-data-monitor contract-v2 monitoring files.

File-contract only: this module reads the manifest + UHI-HDF5 binned
histograms with plain h5py/json/numpy/pandas and must NEVER import
``legend_data_monitor`` (enforced by a test). Layout pinned by lmon's
``tests/test_contract_v2.py::test_v2_readable_with_plain_h5py``:

    l200-<p>-<r>-manifest.json                    schema_version, files, cadences,
                                                  key_vocabulary, flagged_ranges
    <geds file>.hdf::hist/{flag}_{param}/{cad}    group with storage/{counts,values,
                                                  variances} (incl. flow bins),
                                                  min/max sidecars (data region),
                                                  ref_axes/axis_0 attrs bins/lower/upper,
                                                  ref_axes/axis_1/categories,
                                                  attrs unit/label/limits/event_type
    <geds file>.hdf::hist/{flag}_{param}_dist     1-D distribution (storage/values)
    <geds file>.hdf::{flag}_{param}_mean          pandas frame of run means
    <geds file>.hdf::detector_map                 pandas frame (name,rawid,string,...)

All read results are cached keyed on ``(path, st_mtime_ns, ...)`` so a
rebuilt file invalidates naturally; every accessor hands out fresh copies so
plot code can never corrupt the cache.
"""

import dataclasses
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from legenddashboard.util import LRUDict

_manifest_cache = LRUDict(maxsize=64)
_series_cache = LRUDict(maxsize=64)
_frame_cache = LRUDict(maxsize=128)

# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def find_manifest(
    phy_path: str, period: str, run: str, experiment: str = "l200"
) -> dict | None:
    """Load the run manifest, or None (→ v1 fallback) if absent/incompatible.

    Absence is never cached (a later pipeline run may create the file); parsed
    content is cached per (path, mtime).
    """
    path = (
        Path(phy_path)
        / "generated/plt/hit/phy"
        / period
        / run
        / f"{experiment}-{period}-{run}-manifest.json"
    )
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    key = (str(path), mtime)
    if key not in _manifest_cache:
        with open(path) as f:
            _manifest_cache[key] = json.load(f)
    manifest = _manifest_cache[key]
    if manifest.get("schema_version") != 2:
        return None
    return manifest


def geds_file_from_manifest(manifest: dict, run_dir: Path) -> Path | None:
    """Path of the geds data file named by the manifest (never hardcode names)."""
    for name in manifest.get("files", {}):
        if "-geds" in name:
            return Path(run_dir) / name
    return None


def available_keys(manifest: dict) -> set:
    """All ``{flag}_{param}`` bodies present (hist prefix / cadence stripped)."""
    bodies = set()
    for entry in manifest.get("files", {}).values():
        for key in entry.get("keys", []):
            body = key.removeprefix("hist/")
            head, _, tail = body.rpartition("/")
            bodies.add(head or tail)
    return bodies


def flagged_ranges(manifest: dict) -> list:
    """[(from_ts, to_ts, reason)] parsed from the manifest (UTC Timestamps)."""
    out = []
    for entry in manifest.get("flagged_ranges", []):
        try:
            lo = _parse_ts(entry.get("from"))
            hi = _parse_ts(entry.get("to"))
        except (ValueError, TypeError):
            continue
        out.append((lo, hi, entry.get("reason", "flagged")))
    return out


def _parse_ts(value) -> pd.Timestamp:
    try:
        return pd.to_datetime(value, format="%Y%m%dT%H%M%SZ", utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(value, utc=True)


def snap_cadence(minutes: int, cadences: list) -> str:
    """Map the resample slider (minutes) to the nearest available cadence.

    0 means "finest"; ties resolve to the finer cadence.
    """
    def _minutes(cadence: str) -> int:
        return int(cadence.removesuffix("min"))

    ordered = sorted(cadences, key=_minutes)
    if minutes <= 0:
        return ordered[0]
    return min(ordered, key=lambda c: (abs(_minutes(c) - minutes), _minutes(c)))


def label_and_unit(
    manifest: dict, attrs: dict, param_name: str, relative: bool
) -> tuple:
    """(label, unit) with hist attrs primary and manifest vocabulary fallback."""
    label = attrs.get("label")
    unit = attrs.get("unit")
    if label is None or unit is None:
        vocab = manifest.get("key_vocabulary", {}).get("parameters", {})
        entry = vocab.get(_uncamel(param_name), {})
        label = label or entry.get("label") or param_name
        unit = unit or entry.get("unit") or ""
    if relative:
        unit = "%"
    return label, unit


def _uncamel(name: str) -> str:
    """TrapemaxCtcCal -> trapemax_ctc_cal (inverse of the producer's _camel)."""
    out = []
    for ch in name:
        if ch.isupper() and out:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# binned series
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BinnedSeries:
    """(time × detector) binned stats read from one hist group."""

    time_edges: np.ndarray  # unix seconds, length n_bins + 1
    detectors: tuple
    counts: np.ndarray  # (n_bins, n_det)
    values: np.ndarray  # per-bin mean
    variances: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    attrs: dict

    def to_frame(self, stat: str = "mean") -> pd.DataFrame:
        """One statistic as a fresh frame (UTC DatetimeIndex × detector cols).

        Always returns newly allocated data — consumers may mutate freely.
        """
        if stat == "mean":
            data = np.where(self.counts > 0, self.values, np.nan)
        elif stat == "count":
            data = self.counts.copy()
        elif stat == "variance":
            data = np.where(self.counts > 1, self.variances, np.nan)
        elif stat == "std":
            data = np.sqrt(np.where(self.counts > 1, self.variances, np.nan))
        elif stat == "min":
            data = self.mins.copy()
        elif stat == "max":
            data = self.maxs.copy()
        else:
            raise ValueError(f"unknown stat {stat!r}")
        index = pd.to_datetime(self.time_edges[:-1].copy(), unit="s", utc=True)
        index.name = "datetime"
        return pd.DataFrame(
            np.array(data, copy=True), index=index, columns=list(self.detectors)
        )


def read_binned(hdf_path, flag: str, param: str, cadence: str) -> BinnedSeries:
    """Read one (flag, param, cadence) hist group; cached per file mtime."""
    hdf_path = str(hdf_path)
    mtime = os.stat(hdf_path).st_mtime_ns
    hist_key = f"hist/{flag}_{param}/{cadence}"
    cache_key = (hdf_path, mtime, hist_key)
    if cache_key not in _series_cache:
        _series_cache[cache_key] = _load_binned(hdf_path, hist_key)
    return _series_cache[cache_key]


def _load_binned(hdf_path: str, hist_key: str) -> BinnedSeries:
    with h5py.File(hdf_path, "r") as f:
        group = f[hist_key]
        ax0 = group["ref_axes/axis_0"].attrs
        n_bins = int(ax0["bins"])
        time_edges = np.linspace(float(ax0["lower"]), float(ax0["upper"]), n_bins + 1)
        detectors = tuple(
            name.decode() if isinstance(name, bytes) else str(name)
            for name in group["ref_axes/axis_1/categories"][...]
        )
        # storage arrays carry flow bins: [1:-1] on the regular time axis,
        # [:-1] on the category axis (single overflow slot)
        counts = group["storage/counts"][1:-1, : len(detectors)]
        values = group["storage/values"][1:-1, : len(detectors)]
        variances = group["storage/variances"][1:-1, : len(detectors)]
        shape = (n_bins, len(detectors))
        mins = group["min"][...] if "min" in group else np.full(shape, np.nan)
        maxs = group["max"][...] if "max" in group else np.full(shape, np.nan)
        attrs = _decode_attrs(dict(group.attrs))

    for arr in (time_edges, counts, values, variances, mins, maxs):
        arr.setflags(write=False)
    return BinnedSeries(
        time_edges, detectors, counts, values, variances, mins, maxs, attrs
    )


def read_dist(hdf_path, flag: str, param: str) -> tuple:
    """(edges, counts, attrs) of the 1-D distribution histogram."""
    hdf_path = str(hdf_path)
    mtime = os.stat(hdf_path).st_mtime_ns
    key = f"hist/{flag}_{param}_dist"
    cache_key = (hdf_path, mtime, key)
    if cache_key not in _series_cache:
        with h5py.File(hdf_path, "r") as f:
            group = f[key]
            ax0 = group["ref_axes/axis_0"].attrs
            n_bins = int(ax0["bins"])
            edges = np.linspace(float(ax0["lower"]), float(ax0["upper"]), n_bins + 1)
            counts = group["storage/values"][1:-1]
            attrs = _decode_attrs(dict(group.attrs))
        edges.setflags(write=False)
        counts.setflags(write=False)
        _series_cache[cache_key] = (edges, counts, attrs)
    return _series_cache[cache_key]


def _decode_attrs(attrs: dict) -> dict:
    out = {}
    for name, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str) and value[:1] in "[{":
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        out[name] = value
    return out


# ---------------------------------------------------------------------------
# pandas frames
# ---------------------------------------------------------------------------


def read_frame(hdf_path, key: str) -> pd.DataFrame:
    """Cached pd.read_hdf; always returns a copy."""
    hdf_path = str(hdf_path)
    mtime = os.stat(hdf_path).st_mtime_ns
    cache_key = (hdf_path, mtime, key)
    if cache_key not in _frame_cache:
        _frame_cache[cache_key] = pd.read_hdf(hdf_path, key=key)
    return _frame_cache[cache_key].copy()


def read_mean_frame(hdf_path, flag: str, param: str) -> pd.DataFrame:
    return read_frame(hdf_path, f"{flag}_{param}_mean")


def read_detector_map(hdf_path) -> pd.DataFrame:
    return read_frame(hdf_path, "detector_map")
