"""Reader for legend-data-monitor's period contract files.

``<phy>/generated/plt/hit/phy/<period>/l200-<period>-{phy,cal}-monitoring.hdf``
are pandas HDFStores (root attr ``lmon_schema_version=2``) holding the
frames the pipeline's shifter figures are drawn from, e.g.::

    detector_summary/<title>/<run>      ged,string,pos,mean,std,min,max,fwhm,usability
    param_stability/<Param>/<run>       UTC hourly x detector (legacy-scaled)
    gain_shift/{corr,uncorr}/<run>      UTC hourly x detector, keV   (cumulative)
    cal_points/<run>                    detector,string,position,run_start,
                                        fep_diff,cal_const_diff[,res,res_quad]
    ft_summary/<kind>/<run>             hourly x detector | string | 1
    qc_rate_series/<flag>/<run>         UTC hourly x detector, mHz
    qc_average/<run>, qc_classifier_frac/<run>, dead_time/<run>, event_rate_qc/<run>

Plain pandas/h5py only (never imports ``legend_data_monitor``). Everything
is cached per file version ``(path, mtime_ns, size)``; a missing file or key
is never cached and never raises from the ``*_optional``/listing helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import pandas as pd

from legenddashboard.geds.phy import contract_reader
from legenddashboard.util import LRUDict

log = logging.getLogger(__name__)

_keys_cache = LRUDict(maxsize=32)


def period_file(phy_path, period: str, data_type: str = "phy", experiment="l200"):
    """Path of the period contract file (may not exist)."""
    return (
        Path(phy_path)
        / "generated/plt/hit/phy"
        / period
        / f"{experiment}-{period}-{data_type}-monitoring.hdf"
    )


def list_keys(path) -> tuple:
    """All frame keys of a period file, leading slash stripped; () if absent.

    Listing the keys of a large store is slow (~1 s) so the result is cached
    per file version; frames themselves are cheap to read.
    """
    path = Path(path)
    try:
        cache_key = (str(path), *contract_reader._stat_key(path))
    except OSError:
        return ()
    if cache_key not in _keys_cache:
        keys = []
        try:
            handle = h5py.File(path, "r")
        except OSError:
            # the producer may be rewriting the file: report nothing this
            # time, and do not cache it -- the next stat sees a new version
            log.debug("period file %s is not readable right now", path, exc_info=True)
            return ()
        with handle as f:
            # pandas stores a frame as a group carrying the pandas_type attr;
            # walking with h5py avoids opening a full HDFStore
            def visit(name, obj):
                if isinstance(obj, h5py.Group) and "pandas_type" in obj.attrs:
                    keys.append(name)

            f.visititems(visit)
        _keys_cache[cache_key] = tuple(sorted(keys))
    return _keys_cache[cache_key]


def has_key(path, key: str) -> bool:
    return key in list_keys(path)


def children(path, prefix: str) -> list:
    """Direct child names under ``prefix`` (e.g. runs of ``gain_shift/corr``)."""
    prefix = prefix.rstrip("/") + "/"
    out = []
    for key in list_keys(path):
        if key.startswith(prefix):
            child = key[len(prefix) :].split("/", 1)[0]
            if child not in out:
                out.append(child)
    return out


def runs_for(path, prefix: str) -> list:
    """Run ids available under ``prefix`` (``r000`` ...), sorted."""
    return sorted(c for c in children(path, prefix) if c.startswith("r"))


def flags_for(path, run: str) -> list:
    """QC flags with a ``qc_rate_series/<flag>/<run>`` frame."""
    return [
        flag
        for flag in children(path, "qc_rate_series")
        if has_key(path, f"qc_rate_series/{flag}/{run}")
    ]


def read(path, key: str) -> pd.DataFrame:
    """A frame (fresh copy, cached per file version)."""
    return contract_reader.read_frame(path, key)


def read_optional(path, key: str):
    """A frame, or None when the file or key does not exist."""
    if not has_key(path, key):
        return None
    return read(path, key)


def detector_map(phy_path, period: str, run: str, experiment="l200", subsystem="geds"):
    """The run contract's detector map of one subsystem.

    geds: name,rawid,string,position,processable,usability,mass_in_kg;
    spms: name,rawid,barrel,fiber,position,processable,usability.
    """
    manifest = contract_reader.find_manifest(phy_path, period, run, experiment)
    if manifest is None:
        return None
    run_dir = Path(phy_path) / "generated/plt/hit/phy" / period / run
    geds = contract_reader.file_from_manifest(manifest, run_dir, subsystem)
    if geds is None or not geds.exists():
        return None
    try:
        return contract_reader.read_detector_map(geds)
    except KeyError:
        return None
