"""Reader for the calibration plot-data LH5 files (``*-plt_{tier}.lh5``).

The dataflow writes the data behind the calibration plots next to the legacy
plot shelves: ``/<detector>/<section>/...`` plus ``/common/<detector>/...``.
It is plain HDF5 underneath, so it is read with h5py (no lgdo dependency):
groups become dicts, datasets become read-only arrays or scalars. lgdo
splits field names on ``.``, so the writer encodes ``2614.511`` as
``2614p511``; numeric keys are decoded back here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

from legenddashboard.geds.cal.shelf_cache import _stat_key
from legenddashboard.util import LRUDict

_group_cache = LRUDict(maxsize=512)
_keys_cache = LRUDict(maxsize=64)
_NUMERIC_KEY = re.compile(r"^-?\d+p\d+$")


def plt_data_path(shelf_path) -> Path:
    """The plot-data LH5 file that sits next to a plot shelf."""
    return Path(f"{shelf_path}.lh5")


def encode_key(key) -> str:
    """Field name as written by the dataflow (``2614.511`` -> ``2614p511``)."""
    return str(key).replace("/", "_").replace(".", "p")


def decode_key(key: str) -> str:
    """Inverse of :func:`encode_key` for numeric keys."""
    return key.replace("p", ".") if _NUMERIC_KEY.match(key) else key


def _to_python(obj):
    if isinstance(obj, h5py.Group):
        return {decode_key(k): _to_python(v) for k, v in obj.items()}
    value = obj[()]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        value.setflags(write=False)  # shared across sessions
        return value
    return value.item() if isinstance(value, np.generic) else value


def data_keys(path) -> list:
    """Top-level keys (detectors and ``common``) of a plot-data file."""
    key = _stat_key(path)
    if key not in _keys_cache:
        with h5py.File(path, "r") as f:
            _keys_cache[key] = sorted(f.keys())
    return list(_keys_cache[key])


def read_group(path, *keys):
    """
    One group or dataset of a plot-data file, cached per file version.

    Parameters
    ----------
    path : str or Path
        plot-data LH5 file
    *keys
        path components, e.g. ``"V00048A", "ecal", "cuspEmax_ctc_cal"``;
        numeric components are encoded as the writer does

    Returns
    -------
    dict, ndarray, scalar or None
        the decoded content, or None if the file or key does not exist
    """
    path = Path(path)
    if not path.exists():
        return None
    h5_path = "/".join(encode_key(k) for k in keys)
    key = (*_stat_key(path), h5_path)
    if key not in _group_cache:
        with h5py.File(path, "r") as f:
            _group_cache[key] = _to_python(f[h5_path]) if h5_path in f else None
    return _group_cache[key]


class CommonData(Mapping):
    """Lazy per-detector view of ``/common`` (reads only the detectors used)."""

    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, det):
        data = read_group(self.path, "common", det)
        if data is None:
            raise KeyError(det)
        return data

    def __iter__(self):
        return iter(read_group(self.path, "common") or {})

    def __len__(self):
        return len(read_group(self.path, "common") or {})
