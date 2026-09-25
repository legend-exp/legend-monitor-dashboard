"""Process-wide cache for the calibration plot shelves.

The dataflow ``*-plt_hit`` / ``*-plt_dsp`` shelves hold pickled matplotlib
figures; unpickling one channel takes seconds while reading its bytes takes
milliseconds, so the unpickled entries are what is worth keeping. Entries are
keyed on the shelve's data-file stat so a regenerated file invalidates
naturally. Cached figures are shared by every session and matplotlib is not
thread-safe, so they are only ever rasterised under ``render_png``'s lock.
"""

from __future__ import annotations

import io
import pickle as pkl
import shelve
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

from legenddashboard.util import LRUDict

# ~40-60 MB per entry once unpickled; a dozen covers a few runs' worth of
# common dicts plus the channels being browsed.
_entry_cache = LRUDict(maxsize=12)
_keys_cache = LRUDict(maxsize=64)
_png_cache = LRUDict(maxsize=256)
_data_cache = LRUDict(maxsize=512)  # figure-free sub-dicts, ~200 kB each
_render_lock = threading.Lock()

RENDER_DPI = 144  # panel's Matplotlib pane default, kept for visual parity


def _stat_key(shelf_path) -> tuple:
    """Fingerprint of a dbm.dumb shelve: stat of its ``.dat`` file."""
    base = Path(shelf_path)
    dat = base.with_name(base.name + ".dat")
    st = (dat if dat.exists() else base).stat()
    return (str(base), st.st_mtime_ns, st.st_size)


def shelf_keys(shelf_path) -> list:
    """Sorted key names of a shelve (cached per file version)."""
    key = _stat_key(shelf_path)
    if key not in _keys_cache:
        with shelve.open(str(shelf_path), "r", protocol=pkl.HIGHEST_PROTOCOL) as sh:
            _keys_cache[key] = sorted(sh.keys())
    return list(_keys_cache[key])


def shelf_entry(shelf_path, entry: str):
    """One unpickled shelve entry, shared read-only across sessions."""
    key = (*_stat_key(shelf_path), entry)
    if key not in _entry_cache:
        with shelve.open(str(shelf_path), "r", protocol=pkl.HIGHEST_PROTOCOL) as sh:
            _entry_cache[key] = sh[entry]
    return _entry_cache[key]


class _NoFigure:
    """Stand-in for any matplotlib object; swallows its pickled state."""

    def __new__(cls, *_args, **_kwargs):
        return object.__new__(cls)

    def __init__(self, *_args, **_kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *_args, **_kwargs):
        return _NoFigure()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NoFigure()


class _NoFigureUnpickler(pkl.Unpickler):
    """Unpickler that skips matplotlib objects (the slow part of an entry)."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("matplotlib", "mpl_toolkits"):
            return _NoFigure
        return super().find_class(module, name)


def shelf_data(shelf_path, entry: str, keys: tuple):
    """
    Figure-free sub-dict of a shelve entry, cached per file version.

    Parameters
    ----------
    shelf_path : str or Path
        shelve base path
    entry : str
        top-level shelve key, e.g. a detector name
    keys : tuple of str
        nested keys to walk inside the entry

    Returns
    -------
    dict or None
        the sub-dict minus any ``fig`` item, or None if a key is missing
    """
    key = (*_stat_key(shelf_path), entry, tuple(keys))
    if key not in _data_cache:
        with shelve.open(str(shelf_path), "r", protocol=pkl.HIGHEST_PROTOCOL) as sh:
            try:
                raw = sh.dict[entry.encode()]
            except KeyError:
                raw = None
        data = None
        if raw is not None:
            data = _NoFigureUnpickler(io.BytesIO(raw)).load()
            for k in keys:
                data = data.get(k) if isinstance(data, dict) else None
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if not isinstance(v, _NoFigure)}
        _data_cache[key] = data
    return _data_cache[key]


def shelf_data_many(shelf_path, entries, keys: tuple, max_workers: int = 8) -> dict:
    """``shelf_data`` for several entries, read in parallel (IO-bound)."""
    entries = list(entries)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(lambda e: shelf_data(shelf_path, e, keys), entries)
    return {e: d for e, d in zip(entries, results, strict=True) if d is not None}


def render_png(cache_key: tuple, make_figure) -> bytes:
    """PNG bytes for a figure, rendered once per ``cache_key``.

    ``make_figure`` is only called on a miss; it may return a figure living
    in the shared entry cache, which is why rasterisation is serialised.
    """
    if cache_key not in _png_cache:
        # double-checked: figures may live in the shared entry cache, so both
        # building and rasterising must happen at most once per key
        with _render_lock:
            if cache_key not in _png_cache:
                fig = make_figure()
                buf = io.BytesIO()
                FigureCanvasAgg(fig)  # unpickled figures carry no canvas
                fig.canvas.print_figure(buf, format="png", dpi=RENDER_DPI)
                _png_cache[cache_key] = buf.getvalue()
    return _png_cache[cache_key]
