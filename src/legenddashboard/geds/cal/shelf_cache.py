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
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

from legenddashboard.util import LRUDict

# ~40-60 MB per entry once unpickled; a dozen covers a few runs' worth of
# common dicts plus the channels being browsed.
_entry_cache = LRUDict(maxsize=12)
_keys_cache = LRUDict(maxsize=64)
_png_cache = LRUDict(maxsize=256)
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
