"""Reloadable read layer over the editable legend-metadata clone.

Unlike ``util.sort_dets``/``util.sorter`` (whose process-wide caches are never
invalidated -- correct for the immutable production metadata), the editor
needs its reads to reflect staged edits immediately, so every consumer goes
through one shared :class:`MetaDB` per clone path and calls :meth:`reload`
after each write.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from dbetto import Props, TextDB
from legendmeta import LegendMetadata

log = logging.getLogger(__name__)

#: Groupings ("partitions") files editable through the dashboard, relative to
#: the ``datasets/`` directory of the clone.
GROUPING_FILES = {
    "cal": "cal_groupings.yaml",
    "phy": "phy_groupings.yaml",
    "escale": "groupings/escale_cal_groupings.yaml",
    "psd": "groupings/psd_cal_groupings.yaml",
}


def tstamp_to_unix(tstamp: str) -> float:
    """``"20230311T235840Z"`` -> unix seconds (UTC)."""
    return datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()


# Channelmaps live in the shared hardware submodules, which the editor never
# touches -- these caches survive reload() AND are shared between every MetaDB
# of the same clone (each workspace MetaDB would otherwise re-pay the multi-
# second per-column channelmap scan of the matrix views).
_hw_caches: dict[str, tuple[dict, dict]] = {}
_hw_caches_lock = threading.Lock()


class MetaDB:
    """Read access to one datasets tree of the editable clone.

    ``datasets_path`` selects the tree the statuses/runinfo/groupings reads
    come from -- a per-user workspace worktree, or the clone's own
    ``datasets/`` (read-only viewing). Channelmaps always come from the shared
    clone. ``reload()`` after every staged edit; ``version`` increments on
    each reload so views can key their figure caches on it and invalidate
    naturally.
    """

    def __init__(self, meta_path: str | Path, datasets_path: str | Path | None = None):
        self.meta_path = Path(meta_path)
        self.datasets_path = (
            Path(datasets_path) if datasets_path else self.meta_path / "datasets"
        )
        self._lock = threading.Lock()
        self.version = 0
        with _hw_caches_lock:
            self._chmap_cache, self._geds_cache = _hw_caches.setdefault(
                str(self.meta_path.resolve()), ({}, {})
            )
        self._load()

    def _load(self):
        # LegendMetadata for channelmaps (needs the metadata repo root);
        # plain TextDB for the datasets tree (statuses, runinfo, groupings)
        # so no git machinery is involved in the hot read path.
        self.meta = LegendMetadata(str(self.meta_path), lazy=True)
        self.status_db = TextDB(self.datasets_path, lazy=True)
        self._groupings_cache: dict[str, dict] = {}
        self._runinfo = None

    def reload(self):
        """Drop every cached object so subsequent reads see staged edits."""
        with self._lock:
            self._load()
            self.version += 1

    # -- surface consumed by meta_views (duck-types the leds viewer) --------

    @property
    def runinfo(self) -> dict:
        if self._runinfo is None:
            self._runinfo = Props.read_from(self.datasets_path / "runinfo.yaml")
        return self._runinfo

    def available_runs(self) -> dict[str, list[str]]:
        """Every run in the metadata catalogue: ``{period: [run, ...]}``."""
        return {period: sorted(runs) for period, runs in self.runinfo.items()}

    def _channelmap(self, tstamp: str):
        chmap = self._chmap_cache.get(tstamp)
        if chmap is None:
            chmap = self.meta.channelmap(tstamp)
            self._chmap_cache[tstamp] = chmap
        return chmap

    def geds_positions(self, tstamp: str) -> dict[str, tuple[int, int]]:
        """``{ged name: (string, position)}`` in the array at ``tstamp``."""
        cached = self._geds_cache.get(tstamp)
        if cached is None:
            chmap = self._channelmap(tstamp)
            geds = chmap.map("system", unique=False).geds.map("name")
            cached = {
                str(name): (
                    int(geds[name].location.string),
                    int(geds[name].location.position),
                )
                for name in geds
            }
            self._geds_cache[tstamp] = cached
        return cached

    # -- extras for the editor ---------------------------------------------

    def statuses_on(self, tstamp: str, category: str = "all") -> dict:
        """Resolved detector statuses valid at ``tstamp`` for ``category``."""
        return self.status_db.statuses.on(tstamp, system=category)

    def groupings(self, key: str) -> dict:
        """Parsed groupings file for ``key`` ("cal"|"phy"|"escale"|"psd")."""
        cached = self._groupings_cache.get(key)
        if cached is None:
            cached = Props.read_from(self.datasets_path / GROUPING_FILES[key])
            self._groupings_cache[key] = cached
        return cached

    def run_start_key(self, period: str, run: str) -> str:
        """Earliest start key of ``period/run`` across its datatypes.

        Status validity entries are keyed on these runinfo start keys (not on
        the cal-par timestamps used elsewhere in the dashboard).
        """
        entries = self.runinfo[period][run]
        keys = [
            info["start_key"]
            for info in entries.values()
            if isinstance(info, dict) and info.get("start_key")
        ]
        if not keys:
            msg = f"no start_key in runinfo for {period}/{run}"
            raise KeyError(msg)
        return min(keys, key=tstamp_to_unix)

    def next_run(self, period: str, run: str) -> tuple[str, str] | None:
        """The run following ``period/run`` in the catalogue, or None if last."""
        flat = [(p, r) for p in sorted(self.runinfo) for r in sorted(self.runinfo[p])]
        idx = flat.index((period, run))
        return flat[idx + 1] if idx + 1 < len(flat) else None


_meta_db_cache: dict[tuple[str, str], MetaDB] = {}
_meta_db_lock = threading.Lock()


def get_meta_db(path: str | Path, datasets_path: str | Path | None = None) -> MetaDB:
    """Shared :class:`MetaDB` per (clone, datasets tree).

    One instance per process for each combination, shared by every session
    reading the same tree (the clone's own ``datasets/`` by default, or a
    per-user workspace worktree).
    """
    resolved = Path(path).resolve()
    ds = Path(datasets_path).resolve() if datasets_path else resolved / "datasets"
    key = (str(resolved), str(ds))
    with _meta_db_lock:
        cached = _meta_db_cache.get(key)
        if cached is None:
            cached = _meta_db_cache.setdefault(key, MetaDB(path, datasets_path))
        return cached
