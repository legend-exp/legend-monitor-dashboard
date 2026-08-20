from __future__ import annotations

import asyncio
import importlib.resources
import logging
import os
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
import panel as pn
from dbetto import AttrsDict, Props, TextDB
from dbetto.catalog import Catalog
from legendmeta import LegendMetadata

log = logging.getLogger(__name__)

# somehow TUM server needs Agg -> needs fix in the future
mpl.use("Agg")

# Only load the extensions actually used by the dashboard (tabulator for the
# metadata table, plotly for the detailed cal plots); every extension listed
# here ships extra JS to each client.
pn.extension("tabulator", "plotly")

logo_path = importlib.resources.files("legenddashboard") / "logos"

sort_dict = {
    "String": {
        "out_key": "{key}:{k:02}",
        "primary_key": "location.string",
        "secondary_key": "location.position",
    },
    "CC4": {
        "out_key": "{key}:{k}",
        "primary_key": "electronics.cc4.id",
        "secondary_key": "electronics.cc4.channel",
    },
    "HV": {
        "out_key": "{key}:{k:02}",
        "primary_key": "voltage.card.id",
        "secondary_key": "voltage.channel",
    },
    "Det_Type": {"out_key": "{k}", "primary_key": "type", "secondary_key": "name"},
    "DAQ": {"out_key": None, "primary_key": None, "secondary_key": None},
}


def read_config(config: str | dict) -> AttrsDict:
    """
    Parse the config file or dictionary and return an AttrsDict.
    """
    if isinstance(config, str | Path):
        config = AttrsDict(Props.read_from(config))
    else:
        config = AttrsDict(config)

    return config.paths


class sort_dets:
    def __init__(self, path):
        self.prod_config = get_dataflow_config(path)

        meta_path = Path(self.prod_config["paths"]["metadata"])
        chmap_path = Path(self.prod_config["paths"]["chan_map"])
        chmap_catalog = Catalog.read_from(chmap_path / "channelmaps" / "validity.yaml")
        chmap_entries = {}
        meta = LegendMetadata(meta_path, lazy=False)
        for system in chmap_catalog.entries:
            chmap_entries[system] = []
            for entry in chmap_catalog.entries[system]:
                try:
                    db = meta.channelmap(
                        datetime.fromtimestamp(entry.valid_from, tz=UTC), system=system
                    )
                    new_entry = Catalog.Entry(entry.valid_from, db)
                    chmap_entries[system].append(new_entry)
                except RuntimeError:
                    continue

        self.chmaps = Catalog(chmap_entries)

        status_path = Path(self.prod_config["paths"]["detector_status"]) / "statuses"
        status_catalog = Catalog.read_from(status_path / "validity.yaml")
        status_entries = {}
        textdb = TextDB(status_path, lazy=False)
        for system in status_catalog.entries:
            status_entries[system] = []
            for entry in status_catalog.entries[system]:
                db = textdb.on(
                    datetime.fromtimestamp(entry.valid_from, tz=UTC), system=system
                )
                new_entry = Catalog.Entry(entry.valid_from, db)
                status_entries[system].append(new_entry)

        self.statuses = Catalog(status_entries)


class LRUDict(OrderedDict):
    """
    Thread-safe, size-bounded dict with least-recently-used eviction.

    Used to cache the (immutable) parsed parameter files so that they are
    loaded from the high-latency NERSC filesystem at most once and shared
    across all user sessions without growing without bound.
    """

    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        with self._lock:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
            self.move_to_end(key)
            while len(self) > self.maxsize:
                super().__delitem__(next(iter(self)))

    def __contains__(self, key):
        with self._lock:
            contained = super().__contains__(key)
            if contained:
                self.move_to_end(key)
            return contained

    def get(self, key, default=None):
        with self._lock:
            if not super().__contains__(key):
                return default
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value


# Read-only metadata catalogs are expensive to build (they walk every
# channelmap/status validity entry on disk) but never change at runtime, so we
# build one ``sort_dets`` per base path and share it across all sessions.
_sort_dets_cache: dict[str, sort_dets] = {}
_sort_dets_lock = threading.Lock()

# Parsed parameter files are immutable per run; share a single bounded cache
# across every session to bound memory use and avoid repeated disk reads.
_par_cache = {"hit": LRUDict(maxsize=128), "dsp": LRUDict(maxsize=128)}

# dataflow-config.yaml never changes at runtime but is needed by every
# Monitoring instance and inside the per-run plot loops; parse it once per
# base path.
_dataflow_config_cache: dict[str, dict] = {}
_dataflow_config_lock = threading.Lock()

# The run dict requires one filesystem stat per run on the high-latency NERSC
# filesystem; share one scan per base path across all sessions. It is
# refreshed by PeriodRefreshRegistry.scan_and_push (hourly / manual refresh).
_run_dict_cache: dict[str, dict] = {}
_run_dict_lock = threading.Lock()


def get_dataflow_config(path):
    """Return the parsed (read-only) dataflow-config.yaml for ``path``."""
    key = str(Path(path).resolve())
    with _dataflow_config_lock:
        cfg = _dataflow_config_cache.get(key)
    if cfg is None:
        cfg = Props.read_from(Path(path) / "dataflow-config.yaml", subst_pathvar=True)
        with _dataflow_config_lock:
            cfg = _dataflow_config_cache.setdefault(key, cfg)
    return cfg


def get_run_dict(path, refresh=False):
    """Return the shared run dict for ``path``, scanning the filesystem once.

    Pass ``refresh=True`` to force a rescan (used by the periodic/manual
    refresh); all subsequent callers then see the updated dict.
    """
    key = str(Path(path).resolve())
    if not refresh:
        with _run_dict_lock:
            cached = _run_dict_cache.get(key)
        if cached is not None:
            return cached
    scanned = gen_run_dict(path)
    with _run_dict_lock:
        _run_dict_cache[key] = scanned
    return scanned


def get_sort_dets(path):
    """Return a shared, cached :class:`sort_dets` instance for ``path``."""
    key = str(Path(path).resolve())
    with _sort_dets_lock:
        cached = _sort_dets_cache.get(key)
    if cached is None:
        # Build outside the lock so concurrent first-time builds for *different*
        # paths do not serialise; a duplicate build for the same path is rare
        # and harmless.
        built = sort_dets(path)
        with _sort_dets_lock:
            cached = _sort_dets_cache.setdefault(key, built)
    return cached


def get_par_cache():
    """Return the process-wide bounded cache of parsed parameter files."""
    return _par_cache


# Parsing a par_hit yaml takes seconds even with libyaml (millions of scalar
# nodes); unpickling the parsed dict takes milliseconds. The on-disk pickle
# cache lives under the configured tmp path and is keyed on the yaml's stat,
# so it survives restarts and is shared between server processes.
_par_disk_cache: Path | None = None


def configure_par_disk_cache(tmp_path) -> None:
    """Enable the on-disk par cache under ``tmp_path`` (None disables)."""
    global _par_disk_cache  # noqa: PLW0603
    if tmp_path is None:
        _par_disk_cache = None
        return
    path = Path(tmp_path) / "legenddashboard-par-cache"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.warning("par disk cache disabled: cannot create %s", path)
        _par_disk_cache = None
    else:
        _par_disk_cache = path


def _read_pars(pars_path: Path):
    if _par_disk_cache is None:
        return Props.read_from(pars_path)
    st = pars_path.stat()
    pickled = _par_disk_cache / f"{pars_path.stem}-{st.st_mtime_ns}-{st.st_size}.pkl"
    if pickled.exists():
        try:
            with pickled.open("rb") as f:
                return pickle.load(f)
        except Exception:  # corrupt/partial file: fall through and rebuild
            log.warning("discarding unreadable par cache %s", pickled, exc_info=True)
    pars = Props.read_from(pars_path)
    tmp = pickled.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(pars, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pickled)  # atomic: readers never see a partial file
    except OSError:
        log.warning("could not write par cache %s", pickled, exc_info=True)
        tmp.unlink(missing_ok=True)
    return pars


def run_pars_path(prod_config, tier, period, run, run_info) -> Path:
    """Path of the par_<tier> yaml for one calibration run."""
    return (
        Path(prod_config["paths"][f"par_{tier}"])
        / f"cal/{period}/{run}"
        / (
            f"{run_info['experiment']}-{period}-{run}-cal-"
            f"{run_info['timestamp']}-par_{tier}.yaml"
        )
    )


def load_run_pars(prod_config, tier, period, run, run_info, cache_data=None):
    """Load the par_<tier> file for one calibration run via the shared cache.

    Cache keys must include the period because run ids (r000, r001, ...)
    repeat across periods, and ``cache_data`` is shared by every session.
    """
    cache_key = f"{period}/{run}"
    if cache_data is not None:
        cached = cache_data[tier].get(cache_key)
        if cached is not None:
            return cached
    pars = _read_pars(run_pars_path(prod_config, tier, period, run, run_info))
    if cache_data is not None:
        cache_data[tier][cache_key] = pars
    return pars


def prewarm_run_pars(base_path, periods=None, n_periods=1) -> None:
    """Parse (or unpickle) the par files of the latest ``n_periods`` periods.

    Meant to run once at server start so the first clicks of a session do
    not pay the multi-second yaml parse; newest runs first.
    """
    prod_config = get_dataflow_config(base_path)
    periods = periods if periods is not None else get_run_dict(base_path)
    start = time.time()
    for period in sorted(periods)[-n_periods:][::-1]:
        for run in sorted(periods[period])[::-1]:
            for tier in ("hit", "dsp"):
                try:
                    load_run_pars(
                        prod_config, tier, period, run, periods[period][run], _par_cache
                    )
                except Exception:
                    log.warning("prewarm: could not load %s %s/%s", tier, period, run)
    log.info(
        "prewarmed par files for %d period(s) in %.0fs", n_periods, time.time() - start
    )


class PeriodRefreshRegistry:
    """
    Coordinate a single, process-wide scan for new periods/runs.

    Every user session registers its base monitor here. A single scheduled
    task (or a manual refresh) then runs the filesystem scan *once* and pushes
    the result into each live session via that session's Bokeh document, so we
    never scan the (high-latency) filesystem once per user.
    """

    def __init__(self):
        # id(monitor) -> (weakref to monitor, that session's document)
        self._sessions: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self._scan_path = None
        self._scheduled = False

    def register(self, monitor, path):
        """Register a session's monitor; no-op outside a session context."""
        doc = pn.state.curdoc
        if doc is None:
            # No live session (e.g. during warm-up) -> nothing to push to.
            return
        key = id(monitor)
        with self._lock:
            self._sessions[key] = (weakref.ref(monitor), doc)
            self._scan_path = path
        doc.on_session_destroyed(lambda _ctx: self._unregister(key))

    def _unregister(self, key):
        with self._lock:
            self._sessions.pop(key, None)

    def ensure_scheduled(self, period):
        """Schedule the single server-wide refresh task exactly once."""
        with self._lock:
            if self._scheduled:
                return
            self._scheduled = True
        try:
            pn.state.schedule_task(
                "legend_refresh_periods", self._scan_and_push_async, period=period
            )
        except Exception:
            # No running event loop yet (e.g. during warm-up); allow a later
            # session to schedule it.
            with self._lock:
                self._scheduled = False
            log.debug("Could not schedule global period refresh", exc_info=True)

    async def _scan_and_push_async(self):
        """Scheduled-task wrapper: run the blocking scan on a worker thread.

        The scheduled task runs on the server's event loop; running the
        high-latency filesystem scan there would stall every session's
        websocket handling for its duration.
        """
        await asyncio.to_thread(self.scan_and_push)

    def scan_and_push(self):
        """Scan the filesystem once and push new periods to every session."""
        with self._lock:
            path = self._scan_path
            sessions = list(self._sessions.values())
        if path is None or not sessions:
            # Nobody connected -> skip the scan entirely.
            return
        # Force a rescan and refresh the shared cache so new sessions also see
        # the new periods without their own filesystem walk.
        new_periods = get_run_dict(path, refresh=True)
        for ref, doc in sessions:
            monitor = ref()
            if monitor is None:
                continue
            try:
                # Mutate each session's state on its own document/event loop.
                doc.add_next_tick_callback(partial(monitor._apply_periods, new_periods))
            except Exception:
                log.debug("Could not push period refresh to a session", exc_info=True)


# Single, process-wide registry shared by all sessions.
period_refresh_registry = PeriodRefreshRegistry()


def gen_run_dict(path):
    prod_config = get_dataflow_config(path)
    par_file = Path(prod_config["paths"]["par_hit"]) / "validity.yaml"
    run_dict = {}
    file = Props.read_from(par_file)
    for entry in file:
        experiment, period, run, _, _, _ = entry["apply"][0].split("/")[-1].split("-")
        timestamp = entry["valid_from"]
        if (
            Path(prod_config["paths"]["par_hit"])
            / f"cal/{period}/{run}"
            / f"{experiment}-{period}-{run}-cal-{timestamp}-par_hit.yaml"
        ).exists():
            if period in run_dict:
                run_dict[period][run] = {
                    "experiment": experiment,
                    "timestamp": timestamp,
                }
            else:
                run_dict[period] = {
                    run: {"experiment": experiment, "timestamp": timestamp}
                }
    return run_dict


# Sorting the channel map is pure w.r.t. (path, timestamp, key, datatype) but
# is called several times per render; cache the results process-wide. Entries
# are treated as read-only by all callers.
_sorter_cache = LRUDict(maxsize=256)


def sorter(
    path, timestamp, key="String", datatype="cal", spms=False, sort_dets_obj=None
):
    cache_key = (str(path), timestamp, key, datatype, spms)
    cached = _sorter_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _sorter_uncached(path, timestamp, key, datatype, spms, sort_dets_obj)
    _sorter_cache[cache_key] = result
    return result


def _sorter_uncached(path, timestamp, key, datatype, spms, sort_dets_obj):
    if sort_dets_obj is not None:
        chmap = sort_dets_obj.chmaps.valid_for(timestamp, system=datatype)
        det_status = sort_dets_obj.statuses.valid_for(timestamp, system=datatype)
    else:
        prod_config = get_dataflow_config(path)

        cfg_file = prod_config["paths"]["metadata"]
        configs = LegendMetadata(path=cfg_file)
        chmap = configs.channelmap(timestamp)

        det_status_path = prod_config["paths"]["detector_status"]
        det_status = LegendMetadata(path=det_status_path, lazy=True).statuses.on(
            timestamp, system=datatype
        )

    out_dict = {}
    # SiPMs sorting
    if spms:
        chmap = chmap.map("system", unique=False)["spms"]
        if key == "Barrel":
            mapping = chmap.map("name")
            for pos in ["top", "bottom"]:
                for barrel in ["IB", "OB"]:
                    out_dict[f"{barrel}-{pos}"] = [
                        k
                        for k, entry in sorted(mapping.items())
                        if barrel in entry["location"]["fiber"]
                        and pos in entry["location"]["position"]
                    ]
        return out_dict, chmap

    # Daq needs special item as sort on tertiary key
    if key == "DAQ":
        mapping = chmap.map("daq.crate", unique=False)
        for k, entry in sorted(mapping.items()):
            for m, item in sorted(entry.map("daq.card.id", unique=False).items()):
                out_dict[f"DAQ:Cr{k:02},Ch{m:02}"] = [
                    item.map("daq.channel")[pos].name
                    for pos in sorted(item.map("daq.channel"))
                    if item.map("daq.channel")[pos].system == "geds"
                ]
    else:
        out_key = sort_dict[key]["out_key"]
        primary_key = sort_dict[key]["primary_key"]
        secondary_key = sort_dict[key]["secondary_key"]
        mapping = chmap.map(primary_key, unique=False)
        for k, entry in sorted(mapping.items()):
            out_dict[out_key.format(key=key, k=k)] = [
                entry.map(secondary_key)[pos].name
                for pos in sorted(entry.map(secondary_key))
                if entry.map(secondary_key)[pos].system == "geds"
            ]

    out_dict = {
        entry: out_dict[entry] for entry in list(out_dict) if len(out_dict[entry]) > 0
    }
    return out_dict, det_status, chmap


def get_characterization(x, key):
    try:
        return x["manufacturer"][key]
    except KeyError:
        return np.nan


def get_production(x, key):
    try:
        return x[key]
    except KeyError:
        return np.nan
