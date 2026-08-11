"""Pure YAML-editing layer for the metadata editor.

All writes go through ruamel.yaml round-trip so untouched lines, comments and
quoting survive (the reasons in ``ignored_daq_cycles.yaml`` live in end-of-line
comments). Nothing in here touches git or Panel; every function takes the
``datasets/`` directory of the editable clone.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, datetime
from pathlib import Path

from dbetto.catalog import Catalog
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from legenddashboard.metadata.meta_db import GROUPING_FILES, tstamp_to_unix

#: Datatype choice -> status-validity categories (user-specified mapping).
DATATYPE_MAP = {
    "cal": ["cal", "fft", "pzc"],
    "phy": ["all"],
    "both": ["cal", "fft", "pzc", "all"],
}
#: Datatype choice -> datatype token in the status file name.
FILE_TOKEN = {"cal": "cal", "phy": "phy", "both": "all"}

#: PSD flags editable per detector.
PSD_FLAGS = ("low_aoe", "high_aoe", "lq", "ann", "coax_rt")
PSD_FLAG_VALUES = ("valid", "present", "missing")
USABILITY_VALUES = ("on", "off", "ac")

CYCLE_RE = re.compile(r"^l200-p\d{2}-r\d{3}-[a-z0-9]{3}-\d{8}T\d{6}Z$")


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.width = 4096  # never reflow long lines
    # match the repo's "  - item" sequence style, else every untouched line
    # of a list file (e.g. the 800+ ignored cycles) shows up in the diff
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _read(path: Path):
    with path.open() as f:
        return _yaml().load(f)


def _write(path: Path, doc) -> None:
    if isinstance(doc, list):
        # A root-level block sequence (validity.yaml) is emitted with the
        # sequence offset as an artificial root indent; strip it so top-level
        # items sit at column 0 (`- valid_from: ...`) like the originals,
        # while nested sequences keep their 4-space style.
        buf = io.StringIO()
        _yaml().dump(doc, buf)
        text = "\n".join(
            line[2:] if line.startswith("  ") else line
            for line in buf.getvalue().splitlines()
        )
        path.write_text(text + "\n")
        return
    with path.open("w") as f:
        _yaml().dump(doc, f)


def _to_plain(obj):
    """Recursively convert AttrsDict/CommentedMap trees to plain dict/list."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    return obj


def _dq(value: str) -> DoubleQuotedScalarString:
    """Double-quote a scalar (``on``/``off`` are YAML-1.1 booleans unquoted)."""
    return DoubleQuotedScalarString(value)


# ---------------------------------------------------------------------------
# detector statuses
# ---------------------------------------------------------------------------


def _merge_detector_edit(det_entry, edit: dict) -> None:
    """Apply one detector's edit onto its (Commented)Map entry, in place."""
    if "usability" in edit:
        det_entry["usability"] = _dq(str(edit["usability"]))
    if "reason" in edit:
        det_entry["reason"] = _dq(str(edit["reason"]))
    psd_edit = edit.get("psd")
    if psd_edit:
        psd = det_entry.setdefault("psd", CommentedMap())
        if "is_bb_like" in psd_edit:
            psd["is_bb_like"] = str(psd_edit["is_bb_like"])
        flag_edits = psd_edit.get("status", {})
        if flag_edits:
            status = psd.setdefault("status", CommentedMap())
            for flag, value in flag_edits.items():
                status[str(flag)] = str(value)


def _quote_statuses(doc) -> None:
    """Quote every usability/reason scalar (matches repo convention)."""
    for det_entry in doc.values():
        if not isinstance(det_entry, dict):
            continue
        for key in ("usability", "reason"):
            if key in det_entry and not isinstance(
                det_entry[key], DoubleQuotedScalarString
            ):
                det_entry[key] = _dq(str(det_entry[key]))


def status_file_name(period: str, run: str, datatype: str = "both") -> str:
    return f"l200-{period}-{run}-T%-{FILE_TOKEN[datatype]}-config.yaml"


def stage_status_edit(
    datasets_path: str | Path,
    period: str,
    run: str,
    start_key: str,
    edits: dict[str, dict],
    datatype: str = "both",
    resolved_seed: dict | None = None,
) -> Path:
    """Stage detector-status edits for one run onto the working tree.

    ``edits`` maps detector name to a dict that may set ``usability`` +
    ``reason`` and/or ``psd`` (``{"is_bb_like": ..., "status": {flag: value}}``).

    If the target status file exists the edits are merged into it (preserving
    every untouched key and comment). Otherwise a new file is created seeded
    from ``resolved_seed`` (the currently-valid resolved statuses) and
    registered in ``statuses/validity.yaml`` with the categories of
    ``datatype`` (see :data:`DATATYPE_MAP`).
    """
    datasets_path = Path(datasets_path)
    statuses_dir = datasets_path / "statuses"
    fpath = statuses_dir / status_file_name(period, run, datatype)

    if fpath.exists():
        doc = _read(fpath)
        for det, edit in edits.items():
            entry = doc.get(det)
            if entry is None:
                seed = _to_plain((resolved_seed or {}).get(det)) or {
                    "reason": "",
                    "usability": "on",
                    "processable": True,
                }
                doc[det] = entry = CommentedMap(seed)
                _quote_statuses({det: entry})
            _merge_detector_edit(entry, edit)
        _write(fpath, doc)
        return fpath

    if not resolved_seed:
        msg = f"{fpath.name} does not exist and no seed statuses were provided"
        raise ValueError(msg)

    doc = CommentedMap()
    for det, entry in _to_plain(resolved_seed).items():
        doc[det] = CommentedMap(entry)
    _quote_statuses(doc)
    for det, edit in edits.items():
        if det not in doc:
            doc[det] = CommentedMap(
                {"reason": _dq(""), "usability": _dq("on"), "processable": True}
            )
        _merge_detector_edit(doc[det], edit)
    _write(fpath, doc)

    try:
        update_validity(
            statuses_dir / "validity.yaml",
            {  # key order matches the existing entries
                "valid_from": start_key,
                "apply": [fpath.name],
                "category": list(DATATYPE_MAP[datatype]),
                "mode": "reset",
            },
        )
    except Exception:
        fpath.unlink(missing_ok=True)  # don't leave an unregistered file behind
        raise
    return fpath


def update_validity(validity_path: str | Path, entry: dict) -> None:
    """Insert ``entry`` into validity.yaml keeping chronological order.

    Idempotent: an entry with identical ``(valid_from, apply)`` is not
    duplicated. After writing, the file is parsed back with
    ``dbetto.catalog.Catalog`` (the consumer every processing tool uses); on
    failure the previous content is restored and the error re-raised.
    """
    validity_path = Path(validity_path)
    doc = _read(validity_path)

    key = (str(entry["valid_from"]), tuple(entry["apply"]))
    for existing in doc:
        if (str(existing["valid_from"]), tuple(existing["apply"])) == key:
            return

    ts = tstamp_to_unix(str(entry["valid_from"]))
    idx = len(doc)
    for i, existing in enumerate(doc):
        if tstamp_to_unix(str(existing["valid_from"])) > ts:
            idx = i
            break
    doc.insert(idx, CommentedMap(entry))

    backup = validity_path.read_bytes()
    _write(validity_path, doc)
    try:
        Catalog.read_from(str(validity_path))
    except Exception:
        validity_path.write_bytes(backup)
        raise


def apply_status_range(
    datasets_path: str | Path,
    edits: dict[str, dict],
    datatype: str,
    start_point: tuple[str, str, str, dict | None],
    revert_point: tuple[str, str, str, dict | None, dict] | None = None,
) -> list[Path]:
    """Apply ``edits`` from a start run, optionally bounded by a revert.

    ``start_point`` is ``(period, run, start_key, resolved_seed)``;
    ``revert_point`` (for a bounded end run) is ``(period, run, start_key,
    resolved_seed, revert_edits)`` for the run *after* the end run, where
    ``revert_edits`` restore the previously-valid values there.
    """
    paths = [
        stage_status_edit(
            datasets_path,
            *start_point[:3],
            edits=edits,
            datatype=datatype,
            resolved_seed=start_point[3],
        )
    ]
    if revert_point is not None:
        period, run, start_key, seed, revert_edits = revert_point
        paths.append(
            stage_status_edit(
                datasets_path,
                period,
                run,
                start_key,
                edits=revert_edits,
                datatype=datatype,
                resolved_seed=seed,
            )
        )
    return paths


# ---------------------------------------------------------------------------
# partitions / groupings
# ---------------------------------------------------------------------------


def compress_runs(runs: list[str]) -> str | list:
    """Canonical groupings notation for a run list.

    Contiguous ascending spans of >= 3 runs collapse to ``"rXXX..rYYY"``;
    shorter spans stay as individual runs. A single piece is returned as a
    scalar, several as a list. Round-trips with ``expand_run_list``.
    """
    nums = sorted({int(r[1:]) for r in runs})
    spans: list[tuple[int, int]] = []
    for n in nums:
        if spans and n == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], n)
        else:
            spans.append((n, n))
    pieces: list[str] = []
    for lo, hi in spans:
        if hi - lo >= 2:
            pieces.append(f"r{lo:03d}..r{hi:03d}")
        else:
            pieces.extend(f"r{n:03d}" for n in range(lo, hi + 1))
    if len(pieces) == 1:
        return pieces[0]
    return pieces


def set_detector_block(
    datasets_path: str | Path, key: str, detector: str, block: dict | None
) -> Path:
    """Replace one detector's (or ``default``'s) block in a groupings file.

    ``block`` is ``{partition: {period: runs-notation}}``; ``None`` or an
    empty dict removes the detector's override. New detectors are inserted
    alphabetically (after ``default``, which stays first). Other blocks are
    untouched (ruamel round-trip).
    """
    datasets_path = Path(datasets_path)
    fpath = datasets_path / GROUPING_FILES[key]
    doc = _read(fpath)

    if not block:
        if detector in doc:
            del doc[detector]
        _write(fpath, doc)
        return fpath

    new_block = CommentedMap()
    for part, periods in block.items():
        new_block[str(part)] = CommentedMap(
            {str(p): _to_plain(v) for p, v in periods.items()}
        )

    if detector in doc:
        doc[detector] = new_block
    else:
        names = [k for k in doc if k != "default"]
        after = [n for n in sorted([*names, detector]) if n > detector]
        if after:
            doc.insert(list(doc).index(after[0]), detector, new_block)
        else:
            doc[detector] = new_block
    _write(fpath, doc)
    return fpath


def next_letter(part_name: str) -> str:
    """``"calgroup003a"`` -> ``"calgroup003b"``."""
    return part_name[:-1] + chr(ord(part_name[-1]) + 1)


PARTITION_RE = re.compile(r"^[a-z]+group\d{3}[a-z]$")
_PARTITION_SHORT_RE = re.compile(r"^(\d{1,3})([a-z])$")


def normalize_partition(name: str, groupings_key: str = "cal") -> str:
    """Expand shorthand partition names: ``"12a"`` -> ``"calgroup012a"``.

    Full names pass through; the prefix comes from the groupings file being
    edited (``phy`` -> ``phygroup``, everything else -> ``calgroup``).
    """
    name = name.strip().lower()
    if PARTITION_RE.match(name):
        return name
    m = _PARTITION_SHORT_RE.match(name)
    if m:
        prefix = "phygroup" if groupings_key == "phy" else "calgroup"
        return f"{prefix}{int(m.group(1)):03d}{m.group(2)}"
    msg = f"not a valid partition name: {name!r} (e.g. '12a' or 'calgroup012a')"
    raise ValueError(msg)


def block_from_map(mapping: dict[tuple[str, str], str]) -> dict:
    """``{(period, run): partition}`` -> groupings block, runs compressed."""
    by_part: dict[str, dict[str, list[str]]] = {}
    for (period, run), part in sorted(mapping.items()):
        by_part.setdefault(part, {}).setdefault(period, []).append(run)
    return {
        part: {period: compress_runs(runs) for period, runs in periods.items()}
        for part, periods in by_part.items()
    }


def assign_in_block(
    groupings: dict,
    block_name: str,
    cells: list[tuple[str, str]],
    target: str,
) -> dict | None:
    """Reassign ``cells`` within one groupings block, including ``default``.

    For a detector block this is :func:`assign_partition`; the ``default``
    block has no lower layer to merge against, so it is edited against an
    empty default (same minimal-churn semantics).
    """
    if block_name != "default":
        return assign_partition(groupings, block_name, cells, target)
    synthetic = {"default": {}, "_default": groupings.get("default", {})}
    return assign_partition(synthetic, "_default", cells, target)


def assign_partition(
    groupings: dict,
    detector: str,
    cells: list[tuple[str, str]],
    target: str,
) -> dict | None:
    """Reassign ``cells`` (``(period, run)`` pairs) of one detector to ``target``.

    Returns the detector's new override block, or ``None`` when the result is
    identical to what the ``default`` block alone yields (drop the override).
    The override stays *minimal*: only the partitions/periods actually touched
    by the reassignment are materialized (an empty run list masks a default
    period that was fully vacated); everything else keeps coming from the
    default block. Pure: the caller writes the result via
    :func:`set_detector_block`.
    """
    if not PARTITION_RE.match(target):
        msg = f"not a valid partition name: {target!r} (e.g. 'calgroup011a')"
        raise ValueError(msg)
    from legenddashboard.metadata.meta_views import (
        expand_run_list,
        merge_with_defaults,
    )

    default = groupings.get("default", {})
    det = groupings.get(detector, {})

    def default_runs(part: str, period: str) -> set[str]:
        periods = default.get(part, {})
        if not isinstance(periods, dict) or period not in periods:
            return set()
        return set(expand_run_list(periods[period]))

    # working copy of the detector's override: {part: {period: set(runs)}}
    block: dict[str, dict[str, set]] = {
        part: {per: set(expand_run_list(runs)) for per, runs in periods.items()}
        for part, periods in det.items()
        if isinstance(periods, dict)
    }
    merged = merge_with_defaults(det, default)
    touched: set[tuple[str, str]] = set()
    for period, run in cells:
        current = merged.get((period, run))
        if current == target:
            continue
        if current is not None:
            # materialize the losing partition's period (from the default if
            # the override does not mention it) and take the run out
            periods = block.setdefault(current, {})
            if period not in periods:
                periods[period] = default_runs(current, period)
            periods[period].discard(run)
            touched.add((current, period))
        tperiods = block.setdefault(target, {})
        if period not in tperiods:
            tperiods[period] = default_runs(target, period)
        tperiods[period].add(run)
        touched.add((target, period))

    out: dict = {}
    for part, periods in block.items():
        kept: dict = {}
        for period, runs in periods.items():
            if (part, period) not in touched:
                # keep the original notation so untouched entries don't churn
                kept[period] = _to_plain(det[part][period])
            elif runs:
                kept[period] = compress_runs(sorted(runs))
            elif default_runs(part, period):
                kept[period] = []  # masks the default assignment
        if kept:
            out[part] = kept
    if not out or merge_with_defaults(out, default) == merge_with_defaults({}, default):
        return None
    return out


def split_group(block: dict, part: str, period: str, at_run: str) -> dict:
    """Split ``part`` at ``period/at_run`` into a new following partition.

    Runs of ``part``/``period`` >= ``at_run`` move to the next free letter
    after ``part`` (``...a`` -> ``...b``, skipping names already in the
    block), as do all later periods of ``part``. Returns a new plain-dict
    block; run values are re-emitted via :func:`compress_runs`.
    """
    from legenddashboard.metadata.meta_views import expand_run_list

    new_name = next_letter(part)
    while new_name in block:
        new_name = next_letter(new_name)
    out: dict = {}
    for p_name, periods in block.items():
        if p_name != part:
            out[p_name] = _to_plain(periods)
            continue
        kept: dict = {}
        moved: dict = {}
        for per in sorted(periods):
            runs = expand_run_list(periods[per])
            if per < period:
                kept[per] = compress_runs(runs)
            elif per > period:
                moved[per] = compress_runs(runs)
            else:
                before = [r for r in runs if r < at_run]
                after = [r for r in runs if r >= at_run]
                if before:
                    kept[per] = compress_runs(before)
                if after:
                    moved[per] = compress_runs(after)
        if kept:
            out[p_name] = kept
        if moved:
            out[new_name] = moved
    return out


# ---------------------------------------------------------------------------
# runlists
# ---------------------------------------------------------------------------


def set_runlist(
    datasets_path: str | Path,
    dataset: str,
    datatype: str,
    periods: dict | None,
) -> Path:
    """Replace one ``<dataset>.<datatype>`` node of ``runlists.yaml``.

    ``periods`` maps period to a runs value (``"all"``, a run list, or range
    notation); ``None``/empty removes the datatype, and a dataset left with
    no datatypes is removed entirely. Everything else in the file (other
    datasets/datatypes, the header comments) is untouched (ruamel
    round-trip).
    """
    path = Path(datasets_path) / "runlists.yaml"
    doc = _read(path)

    if not periods:
        if dataset in doc and datatype in doc[dataset]:
            del doc[dataset][datatype]
            if not doc[dataset]:
                del doc[dataset]
        _write(path, doc)
        return path

    node = CommentedMap()
    for period, runs in periods.items():
        node[str(period)] = _to_plain(runs)
    if dataset not in doc:
        doc[dataset] = CommentedMap()
    doc[dataset][str(datatype)] = node
    _write(path, doc)
    return path


# ---------------------------------------------------------------------------
# bad cycles
# ---------------------------------------------------------------------------


def _cycles_doc(datasets_path: str | Path):
    path = Path(datasets_path) / "ignored_daq_cycles.yaml"
    return path, _read(path)


def list_ignored_cycles(datasets_path: str | Path) -> list[tuple[str, str]]:
    """``[(cycle_id, reason)]`` from the ``unprocessable`` list (EOL comments)."""
    _, doc = _cycles_doc(datasets_path)
    seq = doc.get("unprocessable") or []
    out = []
    for idx, cycle_id in enumerate(seq):
        reason = ""
        comment = seq.ca.items.get(idx)
        if comment and comment[0] is not None:
            reason = comment[0].value.strip().lstrip("#").strip()
        out.append((str(cycle_id), reason))
    return out


def add_ignored_cycles(
    datasets_path: str | Path, cycle_ids: str | list[str], reason: str
) -> list[str]:
    """Append cycle ids with ``reason`` as an EOL comment; returns those added.

    Invalid ids raise; ids already listed are skipped (a whole time-range
    batch lands in one write).
    """
    if isinstance(cycle_ids, str):
        cycle_ids = [cycle_ids]
    for cid in cycle_ids:
        if not CYCLE_RE.match(cid):
            msg = f"not a valid cycle id: {cid!r}"
            raise ValueError(msg)

    path, doc = _cycles_doc(datasets_path)
    seq = doc["unprocessable"]
    existing = {str(c) for c in seq}
    added = []
    for cid in cycle_ids:
        if cid in existing:
            continue
        seq.append(cid)
        if reason:
            seq.yaml_add_eol_comment(f"# {reason}", key=len(seq) - 1)
        added.append(cid)
    if added:
        _write(path, doc)
    return added


def remove_ignored_cycle(datasets_path: str | Path, cycle_id: str) -> bool:
    """Remove one cycle id from the list (used for staged, un-pushed entries)."""
    path, doc = _cycles_doc(datasets_path)
    seq = doc["unprocessable"]
    for idx, cid in enumerate(seq):
        if str(cid) == cycle_id:
            del seq[idx]
            _write(path, doc)
            return True
    return False


def raw_run_catalogue(
    raw_dirs: list[str | Path],
) -> dict[str, dict[str, list[str]]]:
    """``{period: {run: [cycle ids]}}`` from raw-tier file names.

    Scans ``<dir>/<datatype>/<period>/<run>/*.lh5`` under each existing
    directory in ``raw_dirs`` (filename-only); cycle ids are sorted by
    timestamp. Feeds the period/run/cycle selectors of the bad-cycles page.
    """
    cat: dict[str, dict[str, set[str]]] = {}
    for raw_dir_str in raw_dirs:
        raw_dir = Path(raw_dir_str)
        if not raw_dir.is_dir():
            continue
        for f in raw_dir.glob("*/*/*/*.lh5"):
            parts = f.name.split("-")
            if len(parts) < 5:
                continue
            cycle_id = "-".join(parts[:5])
            if not CYCLE_RE.match(cycle_id):
                continue
            cat.setdefault(parts[1], {}).setdefault(parts[2], set()).add(cycle_id)
    return {
        period: {
            run: sorted(cat[period][run], key=lambda c: c.split("-")[4])
            for run in sorted(cat[period])
        }
        for period in sorted(cat)
    }


def find_cycles_in_range(
    raw_dirs: list[str | Path],
    start_dt: datetime,
    end_dt: datetime,
    datatypes: list[str] | None = None,
) -> list[str]:
    """Cycle ids in ``[start_dt, end_dt]`` from raw-tier file names.

    Scans ``<dir>/<datatype>/<period>/<run>/l200-...-<tstamp>-tier_*.lh5``
    under each existing directory in ``raw_dirs`` (filename-only, no file
    reads). Returns unique ids sorted by timestamp.
    """

    def _aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

    start_dt, end_dt = _aware(start_dt), _aware(end_dt)
    found: dict[str, datetime] = {}
    for raw_dir_str in raw_dirs:
        raw_dir = Path(raw_dir_str)
        if not raw_dir.is_dir():
            continue
        for f in raw_dir.glob("*/*/*/*.lh5"):
            parts = f.name.split("-")
            if len(parts) < 5:
                continue
            cycle_id = "-".join(parts[:5])
            if not CYCLE_RE.match(cycle_id):
                continue
            if datatypes and parts[3] not in datatypes:
                continue
            ts = datetime.fromtimestamp(tstamp_to_unix(parts[4]), UTC)
            if start_dt <= ts <= end_dt:
                found[cycle_id] = ts
    return sorted(found, key=found.get)
