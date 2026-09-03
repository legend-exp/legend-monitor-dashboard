"""Shared detector-issue notes.

One JSON file (``paths.notes``, falling back to ``<paths.tmp>/detector-notes.json``)
holds free-text notes that every session sees: author (typed, not
authenticated -- the Metadata Editor trust model), the period/run the note was
written against, optional detector names and a "seen in" hint naming the plot.
A note's run doubles as its link: Jump points the whole dashboard at it by
assigning the ref-shared period/run, like the metadata click-to-jump.

The store is process-wide (one instance per path). Every read/mutation
reloads when the file's (mtime, size) fingerprint changed, so hand edits and
writes by another process are picked up; writes are atomic renames but there
is no cross-process lock, so concurrent writers in separate processes are
last-write-wins (the deployment serves all sessions from one process). An
unwritable path degrades to a memory-only store with a warning, never a
crashed dashboard.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
import panel as pn
import param

from legenddashboard.base import Monitoring
from legenddashboard.util import sorter

log = logging.getLogger(__name__)

_FIELDS = (
    "id",
    "created",
    "author",
    "period",
    "run",
    "detectors",
    "seen_in",
    "text",
    "status",
)


class NotesStore:
    """All notes in one JSON file; shared by every session of the process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._notes: list[dict] = []
        self._fingerprint: tuple | None = None
        self._writable = True
        self.version = 0
        with self._lock:
            self._reload()

    # -- persistence -------------------------------------------------------

    def _stat(self):
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload(self):
        """Pick up external writes (other process, hand edit); lock held."""
        fingerprint = self._stat()
        if fingerprint is None or fingerprint == self._fingerprint:
            return
        try:
            with self.path.open() as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                msg = f"notes file holds {type(raw).__name__}, not a list"
                raise TypeError(msg)
            self._notes = [n for n in raw if isinstance(n, dict) and n.get("id")]
        except (OSError, json.JSONDecodeError, TypeError):
            log.exception("unreadable notes file %s; keeping current notes", self.path)
        else:
            self._fingerprint = fingerprint
            self.version += 1

    def _write(self):
        """Atomic write; an unwritable path disables persistence. Lock held."""
        if not self._writable:
            return
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w") as f:
                json.dump(self._notes, f, indent=1)
            tmp.replace(self.path)
        except OSError:
            self._writable = False
            log.warning("notes not persisted: cannot write %s", self.path)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        else:
            self._fingerprint = self._stat()

    # -- API ---------------------------------------------------------------

    def all(self) -> pd.DataFrame:
        """Every note, newest first, one column per field."""
        with self._lock:
            self._reload()
            notes = [dict(n) for n in self._notes]
        frame = pd.DataFrame(notes, columns=list(_FIELDS))
        return frame.sort_values("created", ascending=False).reset_index(drop=True)

    def add(self, **fields) -> dict:
        note = {
            "id": uuid.uuid4().hex,
            "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "status": "open",
            **fields,
        }
        with self._lock:
            self._reload()
            self._notes.append(note)
            self._write()
            self.version += 1
        return note

    def set_status(self, note_id: str, status: str) -> bool:
        with self._lock:
            self._reload()
            for note in self._notes:
                if note["id"] == note_id:
                    note["status"] = status
                    self._write()
                    self.version += 1
                    return True
        return False

    def delete(self, note_id: str) -> bool:
        with self._lock:
            self._reload()
            kept = [n for n in self._notes if n["id"] != note_id]
            if len(kept) == len(self._notes):
                return False
            self._notes = kept
            self._write()
            self.version += 1
        return True


_stores: dict[str, NotesStore] = {}
_stores_lock = threading.Lock()


def get_notes_store(path: str | Path) -> NotesStore:
    """Shared :class:`NotesStore` per path (one file, every session)."""
    key = str(Path(path).resolve())
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = _stores.setdefault(key, NotesStore(path))
        return store


class NotesPage(Monitoring):
    """The Notes tab: shared notes with jump-to-run links."""

    notes_path = param.String("")

    _COLUMN_TITLES: ClassVar[dict] = {
        "created": "Created",
        "author": "Author",
        "where": "Run",
        "detectors": "Detectors",
        "seen_in": "Seen in",
        "text": "Note",
        "status": "Status",
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.store = get_notes_store(self.notes_path)

    # -- helpers -----------------------------------------------------------

    def _detector_names(self) -> list[str]:
        """Ged names for the current run, string order (cached sorter)."""
        try:
            strings_dict, _, _ = sorter(
                self.base_path,
                self.run_dict[self.run]["timestamp"],
                key="String",
                sort_dets_obj=self.sort_obj,
            )
        except Exception:
            return []
        return [str(d) for dets in strings_dict.values() for d in dets]

    def _table_frame(self, show_resolved: bool) -> pd.DataFrame:
        frame = self.store.all()
        if not show_resolved:
            frame = frame[frame["status"] != "resolved"]
        frame = frame.assign(
            where=frame["period"].fillna("?") + " " + frame["run"].fillna("?"),
            detectors=frame["detectors"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else ""
            ),
        )
        columns = list(self._COLUMN_TITLES)
        return frame[["id", *columns]].rename(columns=self._COLUMN_TITLES)

    def _note(self, note_id: str) -> dict | None:
        """The note's fields, or None when another session removed it."""
        frame = self.store.all()
        rows = frame[frame["id"] == note_id]
        return None if rows.empty else rows.iloc[0].to_dict()

    def _jump_to(self, period: str, run: str) -> None:
        """Point the whole dashboard at (period, run) if it has data."""
        if period not in self.periods or run not in self.periods[period]:
            return
        if self.period != period:
            self.period = period
        if self.run != run:
            self.run = run

    # -- UI ----------------------------------------------------------------

    def build_notes_pane(self, widget_widths: int = 140):
        alert = pn.pane.Alert("", alert_type="success", visible=False)

        def feedback(message, kind="success"):
            alert.alert_type = kind
            alert.object = message
            alert.visible = True

        author = pn.widgets.TextInput(
            name="Author", placeholder="your name", width=widget_widths
        )
        detectors = pn.widgets.MultiChoice(
            name="Detectors (optional)",
            options=self._detector_names(),
            sizing_mode="stretch_width",
        )
        seen_in = pn.widgets.TextInput(
            name="Seen in (optional)",
            placeholder="e.g. Cal Summary / Energy Spectrum",
            sizing_mode="stretch_width",
        )
        text = pn.widgets.TextAreaInput(
            name="Note", rows=3, sizing_mode="stretch_width"
        )
        add = pn.widgets.Button(
            name="Add note", button_type="primary", width=widget_widths
        )
        show_resolved = pn.widgets.Checkbox(name="show resolved", value=False)
        refresh = pn.widgets.Button(name="Refresh", width=widget_widths)
        jump = pn.widgets.Button(
            name="Jump to run", button_type="primary", width=widget_widths
        )
        resolve = pn.widgets.Button(name="Resolve / reopen", width=widget_widths)
        delete = pn.widgets.Button(
            name="Delete", button_type="danger", width=widget_widths
        )

        table = pn.widgets.Tabulator(
            self._table_frame(show_resolved.value),
            show_index=False,
            selectable=1,
            disabled=True,
            pagination="local",
            page_size=25,
            hidden_columns=["id"],
            widths={"Note": "35%"},
            sizing_mode="stretch_width",
        )

        def refresh_table(*_events):
            table.value = self._table_frame(show_resolved.value)
            table.selection = []

        def selected_id():
            if not table.selection:
                feedback("select a note in the table first", "warning")
                return None
            return table.value.iloc[table.selection[0]]["id"]

        def on_add(_event):
            if not text.value.strip():
                feedback("the note text is empty", "warning")
                return
            self.store.add(
                author=author.value.strip() or "anonymous",
                period=self.period,
                run=self.run,
                detectors=list(detectors.value),
                seen_in=seen_in.value.strip(),
                text=text.value.strip(),
            )
            text.value = ""
            refresh_table()
            feedback(f"note added for {self.period} {self.run}")

        def on_jump(_event):
            note_id = selected_id()
            if note_id is None:
                return
            row = self._note(note_id)
            if row is None:  # deleted by another session since selection
                refresh_table()
                feedback("that note no longer exists", "warning")
                return
            self._jump_to(row["period"], row["run"])
            if self.period == row["period"] and self.run == row["run"]:
                feedback(f"dashboard now at {row['period']} {row['run']}")
            else:
                feedback(f"{row['period']} {row['run']} has no data here", "warning")

        def on_resolve(_event):
            note_id = selected_id()
            if note_id is None:
                return
            row = self._note(note_id)
            if row is None:  # deleted by another session since selection
                refresh_table()
                feedback("that note no longer exists", "warning")
                return
            status = "open" if row["status"] == "resolved" else "resolved"
            self.store.set_status(note_id, status)
            refresh_table()
            feedback(f"note marked {status}")

        def on_delete(_event):
            note_id = selected_id()
            if note_id is not None and self.store.delete(note_id):
                refresh_table()
                feedback("note deleted")

        add.on_click(on_add)
        refresh.on_click(refresh_table)
        jump.on_click(on_jump)
        resolve.on_click(on_resolve)
        delete.on_click(on_delete)
        show_resolved.param.watch(refresh_table, "value")
        # new run selection: refresh the table and offer that run's detectors
        self.param.watch(
            lambda *_e: (
                refresh_table(),
                setattr(detectors, "options", self._detector_names()),
            ),
            ["run_dict", "run"],
        )

        return pn.Column(
            pn.Row(
                "## Detector Notes",
                pn.pane.Markdown(
                    "Notes are shared with everyone; new notes are tagged with "
                    "the currently selected run.",
                    styles={"color": "dimgray"},
                ),
            ),
            alert,
            pn.Row(author, detectors, seen_in),
            pn.Row(text, add),
            pn.Row(refresh, show_resolved, pn.layout.HSpacer(), jump, resolve, delete),
            table,
            name="Notes",
            sizing_mode="stretch_width",
        )
