"""Tests for the shared Notes store and page."""

from __future__ import annotations

import json

import param

from legenddashboard import notes as notes_mod
from legenddashboard.notes import NotesPage, NotesStore, get_notes_store


def test_store_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "notes.json"
    store = NotesStore(path)
    note = store.add(
        author="gm",
        period="p22",
        run="r010",
        detectors=["V05268A"],
        seen_in="Cal Summary / Energy Spectrum",
        text="baseline wobble",
    )
    frame = store.all()
    assert list(frame["text"]) == ["baseline wobble"]
    assert frame.iloc[0]["status"] == "open"
    assert frame.iloc[0]["detectors"] == ["V05268A"]

    # a fresh store on the same file sees the note (restart survival)
    again = NotesStore(path)
    assert list(again.all()["id"]) == [note["id"]]

    assert again.set_status(note["id"], "resolved")
    assert store.all().iloc[0]["status"] == "resolved"  # store 1 reloads
    assert again.delete(note["id"])
    assert store.all().empty
    assert not again.delete(note["id"])  # already gone


def test_store_sees_external_writes(tmp_path):
    path = tmp_path / "notes.json"
    store = NotesStore(path)
    external = [
        {
            "id": "abc",
            "created": "2026-09-03 00:00 UTC",
            "author": "elsewhere",
            "period": "p22",
            "run": "r000",
            "detectors": [],
            "seen_in": "",
            "text": "written by another process",
            "status": "open",
        }
    ]
    path.write_text(json.dumps(external))
    assert list(store.all()["author"]) == ["elsewhere"]


def test_unwritable_path_degrades_to_memory(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    store = NotesStore(blocker / "notes.json")  # parent is a file
    store.add(author="a", period="p22", run="r000", detectors=[], seen_in="", text="x")
    store.add(author="b", period="p22", run="r001", detectors=[], seen_in="", text="y")
    assert len(store.all()) == 2  # both kept in memory, no exception


def test_get_notes_store_is_shared_per_path(tmp_path):
    a = get_notes_store(tmp_path / "n.json")
    b = get_notes_store(tmp_path / "n.json")
    c = get_notes_store(tmp_path / "other.json")
    assert a is b
    assert a is not c


def _page(tmp_path):
    runs = {"r010": {"experiment": "l200", "timestamp": "20260101T000000Z"}}
    page = NotesPage.__new__(NotesPage)
    # bare param layer only: skip Monitoring.__init__'s tree discovery
    param.Parameterized.__init__(
        page, base_path=str(tmp_path), notes_path=str(tmp_path / "notes.json"),
        periods={"p22": runs}, run_dict=runs, period="p22", run="r010",
    )  # fmt: skip
    page.sort_obj = None
    page.store = notes_mod.get_notes_store(page.notes_path)
    return page


def test_page_table_and_resolved_filter(tmp_path):
    page = _page(tmp_path)
    page.store.add(
        author="gm", period="p22", run="r010",
        detectors=["V05268A", "P00574A"], seen_in="", text="open one",
    )  # fmt: skip
    done = page.store.add(
        author="gm", period="p22", run="r010",
        detectors=[], seen_in="", text="fixed one",
    )  # fmt: skip
    page.store.set_status(done["id"], "resolved")

    frame = page._table_frame(show_resolved=False)
    assert list(frame["Note"]) == ["open one"]
    assert frame.iloc[0]["Run"] == "p22 r010"
    assert frame.iloc[0]["Detectors"] == "V05268A, P00574A"
    both = page._table_frame(show_resolved=True)
    assert set(both["Note"]) == {"open one", "fixed one"}


def test_page_jump_guard(tmp_path):
    page = _page(tmp_path)
    page._jump_to("p99", "r000")  # unknown period: ignored
    assert (page.period, page.run) == ("p22", "r010")
    page._jump_to("p22", "r999")  # unknown run: ignored
    assert page.run == "r010"
    page._jump_to("p22", "r010")  # already there: no-op, no error
    assert (page.period, page.run) == ("p22", "r010")


def test_page_detector_names_tolerates_missing_tree(tmp_path):
    assert _page(tmp_path)._detector_names() == []


def test_reload_keeps_notes_on_malformed_file(tmp_path):
    path = tmp_path / "notes.json"
    store = NotesStore(path)
    store.add(author="a", period="p22", run="r000", detectors=[], seen_in="", text="x")
    path.write_text('{"oops": "a dict, not a list"}')
    assert list(store.all()["text"]) == ["x"]  # kept, not silently dropped


def test_table_blank_for_missing_detectors_field(tmp_path):
    page = _page(tmp_path)
    page.store.add(author="a", period="p22", run="r010", seen_in="", text="no dets")
    frame = page._table_frame(show_resolved=True)
    assert frame.iloc[0]["Detectors"] == ""  # not "nan"


def test_note_lookup_tolerates_deleted_id(tmp_path):
    page = _page(tmp_path)
    note = page.store.add(
        author="a", period="p22", run="r010", detectors=[], seen_in="", text="t"
    )
    assert page._note(note["id"])["text"] == "t"
    page.store.delete(note["id"])
    assert page._note(note["id"]) is None
