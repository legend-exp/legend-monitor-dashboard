"""MetaDB caching semantics (stubbed status DB, no metadata tree needed)."""

from __future__ import annotations

import threading

import pytest

from legenddashboard.metadata.meta_db import MetaDB


class _CountingStatuses:
    def __init__(self):
        self.calls = 0

    def on(self, tstamp, category="all"):
        self.calls += 1
        return {"V1": {"usability": "on"}, "tstamp": tstamp, "category": category}


class _StubStatusDB:
    def __init__(self):
        self.statuses = _CountingStatuses()


def _bare_metadb():
    db = MetaDB.__new__(MetaDB)
    db._lock = threading.Lock()
    db.version = 0
    db.status_db = _StubStatusDB()
    db._statuses_cache = {}
    return db


def test_statuses_on_is_cached_per_timestamp_and_category():
    db = _bare_metadb()
    first = db.statuses_on("20260101T000000Z", category="phy")
    again = db.statuses_on("20260101T000000Z", category="phy")
    assert again is first
    assert db.status_db.statuses.calls == 1
    db.statuses_on("20260101T000000Z", category="cal")
    db.statuses_on("20260102T000000Z", category="phy")
    assert db.status_db.statuses.calls == 3


def test_reload_drops_the_status_cache(monkeypatch):
    db = _bare_metadb()
    db.statuses_on("20260101T000000Z")
    stub = _StubStatusDB()
    # reload() rebuilds the DB objects; emulate it the way _load does
    monkeypatch.setattr(
        MetaDB,
        "_load",
        lambda self: (
            setattr(self, "status_db", stub),
            setattr(self, "_statuses_cache", {}),
        ),
    )
    db.reload()
    assert db.version == 1
    db.statuses_on("20260101T000000Z")
    assert stub.statuses.calls == 1  # recomputed, not served from the old cache


def test_get_meta_db_refuses_a_missing_clone(tmp_path):
    from legenddashboard.metadata.meta_db import get_meta_db

    # an empty dir must raise cleanly (-> editor alert), never reach
    # LegendMetadata, whose fallback is to clone the repo itself over SSH
    with pytest.raises(RuntimeError, match="no metadata clone"):
        get_meta_db(tmp_path)
