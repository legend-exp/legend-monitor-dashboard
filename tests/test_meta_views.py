"""Tests for the metadata figures (currently the data-blocks timeline)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from legenddashboard.metadata.meta_views import timeline_cells


class _StubViewer:
    """Just enough of a MetaDB for timeline_cells: runinfo + available_runs."""

    def __init__(self, runinfo):
        self.runinfo = runinfo

    def available_runs(self):
        return {p: sorted(runs) for p, runs in self.runinfo.items()}


def test_timeline_ends_come_from_runinfo():
    viewer = _StubViewer(
        {
            "p01": {
                # modern entry: explicit unix timestamps; livetime is shorter
                # than the wall-clock span and must NOT set the block width
                "r000": {
                    "phy": {
                        "start_key": "20230312T043356Z",
                        "end_key": "20230317T090516Z",
                        "start_timestamp": 1678595644.0,
                        "end_timestamp": 1679043972.0,
                        "livetime_in_s": 443740,
                    }
                },
                # end_key only (no end_timestamp)
                "r001": {
                    "cal": {
                        "start_key": "20230317T100000Z",
                        "end_key": "20230317T120000Z",
                    }
                },
                # legacy entry: no end fields -> livetime fallback
                "r002": {
                    "phy": {
                        "start_key": "20230318T000000Z",
                        "livetime_in_s": 3600,
                    }
                },
            }
        }
    )
    cells, lanes, colors = timeline_cells(viewer)
    assert lanes == ["p01"]
    by_label = {
        (label, dtype): (left, right)
        for label, dtype, left, right in zip(
            cells["label"], cells["datatype"], cells["left"], cells["right"]
        )
    }
    left, right = by_label[("p01 r000", "phy")]
    assert left == datetime.fromtimestamp(1678595644.0, UTC)
    assert right == datetime.fromtimestamp(1679043972.0, UTC)
    left, right = by_label[("p01 r001", "cal")]
    assert right == datetime(2023, 3, 17, 12, 0, tzinfo=UTC)
    left, right = by_label[("p01 r002", "phy")]
    assert right == left + timedelta(seconds=3600)
    assert cells["end"][0] == "2023-03-17 09:06 UTC"
