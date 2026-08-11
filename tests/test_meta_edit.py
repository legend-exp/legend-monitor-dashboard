from __future__ import annotations

from datetime import UTC, datetime

import pytest
from dbetto.catalog import Catalog

from legenddashboard.metadata import meta_edit
from legenddashboard.metadata.meta_views import expand_run_list

STATUS_FILE = """\
V00000A:
  reason: ""
  usability: "on"
  processable: true
  psd:
    is_bb_like: low_aoe & high_aoe
    status:
      low_aoe: valid
      high_aoe: valid
      lq: present # left over from p03
V00000B:
  reason: "noisy"
  usability: "ac"
  processable: true
"""

VALIDITY_FILE = """\
- valid_from: 20230311T235840Z
  apply:
    - l200-p03-r000-T%-all-config.yaml
  category:
    - cal
    - all
  mode: reset
- valid_from: 20240101T000000Z
  apply:
    - l200-p10-r000-T%-all-config.yaml
  category: all
  mode: reset
"""

CYCLES_FILE = """\
# DAQ cycles excluded from processing.
unprocessable:
  - l200-p02-r008-cal-20230111T203016Z # empty
  - l200-p03-r002-phy-20230327T145702Z # empty
"""

GROUPINGS_FILE = """\
default:
  calgroup001a:
    p03: r000..r005
B00000D:
  calgroup003a:
    p07: r001..r002
  calgroup003b:
    p07: r003..r008
"""


@pytest.fixture
def datasets(tmp_path):
    statuses = tmp_path / "statuses"
    statuses.mkdir()
    (statuses / "l200-p03-r000-T%-all-config.yaml").write_text(STATUS_FILE)
    (statuses / "validity.yaml").write_text(VALIDITY_FILE)
    (tmp_path / "ignored_daq_cycles.yaml").write_text(CYCLES_FILE)
    (tmp_path / "cal_groupings.yaml").write_text(GROUPINGS_FILE)
    return tmp_path


# -- statuses ---------------------------------------------------------------


def test_edit_existing_status_file_preserves_rest(datasets):
    path = meta_edit.stage_status_edit(
        datasets,
        "p03",
        "r000",
        "20230311T235840Z",
        edits={"V00000A": {"usability": "off", "reason": "sparking"}},
        datatype="both",
    )
    text = path.read_text()
    assert 'usability: "off"' in text
    assert 'reason: "sparking"' in text
    # untouched detector, psd block and its comment survive
    assert 'usability: "ac"' in text
    assert "lq: present # left over from p03" in text
    # existing file -> validity untouched
    assert (datasets / "statuses" / "validity.yaml").read_text() == VALIDITY_FILE


def test_psd_edit_only_touches_psd_block(datasets):
    path = meta_edit.stage_status_edit(
        datasets,
        "p03",
        "r000",
        "20230311T235840Z",
        edits={
            "V00000A": {"psd": {"is_bb_like": "low_aoe", "status": {"lq": "missing"}}}
        },
        datatype="both",
    )
    text = path.read_text()
    assert "is_bb_like: low_aoe\n" in text
    assert "lq: missing" in text
    assert 'usability: "on"' in text  # usability untouched
    # detector without a psd block gets one created
    meta_edit.stage_status_edit(
        datasets,
        "p03",
        "r000",
        "20230311T235840Z",
        edits={"V00000B": {"psd": {"status": {"ann": "valid"}}}},
        datatype="both",
    )
    assert "ann: valid" in path.read_text()


def test_create_status_file_registers_validity(datasets):
    seed = {
        "V00000A": {"reason": "", "usability": "on", "processable": True},
        "V00000B": {"reason": "noisy", "usability": "ac", "processable": True},
    }
    path = meta_edit.stage_status_edit(
        datasets,
        "p05",
        "r001",
        "20230501T000000Z",
        edits={"V00000B": {"usability": "off", "reason": "dead"}},
        datatype="cal",
        resolved_seed=seed,
    )
    assert path.name == "l200-p05-r001-T%-cal-config.yaml"
    text = path.read_text()
    assert 'usability: "off"' in text
    assert 'usability: "on"' in text  # seeded detector kept

    validity = datasets / "statuses" / "validity.yaml"
    entries = Catalog.read_from(str(validity))  # parse-back must still work
    assert entries is not None
    vtext = validity.read_text()
    # chronological: between the 2023 and 2024 entries
    assert vtext.index("20230311T235840Z") < vtext.index("20230501T000000Z")
    assert vtext.index("20230501T000000Z") < vtext.index("20240101T000000Z")
    assert "- cal\n" in vtext
    assert "- fft\n" in vtext
    assert "- pzc\n" in vtext
    # minimal diff: every original line survives byte-identically
    assert all(line in vtext.splitlines() for line in VALIDITY_FILE.splitlines())
    # top-level items stay at column 0, nested sequences at 4-space indent
    assert "\n- valid_from: 20230501T000000Z\n" in vtext
    assert "\n  apply:\n    - l200-p05-r001-T%-cal-config.yaml\n" in vtext

    # idempotent: creating again does not duplicate the validity entry
    meta_edit.update_validity(
        validity,
        {
            "valid_from": "20230501T000000Z",
            "category": ["cal", "fft", "pzc"],
            "mode": "reset",
            "apply": [path.name],
        },
    )
    assert validity.read_text().count("20230501T000000Z") == 1


def test_create_without_seed_raises(datasets):
    with pytest.raises(ValueError, match="no seed"):
        meta_edit.stage_status_edit(
            datasets, "p05", "r001", "20230501T000000Z", edits={}, datatype="both"
        )


def test_apply_status_range_writes_revert(datasets):
    seed = {"V00000A": {"reason": "", "usability": "on", "processable": True}}
    paths = meta_edit.apply_status_range(
        datasets,
        edits={"V00000A": {"usability": "ac", "reason": "test range"}},
        datatype="both",
        start_point=("p05", "r001", "20230501T000000Z", seed),
        revert_point=(
            "p05",
            "r003",
            "20230520T000000Z",
            seed,
            {"V00000A": {"usability": "on", "reason": ""}},
        ),
    )
    assert [p.name for p in paths] == [
        "l200-p05-r001-T%-all-config.yaml",
        "l200-p05-r003-T%-all-config.yaml",
    ]
    assert 'usability: "ac"' in paths[0].read_text()
    assert 'usability: "on"' in paths[1].read_text()
    vtext = (datasets / "statuses" / "validity.yaml").read_text()
    assert vtext.count("mode: reset") == 4  # 2 original + start + revert


# -- groupings --------------------------------------------------------------


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        (["r000", "r001", "r002", "r003"], "r000..r003"),
        (["r000", "r001"], ["r000", "r001"]),
        (["r004"], "r004"),
        (
            ["r000", "r001", "r002", "r004", "r006", "r007", "r008"],
            ["r000..r002", "r004", "r006..r008"],
        ),
    ],
)
def test_compress_runs_round_trips(runs, expected):
    compressed = meta_edit.compress_runs(runs)
    assert compressed == expected
    assert expand_run_list(compressed) == sorted(runs)


def test_split_group():
    block = {"calgroup003a": {"p07": "r001..r008", "p08": "r000..r002"}}
    out = meta_edit.split_group(block, "calgroup003a", "p07", "r003")
    assert out == {
        "calgroup003a": {"p07": ["r001", "r002"]},
        "calgroup003b": {"p07": "r003..r008", "p08": "r000..r002"},
    }


def test_split_group_skips_taken_letters():
    # the next letter is already used by another partition of this detector:
    # the split must pick the next free one, not clobber the existing block
    block = {
        "calgroup001a": {"p03": "r000..r005"},
        "calgroup001b": {"p04": "r000..r004"},
    }
    out = meta_edit.split_group(block, "calgroup001a", "p03", "r003")
    assert out == {
        "calgroup001a": {"p03": "r000..r002"},
        "calgroup001c": {"p03": "r003..r005"},
        "calgroup001b": {"p04": "r000..r004"},
    }


def test_set_detector_block_inserts_alphabetically(datasets):
    path = meta_edit.set_detector_block(
        datasets,
        "cal",
        "B00000A",
        {"calgroup001a": {"p03": "r000..r002"}, "calgroup001b": {"p03": "r003..r005"}},
    )
    text = path.read_text()
    assert text.index("default:") < text.index("B00000A:") < text.index("B00000D:")
    # existing override untouched
    assert "calgroup003b:" in text
    # removal
    meta_edit.set_detector_block(datasets, "cal", "B00000A", None)
    assert "B00000A" not in path.read_text()


def test_next_letter():
    assert meta_edit.next_letter("calgroup003a") == "calgroup003b"


def test_block_from_map():
    mapping = {
        ("p03", "r000"): "calgroup001a",
        ("p03", "r001"): "calgroup001a",
        ("p03", "r002"): "calgroup001a",
        ("p04", "r000"): "calgroup001b",
    }
    assert meta_edit.block_from_map(mapping) == {
        "calgroup001a": {"p03": "r000..r002"},
        "calgroup001b": {"p04": "r000"},
    }


def test_assign_partition_reassigns_and_covers_unassigned():
    groupings = {
        "default": {"calgroup001a": {"p03": "r000..r005"}},
    }
    # move the tail of p03 (incl. r006, previously unassigned) to a new group
    block = meta_edit.assign_partition(
        groupings,
        "V00000A",
        [("p03", "r004"), ("p03", "r005"), ("p03", "r006")],
        "calgroup001b",
    )
    assert block == {
        "calgroup001a": {"p03": "r000..r003"},
        "calgroup001b": {"p03": "r004..r006"},
    }


def test_assign_partition_drops_redundant_override():
    groupings = {
        "default": {"calgroup001a": {"p03": "r000..r005"}},
        "V00000A": {"calgroup001a": {"p03": "r000..r004"}},
    }
    # re-assigning r005 back to the default partition makes the override moot
    block = meta_edit.assign_partition(
        groupings, "V00000A", [("p03", "r005")], "calgroup001a"
    )
    assert block is None


def test_assign_partition_rejects_bad_name():
    with pytest.raises(ValueError, match="not a valid partition name"):
        meta_edit.assign_partition({"default": {}}, "V00000A", [], "foo")


def test_assign_partition_override_stays_minimal():
    # a late-join detector: only the touched partition/period appears in the
    # override, defaults for other periods are NOT materialized
    groupings = {
        "default": {
            "calgroup001a": {"p03": "r000..r005"},
            "calgroup008a": {"p16": "r000..r003"},
        }
    }
    block = meta_edit.assign_partition(
        groupings, "V14654A", [("p16", "r002"), ("p16", "r003")], "calgroup008b"
    )
    assert block == {
        "calgroup008a": {"p16": ["r000", "r001"]},
        "calgroup008b": {"p16": ["r002", "r003"]},
    }
    assert "calgroup001a" not in block


def test_assign_in_block_edits_the_default_block_itself():
    groupings = {
        "default": {
            "calgroup001a": {"p03": "r000..r005"},
            "calgroup002a": {"p06": "r000..r002"},
        }
    }
    block = meta_edit.assign_in_block(
        groupings,
        "default",
        [("p03", "r004"), ("p03", "r005"), ("p04", "r000")],
        "calgroup001b",
    )
    assert block == {
        "calgroup001a": {"p03": "r000..r003"},
        "calgroup001b": {"p03": ["r004", "r005"], "p04": "r000"},
        "calgroup002a": {"p06": "r000..r002"},  # untouched, original notation
    }


def test_assign_in_block_dispatches_to_detector_override():
    groupings = {"default": {"calgroup001a": {"p03": "r000..r005"}}}
    block = meta_edit.assign_in_block(
        groupings, "V00000A", [("p03", "r005")], "calgroup001b"
    )
    assert block == {
        "calgroup001a": {"p03": "r000..r004"},
        "calgroup001b": {"p03": "r005"},
    }


def test_assign_partition_masks_fully_vacated_default_period():
    groupings = {"default": {"calgroup001a": {"p03": ["r000", "r001"]}}}
    block = meta_edit.assign_partition(
        groupings, "V00000A", [("p03", "r000"), ("p03", "r001")], "calgroup001b"
    )
    # the vacated default period is masked with an empty list, not dropped
    assert block == {
        "calgroup001a": {"p03": []},
        "calgroup001b": {"p03": ["r000", "r001"]},
    }


# -- bad cycles -------------------------------------------------------------


def test_add_ignored_cycles_preserves_existing_comments(datasets):
    added = meta_edit.add_ignored_cycles(
        datasets, "l200-p13-r002-ath-20241216T190730Z", "less than 100 events"
    )
    assert added == ["l200-p13-r002-ath-20241216T190730Z"]
    text = (datasets / "ignored_daq_cycles.yaml").read_text()
    assert "l200-p13-r002-ath-20241216T190730Z # less than 100 events" in text
    assert "l200-p02-r008-cal-20230111T203016Z # empty" in text
    assert text.startswith("# DAQ cycles excluded from processing.")

    # duplicates skipped, invalid ids rejected
    assert (
        meta_edit.add_ignored_cycles(
            datasets, "l200-p13-r002-ath-20241216T190730Z", "again"
        )
        == []
    )
    with pytest.raises(ValueError, match="not a valid cycle id"):
        meta_edit.add_ignored_cycles(datasets, "not-a-cycle", "x")


def test_list_and_remove_ignored_cycles(datasets):
    cycles = meta_edit.list_ignored_cycles(datasets)
    assert cycles[0] == ("l200-p02-r008-cal-20230111T203016Z", "empty")
    assert meta_edit.remove_ignored_cycle(
        datasets, "l200-p03-r002-phy-20230327T145702Z"
    )
    assert len(meta_edit.list_ignored_cycles(datasets)) == 1
    assert not meta_edit.remove_ignored_cycle(
        datasets, "l200-p99-r000-phy-20990101T000000Z"
    )


@pytest.mark.parametrize(
    ("name", "key", "expected"),
    [
        ("12a", "cal", "calgroup012a"),
        ("12a", "escale", "calgroup012a"),
        ("3b", "phy", "phygroup003b"),
        ("calgroup012a", "cal", "calgroup012a"),
        ("  CALGROUP012A ", "cal", "calgroup012a"),
    ],
)
def test_normalize_partition(name, key, expected):
    assert meta_edit.normalize_partition(name, key) == expected


def test_normalize_partition_rejects_garbage():
    with pytest.raises(ValueError, match="not a valid partition name"):
        meta_edit.normalize_partition("group12", "cal")


def test_raw_run_catalogue(tmp_path):
    raw = tmp_path / "raw"
    for period, run, dtype, ts in [
        ("p14", "r001", "phy", "20250502T110000Z"),
        ("p14", "r001", "cal", "20250502T100000Z"),
        ("p14", "r002", "phy", "20250507T140000Z"),
        ("p03", "r000", "phy", "20230312T043356Z"),
    ]:
        d = raw / dtype / period / run
        d.mkdir(parents=True)
        (d / f"l200-{period}-{run}-{dtype}-{ts}-tier_raw.lh5").touch()

    cat = meta_edit.raw_run_catalogue([raw, tmp_path / "missing"])
    assert list(cat) == ["p03", "p14"]
    assert list(cat["p14"]) == ["r001", "r002"]
    # cycles sorted by timestamp: cal at 10:00 before phy at 11:00
    assert cat["p14"]["r001"] == [
        "l200-p14-r001-cal-20250502T100000Z",
        "l200-p14-r001-phy-20250502T110000Z",
    ]


def test_find_cycles_in_range(tmp_path):
    raw = tmp_path / "raw"
    for period, run, dtype, ts in [
        ("p14", "r001", "phy", "20250502T110000Z"),
        ("p14", "r001", "bkg", "20250502T115229Z"),
        ("p14", "r002", "phy", "20250507T140000Z"),
    ]:
        d = raw / dtype / period / run
        d.mkdir(parents=True)
        (d / f"l200-{period}-{run}-{dtype}-{ts}-tier_raw.lh5").touch()

    found = meta_edit.find_cycles_in_range(
        [raw, tmp_path / "missing"],
        datetime(2025, 5, 2, tzinfo=UTC),
        datetime(2025, 5, 3, tzinfo=UTC),
    )
    assert found == [
        "l200-p14-r001-phy-20250502T110000Z",
        "l200-p14-r001-bkg-20250502T115229Z",
    ]
    only_phy = meta_edit.find_cycles_in_range(
        [raw],
        datetime(2025, 5, 1, tzinfo=UTC),
        datetime(2025, 5, 10, tzinfo=UTC),
        datatypes=["phy"],
    )
    assert only_phy == [
        "l200-p14-r001-phy-20250502T110000Z",
        "l200-p14-r002-phy-20250507T140000Z",
    ]
