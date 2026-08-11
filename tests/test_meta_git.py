"""Integration tests for the workspace git layer (real git, no network)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from legenddashboard.metadata import meta_edit, meta_git

CYCLES_FILE = """\
unprocessable:
  - l200-p02-r008-cal-20230111T203016Z # empty
"""


def _run(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path, default_branch="main"):
    path.mkdir(parents=True, exist_ok=True)
    _run(path.parent, "init", "-q", "-b", default_branch, str(path))
    _run(path, "config", "user.name", "tester")
    _run(path, "config", "user.email", "tester@example.com")


@pytest.fixture
def clone(tmp_path):
    """A real clone (superproject + datasets submodule) via ensure_clone."""
    upstream_ds = tmp_path / "origin" / "datasets"
    _init_repo(upstream_ds)
    (upstream_ds / "ignored_daq_cycles.yaml").write_text(CYCLES_FILE)
    (upstream_ds / "statuses").mkdir()
    (upstream_ds / "statuses" / "validity.yaml").write_text("[]\n")
    _run(upstream_ds, "add", "-A")
    _run(upstream_ds, "commit", "-q", "-m", "seed")

    superproject = tmp_path / "origin" / "metadata"
    _init_repo(superproject)
    _run(
        superproject,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(upstream_ds),
        "datasets",
    )
    _run(superproject, "commit", "-q", "-m", "add datasets submodule")

    clone_path = tmp_path / "clone"
    assert meta_git.ensure_clone(clone_path, str(superproject))
    # local file remotes have no origin/HEAD by default; set it so
    # _default_branch resolves the same way it does for real clones
    _run(clone_path / "datasets", "remote", "set-head", "origin", "-a")
    return clone_path


def _commit_upstream_file(clone, fname, content="x\n"):
    """Add a commit to the upstream datasets repo (the clone's origin)."""
    upstream = clone.parent / "origin" / "datasets"
    (upstream / fname).write_text(content)
    _run(upstream, "add", "-A")
    _run(upstream, "commit", "-q", "-m", f"add {fname}")


def test_ensure_workspace_creates_worktree(clone):
    ws = meta_git.ensure_workspace(clone, "alice")
    assert ws == (clone.parent / "clone-workspaces" / "alice").resolve()
    # linked worktree of the gitfile submodule: .git is a file, metadata
    # lives under the superproject's modules dir
    assert (ws / ".git").is_file()
    assert (clone / ".git" / "modules" / "datasets" / "worktrees" / "alice").is_dir()
    head = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == "workspace/alice"
    assert (ws / "ignored_daq_cycles.yaml").exists()


def test_workspace_isolated(clone):
    alice = meta_git.ensure_workspace(clone, "alice")
    bob = meta_git.ensure_workspace(clone, "bob")
    meta_edit.add_ignored_cycles(alice, "l200-p13-r002-ath-20241216T190730Z", "test")
    assert meta_git.status(alice) == [("M", "ignored_daq_cycles.yaml")]
    assert meta_git.status(clone / "datasets") == []
    assert meta_git.status(bob) == []


def test_reattach_preserves_dirty(clone):
    ws = meta_git.ensure_workspace(clone, "alice")
    meta_edit.add_ignored_cycles(ws, "l200-p13-r002-ath-20241216T190730Z", "keep")
    again = meta_git.ensure_workspace(clone, "alice")
    assert again == ws
    assert meta_git.status(ws) != []


def test_reattach_clean_resets_to_upstream(clone):
    ws = meta_git.ensure_workspace(clone, "alice")
    _commit_upstream_file(clone, "new_file.yaml")
    assert not (ws / "new_file.yaml").exists()
    meta_git.ensure_workspace(clone, "alice")
    assert (ws / "new_file.yaml").exists()


def test_recreate_after_dir_deleted(clone):
    ws = meta_git.ensure_workspace(clone, "alice")
    shutil.rmtree(ws)
    again = meta_git.ensure_workspace(clone, "alice")
    assert again == ws
    assert meta_git.status(ws) == []
    assert (ws / "ignored_daq_cycles.yaml").exists()


@pytest.mark.parametrize("name", ["", "../x", "a b", "-alice", "x" * 40, "a/b", "a..b"])
def test_workspace_name_validation(clone, name):
    with pytest.raises(meta_git.GitError, match="invalid workspace name"):
        meta_git.ensure_workspace(clone, name)


def test_commit_and_push_from_workspace(clone, tmp_path):
    fork = tmp_path / "fork.git"
    _run(tmp_path, "init", "-q", "--bare", str(fork))
    ws = meta_git.ensure_workspace(clone, "alice")
    meta_edit.add_ignored_cycles(ws, "l200-p13-r002-ath-20241216T190730Z", "bad")

    url = meta_git.commit_and_push(ws, "ignore a cycle", "alice", "tok", str(fork))
    assert "compare/main...alice:legend-datasets:metaedit/alice/" in url

    refs = subprocess.run(
        ["git", "-C", str(fork), "for-each-ref", "--format=%(refname)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(refs) == 1
    assert refs[0].startswith("refs/heads/metaedit/alice/")

    # workspace back to a clean upstream state, still on its own branch
    assert meta_git.status(ws) == []
    head = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == "workspace/alice"


def test_push_failure_soft_resets(clone, tmp_path):
    ws = meta_git.ensure_workspace(clone, "alice")
    meta_edit.add_ignored_cycles(ws, "l200-p13-r002-ath-20241216T190730Z", "bad")
    with pytest.raises(meta_git.GitError, match="still pending"):
        meta_git.commit_and_push(
            ws, "msg", "alice", "tok", str(tmp_path / "missing-fork")
        )
    # the edit is back as a pending working-tree change, no stray commit
    assert meta_git.status(ws) == [("M", "ignored_daq_cycles.yaml")]
    head = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    upstream = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == upstream


def test_commit_and_push_nothing_pending(clone, tmp_path):
    ws = meta_git.ensure_workspace(clone, "alice")
    with pytest.raises(meta_git.GitError, match="no pending changes"):
        meta_git.commit_and_push(ws, "msg", "alice", "tok", str(tmp_path / "f"))


def test_startup_cleanup(clone):
    alice = meta_git.ensure_workspace(clone, "alice")
    bob = meta_git.ensure_workspace(clone, "bob")
    meta_edit.add_ignored_cycles(bob, "l200-p13-r002-ath-20241216T190730Z", "keep")

    assert meta_git.ensure_clone(clone)  # rerun startup
    assert not alice.exists()  # clean workspace removed
    assert not meta_git._branch_exists(clone / "datasets", "workspace/alice")
    assert bob.exists()  # dirty workspace kept
    assert meta_git.status(bob) != []


def test_status_diff_discard_on_workspace(clone):
    ws = meta_git.ensure_workspace(clone, "alice")
    meta_edit.add_ignored_cycles(ws, "l200-p13-r002-ath-20241216T190730Z", "bad")
    (ws / "statuses" / "new-file.yaml").write_text("A: 1\n")

    d = meta_git.diff(ws)
    assert "+  - l200-p13-r002-ath-20241216T190730Z # bad" in d
    assert "new-file.yaml" in d  # untracked files show via --no-index

    meta_git.discard_all(ws)
    assert meta_git.status(ws) == []
    assert not (ws / "statuses" / "new-file.yaml").exists()


def test_branch_collision_with_old_push_branches(clone):
    # relic of the old failure path: a local metaedit/<user>/<ts> branch
    _run(clone / "datasets", "branch", "metaedit/alice/20250101-000000")
    # the workspace/<name> namespace avoids the D/F collision entirely
    ws = meta_git.ensure_workspace(clone, "alice")
    assert (ws / ".git").is_file()
    # and startup cleanup deletes the relic
    assert meta_git.ensure_clone(clone)
    assert not meta_git._branch_exists(
        clone / "datasets", "metaedit/alice/20250101-000000"
    )


def test_ensure_clone_recovers_uninitialised_submodule(clone):
    # An interrupted first startup can leave the superproject cloned but the
    # datasets submodule absent. The porcelain check then runs `git -C` on a
    # path that does not exist, which raises and is swallowed by the outer
    # handler -- and because `.git` still exists the same branch is taken on
    # every restart, so the page stayed disabled forever. ensure_clone must
    # re-initialise instead.
    # (An *empty* datasets/ dir already recovered: git reports the
    # uninitialised submodule as clean, so the pull + submodule-init ran.)
    shutil.rmtree(clone / "datasets")
    assert not (clone / "datasets").exists()

    assert meta_git.ensure_clone(clone) is True

    assert (clone / "datasets" / ".git").exists()
    assert (clone / "datasets" / "ignored_daq_cycles.yaml").exists()
    # and the recovered clone is usable
    ws = meta_git.ensure_workspace(clone, "alice")
    assert (ws / "ignored_daq_cycles.yaml").exists()
