"""Git layer for the editable legend-metadata clone and per-user workspaces.

The editable files all live in the ``datasets/`` submodule (legend-datasets).
Each user edits in their own **workspace** -- a git worktree of that submodule
at ``<clone>-workspaces/<name>`` on branch ``workspace/<name>`` -- so staged
edits are private until pushed and survive page reloads (re-opening the same
name reattaches). The superproject and the hardware submodules (channelmaps)
stay shared read-only.

Users cannot push to legend-exp upstream, so pushes go to the user's own fork
over HTTPS with a one-shot token (via ``GIT_ASKPASS`` -- never on disk, argv,
or in a stored remote URL) as a remote-only ``metaedit/<user>/<timestamp>``
branch. The local workspace branch namespace (``workspace/*``) is deliberately
disjoint from the push namespace: ``refs/heads/metaedit/<user>`` could not
coexist with ``refs/heads/metaedit/<user>/<ts>``.

One process-wide lock serialises every git operation (all worktrees share one
object store).
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_URL = "https://github.com/legend-exp/legend-metadata"
#: Upstream repo of the datasets submodule (fork target for pushes).
DATASETS_REPO = "legend-datasets"
DATASETS_UPSTREAM = "legend-exp"
#: Submodules the editor needs (statuses/groupings + channelmaps/detectors
#: for the viewer); the rest of legend-metadata's submodules stay
#: uninitialised.
EDIT_SUBMODULES = ("datasets", "hardware/configuration", "hardware/detectors")

WORKSPACES_SUFFIX = "-workspaces"
#: GitHub-username shape; also a valid single git ref component, and immune to
#: path traversal (no separators, no leading '-').
_WS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")

_GIT_LOCK = threading.Lock()


class GitError(RuntimeError):
    """A git command failed; message carries the (token-redacted) stderr."""


def _https(url: str) -> str:
    """Rewrite an SSH github URL to HTTPS; anything else passes through."""
    url = url.strip()
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    return url


def _git(
    repo: str | Path, *args: str, env: dict | None = None, ok_codes: tuple = (0,)
) -> str:
    """Run one git command in ``repo``; raise :class:`GitError` on failure."""
    cmd = ["git", "-C", str(repo), *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode not in ok_codes:
        msg = f"`git {' '.join(args[:2])}` failed: {result.stderr.strip()}"
        raise GitError(msg)
    return result.stdout


def _submodule_url(path: Path, name: str) -> str:
    url = _git(path, "config", "-f", ".gitmodules", f"submodule.{name}.url").strip()
    return _https(url)


# ---------------------------------------------------------------------------
# clone setup (startup)
# ---------------------------------------------------------------------------


def ensure_clone(path: str | Path, url: str | None = None) -> bool:
    """Clone the editable metadata repo, or update it, on startup.

    Also prunes stale worktree registrations and removes *clean* workspaces
    (dirty ones -- un-pushed edits -- are kept). Never raises: on failure the
    error is logged and ``False`` returned, so a broken clone disables the
    editor page instead of crashing every session.
    """
    path = Path(path)
    url = _https(url or DEFAULT_URL)
    with _GIT_LOCK:
        try:
            if not (path / ".git").exists():
                log.info("cloning metadata edit repo %s -> %s", url, path)
                path.parent.mkdir(parents=True, exist_ok=True)
                _git(
                    path.parent,
                    "clone",
                    "--filter=blob:none",
                    url,
                    str(path),
                )
                _init_submodules(path)
            elif _git(path / "datasets", "status", "--porcelain").strip():
                # pre-workspace leftovers in the shared tree: keep them visible
                log.warning(
                    "shared datasets tree in %s has un-pushed edits; " "skipping pull",
                    path,
                )
            else:
                _git(path, "pull", "--ff-only")
                _init_submodules(path)  # picks up newly added submodules too
            _cleanup_workspaces(path)
        except (GitError, OSError):
            log.exception("could not set up the metadata edit clone at %s", path)
            return False
    return True


def _init_submodules(path: Path) -> None:
    """Init/update the needed submodules with SSH->HTTPS rewritten URLs."""
    declared = _git(
        path,
        "config",
        "-f",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
        ok_codes=(0, 1),
    )
    declared_paths = {line.split()[-1] for line in declared.splitlines()}
    wanted = [n for n in EDIT_SUBMODULES if n in declared_paths]
    if "datasets" not in wanted:
        msg = "the metadata repo has no `datasets` submodule"
        raise GitError(msg)
    local = False
    for name in wanted:
        _git(path, "submodule", "init", "--", name)
        url = _submodule_url(path, name)
        _git(path, "config", f"submodule.{name}.url", url)
        local = local or "://" not in url
    # local-path submodule URLs (test clones of a local mock tree) need the
    # file protocol re-enabled; never needed for the real https URLs
    protocol = ["-c", "protocol.file.allow=always"] if local else []
    _git(path, *protocol, "submodule", "update", "--", *wanted)
    # workspace worktrees branch off origin/<default>: make sure the main
    # datasets checkout is on a local default branch, not the detached HEAD
    # `submodule update` leaves behind
    datasets = path / "datasets"
    head = _git(datasets, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if head == "HEAD":
        branch = _default_branch(datasets)
        _git(datasets, "checkout", branch)
        _git(datasets, "pull", "--ff-only")


def _cleanup_workspaces(path: Path) -> None:
    """Prune worktrees, drop clean workspaces, delete relic push branches.

    Called under ``_GIT_LOCK``; logs instead of raising (startup must not
    break on a mangled workspace).
    """
    base = path / "datasets"
    try:
        _git(base, "worktree", "prune")
    except GitError:
        log.warning("could not prune datasets worktrees", exc_info=True)
        return
    root = workspaces_root(path)
    if root.is_dir():
        for ws in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                dirty = bool(_git(ws, "status", "--porcelain").strip())
            except GitError:
                dirty = False  # broken tree -> removable
            if dirty:
                log.info("keeping workspace with un-pushed edits: %s", ws)
                continue
            log.info("removing clean workspace %s", ws)
            try:
                _git(base, "worktree", "remove", "--force", str(ws))
            except GitError:
                shutil.rmtree(ws, ignore_errors=True)
                _git(base, "worktree", "prune", ok_codes=(0, 1))
            _git(base, "branch", "-D", f"workspace/{ws.name}", ok_codes=(0, 1))
    # relic local metaedit/* branches from the old (pre-workspace) push flow;
    # they would collide with nothing now but are dead weight
    relics = _git(
        base, "for-each-ref", "--format=%(refname:short)", "refs/heads/metaedit"
    ).splitlines()
    for branch in relics:
        _git(base, "branch", "-D", branch, ok_codes=(0, 1))


# ---------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------


def workspaces_root(meta_path: str | Path) -> Path:
    """Directory holding the per-user worktrees (sibling of the clone)."""
    p = Path(meta_path)
    return p.parent / (p.name + WORKSPACES_SUFFIX)


def _check_workspace_name(name: str) -> str:
    name = name.strip()
    if not _WS_NAME_RE.match(name):
        msg = (
            f"invalid workspace name {name!r}: use your GitHub username "
            "(letters, digits and dashes, not starting with a dash)"
        )
        raise GitError(msg)
    return name


def _default_branch(datasets: Path) -> str:
    """Default branch of the datasets repo ("main" in practice)."""
    remote_head = _git(
        datasets, "rev-parse", "--abbrev-ref", "origin/HEAD", ok_codes=(0, 128)
    ).strip()
    if remote_head and "/" in remote_head:
        return remote_head.split("/", 1)[1]
    head = _git(datasets, "rev-parse", "--abbrev-ref", "HEAD").strip()
    return head if head != "HEAD" else "main"


def _branch_exists(datasets: Path, branch: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(datasets),
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _fetch_tolerant(repo: Path) -> None:
    """Fetch origin, tolerating failure (offline dev must not break opening)."""
    try:
        _git(repo, "fetch", "origin")
    except GitError:
        log.warning("could not fetch origin in %s; using stale refs", repo)


def ensure_workspace(meta_path: str | Path, name: str) -> Path:
    """Create or reattach the per-user worktree of the datasets submodule.

    A dirty workspace is returned untouched (un-pushed edits survive
    reloads and restarts); a clean one is reset to the upstream default
    branch so reattaching always starts from the current tip.
    """
    meta_path = Path(meta_path)
    name = _check_workspace_name(name)
    base = meta_path / "datasets"
    ws = workspaces_root(meta_path) / name
    branch = f"workspace/{name}"

    with _GIT_LOCK:
        if ws.exists():
            healthy = (ws / ".git").exists()
            if healthy:
                try:
                    dirty = bool(_git(ws, "status", "--porcelain").strip())
                except GitError:
                    healthy = False
            if healthy:
                if not dirty:
                    _fetch_tolerant(ws)
                    default = _default_branch(base)
                    _git(ws, "reset", "--hard", f"origin/{default}")
                    _git(ws, "clean", "-fd")
                return ws.resolve()
            log.warning("workspace %s is broken; recreating it", ws)
            shutil.rmtree(ws, ignore_errors=True)

        _git(base, "worktree", "prune")
        ws.parent.mkdir(parents=True, exist_ok=True)
        _fetch_tolerant(base)
        default = _default_branch(base)
        if _branch_exists(base, branch):
            # leftover branch from a removed worktree: reuse it, then make its
            # state deterministic
            _git(base, "worktree", "add", str(ws), branch)
            _git(ws, "reset", "--hard", f"origin/{default}")
        else:
            _git(base, "worktree", "add", "-b", branch, str(ws), f"origin/{default}")
    return ws.resolve()


# ---------------------------------------------------------------------------
# working-tree state (workspace dir or <clone>/datasets)
# ---------------------------------------------------------------------------


def status(datasets_path: str | Path) -> list[tuple[str, str]]:
    """``[(state, file)]`` of pending changes in a datasets tree."""
    with _GIT_LOCK:
        out = _git(datasets_path, "status", "--porcelain")
    entries = []
    for line in out.splitlines():
        entries.append((line[:2].strip() or "??", line[3:]))
    return entries


def diff(datasets_path: str | Path) -> str:
    """Unified diff of the pending changes in a datasets tree."""
    with _GIT_LOCK:
        tracked = _git(datasets_path, "diff")
        untracked = _git(
            datasets_path, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        for f in untracked:
            # --no-index exits 1 when the files differ; that's the point here
            tracked += _git(
                datasets_path,
                "diff",
                "--no-index",
                "--",
                "/dev/null",
                f"./{f}",
                ok_codes=(0, 1),
            )
    return tracked


def discard_all(datasets_path: str | Path) -> None:
    """Throw away every pending change in a datasets tree."""
    with _GIT_LOCK:
        _git(datasets_path, "checkout", "--", ".")
        _git(datasets_path, "clean", "-fd")


# ---------------------------------------------------------------------------
# commit & push
# ---------------------------------------------------------------------------


def _redact(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def commit_and_push(
    datasets_path: str | Path,
    message: str,
    username: str,
    token: str,
    fork_url: str | None = None,
) -> str:
    """Commit a workspace's pending changes and push them to the user's fork.

    Returns the GitHub compare URL to open a pull request. The commit is made
    on the workspace branch and pushed as a remote-only
    ``metaedit/<user>/<timestamp>`` branch to
    ``https://github.com/<user>/legend-datasets``. On success the workspace is
    reset to the upstream default branch (the edits now live only on the
    pushed branch); on push failure the commit is soft-reset so the changes
    come straight back as pending edits -- the workspace never accumulates
    committed-but-unpushed state.

    ``fork_url`` overrides the fork remote (used by tests).
    """
    datasets_path = Path(datasets_path)
    push_branch = f"metaedit/{username}/{time.strftime('%Y%m%d-%H%M%S')}"
    url = fork_url or f"https://github.com/{username}/{DATASETS_REPO}.git"
    askpass = (
        importlib.resources.files("legenddashboard") / "metadata" / "git_askpass.sh"
    )
    env = {
        **os.environ,
        "GIT_ASKPASS": str(askpass),
        "GIT_USERNAME": username,
        "GIT_PASSWORD": token,
        "GIT_TERMINAL_PROMPT": "0",
    }

    with _GIT_LOCK:
        if not _git(datasets_path, "status", "--porcelain").strip():
            msg = "no pending changes to commit"
            raise GitError(msg)
        default = _default_branch(datasets_path)
        _git(datasets_path, "add", "-A")
        try:
            _git(
                datasets_path,
                "-c",
                f"user.name={username} via dashboard",
                "-c",
                f"user.email={username}@users.noreply.github.com",
                "commit",
                "-m",
                message,
            )
        except GitError:
            _git(datasets_path, "reset", ok_codes=(0, 1))  # unstage, keep edits
            raise
        try:
            _git(datasets_path, "push", url, f"HEAD:refs/heads/{push_branch}", env=env)
        except GitError as exc:
            # bring the edits back as pending working-tree changes
            _git(datasets_path, "reset", "--soft", "HEAD~1")
            _git(datasets_path, "reset", ok_codes=(0, 1))
            msg = (
                f"{_redact(str(exc), token)} — your changes are still pending "
                "in your workspace"
            )
            raise GitError(msg) from None
        _git(datasets_path, "reset", "--hard", f"origin/{default}")

    return (
        f"https://github.com/{DATASETS_UPSTREAM}/{DATASETS_REPO}/compare/"
        f"{default}...{username}:{DATASETS_REPO}:{push_branch}?expand=1"
    )
