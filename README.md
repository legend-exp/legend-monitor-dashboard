# LEGEND monitoring dashboard

A [Panel](https://panel.holoviz.org/)-based monitoring dashboard for
LEGEND-200 data production, served to multiple users with independent
sessions. It runs in production inside a lightweight Docker image on
[NERSC Spin](https://docs.nersc.gov/services/spin/) and can be run locally
against any production cycle directory.

## Setup

### Requirements

- Python >= 3.11 and `git` on the `PATH` (the metadata pages open git
  repositories).
- A **production cycle directory** — a directory containing
  `dataflow-config.yaml`, the `inputs/` metadata checkout and the
  `generated/` tiers. Every page derives its data paths from it.
- (Metadata Editor only) network access to clone
  [legend-metadata](https://github.com/legend-exp/legend-metadata), or a
  local repository to clone from.

### Install

Using [uv](https://docs.astral.sh/uv/) (recommended — commands below use
`uv run`, which resolves the environment automatically):

```sh
git clone https://github.com/legend-exp/legend-monitor-dashboard
cd legend-monitor-dashboard
```

or install a release from PyPI into any environment:

```sh
pip install legend_dashboard
```

### Configuration

All paths live in `dashboard-config.yaml` at the project root:

```yaml
paths:
  base: .. # production cycle (period/run discovery)
  cal: .. # production cycle used by the cal pages
  phy: /path/to/phy # physics monitoring data
  muon: /path/to/muon # muon monitoring data
  llama: /path/to/llama # llama DAQ data
  tmp: /tmp # writable dir (CSV downloads, caches)
  # editable legend-metadata clone for the Metadata Editor page; cloned /
  # updated automatically on startup (see METADATA_EDIT_URL below)
  metadata_edit: ../metadata-edit
```

`base`/`cal` must point at a production cycle directory. Paths for disabled
pages can stay as placeholders. If `metadata_edit` is omitted, the Metadata
Editor page is simply not built.

## Running the dashboard

```sh
uv run dashboard dashboard-config.yaml -p 9009
```

then open `http://localhost:9009` (forward the port when running remotely).

Useful options:

| option                        | effect                                                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `-p/--port`                   | port to serve on (default 9000)                                                                                                             |
| `-d/--disable-page ...`       | skip pages: `cal` `phy` `spm` `muon` `llama` `meta` `metaedit`                                                                              |
| `--num-threads N`             | server thread pool (default 4)                                                                                                              |
| `--websocket-origin HOST ...` | allowed websocket origin(s); **required behind a reverse proxy** (use the public hostname, or `'*'` to disable the check for local testing) |
| `-w/--widget-widths N`        | widget width tuning                                                                                                                         |

The production Spin deployment runs
`dashboard ./dashboard-config.yaml -p 5000 --num-threads 4 -d spm muon llama phy`,
serving the calibration pages, the read-only MetaData page, the Metadata
Editor and the Information page.

### Authentication (optional)

Set environment variables to put the dashboard behind a shared-password
login (on Spin these come from secrets):

- `DASHBOARD_PASSWORD` — enables the login page.
- `DASHBOARD_USERNAME` — if set, only this username/password pair is
  accepted; otherwise any username with the matching password works.
- `DASHBOARD_COOKIE_SECRET` — signs the login cookie; set it for stable
  sessions across restarts/replicas (an ephemeral one is generated
  otherwise).

Without `DASHBOARD_PASSWORD` the dashboard is served unauthenticated.

### Individual components

Each page group also has a standalone entry point:

```sh
uv run dashboard-cal      dashboard-config.yaml -p 9009
uv run dashboard-phy      dashboard-config.yaml -p 9009
uv run dashboard-muon     dashboard-config.yaml -p 9009
uv run dashboard-llama    dashboard-config.yaml -p 9009
uv run dashboard-meta     dashboard-config.yaml -p 9009   # read-only metadata pages
uv run dashboard-metaedit dashboard-config.yaml -p 9009   # Metadata Editor only
```

`dashboard-metaedit` serves just the Metadata Editor (with the period/run
sidebar) and also accepts `--websocket-origin`. It still needs `paths: cal`
(production cycle) and `paths: metadata_edit` in the config.

## The Metadata Editor

The Metadata Editor page views and edits
[legend-metadata](https://github.com/legend-exp/legend-metadata) — detector
statuses (usability and PSD), analysis partitions (groupings), run lists and
ignored DAQ cycles — in a **dedicated editable clone** (`paths:
metadata_edit`), separate from the production cycle's read-only copy.

- **Startup**: the clone is created or fast-forwarded automatically before
  serving. `METADATA_EDIT_URL` overrides the upstream URL (defaults to the
  legend-exp repository; SSH submodule URLs are rewritten to HTTPS
  automatically).
- **Workspaces**: the page is read-only until you open a _workspace_ (your
  GitHub username). Each workspace is a git worktree of the `datasets`
  submodule, so staged edits are isolated per user and survive page reloads —
  re-open the same name to reattach. Un-pushed edits also survive server
  restarts (clean workspaces are pruned at startup, dirty ones kept), but not
  the loss of the container filesystem — push finished work.
- **Commit & Push**: commits your workspace's changes and pushes them as a
  `metaedit/<user>/<timestamp>-<suffix>` branch to **your fork** of
  [legend-datasets](https://github.com/legend-exp/legend-datasets) (users
  cannot push to legend-exp directly). You need a fork and a token with
  write access to it; the token is used for a single push and never stored.
  The success message links to the ready-made pull-request page.

## Docker / Spin

The production image is built from the `Dockerfile` at the repo root. It
installs a pinned `legend_dashboard` release from PyPI (local changes need a
release + pin bump to reach production), generates the container's
`dashboard-config.yaml`, and expects the production cycle to be mounted at
`/srv/tmp-auto`. Relevant environment variables in the deployment:
`WEBSOCKET_ORIGIN` (public hostname), the authentication variables above,
and `METADATA_EDIT_URL`. The editable metadata clone and the per-user
workspaces live under the container user's home directory and are ephemeral
per pod.

## Developing

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

- Code style: `pre-commit run -a` (ruff + formatting; also runs in CI).
- Tests: `pytest` (`tests/` covers the metadata editing layer with pure
  YAML round-trip tests and the git/workspace layer with real-git
  integration tests in temporary repositories).
- The dashboard classes all expose `build_*_pane(s)` functions returning
  Panel panes, so pages can be displayed in a Jupyter notebook; pass
  `notebook=True` when constructing them to make displaying work better.
- A mock production tree (for exercising the dashboard without real data)
  is expected as the parent directory in the default `dashboard-config.yaml`
  (`base: ..`); point `metadata_edit` at a scratch clone when testing the
  editor, never at the production `inputs/`.

Releases are published to PyPI by creating a GitHub Release
(`.github/workflows/distribute.yml`).
