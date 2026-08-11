FROM python:3.11-slim

LABEL maintainer.name="George Marshall"
LABEL maintainer.email="ggmarsh@uw.edu"

# Only git is needed at runtime (legendmeta opens the metadata git repo).
# Build tools and editors do not belong in the production image.
RUN apt-get update && apt-get install --no-install-recommends --yes \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.8,<1.0"

# uv resolves the target environment from VIRTUAL_ENV (not from PATH), so both
# must be set for `uv pip install` below to install into /opt/venv.
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Line-buffered stdout/stderr so startup and auth warnings appear in spin logs.
ENV PYTHONUNBUFFERED=1

# Install the dashboard. 0.0.8 includes the per-user session independence fix,
# the shared metadata/par caches, the hourly shared refresh + Refresh button,
# threading defaults and exception logging. Bump this pin to pick up future
# releases (publishing a GitHub Release triggers .github/workflows/distribute.yml).
# The cal/dsp shelve files are pickled matplotlib figures produced by the data
# production pipeline. Pickled matplotlib objects are NOT portable across
# matplotlib versions, so the dashboard MUST use the same matplotlib version
# that wrote them. These v2.0.0 files import `matplotlib.colorizer` (added in
# 3.10), so they were written by matplotlib >= 3.10 -- NOT the 3.9.2 reported by
# the current dataflow env. Pin to the exact version that generated the plots;
# confirm it and adjust the patch level if rendering still fails.
# ruamel.yaml is a dependency of the (unreleased) Metadata editor page; keep
# it pre-installed so the next pin bump only swaps the dashboard package.
RUN uv pip install legend_dashboard==0.0.8 "matplotlib>=3.10,<3.11" "ruamel.yaml>=0.18"

WORKDIR /app

# tmp must be a writable directory: CSV downloads are written there. An empty
# string would make them land in the working directory.
# metadata_edit is the editable legend-metadata clone for the Metadata editor
# page; it lives in the (always writable) home directory and is cloned /
# pulled on startup, so it is ephemeral per pod -- fine, finished edits are
# pushed to the user's fork, and the page warns about un-pushed edits.
# Per-user workspaces (git worktrees of the datasets submodule) are created at
# runtime under /home/dashboard/metadata-edit-workspaces by the same uid that
# runs git, so no safe.directory entries are needed for them (and the
# safe.directory `/*` glob would be silently ignored by this image's git 2.39
# anyway -- it needs git >= 2.46).
RUN python3 -c 'import yaml; yaml.dump({"paths":{"base":"/srv/tmp-auto", "cal":"/srv/tmp-auto", "phy":"", "llama":"", "sipm":"", "muon":"", "tmp":"/tmp", "metadata_edit":"/home/dashboard/metadata-edit"}}, open("dashboard-config.yaml","w"))'
RUN git config --system --add safe.directory /srv/tmp-auto/inputs \
    && git config --system --add safe.directory /home/dashboard/metadata-edit \
    && git config --system --add safe.directory /home/dashboard/metadata-edit/datasets \
    && git config --system --add safe.directory /home/dashboard/metadata-edit/hardware/configuration \
    && git config --system --add safe.directory /home/dashboard/metadata-edit/hardware/detectors

# Upstream URL of the metadata repo cloned for the editor (submodule URLs are
# rewritten from SSH to HTTPS automatically).
ENV METADATA_EDIT_URL="https://github.com/legend-exp/legend-metadata"

# NERSC spin requires containers to run as a non-root user.
RUN useradd --uid 1000 --create-home dashboard \
    && chown -R dashboard /app
USER dashboard

EXPOSE 5000

# When serving behind the spin reverse proxy, the public hostname MUST be
# passed via --websocket-origin or the websocket is rejected and the dashboard
# never updates. Set WEBSOCKET_ORIGIN (e.g. your-spin-host.nersc.gov) in the
# spin workload configuration; it is appended to the command when present.
ENV WEBSOCKET_ORIGIN=""
CMD dashboard ./dashboard-config.yaml \
    -p 5000 --num-threads 4 \
    -d spm muon llama phy \
    ${WEBSOCKET_ORIGIN:+--websocket-origin $WEBSOCKET_ORIGIN}
