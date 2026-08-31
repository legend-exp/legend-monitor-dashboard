"""Panel page for viewing and editing the metadata catalogue.

Operates on a dedicated editable clone of legend-metadata (``meta_path``),
separate from the production cycle's read-only copy. Five sub-tabs: Overview
(leds-style matrix/timeline figures), Detector status editor, Partitions
(groupings) editor, Bad cycles, and Commit & Push (to the user's fork of
legend-datasets).

Edits happen in a per-user **workspace** (a git worktree of the datasets
submodule, see ``meta_git.ensure_workspace``): the page is read-only until a
workspace name (the user's GitHub username) is opened; staged edits are then
isolated to that workspace and survive page reloads -- re-opening the same
name reattaches. Workspaces are isolated from one another but not
access-controlled -- see the note in ``meta_git``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pandas as pd
import panel as pn
import param

from legenddashboard.base import Monitoring
from legenddashboard.metadata import meta_edit, meta_git, meta_views
from legenddashboard.metadata.meta_db import GROUPING_FILES, get_meta_db
from legenddashboard.util import logo_path, read_config

log = logging.getLogger(__name__)

OPEN_ENDED = "(open-ended)"
UNCHANGED = "(unchanged)"

#: Groupings file -> datatype of the run grid its matrix is drawn on.
_GROUPINGS_GRID = {"cal": "cal", "phy": "phy", "escale": "cal", "psd": "cal"}


class MetaMonitoring(Monitoring):
    """Metadata viewer/editor operating on the editable clone."""

    meta_path = param.String("", allow_refs=True, nested_refs=True)
    #: Bumped after every staged edit; every view depends on it.
    meta_version = param.Integer(0)

    overview_plot = param.Selector(
        default="data", objects=list(meta_views.PLOTS), label="Plot"
    )
    overview_datatype = param.Selector(
        default="phy", objects=list(meta_views.DATATYPES), label="Datatype"
    )
    overview_dataset = param.Selector(
        default="all", objects=["all"], label="Dataset (runlists)"
    )
    groupings_key = param.Selector(
        default="cal", objects=list(GROUPING_FILES), label="Groupings file"
    )

    edit_type = param.Selector(default="Usability", objects=["Usability", "PSD"])
    edit_detector = param.Selector(default=None, objects=[], label="Detector")
    edit_status = param.Selector(
        default="on", objects=list(meta_edit.USABILITY_VALUES), label="Status"
    )
    edit_start = param.Selector(default=None, objects=[], label="Start run")
    edit_end = param.Selector(default=OPEN_ENDED, objects=[], label="End run")
    edit_datatype = param.Selector(
        default="both", objects=list(meta_edit.DATATYPE_MAP), label="Datatype"
    )

    #: Open workspace name; "" = read-only mode (no edits possible).
    workspace = param.String("")

    def __init__(self, **params):
        super().__init__(**params)
        self.alert = pn.pane.Alert(
            "", alert_type="danger", visible=False, sizing_mode="stretch_width"
        )
        self._figs: dict = {}
        self._partition_selection = (None, [])
        self._selection_info = pn.pane.Markdown("*nothing selected*")
        self._workspace_path = None
        self._username_in = None
        self.metadb = None
        try:
            self.metadb = get_meta_db(self.meta_path)
        except Exception:
            log.exception("metadata edit clone unusable at %s", self.meta_path)
            return

        run_labels = [
            f"{p} {r}"
            for p in sorted(self.metadb.available_runs())
            for r in self.metadb.available_runs()[p]
        ]
        self.param.edit_start.objects = run_labels
        self.edit_start = run_labels[-1]
        self.param.watch(self._filter_end_runs, ["edit_start"])
        self._filter_end_runs()

        detectors = self._detector_names()
        self.param.edit_detector.objects = detectors
        self.edit_detector = detectors[0]

        self.param.watch(self._refresh_datasets, ["meta_version"])
        self._refresh_datasets()

    def _refresh_datasets(self, *_events):
        """Dataset choices for the overview filter: "all" + runlists keys."""
        try:
            options = ["all", *self.metadb.runlists()]
        except Exception:
            log.exception("could not read runlists.yaml")
            options = ["all"]
        self.param.overview_dataset.objects = options
        if self.overview_dataset not in options:
            self.overview_dataset = "all"

    def _filter_end_runs(self, *_events):
        """End-run choices: open-ended or any run from the start run onwards."""
        labels = self.param.edit_start.objects
        options = [OPEN_ENDED, *labels[labels.index(self.edit_start) :]]
        self.param.edit_end.objects = options
        if self.edit_end not in options:
            self.edit_end = OPEN_ENDED

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _detector_names(self) -> list[str]:
        tstamp = self.metadb.run_start_key(*self._parse(self.edit_start))
        return sorted(self.metadb.geds_positions(tstamp))

    @staticmethod
    def _scrollable(pane):
        """Wrap a fixed-width matrix pane in a horizontal scroll container.

        ``scroll=True`` (not raw CSS): Bokeh's layout engine propagates a
        fixed child width as a hard min-width up every ancestor, so a plain
        ``overflow-x`` style never gets to clip -- a scrollable layout is the
        one container whose content size Bokeh excludes from that
        propagation.
        """
        return pn.Row(
            pane,
            sizing_mode="stretch_width",
            scroll=True,
        )

    #: CSS for an edit panel that stays on screen while the (much bigger)
    #: matrix next to it scrolls. Both scroll axes live on the template's
    #: `.main` container, so a sticky panel with both `top` and `right`
    #: offsets stays glued to the visible corner in either direction --
    #: Bokeh's layout engine propagates the matrix's fixed width up as a hard
    #: min-width, so in-flow containment (overflow wrappers) cannot work;
    #: sticking the panel to the scrollport can. `align-self: flex-start`
    #: keeps the natural height (a stretched flex child cannot stick); the
    #: background/shadow keep it readable when it floats over wide content.
    _STICKY: ClassVar[dict[str, str]] = {
        "position": "sticky",
        "top": "0px",
        # overshoot the viewport edge: the template's main area keeps a
        # padding/scrollbar gutter that would otherwise show a strip of plot
        # to the right of the panel; the extra right padding compensates
        "right": "-30px",
        "align-self": "flex-start",
        "background": "#ffffff",
        "z-index": "100",
        "box-shadow": "-2px 0 8px rgba(0, 0, 0, 0.15)",
        "padding": "0 40px 10px 10px",
        "border-radius": "4px 0 0 4px",
        "margin": "0",
        # cover the full visible column so no plot peeks out below the form
        "min-height": "85vh",
    }

    @staticmethod
    def _parse(run_label: str) -> tuple[str, str]:
        period, run = run_label.split()
        return period, run

    def _fail(self, msg: str) -> None:
        self.alert.alert_type = "danger"
        self.alert.object = msg
        self.alert.visible = True

    def _ok(self, msg: str) -> None:
        self.alert.alert_type = "success"
        self.alert.object = msg
        self.alert.visible = True

    def _bump(self) -> None:
        self.metadb.reload()
        # Every _figs key embeds meta_version, so bumping it orphans the whole
        # cache: drop it rather than let a long editing session accumulate one
        # set of figures per edit.
        self._figs.clear()
        self.meta_version += 1

    def _require_workspace(self) -> bool:
        if self.workspace:
            return True
        self._fail(
            "**open a workspace first** — enter your GitHub username at the "
            "top of the page and press *Open workspace*"
        )
        return False

    def _resolved_seed(self, start_key: str) -> dict:
        """Plain-dict resolved statuses at ``start_key`` (file-creation seed)."""
        statuses = self.metadb.statuses_on(start_key, category="all")
        return meta_edit._to_plain({k: dict(v) for k, v in statuses.items()})

    def _jump_to(self, period: str, run: str) -> None:
        """Point the whole dashboard at (period, run) if it has data."""
        if period not in self.periods or run not in self.periods[period]:
            return
        if self.period != period:
            self.period = period
        if self.run != run:
            self.run = run

    def _on_matrix_tap(self, source, indices) -> None:
        if not indices:
            return
        period, run = source.data["x"][indices[0]]
        self._jump_to(period, run)

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    @param.depends(
        "overview_plot",
        "overview_datatype",
        "overview_dataset",
        "groupings_key",
        "meta_version",
    )
    def view_overview(self):
        if self.metadb is None:
            return pn.pane.Markdown("**metadata clone unavailable**")
        key = (
            self.overview_plot,
            self.overview_datatype,
            self.overview_dataset,
            self.groupings_key,
            self.meta_version,
        )
        fig = self._figs.get(key)
        if fig is None:
            try:
                kwargs = {}
                if self.overview_plot == "partitions":
                    kwargs["groupings_key"] = self.groupings_key
                    datatype = _GROUPINGS_GRID[self.groupings_key]
                else:
                    datatype = self.overview_datatype
                fig, source = meta_views.dataset_figure(
                    self.metadb,
                    plot=self.overview_plot,
                    datatype=datatype,
                    run_filter=self.metadb.runlist_filter(self.overview_dataset),
                    **kwargs,
                )
            except Exception as exc:
                log.exception("could not build overview figure")
                return pn.pane.Markdown(f"**{type(exc).__name__}**: {exc}")
            source.selected.on_change(
                "indices", lambda _a, _o, new, s=source: self._on_matrix_tap(s, new)
            )
            self._figs[key] = fig
        if self.overview_plot in ("usabilities", "partitions", "psd"):
            return self._scrollable(pn.pane.Bokeh(fig))
        return pn.pane.Bokeh(fig, sizing_mode="stretch_width")

    def _build_overview_tab(self, widget_widths):
        plot_sel = pn.widgets.Select.from_param(
            self.param.overview_plot, width=widget_widths
        )
        dtype_sel = pn.widgets.Select.from_param(
            self.param.overview_datatype, width=widget_widths
        )
        groupings_sel = pn.widgets.Select.from_param(
            self.param.groupings_key, width=widget_widths
        )
        dataset_sel = pn.widgets.Select.from_param(
            self.param.overview_dataset, width=widget_widths
        )
        return pn.Column(
            pn.Row(plot_sel, dtype_sel, dataset_sel, groupings_sel),
            pn.pane.Markdown(
                "Tap a cell/bar/block to point the whole dashboard at that run "
                "(runs without processed data are ignored). The dataset "
                "selector restricts every plot to the runs of one "
                "`runlists.yaml` dataset."
            ),
            pn.param.ParamMethod(self.view_overview, lazy=True),
            name="Overview",
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Detector status editor
    # ------------------------------------------------------------------

    @param.depends("edit_type", "edit_datatype", "meta_version")
    def view_status_matrix(self):
        """Detector x run status matrix over the whole catalogue.

        Usability or PSD colours depending on the edit mode; tapping a cell
        pre-fills the Detector and Start run of the edit form.
        """
        if self.metadb is None:
            return pn.pane.Markdown("**metadata clone unavailable**")
        # "both" edits use categories of both grids; show the phy one
        datatype = "cal" if self.edit_datatype == "cal" else "phy"
        plot = "usabilities" if self.edit_type == "Usability" else "psd"
        key = ("status-tab", plot, datatype, self.meta_version)
        fig = self._figs.get(key)
        if fig is None:
            try:
                fig, source = meta_views.dataset_figure(
                    self.metadb, plot=plot, datatype=datatype
                )
            except Exception as exc:
                log.exception("could not build the status matrix")
                return pn.pane.Markdown(f"**{type(exc).__name__}**: {exc}")

            def _on_tap(_attr, _old, new, s=source):
                if new:
                    period, run = s.data["x"][new[0]]
                    self.edit_detector = s.data["y"][new[0]]
                    self.edit_start = f"{period} {run}"

            source.selected.on_change("indices", _on_tap)
            self._figs[key] = fig
        return self._scrollable(pn.pane.Bokeh(fig))

    @param.depends(
        "edit_type",
        "edit_detector",
        "edit_start",
        "edit_end",
        "edit_datatype",
        "meta_version",
    )
    def view_status_preview(self):
        if self.metadb is None:
            return pn.pane.Markdown("")
        try:
            period, run = self._parse(self.edit_start)
            fname = meta_edit.status_file_name(period, run, self.edit_datatype)
            exists = (self.metadb.datasets_path / "statuses" / fname).exists()
            action = "will edit" if exists else "will **CREATE** (+ validity.yaml)"
            lines = [f"{action} `{fname}`"]
            if self.edit_end != OPEN_ENDED:
                end = self._parse(self.edit_end)
                nxt = self.metadb.next_run(*end)
                if nxt is None:
                    lines.append(
                        "end run is the last in the catalogue — no revert file"
                    )
                else:
                    rname = meta_edit.status_file_name(*nxt, self.edit_datatype)
                    rexists = (self.metadb.datasets_path / "statuses" / rname).exists()
                    raction = "revert edit in" if rexists else "revert file **CREATE**"
                    lines.append(f"{raction} `{rname}` (restores previous status)")
            cur = self.metadb.statuses_on(
                self.metadb.run_start_key(period, run), category=self.edit_datatype
            ).get(self.edit_detector)
            if cur is not None:
                lines.append(
                    f"current: `{cur.get('usability')}` "
                    f"(reason: {cur.get('reason') or '—'})"
                )
            return pn.pane.Markdown("<br>".join(lines))
        except Exception as exc:
            return pn.pane.Markdown(f"**{type(exc).__name__}**: {exc}")

    def _current_psd(self) -> dict:
        period, run = self._parse(self.edit_start)
        entry = self.metadb.statuses_on(
            self.metadb.run_start_key(period, run), category=self.edit_datatype
        ).get(self.edit_detector)
        return dict((entry or {}).get("psd") or {})

    def _build_status_tab(self, widget_widths):
        w = max(widget_widths, 160)
        type_toggle = pn.widgets.RadioButtonGroup.from_param(
            self.param.edit_type, button_type="primary"
        )
        detector_sel = pn.widgets.Select.from_param(self.param.edit_detector, width=w)
        start_sel = pn.widgets.Select.from_param(self.param.edit_start, width=w)
        end_sel = pn.widgets.Select.from_param(self.param.edit_end, width=w)
        dtype_sel = pn.widgets.Select.from_param(self.param.edit_datatype, width=w)
        reason_in = pn.widgets.TextInput(
            name="Reason", placeholder="why the change", width=w
        )

        status_group = pn.widgets.RadioButtonGroup.from_param(
            self.param.edit_status, button_type="success"
        )
        usability_box = pn.Column("**Usability**", status_group)

        flag_sels = {
            flag: pn.widgets.Select(
                name=flag,
                options=[UNCHANGED, *meta_edit.PSD_FLAG_VALUES],
                value=UNCHANGED,
                width=w,
            )
            for flag in meta_edit.PSD_FLAGS
        }
        bb_like_in = pn.widgets.TextInput(
            name="is_bb_like",
            placeholder="e.g. low_aoe & high_aoe (empty = unchanged)",
            width=w,
        )
        psd_box = pn.Column(
            "**PSD flags** (hover a selector for the current value; pick one to change it)",
            *flag_sels.values(),
            bb_like_in,
            visible=False,
        )

        def _toggle_type(event):
            usability_box.visible = event.new == "Usability"
            psd_box.visible = event.new == "PSD"

        self.param.watch(_toggle_type, "edit_type")

        def _prefill_psd(*_events):
            try:
                psd = self._current_psd()
            except Exception:
                psd = {}
            flags = psd.get("status") or {}
            for flag, sel in flag_sels.items():
                cur = str(flags.get(flag, ""))
                sel.description = f"current: {cur or '—'}"
            bb_like_in.placeholder = (
                f"current: {psd.get('is_bb_like') or '—'} (empty = unchanged)"
            )

        self.param.watch(_prefill_psd, ["edit_detector", "edit_start", "edit_datatype"])
        _prefill_psd()

        apply_btn = pn.widgets.Button(
            name="Apply", button_type="primary", width=w, icon="pencil"
        )

        def _collect_edit(self) -> dict:
            if self.edit_type == "Usability":
                return {
                    "usability": self.edit_status,
                    "reason": reason_in.value,
                }
            flags = {
                flag: sel.value
                for flag, sel in flag_sels.items()
                if sel.value != UNCHANGED
            }
            psd: dict = {}
            if flags:
                psd["status"] = flags
            if bb_like_in.value.strip():
                psd["is_bb_like"] = bb_like_in.value.strip()
            if not psd:
                msg = "no PSD flag or is_bb_like value selected"
                raise ValueError(msg)
            return {"psd": psd}

        def _revert_edit(self, prior: dict | None) -> dict:
            prior = prior or {}
            if self.edit_type == "Usability":
                return {
                    "usability": str(prior.get("usability", "on")),
                    "reason": str(prior.get("reason", "")),
                }
            psd = dict(prior.get("psd") or {})
            edited = _collect_edit(self)["psd"]
            out: dict = {}
            if "status" in edited:
                flags = psd.get("status") or {}
                out["status"] = {
                    flag: str(flags.get(flag, "missing")) for flag in edited["status"]
                }
            if "is_bb_like" in edited:
                out["is_bb_like"] = str(psd.get("is_bb_like", "missing"))
            return {"psd": out}

        def _apply(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            try:
                edits = {self.edit_detector: _collect_edit(self)}
                start = self._parse(self.edit_start)
                start_key = self.metadb.run_start_key(*start)
                start_point = (*start, start_key, self._resolved_seed(start_key))

                revert_point = None
                if self.edit_end != OPEN_ENDED:
                    end = self._parse(self.edit_end)
                    if end < start:
                        msg = "end run precedes start run"
                        raise ValueError(msg)
                    nxt = self.metadb.next_run(*end)
                    if nxt is not None:
                        nxt_key = self.metadb.run_start_key(*nxt)
                        prior = self.metadb.statuses_on(
                            nxt_key, category=self.edit_datatype
                        ).get(self.edit_detector)
                        revert_point = (
                            *nxt,
                            nxt_key,
                            self._resolved_seed(nxt_key),
                            {self.edit_detector: _revert_edit(self, prior)},
                        )

                paths = meta_edit.apply_status_range(
                    self.metadb.datasets_path,
                    edits=edits,
                    datatype=self.edit_datatype,
                    start_point=start_point,
                    revert_point=revert_point,
                )
            except Exception as exc:
                log.exception("could not stage the status edit")
                self._fail(f"**status edit failed** — {exc}")
                return
            self._bump()
            self._ok(
                f"staged edit for **{self.edit_detector}** in "
                + ", ".join(f"`{p.name}`" for p in paths)
                + " — review it in the *Commit & Push* tab"
            )

        apply_btn.on_click(_apply)

        form = pn.Column(
            type_toggle,
            detector_sel,
            start_sel,
            end_sel,
            dtype_sel,
            usability_box,
            psd_box,
            reason_in,
            pn.param.ParamMethod(self.view_status_preview, lazy=True),
            apply_btn,
            width=max(w + 40, 300),
            styles=self._STICKY,
        )
        return pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Tap a cell to pick that detector and start run; the "
                    "matrix follows the edit mode (usability / PSD)."
                ),
                pn.param.ParamMethod(self.view_status_matrix, lazy=True),
                sizing_mode="stretch_width",
                styles={"min-width": "0"},
            ),
            form,
            name="Detector status",
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Partitions (groupings) editor
    # ------------------------------------------------------------------

    def _grouping_block_df(self, key: str, detector: str) -> pd.DataFrame:
        groupings = self.metadb.groupings(key)
        block = groupings.get(detector, {})
        rows = [
            {
                "partition": part,
                "period": period,
                "runs": ", ".join(meta_views.expand_run_list(runs)),
            }
            for part, periods in block.items()
            for period, runs in periods.items()
        ]
        return pd.DataFrame(rows, columns=["partition", "period", "runs"])

    @param.depends("groupings_key", "meta_version")
    def view_partitions_matrix(self):
        if self.metadb is None:
            return pn.pane.Markdown("**metadata clone unavailable**")
        key = ("partitions-tab", self.groupings_key, self.meta_version)
        fig = self._figs.get(key)
        if fig is None:
            try:
                fig, source = meta_views.partitions_figure(
                    self.metadb,
                    datatype=_GROUPINGS_GRID[self.groupings_key],
                    groupings_key=self.groupings_key,
                    box_select=True,
                )
            except Exception as exc:
                log.exception("could not build the partitions matrix")
                return pn.pane.Markdown(f"**{type(exc).__name__}**: {exc}")

            def _on_select(_attr, _old, new, s=source):
                self._partition_selection = (s, list(new))
                if not new:
                    self._selection_info.object = "*nothing selected*"
                    return
                dets = {s.data["y"][i] for i in new}
                runs = {tuple(s.data["x"][i]) for i in new}
                lo, hi = min(runs), max(runs)
                self._selection_info.object = (
                    f"**{len(new)} cells** selected — {len(dets)} detector(s), "
                    f"runs {' '.join(lo)} … {' '.join(hi)}"
                )

            source.selected.on_change("indices", _on_select)
            self._figs[key] = fig
            self._partition_selection = (source, [])
        return self._scrollable(pn.pane.Bokeh(fig))

    @param.depends("groupings_key")
    def view_groupings_warning(self):
        if self.groupings_key in ("cal", "phy"):
            return pn.pane.Alert(
                f"⚠️ **You are editing the core `{self.groupings_key}` "
                "partitions used by the data production.** Most users should "
                "only edit the `escale` or `psd` groupings — change these "
                "only if you know what you are doing.",
                alert_type="warning",
                sizing_mode="stretch_width",
            )
        return pn.pane.Markdown("", height=0, margin=0)

    def _build_partitions_tab(self, widget_widths):
        w = max(widget_widths, 160)
        groupings_sel = pn.widgets.Select.from_param(self.param.groupings_key, width=w)
        detector_sel = pn.widgets.Select(name="Block", options=["default"], width=w)
        table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["partition", "period", "runs"]),
            show_index=False,
            selectable=1,
            height=300,
            sizing_mode="stretch_width",
        )

        def _refresh_blocks(*_events):
            try:
                groupings = self.metadb.groupings(self.groupings_key)
            except Exception:
                log.exception("could not read groupings %s", self.groupings_key)
                return
            names = ["default", *sorted(k for k in groupings if k != "default")]
            detector_sel.options = names
            if detector_sel.value not in names:
                detector_sel.value = "default"
            _refresh_table()

        def _refresh_table(*_events):
            table.value = self._grouping_block_df(
                self.groupings_key, detector_sel.value
            )

        self.param.watch(_refresh_blocks, ["groupings_key", "meta_version"])
        detector_sel.param.watch(_refresh_table, "value")
        _refresh_blocks()

        add_row_btn = pn.widgets.Button(name="Add row", width=w)

        def _add_row(_event):
            table.value = pd.concat(
                [
                    table.value,
                    pd.DataFrame([{"partition": "", "period": "", "runs": ""}]),
                ],
                ignore_index=True,
            )

        add_row_btn.on_click(_add_row)

        del_row_btn = pn.widgets.Button(name="Delete selected row", width=w)

        def _del_row(_event):
            if table.selection:
                table.value = table.value.drop(
                    table.value.index[table.selection]
                ).reset_index(drop=True)

        del_row_btn.on_click(_del_row)

        split_run_in = pn.widgets.TextInput(
            name="Split at (period run)", placeholder="e.g. p07 r003", width=w
        )
        split_btn = pn.widgets.Button(name="Split group", width=w)

        def _split(_event):
            self.alert.visible = False
            try:
                period, at_run = split_run_in.value.split()
                if not table.selection:
                    msg = "select the row of the partition to split"
                    raise ValueError(msg)
                part = table.value.iloc[table.selection[0]]["partition"]
                block = self._df_to_block(table.value)
                new_block = meta_edit.split_group(block, part, period, at_run)
                table.value = pd.DataFrame(
                    [
                        {
                            "partition": p,
                            "period": per,
                            "runs": ", ".join(meta_views.expand_run_list(runs)),
                        }
                        for p, periods in new_block.items()
                        for per, runs in periods.items()
                    ]
                )
            except Exception as exc:
                self._fail(f"**split failed** — {exc}")

        split_btn.on_click(_split)

        new_det_in = pn.widgets.TextInput(
            name="New override for detector", placeholder="e.g. V01234A", width=w
        )
        new_det_btn = pn.widgets.Button(name="Create override", width=w)

        def _new_override(_event):
            name = new_det_in.value.strip()
            if not name:
                return
            detector_sel.options = [*detector_sel.options, name]
            detector_sel.value = name
            # start from the default block (the usual base for an override)
            table.value = self._grouping_block_df(self.groupings_key, "default")

        new_det_btn.on_click(_new_override)

        # quick add: assign a whole run range to a partition, no table editing
        run_labels = self.param.edit_start.objects
        range_part_in = pn.widgets.TextInput(
            name="Partition", placeholder="e.g. 12a", width=w
        )
        range_start_sel = pn.widgets.Select(
            name="Start run", options=run_labels, value=run_labels[-1], width=w
        )
        range_end_sel = pn.widgets.Select(
            name="End run", options=run_labels, value=run_labels[-1], width=w
        )
        range_add_btn = pn.widgets.Button(
            name="Add", button_type="primary", width=w, icon="plus"
        )

        def _add_range(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            try:
                target = meta_edit.normalize_partition(
                    range_part_in.value, self.groupings_key
                )
                lo = run_labels.index(range_start_sel.value)
                hi = run_labels.index(range_end_sel.value)
                if hi < lo:
                    msg = "end run precedes start run"
                    raise ValueError(msg)
                cells = [tuple(label.split()) for label in run_labels[lo : hi + 1]]
                groupings = self.metadb.groupings(self.groupings_key)
                block = meta_edit.assign_in_block(
                    groupings, detector_sel.value, cells, target
                )
                meta_edit.set_detector_block(
                    self.metadb.datasets_path,
                    self.groupings_key,
                    detector_sel.value,
                    block,
                )
            except Exception as exc:
                log.exception("could not add the partition range")
                self._fail(f"**add partition failed** — {exc}")
                return
            self._bump()
            self._ok(
                f"assigned {range_start_sel.value} … {range_end_sel.value} of "
                f"block **{detector_sel.value}** to **{target}** — review in "
                "*Commit & Push*"
            )

        range_add_btn.on_click(_add_range)

        apply_btn = pn.widgets.Button(
            name="Apply block", button_type="primary", width=w, icon="pencil"
        )

        def _apply(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            try:
                block = self._df_to_block(table.value)
                path = meta_edit.set_detector_block(
                    self.metadb.datasets_path,
                    self.groupings_key,
                    detector_sel.value,
                    block,
                )
            except Exception as exc:
                log.exception("could not stage the groupings edit")
                self._fail(f"**groupings edit failed** — {exc}")
                return
            self._bump()
            self._ok(
                f"staged block **{detector_sel.value}** in `{path.name}` — "
                "review it in the *Commit & Push* tab"
            )

        apply_btn.on_click(_apply)

        # drag-to-assign on the matrix
        target_in = pn.widgets.TextInput(
            name="Assign selection to partition",
            placeholder="e.g. 12a or calgroup012a",
            width=w,
        )
        assign_btn = pn.widgets.Button(
            name="Assign selected cells", button_type="primary", width=w
        )

        def _assign(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            source, indices = getattr(self, "_partition_selection", (None, []))
            if not indices:
                self._fail("drag-select some cells in the matrix first")
                return
            try:
                target = meta_edit.normalize_partition(
                    target_in.value, self.groupings_key
                )
                cells_by_det: dict[str, list[tuple[str, str]]] = {}
                for i in indices:
                    det = str(source.data["y"][i])
                    cells_by_det.setdefault(det, []).append(tuple(source.data["x"][i]))
                groupings = self.metadb.groupings(self.groupings_key)
                for det, cells in cells_by_det.items():
                    block = meta_edit.assign_partition(groupings, det, cells, target)
                    meta_edit.set_detector_block(
                        self.metadb.datasets_path, self.groupings_key, det, block
                    )
            except Exception as exc:
                log.exception("could not assign the selection")
                self._fail(f"**assign failed** — {exc}")
                return
            self._bump()
            self._ok(
                f"assigned {len(indices)} cells across {len(cells_by_det)} "
                f"detector(s) to **{target}** — review in *Commit & Push*"
            )

        assign_btn.on_click(_assign)

        # everything that *edits* lives in one panel pinned to the right of
        # the screen, so it stays visible while scrolling the tall matrix
        edit_panel = pn.Column(
            pn.pane.Markdown("### Edit partitions"),
            groupings_sel,
            detector_sel,
            new_det_in,
            new_det_btn,
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Add by run range** — assigns every catalogue run from "
                "start to end (inclusive) in this block:"
            ),
            range_part_in,
            range_start_sel,
            range_end_sel,
            range_add_btn,
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Assign matrix selection** — drag (box-select) cells on "
                "the matrix (grey = in the array but unassigned; shift-drag "
                "extends):"
            ),
            self._selection_info,
            target_in,
            assign_btn,
            width=max(w + 60, 320),
            styles=self._STICKY,
        )
        content = pn.Column(
            pn.param.ParamMethod(self.view_groupings_warning, lazy=True),
            pn.pane.Markdown(
                "**Edit the block as a table** (one row per partition and "
                "period; runs as a comma-separated list, ranges like "
                "`r000..r005` also work; an empty table removes the "
                "override):"
            ),
            table,
            pn.Row(add_row_btn, del_row_btn, split_run_in, split_btn, apply_btn),
            pn.layout.Divider(),
            pn.param.ParamMethod(self.view_partitions_matrix, lazy=True),
            sizing_mode="stretch_width",
            styles={"min-width": "0"},
        )
        return pn.Row(
            content,
            edit_panel,
            name="Partitions",
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _df_to_block(df: pd.DataFrame) -> dict:
        block: dict = {}
        for _, row in df.iterrows():
            part = str(row["partition"]).strip()
            period = str(row["period"]).strip()
            runs_str = str(row["runs"]).strip()
            if not part and not period and not runs_str:
                continue  # blank row
            if not (part and period and runs_str):
                msg = f"incomplete row: {dict(row)}"
                raise ValueError(msg)
            runs = meta_views.expand_run_list(
                [tok.strip() for tok in runs_str.split(",") if tok.strip()]
            )
            for r in runs:
                if not re.match(r"^r\d{3}$", r):
                    msg = f"not a run id: {r!r}"
                    raise ValueError(msg)
            block.setdefault(part, {})[period] = meta_edit.compress_runs(runs)
        return block

    # ------------------------------------------------------------------
    # Runlists
    # ------------------------------------------------------------------

    def _runlist_df(self, dataset: str, datatype: str) -> pd.DataFrame:
        periods = self.metadb.runlists().get(dataset, {}).get(datatype, {})
        rows = []
        if isinstance(periods, dict):
            for period, runs in periods.items():
                value = (
                    "all"
                    if str(runs) == "all"
                    else ", ".join(meta_views.expand_run_list(runs))
                )
                rows.append({"period": period, "runs": value})
        return pd.DataFrame(rows, columns=["period", "runs"])

    @staticmethod
    def _runlist_mapping(df: pd.DataFrame) -> dict:
        """Table rows -> ``{period: "all" | runs-notation}`` (validated)."""
        mapping: dict = {}
        for _, row in df.iterrows():
            period = str(row["period"]).strip()
            runs_str = str(row["runs"]).strip()
            if not period and not runs_str:
                continue  # blank row
            if not (period and runs_str):
                msg = f"incomplete row: {dict(row)}"
                raise ValueError(msg)
            if not re.match(r"^p\d{2}$", period):
                msg = f"not a period id: {period!r}"
                raise ValueError(msg)
            if runs_str == "all":
                mapping[period] = "all"
                continue
            runs = meta_views.expand_run_list(
                [tok.strip() for tok in runs_str.split(",") if tok.strip()]
            )
            for r in runs:
                if not re.match(r"^r\d{3}$", r):
                    msg = f"not a run id: {r!r}"
                    raise ValueError(msg)
            mapping[period] = meta_edit.compress_runs(runs)
        return mapping

    def _build_runlists_tab(self, widget_widths):
        w = max(widget_widths, 160)
        dataset_sel = pn.widgets.Select(name="Dataset", options=[], width=w)
        datatype_sel = pn.widgets.Select(name="Datatype", options=[], width=w)
        new_dataset_in = pn.widgets.TextInput(
            name="New dataset", placeholder="e.g. mycheck26", width=w
        )
        new_datatype_in = pn.widgets.TextInput(
            name="New datatype", placeholder="e.g. phy", width=w
        )
        table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["period", "runs"]),
            show_index=False,
            selectable=1,
            height=350,
            sizing_mode="stretch_width",
        )

        def _refresh_datasets(*_events):
            try:
                runlists = self.metadb.runlists()
            except Exception:
                log.exception("could not read runlists.yaml")
                return
            names = list(runlists)
            extra = [v for v in (dataset_sel.value,) if v and v not in names]
            dataset_sel.options = names + extra
            if dataset_sel.value not in dataset_sel.options and names:
                dataset_sel.value = names[0]
            _refresh_datatypes()

        def _refresh_datatypes(*_events):
            datatypes = list(self.metadb.runlists().get(dataset_sel.value, {}))
            extra = [v for v in (datatype_sel.value,) if v and v not in datatypes]
            datatype_sel.options = datatypes + extra
            if datatype_sel.value not in datatype_sel.options and datatypes:
                datatype_sel.value = datatypes[0]
            _refresh_table()

        def _refresh_table(*_events):
            table.value = self._runlist_df(dataset_sel.value, datatype_sel.value)

        self.param.watch(_refresh_datasets, ["meta_version"])
        dataset_sel.param.watch(_refresh_datatypes, "value")
        datatype_sel.param.watch(_refresh_table, "value")
        _refresh_datasets()

        def _new_dataset(_event):
            name = new_dataset_in.value.strip()
            if not name:
                return
            dataset_sel.options = [*dataset_sel.options, name]
            dataset_sel.value = name

        new_dataset_btn = pn.widgets.Button(name="Create dataset", width=w)
        new_dataset_btn.on_click(_new_dataset)

        def _new_datatype(_event):
            name = new_datatype_in.value.strip()
            if not name:
                return
            datatype_sel.options = [*datatype_sel.options, name]
            datatype_sel.value = name
            table.value = pd.DataFrame(columns=["period", "runs"])

        new_datatype_btn = pn.widgets.Button(name="Create datatype", width=w)
        new_datatype_btn.on_click(_new_datatype)

        add_row_btn = pn.widgets.Button(name="Add row", width=w)
        add_row_btn.on_click(
            lambda _e: setattr(
                table,
                "value",
                pd.concat(
                    [table.value, pd.DataFrame([{"period": "", "runs": ""}])],
                    ignore_index=True,
                ),
            )
        )
        del_row_btn = pn.widgets.Button(name="Delete selected row", width=w)

        def _del_row(_event):
            if table.selection:
                table.value = table.value.drop(
                    table.value.index[table.selection]
                ).reset_index(drop=True)

        del_row_btn.on_click(_del_row)

        apply_btn = pn.widgets.Button(
            name="Apply", button_type="primary", width=w, icon="pencil"
        )

        def _apply(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            try:
                mapping = self._runlist_mapping(table.value)
                path = meta_edit.set_runlist(
                    self.metadb.datasets_path,
                    dataset_sel.value,
                    datatype_sel.value,
                    mapping,
                )
            except Exception as exc:
                log.exception("could not stage the runlist edit")
                self._fail(f"**runlist edit failed** — {exc}")
                return
            self._bump()
            what = f"**{dataset_sel.value}.{datatype_sel.value}**"
            self._ok(
                (
                    f"removed {what} from `{path.name}`"
                    if not mapping
                    else f"staged {what} in `{path.name}`"
                )
                + " — review in *Commit & Push*"
            )

        apply_btn.on_click(_apply)

        edit_panel = pn.Column(
            pn.pane.Markdown("### Edit runlists"),
            dataset_sel,
            datatype_sel,
            pn.layout.Divider(),
            new_dataset_in,
            new_dataset_btn,
            new_datatype_in,
            new_datatype_btn,
            pn.layout.Divider(),
            add_row_btn,
            del_row_btn,
            apply_btn,
            width=max(w + 60, 320),
            styles=self._STICKY,
        )
        content = pn.Column(
            pn.pane.Markdown(
                "**Run lists** (`runlists.yaml`) define named datasets for "
                "analyses (`snakemake valid-l200-...`). Edit one "
                "dataset/datatype at a time: one row per period, runs as a "
                "comma-separated list, ranges like `r000..r005`, or the "
                "single word `all` for every run of the period. An **empty "
                "table removes the datatype** (and an emptied dataset). New "
                "datasets/datatypes appear in the file on Apply."
            ),
            table,
            sizing_mode="stretch_width",
            styles={"min-width": "0"},
        )
        return pn.Row(
            content,
            edit_panel,
            name="Runlists",
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Bad cycles
    # ------------------------------------------------------------------

    def _raw_dirs(self) -> list[Path]:
        paths = self.prod_config.get("paths", {})
        candidates = [
            paths.get("tier_raw"),
            Path(paths["tier"]) / "raw" if paths.get("tier") else None,
            paths.get("tier_daq"),
        ]
        return [Path(c) for c in candidates if c and Path(c).is_dir()]

    def _staged_cycles(self) -> set[str]:
        """Cycle ids added to ignored_daq_cycles.yaml but not yet pushed."""
        staged = set()
        try:
            for line in meta_git.diff(self.metadb.datasets_path).splitlines():
                if line.startswith("+  - "):
                    cid = line[5:].split("#")[0].strip()
                    if meta_edit.CYCLE_RE.match(cid):
                        staged.add(cid)
        except meta_git.GitError:
            log.debug("could not read the staged-cycles diff", exc_info=True)
        return staged

    @param.depends("meta_version")
    def view_cycles_table(self):
        if self.metadb is None:
            return pn.pane.Markdown("**metadata clone unavailable**")
        cycles = meta_edit.list_ignored_cycles(self.metadb.datasets_path)
        staged = self._staged_cycles()
        cycles_df = pd.DataFrame(
            [
                {"cycle": cid, "reason": reason, "staged": cid in staged}
                for cid, reason in cycles
            ],
            columns=["cycle", "reason", "staged"],
        )
        table = pn.widgets.Tabulator(
            cycles_df,
            show_index=False,
            selectable=1,
            disabled=True,
            pagination="local",
            page_size=25,
            sizing_mode="stretch_width",
        )
        remove_btn = pn.widgets.Button(
            name="Remove selected (staged entries only)", width=260
        )

        def _remove(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            if not table.selection:
                return
            row = table.value.iloc[table.selection[0]]
            if not row["staged"]:
                self._fail("only staged (not yet pushed) entries can be removed here")
                return
            meta_edit.remove_ignored_cycle(self.metadb.datasets_path, row["cycle"])
            self._bump()

        remove_btn.on_click(_remove)
        return pn.Column(table, remove_btn, sizing_mode="stretch_width")

    def _build_cycles_tab(self, widget_widths):
        w = max(widget_widths, 200)
        reason_in = pn.widgets.TextInput(
            name="Reason", placeholder="e.g. empty / DAQ crash", width=2 * w
        )

        raw_dirs = self._raw_dirs()
        # directory names only: runs and cycle ids are read for the chosen
        # period/run below, so opening this tab never walks the whole tier
        datatypes, periods = meta_edit.raw_datatypes_and_periods(raw_dirs)
        period_sel = pn.widgets.Select(name="Period", options=periods or ["—"], width=w)
        run_sel = pn.widgets.Select(name="Run", options=[], width=w)

        now = datetime.now(UTC).replace(tzinfo=None)
        start_pick = pn.widgets.DatetimePicker(
            name="From (UTC)", value=now - timedelta(days=1), width=w
        )
        end_pick = pn.widgets.DatetimePicker(name="To (UTC)", value=now, width=w)
        dtype_choice = pn.widgets.MultiChoice(
            name="Datatypes (empty = all)", options=[], width=2 * w
        )
        find_btn = pn.widgets.Button(name="Find cycles", width=w)
        preview = pn.widgets.Tabulator(
            pd.DataFrame(columns=["cycle"]),
            show_index=False,
            selectable="checkbox",
            height=250,
            sizing_mode="stretch_width",
        )
        add_sel_btn = pn.widgets.Button(
            name="Add selected", button_type="primary", width=w, disabled=True
        )

        dtype_choice.options = datatypes

        def _show_cycles(cycles: list[str]) -> None:
            preview.value = pd.DataFrame({"cycle": cycles})
            preview.selection = []
            add_sel_btn.disabled = not cycles

        def _on_period(*_events):
            runs = meta_edit.raw_runs(raw_dirs, period_sel.value)
            run_sel.options = runs
            if runs:
                run_sel.value = runs[0]
            else:
                _show_cycles([])

        def _on_run(*_events):
            if not run_sel.value:
                _show_cycles([])
                return
            _show_cycles(
                meta_edit.raw_cycles(raw_dirs, period_sel.value, run_sel.value)
            )

        period_sel.param.watch(_on_period, "value")
        run_sel.param.watch(_on_run, "value")
        if periods:
            _on_period()

        def _find(_event):
            self.alert.visible = False
            try:
                found = meta_edit.find_cycles_in_range(
                    raw_dirs,
                    start_pick.value,
                    end_pick.value,
                    datatypes=dtype_choice.value or None,
                )
            except Exception as exc:
                self._fail(f"**cycle search failed** — {exc}")
                return
            _show_cycles(found)
            preview.selection = list(range(len(found)))
            if not found:
                self._fail("no cycles found in that time range")

        find_btn.on_click(_find)

        def _add_selected(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            ids = list(preview.value.iloc[preview.selection]["cycle"])
            if not ids:
                return
            try:
                added = meta_edit.add_ignored_cycles(
                    self.metadb.datasets_path, ids, reason_in.value
                )
            except Exception as exc:
                self._fail(f"**could not add cycles** — {exc}")
                return
            _show_cycles([])
            self._bump()
            self._ok(f"staged {len(added)} bad cycle(s)")

        add_sel_btn.on_click(_add_selected)

        if not raw_dirs:
            find_btn.disabled = True
            period_sel.disabled = run_sel.disabled = True
            find_note = "no raw-tier directory found in this production cycle"
        else:
            find_note = "cycles are listed from raw-tier file names in " + ", ".join(
                f"`{d}`" for d in raw_dirs
            )

        return pn.Column(
            pn.pane.Markdown(f"*{find_note}*"),
            reason_in,
            pn.pane.Markdown(
                "**Pick cycles by run** — the table below lists that run's "
                "cycles; tick the bad ones"
            ),
            pn.Row(period_sel, run_sel),
            pn.pane.Markdown(
                "**… or find all cycles in a time range** (pre-selects every " "match)"
            ),
            pn.Row(start_pick, end_pick, dtype_choice),
            pn.Row(find_btn, add_sel_btn),
            preview,
            pn.layout.Divider(),
            pn.pane.Markdown("**Currently ignored cycles**"),
            pn.param.ParamMethod(self.view_cycles_table, lazy=True),
            name="Bad cycles",
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Commit & Push
    # ------------------------------------------------------------------

    @param.depends("meta_version", "workspace")
    def view_pending(self):
        if self.metadb is None:
            return pn.pane.Markdown("**metadata clone unavailable**")
        if not self.workspace:
            return pn.pane.Markdown(
                "*open a workspace (top of the page) to stage and review changes*"
            )
        try:
            entries = meta_git.status(self.metadb.datasets_path)
            if not entries:
                return pn.pane.Markdown("*no pending changes*")
            listing = "\n".join(f"- `{state}` {f}" for state, f in entries)
            diff_text = meta_git.diff(self.metadb.datasets_path)
        except meta_git.GitError as exc:
            return pn.pane.Markdown(f"**git error**: {exc}")
        return pn.Column(
            pn.pane.Markdown(listing),
            pn.pane.Markdown(
                f"```diff\n{diff_text}\n```",
                sizing_mode="stretch_width",
                styles={"max-height": "500px", "overflow-y": "auto"},
            ),
            sizing_mode="stretch_width",
        )

    def _build_git_tab(self, widget_widths):
        w = max(widget_widths, 200)
        refresh_btn = pn.widgets.Button(name="Refresh", icon="refresh", width=w)
        refresh_btn.on_click(lambda _e: self.param.trigger("meta_version"))

        message_in = pn.widgets.TextAreaInput(
            name="Commit message",
            placeholder="what changed and why",
            height=100,
            sizing_mode="stretch_width",
        )
        username_in = pn.widgets.TextInput(
            name="GitHub username", placeholder="your-github-user", width=w
        )
        self._username_in = username_in  # prefilled when a workspace opens
        token_in = pn.widgets.PasswordInput(
            name="GitHub token",
            placeholder="fine-grained token with contents:write on your fork",
            width=2 * w,
        )
        push_btn = pn.widgets.Button(
            name="Commit & Push to my fork", button_type="primary", width=2 * w
        )

        async def _push(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            message = message_in.value.strip()
            username = username_in.value.strip()
            token = token_in.value
            if not (message and username and token):
                self._fail("commit message, username and token are all required")
                return
            push_btn.loading = True
            try:
                pr_url = await asyncio.to_thread(
                    meta_git.commit_and_push,
                    self.metadb.datasets_path,
                    message,
                    username,
                    token,
                )
            except meta_git.GitError as exc:
                self._fail(f"**push failed** — {exc}")
                return
            finally:
                token_in.value = ""
                push_btn.loading = False
                self._bump()
            self._ok(
                f"pushed to your fork — [**open a pull request**]({pr_url}) "
                "against legend-exp/legend-datasets"
            )

        push_btn.on_click(_push)

        discard_confirm = pn.widgets.Checkbox(
            name="yes, throw away every pending change"
        )
        discard_btn = pn.widgets.Button(
            name="Discard all changes", button_type="danger", width=w
        )

        def _discard(_event):
            self.alert.visible = False
            if not self._require_workspace():
                return
            if not discard_confirm.value:
                self._fail("tick the confirmation box to discard")
                return
            try:
                meta_git.discard_all(self.metadb.datasets_path)
            except meta_git.GitError as exc:
                self._fail(f"**discard failed** — {exc}")
                return
            discard_confirm.value = False
            self._bump()
            self._ok("all pending changes discarded")

        discard_btn.on_click(_discard)

        return pn.Column(
            pn.pane.Alert(
                "Edits stay in **your workspace** until pushed, separate from "
                "other workspaces — but anyone signed in who opens the same "
                "workspace name will see them. Un-pushed edits survive page "
                "reloads (open the same workspace name again) but **not "
                "container restarts** — push finished work.",
                alert_type="warning",
                sizing_mode="stretch_width",
            ),
            refresh_btn,
            pn.param.ParamMethod(self.view_pending, lazy=True),
            pn.layout.Divider(),
            message_in,
            pn.Row(username_in, token_in),
            pn.pane.Markdown(
                "Pushes go to `https://github.com/<username>/legend-datasets` — "
                "you need your own fork of legend-exp/legend-datasets and a "
                "token that can write to it. Credentials are used once and "
                "never stored."
            ),
            push_btn,
            pn.layout.Divider(),
            pn.Row(discard_btn, discard_confirm),
            name="Commit & Push",
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # page assembly
    # ------------------------------------------------------------------

    @param.depends("workspace")
    def view_workspace_state(self):
        if not self.workspace:
            return pn.pane.Alert(
                "**No workspace open — the page is read-only.** Enter your "
                "GitHub username and press *Open workspace* to edit. "
                "Re-opening the same name restores un-pushed edits.",
                alert_type="secondary",
                sizing_mode="stretch_width",
            )
        return pn.pane.Alert(
            f"Workspace **`{self.workspace}`** open — edits stay here until pushed.",
            alert_type="primary",
            sizing_mode="stretch_width",
        )

    def _build_workspace_bar(self, widget_widths):
        w = max(widget_widths, 180)
        ws_input = pn.widgets.TextInput(
            name="Workspace", placeholder="your GitHub username", width=w
        )
        open_btn = pn.widgets.Button(
            name="Open workspace", button_type="primary", icon="folder", width=w
        )

        async def _open(_event):
            self.alert.visible = False
            name = ws_input.value.strip()
            open_btn.loading = True
            try:
                ws = await asyncio.to_thread(
                    meta_git.ensure_workspace, self.meta_path, name
                )
            except meta_git.GitError as exc:
                self._fail(f"**could not open workspace** — {exc}")
                return
            finally:
                open_btn.loading = False
            self.metadb = get_meta_db(self.meta_path, datasets_path=ws)
            self._workspace_path = ws
            if self._username_in is not None and not self._username_in.value:
                self._username_in.value = name
            self.workspace = name
            self._bump()
            try:
                pending = meta_git.status(ws)
            except meta_git.GitError:
                pending = []
            restored = (
                f" — {len(pending)} un-pushed change(s) restored" if pending else ""
            )
            self._ok(f"workspace **{name}** open{restored}")

        open_btn.on_click(_open)
        return pn.Row(
            ws_input,
            pn.Column(pn.Spacer(height=18), open_btn),
            pn.Column(pn.Spacer(height=10), self.view_workspace_state),
            sizing_mode="stretch_width",
        )

    def build_metadata_pane(self, widget_widths: int = 140):
        if self.metadb is None:
            return pn.Column(
                pn.pane.Alert(
                    f"The editable metadata clone at `{self.meta_path}` is "
                    "unavailable — the Metadata editor is disabled. See the "
                    "server log for details.",
                    alert_type="danger",
                ),
                name="Metadata Editor",
                sizing_mode="stretch_width",
            )
        tabs = pn.Tabs(
            self._build_overview_tab(widget_widths),
            self._build_status_tab(widget_widths),
            self._build_partitions_tab(widget_widths),
            self._build_runlists_tab(widget_widths),
            self._build_cycles_tab(widget_widths),
            self._build_git_tab(widget_widths),
            dynamic=True,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.Row(
                pn.pane.SVG(logo_path / "Metadata.svg", height=25),
                pn.pane.Markdown("## Metadata viewer & editor"),
            ),
            self._build_workspace_bar(widget_widths),
            self.alert,
            tabs,
            name="Metadata Editor",
            sizing_mode="stretch_width",
        )


def run_dashboard_metaedit() -> None:
    """Standalone entry point: serve only the Metadata editor page.

    Still needs a production cycle at ``paths: cal`` (period/run catalogue,
    raw-tier cycle discovery) next to the editable clone at
    ``paths: metadata_edit``.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    argparser.add_argument(
        "--websocket-origin",
        nargs="*",
        default=None,
        help="allowed websocket origin host(s) when serving behind a proxy",
    )
    args = argparser.parse_args()

    config = read_config(args.config_file)
    meta_git.ensure_clone(
        config.metadata_edit,
        os.environ.get("METADATA_EDIT_URL", meta_git.DEFAULT_URL),
    )

    def _build():
        monitor = MetaMonitoring(
            base_path=config.cal,
            meta_path=config.metadata_edit,
            name="L200 Metadata Editor",
        )
        # same template/branding as the full dashboard (see dashboard.py)
        template = pn.template.FastListTemplate(
            header_background="#f8f8fa",
            header_color="#1A2A5B",
            title="L200 Metadata Editor",
            sidebar_width=300,
            main_layout=None,
            site="",
            logo="https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Images/Logo/logo_legend_tag_next.svg",
            favicon="https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Favicons/android-chrome-96x96.png",
        )
        template.sidebar.append(monitor.build_sidebar())
        template.main.append(monitor.build_metadata_pane(args.widget_widths))
        return template

    serve_kwargs = {"port": args.port, "show": False, "address": "0.0.0.0"}
    if args.websocket_origin:
        serve_kwargs["websocket_origin"] = args.websocket_origin
    print("Starting Metadata editor on port ", args.port)  # noqa: T201
    pn.serve(_build, **serve_kwargs)
