from __future__ import annotations

import bisect
import datetime as dtt
import logging
import time
from datetime import datetime

import numpy as np
import panel as pn
import param
from bokeh.io import output_notebook
from bokeh.resources import INLINE

from legenddashboard.util import (
    get_dataflow_config,
    get_run_dict,
    get_sort_dets,
    logo_path,
)

log = logging.getLogger(__name__)


class Monitoring(param.Parameterized):
    """
    Base class for monitoring dashboards.
    """

    base_path = param.String("", allow_refs=True, nested_refs=True)
    prod_config = param.Dict({}, allow_refs=True, nested_refs=True)
    tier_dict = param.Dict({}, allow_refs=True, nested_refs=True)
    period = param.Selector(
        default="p00",
        objects=[f"p{i:02}" for i in range(100)],
        allow_refs=True,
        nested_refs=True,
    )
    run = param.Selector(
        default="r000",
        objects=[f"r{i:03}" for i in range(100)],
        allow_refs=True,
        nested_refs=True,
    )
    run_dict = param.Dict({}, allow_refs=True, nested_refs=True)
    periods = param.Dict({}, allow_refs=True, nested_refs=True)
    period_objects = param.List(default=[f"p{i:02}" for i in range(100)])
    run_objects = param.List(default=[f"r{i:03}" for i in range(100)])

    # Static default: a per-session range is set in __init__ (a datetime.now()
    # call here would be evaluated once at import and frozen for all sessions).
    date_range = param.DateRange(
        default=(
            datetime(2000, 1, 1, 0, 0, 0),
            datetime(2100, 1, 1, 0, 0, 0),
        ),
        bounds=(
            datetime(2000, 1, 1, 0, 0, 0),
            datetime(2100, 1, 1, 0, 0, 0),
        ),
        allow_refs=True,
        nested_refs=True,
    )

    def __init__(self, base_path, notebook=False, **params):
        if notebook is True:
            output_notebook(INLINE)
        self.base_path = base_path
        self.sort_obj = get_sort_dets(base_path)

        super().__init__(**params)

        if "date_range" not in params:
            now = datetime.now()
            self.date_range = (
                now - dtt.timedelta(minutes=10),
                now + dtt.timedelta(minutes=10),
            )

        self.tier_dict = {
            "raw": "raw",
            "tcm": "tcm",
            "dsp": "dsp",
            "hit": "hit",
            "evt": "evt",
        }

        if "ref-v" in str(self.base_path):
            self.tier_dict["dsp"] = "psp"
            self.tier_dict["hit"] = "pht"
            self.tier_dict["evt"] = "pet"

        self.prod_config = get_dataflow_config(self.base_path)
        if self.period == "p00":
            self.periods = get_run_dict(self.base_path)
            if not self.periods:
                msg = f"No runs found under {self.base_path}"
                raise RuntimeError(msg)
            self.period_objects = list(self.periods)
            self.period = list(self.periods)[-1]
            self._get_period_data()
            # Only the discovering instance owns the period -> run cascade.
            # Instances fed by refs (periods=..., run=...) must never assign
            # those parameters: assigning a ref-bound param deletes the ref.
            self.param.watch(self._get_period_data, ["period"], precedence=0)

    def _full_date_range(self):
        """(first run start - 100 min, last run start + 110 min) over all periods."""
        start_period = sorted(self.periods)[0]
        start_run = sorted(self.periods[start_period])[0]
        end_period = sorted(self.periods)[-1]
        end_run = sorted(self.periods[end_period])[-1]
        return (
            datetime.strptime(
                self.periods[start_period][start_run]["timestamp"], "%Y%m%dT%H%M%SZ"
            )
            - dtt.timedelta(minutes=100),
            datetime.strptime(
                self.periods[end_period][end_run]["timestamp"], "%Y%m%dT%H%M%SZ"
            )
            + dtt.timedelta(minutes=110),
        )

    def _set_run_state(self, run):
        """Apply the current period's run list, ``run`` and the date range.

        One batch: ref-following instances receive run_dict, run and
        date_range together, so their views render once with consistent
        state. Views key on run_dict (not period) so a new period whose
        latest run id repeats the current one still refreshes.
        """
        run_dict = self.periods[self.period]
        date_range = self._full_date_range()
        self.param["date_range"].bounds = date_range
        self.param.update(
            run_dict=run_dict,
            run_objects=list(run_dict),
            run=run,
            date_range=date_range,
        )

    def _get_period_data(self, *events):  # noqa: ARG002
        # Land on the latest run of the (new) period.
        self._set_run_state(list(self.periods[self.period])[-1])

    def _refresh_periods(self):
        """Scan for new periods/runs in this session and apply any changes."""
        self._apply_periods(get_run_dict(self.base_path, refresh=True))

    def _apply_periods(self, new_periods):
        """Apply an already-scanned periods dict to this session's state.

        Kept separate from the scan so a single shared scan can push the same
        result into many sessions (see util.PeriodRefreshRegistry).
        """
        if new_periods == self.periods:
            return
        self.periods = new_periods
        self.period_objects = list(new_periods)
        if self.period not in new_periods:
            self.period = list(new_periods)[-1]  # cascades via the watcher
        elif new_periods[self.period] != self.run_dict:
            # New runs in the viewed period: refresh the run list but keep
            # the user's selection.
            runs = new_periods[self.period]
            self._set_run_state(self.run if self.run in runs else list(runs)[-1])

    def _get_run_dict(self, event=None):  # noqa: ARG002
        start_time = time.time()
        valid_from = [
            datetime.timestamp(
                datetime.strptime(self.run_dict[entry]["timestamp"], "%Y%m%dT%H%M%SZ")
            )
            for entry in self.run_dict
        ]
        # A plain ``date`` (from the date-picker) needs expanding to the full
        # day; note ``datetime`` is a subclass of ``date``, so check it first.
        if isinstance(self.date_range[0], datetime):
            low_range = datetime.timestamp(self.date_range[0])
        else:
            low_range = datetime.timestamp(
                datetime.combine(self.date_range[0], datetime.min.time())
            )
        if isinstance(self.date_range[1], datetime):
            high_range = datetime.timestamp(self.date_range[1])
        else:
            high_range = datetime.timestamp(
                datetime.combine(self.date_range[1], datetime.max.time())
            )
        pos1 = bisect.bisect_right(valid_from, low_range)
        pos2 = bisect.bisect_left(valid_from, high_range)
        valid_idxs = np.arange(pos1, pos2, 1)
        valid_keys = np.array(list(self.run_dict))[valid_idxs]
        out_dict = {key: self.run_dict[key] for key in valid_keys}
        log.debug("Time to get run dict: %.3fs", time.time() - start_time)
        return out_dict

    def build_sidebar(self):
        run_param = pn.widgets.MenuButton(
            name=f"Run {int(self.run[1:]):02d}",
            button_type="primary",
            sizing_mode="stretch_width",
            items=self.run_objects,
        )

        def update_run(event):
            self.run = event.new
            run_param.name = f"Run {int(self.run[1:]):02d}"

        run_param.on_click(update_run)
        # run_param        = pn.Param(self.param, widgets={'run': {'widget_type': pn.widgets.Select, 'width': 100}}, parameters=['run'], show_labels=False, show_name=False, design=Bootstrap)
        period_param = pn.widgets.MenuButton(
            name=f"Period {int(self.period[1:]):02d}",
            button_type="primary",
            sizing_mode="stretch_width",
            items=self.period_objects,
        )

        def update_period(event):
            self.period = event.new
            run_param.items = self.run_objects
            run_param.name = f"Run {int(self.run[1:]):02d}"
            period_param.name = f"Period {int(self.period[1:]):02d}"

        period_param.on_click(update_period)

        # Keep the menu buttons in sync when the underlying parameters change
        # outside of a direct click, e.g. after a manual or periodic refresh
        # discovers new periods/runs (see _refresh_periods).
        self.param.watch(
            lambda event: setattr(period_param, "items", event.new), "period_objects"
        )
        self.param.watch(
            lambda event: setattr(
                period_param, "name", f"Period {int(event.new[1:]):02d}"
            ),
            "period",
        )
        self.param.watch(
            lambda event: setattr(run_param, "items", event.new), "run_objects"
        )
        self.param.watch(
            lambda event: setattr(run_param, "name", f"Run {int(event.new[1:]):02d}"),
            "run",
        )

        return pn.Column(
            pn.pane.SVG(
                logo_path / "Period.svg",
                height=25,
            ),
            period_param,
            pn.pane.SVG(
                logo_path / "Run.svg",
                height=25,
            ),
            run_param,
            sizing_mode="stretch_width",
        )

    @staticmethod
    def build_base(path, notebook=False):
        monitor = Monitoring(
            base_path=path,
            notebook=notebook,
        )

        return monitor.build_sidebar()
