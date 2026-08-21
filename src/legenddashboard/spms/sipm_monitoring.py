"""SiPM page: the spms flavour of the run contract, drawn like the pipeline.

The run manifest lists ``l200-<p>-<r>-phy-spms-schema2.hdf`` next to the
geds file; its ``hist/<flag>_<param>/<cadence>`` groups have a SiPM-name
axis and ``/detector_map`` carries barrel, fiber and position. The page
reuses the geds contract reader and time-series builder and groups SiPMs
like the pipeline's headline PNGs (barrel x position) or by fiber (the
top/bottom pair). The period file adds ``spms_noise/<run>`` (per-cycle
baseline_curr_fwhm from the par files), ``spms_calibration/<run>`` and
the run's ``qcp_summary.yaml`` carries the channel-health verdicts.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import panel as pn
import param
import yaml
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.models.formatters import PrintfTickFormatter

from legenddashboard.base import Monitoring
from legenddashboard.geds.phy import contract_reader, period_reader, plot_style
from legenddashboard.geds.phy.phy_plots import (
    PlotMeta,
    phy_plot_binned_vsTime,
    phy_plot_dist_histogram,
)
from legenddashboard.util import LRUDict, logo_path

log = logging.getLogger(__name__)

GROUPINGS = ("Barrel x position", "Fiber")
STYLES = ("Time", "Histogram")
UNITS = ("Absolute", "Relative")
SUMMARIES = ("Explorer", "Noise cross-check", "Channel health", "PE calibration")

_qcp_cache = LRUDict(maxsize=32)


def _stat_key(path):
    st = Path(path).stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def read_qcp_summary(path):
    """The run's qcp_summary.yaml as a dict (cached per file version); {} if absent."""
    try:
        key = _stat_key(path)
    except OSError:
        return {}
    if key not in _qcp_cache:
        with Path(path).open() as f:
            _qcp_cache[key] = yaml.safe_load(f) or {}
    return _qcp_cache[key]


def group_labels(detector_map, grouping):
    """Ordered {group label: [SiPM names]} from the spms detector map."""
    if detector_map is None or detector_map.empty:
        return {}
    dmap = detector_map.copy()
    dmap["position"] = dmap["position"].astype(str)
    dmap = dmap.sort_values(
        ["barrel", "fiber", "position"], ascending=[True, True, False]
    )
    groups = {}
    if grouping == "Fiber":
        for fiber, rows in dmap.groupby("fiber", sort=True):
            groups[str(fiber)] = [str(n) for n in rows["name"]]
        return groups
    for barrel in ("IB", "OB"):
        for position in ("top", "bottom"):
            rows = dmap[(dmap["barrel"] == barrel) & (dmap["position"] == position)]
            if len(rows):
                groups[f"{barrel} {position}"] = [str(n) for n in rows["name"]]
    return groups


def calibration_staleness(source, period):
    """(source period/run, periods stale) parsed from a calibration source path."""
    match = re.search(r"(p\d{2})/(r\d{3})", str(source))
    if not match:
        return None, None
    src_period, src_run = match.groups()
    try:
        stale = int(period[1:]) - int(src_period[1:])
    except ValueError:
        stale = None
    return f"{src_period}/{src_run}", stale


class SiPMMonitoring(Monitoring):
    """SiPM explorer and summaries from the spms contract."""

    phy_path = param.String("")
    sipm_view = param.ObjectSelector(default=SUMMARIES[0], objects=list(SUMMARIES))
    sipm_group_by = param.ObjectSelector(default=GROUPINGS[0], objects=list(GROUPINGS))
    sipm_group = param.ObjectSelector(default=None, objects=[], label="Group")
    sipm_plots_types = param.ObjectSelector(
        default=next(iter(plot_style.SPMS_TYPES)), objects=list(plot_style.SPMS_TYPES)
    )
    sipm_plots = param.ObjectSelector(
        default=next(iter(plot_style.SPMS_VALUES)), objects=list(plot_style.SPMS_VALUES)
    )
    sipm_units = param.ObjectSelector(default=UNITS[0], objects=list(UNITS))
    sipm_plot_style = param.ObjectSelector(default=STYLES[0], objects=list(STYLES))
    sipm_resampled = param.Integer(default=10, bounds=(0, 60))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._value_menu = None
        self._group_menu = None
        # menus follow the run's manifest and detector map; registered before
        # the view pane so a run change re-populates them before rendering
        self.param.watch(
            self._update_menus,
            ["run_dict", "run", "sipm_group_by", "sipm_plots_types", "sipm_plots"],
        )
        if self.run_dict:
            self._update_menus()

    # ------------------------------------------------------------------
    # contract lookups
    # ------------------------------------------------------------------

    def _run_dir(self):
        return Path(self.phy_path) / "generated/plt/hit/phy" / self.period / self.run

    def _manifest(self):
        if not self.run_dict or self.run not in self.run_dict:
            return None
        return contract_reader.find_manifest(
            self.phy_path, self.period, self.run, self.run_dict[self.run]["experiment"]
        )

    def _spms_file(self, manifest=None):
        manifest = self._manifest() if manifest is None else manifest
        if manifest is None:
            return None
        path = contract_reader.file_from_manifest(manifest, self._run_dir(), "spms")
        return path if path is not None and path.exists() else None

    def _detector_map(self):
        path = self._spms_file()
        if path is None:
            return None
        try:
            return contract_reader.read_detector_map(path)
        except KeyError:
            return None

    def _groups(self):
        return group_labels(self._detector_map(), self.sipm_group_by)

    def _param_key(self):
        return plot_style.SPMS_VALUES[self.sipm_plots] + (
            "_var" if self.sipm_units == "Relative" else ""
        )

    def _update_menus(self, *events):  # noqa: ARG002
        manifest = self._manifest()
        keys = contract_reader.available_keys(manifest, "spms") if manifest else set()
        if keys:
            types = [
                t
                for t, flag in plot_style.SPMS_TYPES.items()
                if any(f"{flag}_{p}" in keys for p in plot_style.SPMS_VALUES.values())
            ] or list(plot_style.SPMS_TYPES)
            flag = plot_style.SPMS_TYPES[
                self.sipm_plots_types if self.sipm_plots_types in types else types[0]
            ]
            values = [
                v for v, p in plot_style.SPMS_VALUES.items() if f"{flag}_{p}" in keys
            ] or list(plot_style.SPMS_VALUES)
        else:
            types, values = list(plot_style.SPMS_TYPES), list(plot_style.SPMS_VALUES)
        self.param.sipm_plots_types.objects = types
        self.param.sipm_plots.objects = values
        if self._value_menu is not None:
            self._value_menu.items = values
        if self.sipm_plots_types not in types:
            self.sipm_plots_types = types[0]
        if self.sipm_plots not in values:
            self.sipm_plots = values[0]
        groups = list(self._groups())
        self.param.sipm_group.objects = groups
        if self._group_menu is not None:
            self._group_menu.items = groups
        if groups and self.sipm_group not in groups:
            self.sipm_group = groups[0]
        elif not groups:
            self.sipm_group = None

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------

    def _empty(self, reason):
        return plot_style.empty_figure(
            f"{reason} for {self.period} {self.run}", height=600
        )

    @param.depends(
        "run_dict",
        "run",
        "sipm_view",
        "sipm_group_by",
        "sipm_group",
        "sipm_plots_types",
        "sipm_plots",
        "sipm_units",
        "sipm_plot_style",
        "sipm_resampled",
    )
    def update_sipm_plot(self):
        start = time.time()
        try:
            if self._spms_file() is None:
                return self._empty("No spms contract (backfill pending?)")
            if self.sipm_view == "Explorer":
                pane = self._explorer()
            elif self.sipm_view == "Noise cross-check":
                pane = self._noise_cross_check()
            elif self.sipm_view == "Channel health":
                pane = self._channel_health()
            else:
                pane = self._calibration()
        except Exception:
            log.exception(
                "Failed to build SiPM view %s for %s %s",
                self.sipm_view,
                self.period,
                self.run,
            )
            pane = self._empty(f"Could not build {self.sipm_view}")
        log.debug("Time to get sipm plot: %.3fs", time.time() - start)
        return pane

    def _meta(self, label, unit, abs_unit, param_display=None, flag_display=None):
        return PlotMeta(
            label=label,
            unit=unit,
            abs_unit=abs_unit,
            flag_display=flag_display or f"SiPM {self.sipm_plots_types}",
            param_display=param_display or self.sipm_plots,
            string=self.sipm_group or "",
            run=self.run,
            period=self.period,
            experiment=self.run_dict[self.run]["experiment"],
        )

    def _explorer(self):
        manifest = self._manifest()
        data_file = self._spms_file(manifest)
        flag = plot_style.SPMS_TYPES[self.sipm_plots_types]
        key_param = self._param_key()
        key = f"{flag}_{key_param}"
        if key not in contract_reader.available_keys(manifest, "spms"):
            return self._empty(f"Key {key} missing")
        names = self._groups().get(self.sipm_group, [])
        if not names:
            return self._empty(f"No SiPMs in group {self.sipm_group}")
        relative = self.sipm_units == "Relative"
        cadence = contract_reader.snap_cadence(
            self.sipm_resampled, manifest.get("cadences", ["1min"])
        )
        series = contract_reader.read_binned(data_file, flag, key_param, cadence)
        label, unit = contract_reader.label_and_unit(
            manifest, series.attrs, key_param, relative
        )
        meta = self._meta(label, unit, series.attrs.get("unit", unit))
        # the contract's own limits are null for spms keys: the grading bands
        # of mtg-plot-settings apply (keyed by the graded key kind)
        limits = contract_reader.limits(series.attrs)
        band = plot_style.MTG_PLOT_INFO.get(key, {}).get("limits")
        if band is not None and limits == (None, None):
            limits = band
        if self.sipm_plot_style == "Histogram":
            if f"{key}_dist" not in contract_reader.available_keys(manifest, "spms"):
                return self._empty("No distribution stored")
            return phy_plot_dist_histogram(
                contract_reader.read_dist(data_file, flag, key_param), meta, limits
            )
        names = [n for n in names if n in series.detectors]
        if not names:
            return self._empty(f"No SiPMs of {self.sipm_group} in the contract")
        dashes = {}
        if self.sipm_group_by == "Fiber":
            dmap = self._detector_map().set_index("name")
            dashes = {
                n: "dashed" for n in names if str(dmap.loc[n, "position"]) == "bottom"
            }
        return phy_plot_binned_vsTime(
            mean_df=series.to_frame("mean")[names],
            std_df=series.to_frame("std")[names],
            min_df=series.to_frame("min")[names],
            max_df=series.to_frame("max")[names],
            meta=meta,
            flagged_ranges=contract_reader.flagged_ranges(manifest),
            cadence_label=cadence,
            limits=limits,
            envelope="HasAnyNoise" not in key_param,  # boolean rate: min/max are 0/1
            line_dashes=dashes,
        )

    def _noise_cross_check(self):
        """Par-file baseline_curr_fwhm (dotted) over the contract's CurrFwhm mean."""
        manifest = self._manifest()
        data_file = self._spms_file(manifest)
        names = self._groups().get(self.sipm_group, [])
        if not names:
            return self._empty(f"No SiPMs in group {self.sipm_group}")
        noise = period_reader.read_optional(
            period_reader.period_file(self.phy_path, self.period),
            f"spms_noise/{self.run}",
        )
        if "IsBsln_CurrFwhm" not in contract_reader.available_keys(manifest, "spms"):
            return self._empty("Key IsBsln_CurrFwhm missing")
        series = contract_reader.read_binned(data_file, "IsBsln", "CurrFwhm", "60min")
        names = [n for n in names if n in series.detectors]
        label, unit = contract_reader.label_and_unit(
            manifest, series.attrs, "CurrFwhm", False
        )
        meta = self._meta(
            label, unit, unit, "Current noise FWHM", "SiPM Forced trigger"
        )
        p = phy_plot_binned_vsTime(
            mean_df=series.to_frame("mean")[names],
            std_df=series.to_frame("std")[names],
            min_df=series.to_frame("min")[names],
            max_df=series.to_frame("max")[names],
            meta=meta,
            cadence_label="60min",
            envelope=False,
        )
        p.title.text += " | dotted: par-file baseline_curr_fwhm"
        if noise is None:
            p.add_layout(
                plot_style.Label(
                    x=10,
                    y=-10,
                    x_units="screen",
                    y_units="screen",
                    text=f"spms_noise/{self.run} not in the period file",
                    text_font_size="9px",
                    text_color="dimgray",
                )  # fmt: skip
            )
            return p
        index = plot_style.utc_naive(noise.index)
        for n in names:
            if n in noise.columns:
                p.line(
                    index, noise[n].to_numpy(dtype=float), line_dash="dotted",
                    line_width=1.5, color="black", alpha=0.6, legend_label=n,
                )  # fmt: skip
        plot_style.finish_legend(p, "bottom_left")
        return p

    def _channel_health(self):
        """qcp_summary spms_* verdicts per SiPM (pass/fail marker rows)."""
        qcp = read_qcp_summary(
            self._run_dir()
            / f"{self.run_dict[self.run]['experiment']}-{self.period}-{self.run}-qcp_summary.yaml"
        )
        dmap = self._detector_map()
        if dmap is None:
            return self._empty("No spms detector map")
        order = group_labels(dmap, "Barrel x position")
        factors, groups = [], []
        for group, names in order.items():
            for n in names:
                factors.append(n)
                groups.append(group)
        verdicts = {n: (qcp.get(n, {}) or {}).get("phy", {}) or {} for n in factors}
        checks = [
            c for c in plot_style.SPMS_CHECKS if any(c in v for v in verdicts.values())
        ]
        if not checks:
            return self._empty("No spms_* verdicts in qcp_summary.yaml")
        failing = sorted(
            n for n, v in verdicts.items() if any(v.get(c) is False for c in checks)
        )
        title = f"{self.period} {self.run} - SiPM channel health"
        title += f" - failing: {', '.join(failing)}" if failing else " - all SiPMs pass"
        p = plot_style.make_figure(
            title, x_range=FactorRange(*factors), y_range=FactorRange(*checks),
            height=320, tools="pan,reset,save",
        )  # fmt: skip
        xs, ys, colors, states = [], [], [], []
        for n in factors:
            for c in checks:
                verdict = verdicts[n].get(c)
                if verdict is None:
                    continue
                xs.append(n)
                ys.append(c)
                colors.append("seagreen" if verdict else "red")
                states.append("pass" if verdict else "FAIL")
        source = ColumnDataSource({"x": xs, "y": ys, "color": colors, "state": states})
        glyph = p.scatter(
            x="x", y="y", source=source, marker="square", size=9, color="color"
        )
        p.add_tools(
            HoverTool(
                renderers=[glyph],
                tooltips=[("SiPM", "@x"), ("check", "@y"), ("", "@state")],
            )
        )
        p.xaxis.major_label_orientation = 1.2
        p.xaxis.major_label_text_font_size = "8px"
        # group separators
        start = 0
        for i in range(1, len(factors) + 1):
            if i == len(factors) or groups[i] != groups[start]:
                p.add_layout(
                    plot_style.Label(
                        x=start,
                        y=len(checks) - 0.4,
                        x_units="data",
                        y_units="data",
                        text=groups[start],
                        text_font_size="9px",
                        text_color="dimgray",
                    )  # fmt: skip
                )
                start = i
        return p

    def _calibration(self):
        """The PE calibration in force, with its source and staleness."""
        cal = period_reader.read_optional(
            period_reader.period_file(self.phy_path, self.period),
            f"spms_calibration/{self.run}",
        )
        if cal is None:
            return self._empty(f"spms_calibration/{self.run} not in the period file")
        dmap = self._detector_map()
        table = cal.reset_index().rename(columns={"index": "name"})
        if "name" not in table and "level_0" in table:
            table = table.rename(columns={"level_0": "name"})
        if dmap is not None:
            table = table.merge(
                dmap[["name", "barrel", "fiber", "position"]], on="name", how="left"
            )
        cols = [
            c
            for c in (
                "name",
                "barrel",
                "fiber",
                "position",
                "pe_a",
                "pe_m",
                "threshold_a",
            )
            if c in table
        ]
        source, stale = calibration_staleness(
            table["source"].iloc[0] if "source" in table and len(table) else "",
            self.period,
        )
        if source is None:
            banner = "PE calibration source unknown"
        else:
            banner = f"PE calibration source: {source}"
            if stale:
                banner += f" — **{stale} period(s) stale**"
        return pn.Column(
            pn.pane.Markdown(f"### {self.period} {self.run} - {banner}"),
            pn.widgets.Tabulator(
                table[cols].round(5),
                disabled=True,
                height=500,
                pagination=None,
                sizing_mode="stretch_width",
            ),  # fmt: skip
        )

    # ------------------------------------------------------------------
    # pane
    # ------------------------------------------------------------------

    def build_sipm_pane(self, widget_widths=140):
        """The SiPM tab: view / grouping / type / value selectors and the figure."""

        def radio(name, **kw):
            return pn.Param(
                self.param,
                widgets={
                    name: {
                        "widget_type": pn.widgets.RadioBoxGroup,
                        "inline": True,
                        **kw,
                    }
                },
                parameters=[name],
                show_labels=False,
                show_name=False,
            )

        def header(name):
            return pn.widgets.Button(
                name=name, button_type="primary", width=widget_widths, disabled=True
            )

        types = pn.Param(
            self.param,
            widgets={
                "sipm_plots_types": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                    "width": widget_widths,
                }
            },
            parameters=["sipm_plots_types"],
            show_labels=False,
            show_name=False,
        )
        resampled = pn.Param(
            self.param,
            widgets={
                "sipm_resampled": {
                    "widget_type": pn.widgets.IntSlider,
                    "width": widget_widths,
                    "format": PrintfTickFormatter(format="%d min"),
                    "value_throttled": True,
                }
            },
            parameters=["sipm_resampled"],
            show_labels=False,
            show_name=False,
        )
        current = pn.pane.Markdown(f"## {self.sipm_plots}")
        value_menu = pn.widgets.MenuButton(
            name="SiPM value", button_type="primary", width=widget_widths,
            items=self.param.sipm_plots.objects,
        )  # fmt: skip

        def pick_value(event):
            self.sipm_plots = event.new
            current.object = f"## {event.new}"

        value_menu.on_click(pick_value)
        self._value_menu = value_menu
        group_menu = pn.widgets.MenuButton(
            name="Group", button_type="success", width=widget_widths,
            items=self.param.sipm_group.objects,
        )  # fmt: skip
        current_group = pn.pane.Markdown(f"## {self.sipm_group or ''}")

        def pick_group(event):
            self.sipm_group = event.new
            current_group.object = f"## {event.new}"

        group_menu.on_click(pick_group)
        self._group_menu = group_menu
        self.param.watch(
            lambda e: setattr(current_group, "object", f"## {e.new or ''}"),
            "sipm_group",
        )

        grid = pn.GridSpec(width=5 * widget_widths + 20, max_height=800)
        grid[:, 0] = types
        grid[:, 1] = pn.Spacer(width=5)
        grid[0, 2] = header("Units")
        grid[1, 2] = radio("sipm_units")
        grid[:, 3] = pn.Spacer(width=5)
        grid[0, 4] = header("Resampled")
        grid[1, 4] = resampled
        grid[:, 5] = pn.Spacer(width=5)
        grid[0, 6] = header("Style")
        grid[1, 6] = radio("sipm_plot_style")
        grid[:, 7] = pn.Spacer(width=5)
        grid[0, 8] = header("Group by")
        grid[1, 8] = radio("sipm_group_by")

        return pn.Column(
            pn.Row(
                pn.pane.SVG(logo_path / "Physics.svg", height=25),
                value_menu,
                group_menu,
            ),
            pn.Row(
                "## Current Plot:",
                current,
                pn.Spacer(width=20),
                "## Group:",
                current_group,
            ),
            pn.Row("View:", radio("sipm_view")),
            pn.Row(grid),
            pn.param.ParamMethod(
                self.update_sipm_plot, lazy=True, loading_indicator=True
            ),
            name="SiPM",
            sizing_mode="stretch_width",
        )
