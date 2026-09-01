"""Muon-veto page: the pmts flavour of the run contract.

The run manifest lists ``l200-<p>-<r>-phy-pmts-schema2.hdf`` next to the
geds and spms files; its ``hist/<flag>_<param>/<cadence>`` groups have a
PMT-name axis and ``/detector_map`` carries the location (pillbox, floor,
wall). The page reuses the contract reader and time-series builders. The
period file adds ``muon_veto/<run>`` (hourly muon rate, ge-coincidence
fractions, multiplicity and light-sum medians), and per-PMT pulse-height
spectra come from ``All_Pulseheight_dist2d`` — the calibration check that
replaces the retired muon pipeline's "Cal. Spectra".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import panel as pn
import param
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d, Span

from legenddashboard.base import Monitoring
from legenddashboard.geds.phy import contract_reader, period_reader, plot_style
from legenddashboard.geds.phy.phy_plots import (
    PlotMeta,
    phy_plot_binned_vsTime,
    phy_plot_dist_histogram,
)
from legenddashboard.spms.sipm_plots import spe_centroid
from legenddashboard.util import logo_path

log = logging.getLogger(__name__)

VIEWS = ("Explorer", "PMT spectra", "Veto summary")
GROUPS = ("pillbox", "floor", "wall")
STYLES = ("Time", "Histogram")
UNITS = ("Absolute", "Relative")

# display name -> contract parameter (labels/units come from the attrs)
MUON_VALUES = {
    "Event rate": "EventRate",
    "Baseline mean": "BlMean",
    "Baseline sigma": "BlSig",
    "Pulse height": "Pulseheight",
    "Max. peak height": "MaxPeakHeight",
    "Mean peak height": "MeanPeakHeight",
    "Pulse fraction": "Containspulse",
    "Muon light sum": "MuonLightSum",
    "Muon multiplicity": "MuonMultiplicity",
}
MUON_TYPES = {"All events": "All", "Muon events": "IsMuon"}

#: search settings for the single-photon peak in the 0-100 LSB spectra:
#: gains are not equalised, so the position is reported, not judged
SPP_SEARCH = {"window": (3.0, 90.0), "smooth": 15, "half": 2.0, "valley_max": 30.0}


class MuonMonitoring(Monitoring):
    """Muon-veto explorer, PMT spectra and veto summary from the pmts contract."""

    phy_path = param.String("")
    muon_view = param.ObjectSelector(default=VIEWS[0], objects=list(VIEWS))
    muon_group = param.ObjectSelector(default=GROUPS[0], objects=list(GROUPS))
    muon_plots_types = param.ObjectSelector(
        default=next(iter(MUON_TYPES)), objects=list(MUON_TYPES)
    )
    muon_plots = param.ObjectSelector(
        default=next(iter(MUON_VALUES)), objects=list(MUON_VALUES)
    )
    muon_units = param.ObjectSelector(default=UNITS[0], objects=list(UNITS))
    muon_plot_style = param.ObjectSelector(default=STYLES[0], objects=list(STYLES))
    muon_resampled = param.Integer(default=10, bounds=(0, 60))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._value_menu = None
        # menus follow the run's manifest; registered before the view pane so
        # a run change re-populates them before the figure renders
        self.param.watch(
            self._update_menus,
            ["run_dict", "run", "muon_plots_types", "muon_plot_style"],
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

    def _pmts_file(self, manifest=None):
        manifest = self._manifest() if manifest is None else manifest
        if manifest is None:
            return None
        path = contract_reader.file_from_manifest(manifest, self._run_dir(), "pmts")
        return path if path is not None and path.exists() else None

    def _detector_map(self):
        path = self._pmts_file()
        if path is None:
            return None
        try:
            return contract_reader.read_detector_map(path)
        except KeyError:
            return None

    def _group_names(self, group=None):
        """PMT names of one location group, in detector-map order."""
        dmap = self._detector_map()
        if dmap is None or "location" not in dmap:
            return []
        rows = dmap[dmap["location"].astype(str) == (group or self.muon_group)]
        return [str(n) for n in rows["name"]]

    def _param_key(self):
        return MUON_VALUES[self.muon_plots] + (
            "_var" if self.muon_units == "Relative" else ""
        )

    def _update_menus(self, *events):  # noqa: ARG002
        manifest = self._manifest()
        keys = contract_reader.available_keys(manifest, "pmts") if manifest else set()
        if keys:
            types = [
                t
                for t, flag in MUON_TYPES.items()
                if any(
                    f"{flag}_{p}" in keys or f"{flag}_{p}_dist" in keys
                    for p in MUON_VALUES.values()
                )
            ] or list(MUON_TYPES)
            flag = MUON_TYPES[
                self.muon_plots_types if self.muon_plots_types in types else types[0]
            ]

            # a parameter may be dist-only (e.g. MuonMultiplicity): offer it
            # for the Histogram style but not for the Time style
            def offered(p):
                if self.muon_plot_style == "Histogram":
                    return f"{flag}_{p}_dist" in keys
                return f"{flag}_{p}" in keys

            values = [v for v, p in MUON_VALUES.items() if offered(p)] or list(
                MUON_VALUES
            )
        else:
            types, values = list(MUON_TYPES), list(MUON_VALUES)
        self.param.muon_plots_types.objects = types
        self.param.muon_plots.objects = values
        if self._value_menu is not None:
            self._value_menu.items = values
        if self.muon_plots_types not in types:
            self.muon_plots_types = types[0]
        if self.muon_plots not in values:
            self.muon_plots = values[0]

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
        "muon_view",
        "muon_group",
        "muon_plots_types",
        "muon_plots",
        "muon_units",
        "muon_plot_style",
        "muon_resampled",
    )
    def update_muon_plot(self):
        start = time.time()
        try:
            if self._pmts_file() is None:
                pane = self._empty("No pmts contract (producer run needed?)")
            elif self.muon_view == "PMT spectra":
                pane = self._spectra()
            elif self.muon_view == "Veto summary":
                pane = self._veto_summary()
            else:
                pane = self._explorer()
        except Exception:
            log.exception(
                "Failed to build muon view %s for %s %s",
                self.muon_view,
                self.period,
                self.run,
            )
            pane = self._empty(f"Could not build {self.muon_view}")
        log.debug("Time to get muon plot: %.3fs", time.time() - start)
        return pane

    def _meta(self, label, unit, abs_unit):
        return PlotMeta(
            label=label,
            unit=unit,
            abs_unit=abs_unit,
            flag_display=f"Muon {self.muon_plots_types}",
            param_display=self.muon_plots,
            string=self.muon_group,
            run=self.run,
            period=self.period,
            experiment=self.run_dict[self.run]["experiment"],
        )

    def _explorer(self):
        manifest = self._manifest()
        data_file = self._pmts_file(manifest)
        flag = MUON_TYPES[self.muon_plots_types]
        key_param = self._param_key()
        key = f"{flag}_{key_param}"
        keys = contract_reader.available_keys(manifest, "pmts")
        relative = self.muon_units == "Relative"
        names = self._group_names()
        if not names:
            return self._empty(f"No PMTs in group {self.muon_group}")
        if self.muon_plot_style == "Histogram":
            if f"{key}_dist" not in keys:
                return self._empty(f"No distribution stored for {key}")
            edges, counts, attrs = contract_reader.read_dist(data_file, flag, key_param)
            label, unit = contract_reader.label_and_unit(
                manifest, attrs, key_param, relative
            )
            return phy_plot_dist_histogram(
                (edges, counts, attrs),
                self._meta(label, unit, attrs.get("unit", unit)),
                contract_reader.limits(attrs),
            )
        if key not in keys:
            return self._empty(f"Key {key} missing (distribution only?)")
        cadence = contract_reader.snap_cadence(
            self.muon_resampled, manifest.get("cadences", ["1min"])
        )
        series = contract_reader.read_binned(data_file, flag, key_param, cadence)
        label, unit = contract_reader.label_and_unit(
            manifest, series.attrs, key_param, relative
        )
        names = [n for n in names if n in series.detectors]
        if not names:
            return self._empty(f"No PMTs of {self.muon_group} in the contract")
        return phy_plot_binned_vsTime(
            mean_df=series.to_frame("mean")[names],
            std_df=series.to_frame("std")[names],
            min_df=series.to_frame("min")[names],
            max_df=series.to_frame("max")[names],
            meta=self._meta(label, unit, series.attrs.get("unit", unit)),
            flagged_ranges=contract_reader.flagged_ranges(manifest),
            cadence_label=cadence,
            limits=contract_reader.limits(series.attrs),
        )

    def _spectra(self):
        """Per-PMT pulse-height spectra with the located SPP position.

        PMT gains are not equalised, so the single-photon-peak position is
        reported per PMT (grey when it cannot be located), never graded.
        """
        manifest = self._manifest()
        data_file = self._pmts_file(manifest)
        if "All_Pulseheight_dist2d" not in contract_reader.available_keys(
            manifest, "pmts"
        ):
            return self._empty("All_Pulseheight_dist2d not in the pmts contract")
        edges, by_det = contract_reader.read_dist2d(data_file, "All", "Pulseheight")
        names = [n for n in self._group_names() if n in by_det]
        if not names:
            return self._empty(f"No spectra for group {self.muon_group}")
        figures, positions = [], {}
        for name in names:
            counts = by_det[name]
            spp = spe_centroid(edges, counts, **SPP_SEARCH)
            positions[name] = spp
            p = plot_style.make_figure(
                f"{name} - SPP ≈ {spp:.1f} LSB" if spp is not None else f"{name} - no SPP found",
                width=420, height=250, x_range=Range1d(0, 100), y_axis_type="log",
                tools="pan,box_zoom,wheel_zoom,reset,save",
            )  # fmt: skip
            p.sizing_mode = "fixed"
            p.title.text_color = "black" if spp is not None else "dimgray"
            y = np.where(np.asarray(counts, float) > 0, counts, np.nan)
            p.step(x=edges[:-1], y=y, mode="after", color="steelblue", line_width=1.2)
            if spp is not None:
                p.add_layout(
                    Span(
                        location=spp,
                        dimension="height",
                        line_color="seagreen",
                        line_width=2,
                    )  # fmt: skip
                )
            p.xaxis.axis_label = "Pulse height [LSB]"
            p.yaxis.axis_label = "Pulses"
            p.grid.grid_line_color = None
            figures.append(p)
        found = [v for v in positions.values() if v is not None]
        heading = (
            f"### {self.period} {self.run} - {self.muon_group}: "
            f"SPP located for {len(found)}/{len(names)} PMTs"
            + (f", median {np.median(found):.1f} LSB" if found else "")
        )
        ncols = max(1, int(np.ceil(np.sqrt(len(figures)))))
        return pn.Column(
            pn.pane.Markdown(heading),
            pn.pane.Bokeh(gridplot(figures, ncols=ncols, toolbar_location="right")),
        )

    def _veto_summary(self):
        """Muon rate + ge-coincidence from the period file, veto distributions."""
        frame = period_reader.read_optional(
            period_reader.period_file(self.phy_path, self.period),
            f"muon_veto/{self.run}",
        )
        if frame is None:
            return self._empty(f"muon_veto/{self.run} not in the period file")
        index = plot_style.utc_naive(frame.index)
        p = plot_style.make_figure(
            f"{self.period} {self.run} - muon veto", x_datetime=True, height=420,
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )  # fmt: skip
        if "muon_rate_hz" in frame:
            p.line(
                index, frame["muon_rate_hz"].to_numpy(dtype=float), color="dodgerblue",
                line_width=1.5, legend_label="muon rate",
            )  # fmt: skip
            p.yaxis.axis_label = "Muon rate [Hz]"
        frac_cols = [c for c in frame.columns if c.startswith("ge_coincidence_frac")]
        if frac_cols:
            p.extra_y_ranges = {"frac": Range1d(0, 1.05)}
            p.add_layout(
                LinearAxis(y_range_name="frac", axis_label="Ge coincidence fraction"),
                "right",
            )
            for col, color in zip(frac_cols, ("crimson", "darkorange"), strict=False):
                p.line(
                    index, frame[col].to_numpy(dtype=float), color=color,
                    line_dash="dashed", y_range_name="frac", legend_label=col,
                )  # fmt: skip
        plot_style.finish_legend(p, "top_right")

        manifest = self._manifest()
        data_file = self._pmts_file(manifest)
        keys = contract_reader.available_keys(manifest, "pmts")
        dists = []
        for parameter, display in (
            ("MuonLightSum", "Muon light sum"),
            ("MuonMultiplicity", "Muon multiplicity"),
        ):
            if f"All_{parameter}_dist" not in keys:
                continue
            edges, counts, attrs = contract_reader.read_dist(
                data_file, "All", parameter
            )
            label, unit = contract_reader.label_and_unit(
                manifest, attrs, parameter, False
            )
            d = plot_style.make_figure(
                display, width=480, height=300, y_axis_type="log",
                tools="pan,box_zoom,wheel_zoom,reset,save",
            )  # fmt: skip
            d.sizing_mode = "fixed"
            d.quad(
                top=np.where(np.asarray(counts, float) > 0, counts, np.nan), bottom=0.5,
                left=edges[:-1], right=edges[1:], fill_color="steelblue",
                line_color="white", fill_alpha=0.8,
            )  # fmt: skip
            d.xaxis.axis_label = f"{label} [{unit}]"
            d.yaxis.axis_label = "Events"
            dists.append(d)
        column = pn.Column(p, sizing_mode="stretch_width")
        if dists:
            column.append(pn.Row(*[pn.pane.Bokeh(d) for d in dists]))
        return column

    # ------------------------------------------------------------------
    # pane
    # ------------------------------------------------------------------

    def build_muon_pane(self, widget_widths=140):
        """The Muon tab: view/group/type selectors and the figure."""

        def radio(name):
            return pn.Param(
                self.param,
                widgets={
                    name: {"widget_type": pn.widgets.RadioBoxGroup, "inline": True}
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
                "muon_plots_types": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                    "width": widget_widths,
                }
            },
            parameters=["muon_plots_types"],
            show_labels=False,
            show_name=False,
        )
        from bokeh.models.formatters import PrintfTickFormatter

        resampled = pn.Param(
            self.param,
            widgets={
                "muon_resampled": {
                    "widget_type": pn.widgets.IntSlider,
                    "width": widget_widths,
                    "format": PrintfTickFormatter(format="%d min"),
                    "value_throttled": True,
                }
            },
            parameters=["muon_resampled"],
            show_labels=False,
            show_name=False,
        )
        current = pn.pane.Markdown(f"## {self.muon_plots}")
        value_menu = pn.widgets.MenuButton(
            name="PMT value", button_type="primary", width=widget_widths,
            items=self.param.muon_plots.objects,
        )  # fmt: skip

        def pick_value(event):
            self.muon_plots = event.new
            current.object = f"## {event.new}"

        value_menu.on_click(pick_value)
        self._value_menu = value_menu

        grid = pn.GridSpec(width=5 * widget_widths + 20, max_height=800)
        grid[:, 0] = types
        grid[:, 1] = pn.Spacer(width=5)
        grid[0, 2] = header("Units")
        grid[1, 2] = radio("muon_units")
        grid[:, 3] = pn.Spacer(width=5)
        grid[0, 4] = header("Resampled")
        grid[1, 4] = resampled
        grid[:, 5] = pn.Spacer(width=5)
        grid[0, 6] = header("Style")
        grid[1, 6] = radio("muon_plot_style")
        grid[:, 7] = pn.Spacer(width=5)
        grid[0, 8] = header("Group")
        grid[1, 8] = radio("muon_group")

        return pn.Column(
            pn.Row(pn.pane.SVG(logo_path / "Physics.svg", height=25), value_menu),
            pn.Row("## Current Plot:", current),
            pn.Row("View:", radio("muon_view")),
            pn.Row(grid),
            pn.param.ParamMethod(
                self.update_muon_plot,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Muon",
            sizing_mode="stretch_width",
        )
