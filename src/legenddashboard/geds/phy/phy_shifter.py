"""Phy. Shifter page: the pipeline's shifter figures from the period contract."""

from __future__ import annotations

import logging
import time

import panel as pn
import param

from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.geds.phy import period_reader, plot_style, shifter_plots
from legenddashboard.util import logo_path

log = logging.getLogger(__name__)

# family -> (contract prefix whose children are the metric options, label)
DETECTOR_SUMMARY_TITLES = (
    "pulser_stab",
    "baseln_stab",
    "baseln_spike",
    "pulser_stab_uncalib",
)
FAMILIES = ("Detector summary",)


class PhyShifterMonitoring(GedMonitoring):
    """One figure family at a time, selected from what the period file holds."""

    phy_path = param.String("")
    shifter_family = param.ObjectSelector(
        default=FAMILIES[0], objects=list(FAMILIES), label="Figure"
    )
    shifter_metric = param.ObjectSelector(default=None, objects=[], label="Metric")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # menus follow the period file; registered before the view pane so a
        # run change re-populates them before the figure renders
        self.param.watch(self._update_menus, ["run_dict", "run", "shifter_family"])
        if self.run_dict:
            self._update_menus()

    # ------------------------------------------------------------------
    # contract lookups
    # ------------------------------------------------------------------

    def _phy_file(self):
        return period_reader.period_file(self.phy_path, self.period)

    def _metric_options(self):
        """Metric names available for the current family and run."""
        if self.shifter_family == "Detector summary":
            present = period_reader.children(self._phy_file(), "detector_summary")
            return [
                t
                for t in DETECTOR_SUMMARY_TITLES
                if t in present
                and period_reader.has_key(
                    self._phy_file(), f"detector_summary/{t}/{self.run}"
                )
            ]
        return []

    def _update_menus(self, *events):  # noqa: ARG002
        options = self._metric_options()
        self.param.shifter_metric.objects = options
        if options and self.shifter_metric not in options:
            self.shifter_metric = options[0]
        elif not options:
            self.shifter_metric = None

    # ------------------------------------------------------------------
    # view
    # ------------------------------------------------------------------

    @param.depends("run_dict", "run", "string", "shifter_family", "shifter_metric")
    def update_shifter_plot(self):
        start = time.time()
        try:
            pane = self._build()
        except Exception:
            log.exception(
                "Failed to build shifter figure %s/%s for %s %s",
                self.shifter_family,
                self.shifter_metric,
                self.period,
                self.run,
            )
            pane = plot_style.empty_figure(
                f"Could not build {self.shifter_family} for {self.period} {self.run}"
            )
        log.debug("Time to get shifter plot: %.3fs", time.time() - start)
        return pane

    def _missing(self, key):
        return plot_style.empty_figure(
            f"{key} not in {self._phy_file().name} (older pipeline output?)"
        )

    def _build(self):
        if not self._phy_file().exists():
            return plot_style.empty_figure(
                f"No period contract for {self.period}: {self._phy_file()}"
            )
        if self.shifter_family == "Detector summary":
            if self.shifter_metric is None:
                return self._missing(f"detector_summary/*/{self.run}")
            key = f"detector_summary/{self.shifter_metric}/{self.run}"
            frame = period_reader.read_optional(self._phy_file(), key)
            if frame is None:
                return self._missing(key)
            return shifter_plots.detector_summary(
                frame, self.shifter_metric, self.period, self.run
            )
        return plot_style.empty_figure(f"{self.shifter_family}: not implemented")

    # ------------------------------------------------------------------
    # pane
    # ------------------------------------------------------------------

    def build_shifter_pane(self, widget_widths=140):
        """The Phy. Shifter tab: family selector, metric selector, figure."""
        family = pn.Param(
            self.param,
            widgets={
                "shifter_family": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                    "width": widget_widths,
                }
            },
            parameters=["shifter_family"],
            show_labels=False,
            show_name=False,
        )
        metric = pn.Param(
            self.param,
            widgets={
                "shifter_metric": {
                    "widget_type": pn.widgets.Select,
                    "width": 2 * widget_widths,
                }
            },
            parameters=["shifter_metric"],
            show_labels=False,
            show_name=False,
        )
        return pn.Column(
            pn.Row(
                pn.pane.SVG(logo_path / "Physics.svg", height=25), "## Shifter figures"
            ),
            pn.Row(family, pn.Spacer(width=10), pn.Column("Metric", metric)),
            pn.param.ParamMethod(
                self.update_shifter_plot, lazy=True, loading_indicator=True
            ),
            name="Phy. Shifter",
            sizing_mode="stretch_width",
        )
