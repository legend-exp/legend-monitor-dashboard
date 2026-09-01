"""Phy. Shifter page: the pipeline's shifter figures from the period contract."""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path

import numpy as np
import panel as pn
import param

from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.geds.phy import (
    contract_reader,
    period_reader,
    plot_style,
    shifter_plots,
)
from legenddashboard.util import logo_path

log = logging.getLogger(__name__)

# family -> (contract prefix whose children are the metric options, label)
DETECTOR_SUMMARY_TITLES = (
    "pulser_stab",
    "baseln_stab",
    "baseln_spike",
    "pulser_stab_uncalib",
)
STABILITY_PARAMETERS = ("TrapemaxCtcCal", "Trapemax", "Baseline", "BlStd")
FAMILIES = (
    "Detector summary",
    "Param. stability",
    "Gain shift",
    "FT summary",
    "Event rate QC",
    "QC rates",
    "QC average",
    "QC classifiers",
)
CLASSIFIERS = (
    "IsValidBlSlopeClassifier",
    "IsValidTailRmsClassifier",
    "IsValidPzSlopeClassifier",
    "IsValidBlSlopeRmsClassifier",
    "IsValidBlPolyRmsClassifier",
    "IsValidCuspeminClassifier",
    "IsValidCuspemaxClassifier",
)
FT_KINDS = ("per string", "all strings", "survival fraction")
PER_DETECTOR_FAMILIES = ("Param. stability", "Gain shift")


class PhyShifterMonitoring(GedMonitoring):
    """One figure family at a time, selected from what the period file holds."""

    phy_path = param.String("")
    shifter_family = param.ObjectSelector(
        default=FAMILIES[0], objects=list(FAMILIES), label="Figure"
    )
    shifter_metric = param.ObjectSelector(default=None, objects=[], label="Metric")
    shifter_detector = param.ObjectSelector(default=None, objects=[], label="Detector")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # menus follow the period file; registered before the view pane so a
        # run change re-populates them before the figure renders
        self.param.watch(
            self._update_menus, ["run_dict", "run", "string", "shifter_family"]
        )
        if self.run_dict:
            self._update_menus()

    # ------------------------------------------------------------------
    # contract lookups
    # ------------------------------------------------------------------

    def _phy_file(self):
        return period_reader.period_file(self.phy_path, self.period)

    def _metric_options(self):
        """Metric names available for the current family and run."""
        f = self._phy_file()
        if self.shifter_family == "Detector summary":
            return [
                t
                for t in DETECTOR_SUMMARY_TITLES
                if period_reader.has_key(f, f"detector_summary/{t}/{self.run}")
            ]
        if self.shifter_family == "Param. stability":
            return [
                par
                for par in STABILITY_PARAMETERS
                if period_reader.has_key(f, f"param_stability/{par}/{self.run}")
            ]
        if self.shifter_family == "Gain shift":
            return [
                kind
                for kind in ("corr", "uncorr")
                if period_reader.has_key(f, f"gain_shift/{kind}/{self.run}")
            ]
        if self.shifter_family == "FT summary":
            keys = {
                "per string": "per_detector",
                "all strings": "per_string",
                "survival fraction": "survival_fraction",
            }
            return [
                kind
                for kind in FT_KINDS
                if period_reader.has_key(f, f"ft_summary/{keys[kind]}/{self.run}")
            ]
        if self.shifter_family in ("QC rates", "QC average"):
            flags = period_reader.flags_for(f, self.run)
            if self.shifter_family == "QC average":
                avg = period_reader.read_optional(f, f"qc_average/{self.run}")
                if avg is not None and "flag" in avg:
                    flags = list(dict.fromkeys(avg["flag"].astype(str)))
            ordered = [fl for fl in plot_style.DEFAULT_QC_FLAGS if fl in flags]
            return ordered + [fl for fl in flags if fl not in ordered]
        if self.shifter_family == "QC classifiers":
            frac = period_reader.read_optional(f, f"qc_classifier_frac/{self.run}")
            present = set(frac["classifier"].astype(str)) if frac is not None else set()
            return [c for c in CLASSIFIERS if c in present]
        return []

    def _detector_options(self):
        """Detectors of the selected string (sidebar), in position order."""
        if self.shifter_family not in PER_DETECTOR_FAMILIES:
            return []
        return [str(d) for d in self.strings_dict.get(self.string, [])]

    def _string_position(self, detector):
        """(string, position) of a detector from the run's detector map."""
        dmap = period_reader.detector_map(self.phy_path, self.period, self.run)
        if dmap is not None and "name" in dmap:
            rows = dmap[dmap["name"] == detector]
            if len(rows):
                return rows.iloc[0]["string"], rows.iloc[0]["position"]
        cal = period_reader.read_optional(self._phy_file(), f"cal_points/{self.run}")
        if cal is not None:
            rows = cal[cal["detector"] == detector]
            if len(rows):
                return rows.iloc[0]["string"], rows.iloc[0]["position"]
        return self.string, "?"

    def _update_menus(self, *events):  # noqa: ARG002
        options = self._metric_options()
        self.param.shifter_metric.objects = options
        if options and self.shifter_metric not in options:
            self.shifter_metric = options[0]
        elif not options:
            self.shifter_metric = None
        dets = self._detector_options()
        self.param.shifter_detector.objects = dets
        if dets and self.shifter_detector not in dets:
            self.shifter_detector = dets[0]
        elif not dets:
            self.shifter_detector = None

    # ------------------------------------------------------------------
    # view
    # ------------------------------------------------------------------

    @param.depends(
        "run_dict",
        "run",
        "string",
        "shifter_family",
        "shifter_metric",
        "shifter_detector",
    )
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
        if self.shifter_family in PER_DETECTOR_FAMILIES:
            return self._build_per_detector()
        if self.shifter_family == "FT summary":
            return self._build_ft()
        if self.shifter_family == "Event rate QC":
            frame = period_reader.read_optional(
                self._phy_file(), f"event_rate_qc/{self.run}"
            )
            if frame is None:
                return self._missing(f"event_rate_qc/{self.run}")
            return shifter_plots.event_rate_qc(frame, self.period, self.run)
        if self.shifter_family == "QC rates":
            return self._build_qc_rates()
        if self.shifter_family == "QC average":
            return self._build_qc_average()
        if self.shifter_family == "QC classifiers":
            return self._build_classifiers()
        return plot_style.empty_figure(f"{self.shifter_family}: not implemented")

    def _string_groups(self):
        """{string: [(name, position)]} from the run's detector map (processable)."""
        dmap = period_reader.detector_map(self.phy_path, self.period, self.run)
        groups = {}
        if dmap is not None and {"name", "string", "position"} <= set(dmap.columns):
            if "processable" in dmap:
                dmap = dmap[dmap["processable"].astype(bool)]
            for string, group in dmap.sort_values(["string", "position"]).groupby(
                "string", sort=True
            ):
                groups[int(string)] = [
                    (str(n), int(p))
                    for n, p in zip(group["name"], group["position"], strict=False)
                ]
        else:
            for string, names in self.strings_dict.items():
                digits = "".join(ch for ch in str(string) if ch.isdigit())
                groups[int(digits) if digits else string] = [
                    (str(n), i + 1) for i, n in enumerate(names)
                ]
        return groups

    def _build_qc_rates(self):
        f = self._phy_file()
        flag = self.shifter_metric
        rates = period_reader.read_optional(f, f"qc_rate_series/{flag}/{self.run}")
        if flag is None or rates is None:
            return self._missing(f"qc_rate_series/{flag}/{self.run}")
        avg = period_reader.read_optional(f, f"qc_average/{self.run}")
        avg_rates = {}
        if avg is not None:
            sub = avg[avg["flag"].astype(str) == flag]
            avg_rates = dict(
                zip(sub["detector"].astype(str), sub["rate_mhz"], strict=False)
            )
        groups = self._string_groups()
        string = self._string_number()
        # the tab20 cycle advances over all (flag, string) figures before this one
        colors = itertools.cycle(plot_style.TAB20)
        flags = self.param.shifter_metric.objects
        for fl in flags:
            for st, dets in groups.items():
                if fl == flag and st == string:
                    dets_here = dets
                    break
                for name, _ in dets:
                    if name in rates.columns:
                        next(colors)
            else:
                continue
            break
        else:
            return self._missing(f"string {self.string} in the detector map")
        return shifter_plots.qc_rate_series(
            rates, dets_here, avg_rates, flag, self.period, self.run, string, colors
        )

    def _build_qc_average(self):
        f = self._phy_file()
        flag = self.shifter_metric
        avg = period_reader.read_optional(f, f"qc_average/{self.run}")
        if flag is None or avg is None:
            return self._missing(f"qc_average/{self.run}")
        sub = avg[avg["flag"].astype(str) == flag]
        rate_by_name = dict(
            zip(sub["detector"].astype(str), sub["rate_mhz"], strict=False)
        )
        groups = {
            st: [n for n, _ in dets] for st, dets in self._string_groups().items()
        }
        dead = period_reader.read_optional(f, f"dead_time/{self.run}")
        dead_pct = (
            float(dead["dead_time_pct"].iloc[0])
            if dead is not None and "dead_time_pct" in dead and len(dead)
            else None
        )
        return shifter_plots.qc_average(
            rate_by_name, groups, flag, dead_pct, self.period, self.run
        )

    def _build_classifiers(self):
        f = self._phy_file()
        par = self.shifter_metric
        if par is None:
            return self._missing(f"qc_classifier_frac/{self.run}")
        frac = period_reader.read_optional(f, f"qc_classifier_frac/{self.run}")
        string = self._string_number()
        dets = self._string_groups().get(string, [])
        fracs = {}
        frac_rows = None
        if frac is not None:
            frac_rows = frac[frac["classifier"].astype(str) == par]
            fracs = {
                (str(d), str(t)): float(v)
                for d, t, v in zip(
                    frac_rows["detector"],
                    frac_rows["event_type"],
                    frac_rows["percent_in_range"],
                    strict=False,
                )  # fmt: skip
            }
        # per-detector histograms need the run contract's dist2d groups
        manifest = contract_reader.find_manifest(
            self.phy_path, self.period, self.run, self.run_dict[self.run]["experiment"]
        )
        if manifest is not None:
            run_dir = (
                Path(self.phy_path) / "generated/plt/hit/phy" / self.period / self.run
            )
            geds = contract_reader.geds_file_from_manifest(manifest, run_dir)
            keys = contract_reader.available_keys(manifest, "geds")
            if geds is not None and f"All_{par}_dist2d" in keys:
                counts_by_flag, edges = {}, None
                for flag in plot_style.CLASSIFIER_FLAG_LABELS:
                    if f"{flag}_{par}_dist2d" in keys:
                        edges, counts_by_flag[flag] = contract_reader.read_dist2d(
                            geds, flag, par
                        )
                return shifter_plots.classifier_grid(
                    edges,
                    counts_by_flag,
                    dets,
                    fracs,
                    par,
                    self.period,
                    self.run,
                    string,
                )
        if frac_rows is None or frac_rows.empty:
            return self._missing(f"qc_classifier_frac/{self.run} for {par}")
        names = {n for n, _ in dets}
        rows = frac_rows[frac_rows["detector"].astype(str).isin(names)]
        if rows.empty:
            return self._missing(f"{par} fractions for string {self.string}")
        return shifter_plots.classifier_fraction_bars(
            rows, par, self.period, self.run, string
        )

    def _build_ft(self):
        f = self._phy_file()
        kind = self.shifter_metric
        if kind == "all strings":
            frame = period_reader.read_optional(f, f"ft_summary/per_string/{self.run}")
            if frame is None:
                return self._missing(f"ft_summary/per_string/{self.run}")
            return shifter_plots.ft_all_strings(frame, self.period, self.run)
        if kind == "survival fraction":
            frame = period_reader.read_optional(
                f, f"ft_summary/survival_fraction/{self.run}"
            )
            if frame is None:
                return self._missing(f"ft_summary/survival_fraction/{self.run}")
            return shifter_plots.ft_survival(frame, self.period)
        rates = period_reader.read_optional(f, f"ft_summary/per_detector/{self.run}")
        if rates is None:
            return self._missing(f"ft_summary/per_detector/{self.run}")
        total = period_reader.read_optional(f, f"ft_summary/total_forced/{self.run}")
        avg_mhz = None
        if total is not None and total.shape[1]:
            avg = float(total.iloc[:, 0].mean()) / 3600 * 1000
            if np.isfinite(avg):
                avg_mhz = avg
        # one tab20 cycle across all strings, advanced to the selected string
        dmap = period_reader.detector_map(self.phy_path, self.period, self.run)
        colors = itertools.cycle(plot_style.TAB20)
        string_label = self._string_number()
        if dmap is not None and "string" in dmap:
            ordered = dmap.sort_values(["string", "position"])
            dets = []
            for string, group in ordered.groupby("string", sort=True):
                names = [d for d in group["name"] if d in rates.columns]
                if str(string) == str(string_label):
                    dets = names
                    break
                for _ in names:
                    next(colors)
        else:
            dets = [
                d for d in self.strings_dict.get(self.string, []) if d in rates.columns
            ]
        if not dets:
            return self._missing(f"FT rates for string {self.string}")
        return shifter_plots.ft_per_string(
            rates[dets], self.period, self.run, string_label, avg_mhz, colors
        )

    def _string_number(self):
        """The sidebar string ("String:01") as the pipeline's plain number."""
        digits = "".join(ch for ch in str(self.string) if ch.isdigit())
        return int(digits) if digits else self.string

    def _build_per_detector(self):
        f = self._phy_file()
        det = self.shifter_detector
        if det is None or self.shifter_metric is None:
            return self._missing(f"{self.shifter_family} for string {self.string}")
        string, position = self._string_position(det)
        cal = period_reader.read_optional(f, f"cal_points/{self.run}")
        det_cal = None
        if cal is not None and "detector" in cal:
            rows = cal[cal["detector"] == det].sort_values("run_start")
            det_cal = rows if len(rows) else None
        pul = period_reader.read_optional(f, f"pul_cusp/kevdiff/{self.run}")
        pul_trace = pul[det] if pul is not None and det in pul.columns else None

        if self.shifter_family == "Gain shift":
            kind = self.shifter_metric
            frame = period_reader.read_optional(f, f"gain_shift/{kind}/{self.run}")
            if frame is None or det not in frame.columns or frame[det].dropna().empty:
                return self._missing(f"gain_shift/{kind}/{self.run} for {det}")
            std = period_reader.read_optional(f, f"gain_shift/{kind}_std/{self.run}")
            # the period-to-date trace; the selected run is highlighted
            run_start = det_cal.iloc[-1]["run_start"] if det_cal is not None else None
            highlight = (
                (run_start, frame.index.max()) if run_start is not None else None
            )
            return shifter_plots.gain_shift(
                frame[det],
                std[det] if std is not None and det in std.columns else None,
                pul_trace,
                det_cal,
                self.period,
                det,
                string,
                position,
                corrected=(kind == "corr" and pul_trace is not None),
                highlight=highlight,
            )

        par = self.shifter_metric
        frame = period_reader.read_optional(f, f"param_stability/{par}/{self.run}")
        if frame is None or det not in frame.columns or frame[det].dropna().empty:
            return self._missing(f"param_stability/{par}/{self.run} for {det}")
        std = period_reader.read_optional(f, f"param_stability/{par}_std/{self.run}")
        trace = frame[det]
        t0, res0 = None, float("nan")
        if det_cal is not None:
            last = det_cal.iloc[-1]  # cal_points has no run column: latest point
            t0 = last["run_start"]
            res0 = float(last["res"]) if "res" in det_cal else float("nan")
        if par == "TrapemaxCtcCal" and pul_trace is not None:
            lo, hi = trace.index.min(), trace.index.max()
            pul_trace = pul_trace[(pul_trace.index >= lo) & (pul_trace.index <= hi)]
        else:
            pul_trace = None
        return shifter_plots.param_stability(
            trace,
            std[det] if std is not None and det in std.columns else None,
            pul_trace,
            t0,
            res0,
            par,
            self.period,
            det,
            string,
            position,
        )

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
        detector = pn.Param(
            self.param,
            widgets={
                "shifter_detector": {
                    "widget_type": pn.widgets.Select,
                    "width": 2 * widget_widths,
                }
            },
            parameters=["shifter_detector"],
            show_labels=False,
            show_name=False,
        )
        return pn.Column(
            pn.Row(
                pn.pane.SVG(logo_path / "Physics.svg", height=25), "## Shifter figures"
            ),
            pn.Row(
                family,
                pn.Spacer(width=10),
                pn.Column("Metric", metric),
                pn.Spacer(width=10),
                pn.Column("Detector (string from sidebar)", detector),
            ),
            pn.param.ParamMethod(
                self.update_shifter_plot,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Phy. Shifter",
            sizing_mode="stretch_width",
        )
