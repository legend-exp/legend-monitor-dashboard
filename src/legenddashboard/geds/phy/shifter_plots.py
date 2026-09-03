"""Bokeh twins of legend-data-monitor's shifter figures.

Pure builders: frames from the period contract in, a figure (or Panel
layout) out. Each function mirrors one figure of the pipeline's
``plots/{summary,stability,qc}.py`` element by element; what cannot be
reproduced in Bokeh (per-tick label colours, the legacy "last cycle"
legend title) is replaced as noted in the docstrings.
"""

from __future__ import annotations

import itertools
from math import pi

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    HoverTool,
    Label,
    LinearAxis,
    Range1d,
    Span,
    Whisker,
)
from bokeh.transform import dodge

from legenddashboard.geds.phy import plot_style

# fixed y ranges the pipeline applies in plots/summary.py (not in its settings)
_SUMMARY_YRANGE = {
    "baseln_stab": (-20, 20),
    "baseln_spike": (0, 100),
    "pulser_stab": (-6, 6),
    "FEP_gain_stab": (-6, 6),
}
_USABILITY_COLORS = {"off": "red", "false": "red", "ac": "darkorange"}


def _ordered(frame, det_col="ged", string_col="string", pos_col="pos"):
    """Rows sorted by (string, position) with unique detector factors."""
    frame = frame.copy()
    frame[string_col] = pd.to_numeric(frame[string_col], errors="coerce")
    frame[pos_col] = pd.to_numeric(frame[pos_col], errors="coerce")
    frame = frame.sort_values([string_col, pos_col], kind="stable")
    frame = frame.drop_duplicates(det_col, keep="first")
    frame[det_col] = frame[det_col].astype(str)
    return frame.reset_index(drop=True)


def detector_summary(frame, title, period, run):
    """Per-detector run summary (lmon ``plots/summary._detector_summary_figure``).

    Categorical x (detectors by string/position); orange ±FWHM bar, skyblue
    ±1 sigma bar, black mean, min/max whiskers; string separators; the
    metric's threshold lines with the excluded region shaded and the
    pipeline's fixed y range. Usability is shown as a coloured marker row
    under the data (red off / orange ac) instead of coloured tick labels.
    """
    info = plot_style.MTG_BY_TITLE.get(title, {})
    rows = _ordered(frame)
    if rows.empty:
        return plot_style.empty_figure(f"{title}: no detectors for {period} {run}")
    dets = list(rows["ged"])
    mean = rows["mean"].astype(float).to_numpy()
    std = rows["std"].astype(float).fillna(0.0).to_numpy()
    fwhm = (
        rows["fwhm"].astype(float).to_numpy()
        if "fwhm" in rows
        else np.full(len(rows), np.nan)
    )
    usability = (
        rows["usability"].astype(str).str.lower()
        if "usability" in rows
        else pd.Series([""] * len(rows))
    )

    source = ColumnDataSource(
        {
            "det": dets,
            "string": rows["string"].to_numpy(),
            "pos": rows["pos"].to_numpy(),
            "mean": mean,
            "std": std,
            "lo": mean - std,
            "hi": mean + std,
            "min": rows["min"].astype(float).to_numpy(),
            "max": rows["max"].astype(float).to_numpy(),
            "fwhm": fwhm,
            "fwhm_lo": -fwhm / 2,
            "fwhm_hi": fwhm / 2,
            "usability": usability.to_numpy(),
        }
    )

    p = plot_style.make_figure(
        f"{period} {run} - {title}",
        x_range=FactorRange(*dets),
        tools="pan,box_zoom,wheel_zoom,reset,save",
        height=450,
    )
    if title in ("FEP_gain_stab", "pulser_stab", "baseln_stab"):
        p.add_layout(
            Span(location=0, dimension="width", line_color="gray", line_width=0.5)
        )

    if np.isfinite(fwhm).any():
        label = "±FWHM/2" if title == "FEP_gain_stab" else "±FWHM (threshold)"
        p.vbar(
            x="det", width=0.4, bottom="fwhm_lo", top="fwhm_hi", source=source,
            color="orange", alpha=0.2, legend_label=label,
        )  # fmt: skip
    p.vbar(
        x="det", width=0.6, bottom="lo", top="hi", source=source,
        color="skyblue", alpha=0.7, legend_label="±1 sigma",
    )  # fmt: skip
    whisker = Whisker(
        base="det", upper="max", lower="min", source=source,
        line_color="red" if title == "FEP_gain_stab" else "#0266c9",
    )  # fmt: skip
    whisker.upper_head.size = whisker.lower_head.size = 8
    p.add_layout(whisker)
    plot_style.legend_proxy(p, "Min/Max", color=whisker.line_color)
    mean_glyph = p.scatter(
        x="det", y="mean", source=source, color="black", size=6, legend_label="Mean"
    )
    p.add_tools(
        HoverTool(
            renderers=[mean_glyph],
            tooltips=[
                ("Detector", "@det (string @string, pos @pos)"),
                ("Mean", "@mean{0.3f}"),
                ("Std", "@std{0.3f}"),
                ("Min / Max", "@min{0.3f} / @max{0.3f}"),
                ("FWHM(Qββ)", "@fwhm{0.2f} keV"),
                ("Usability", "@usability"),
            ],
        )
    )

    # thresholds + the pipeline's fixed y ranges
    lo_lim, hi_lim = info.get("limits", (None, None))
    unit = info.get("unit", "")
    if title == "FEP_gain_stab":
        lo_lim, hi_lim = -2, 2
    if hi_lim is not None:
        plot_style.threshold(p, hi_lim, above=True, label=f"{hi_lim} {unit} threshold")
    if lo_lim is not None:
        plot_style.threshold(
            p,
            lo_lim,
            above=False,
            label=None if hi_lim is not None else f"{lo_lim} {unit} threshold",
        )
    yr = _SUMMARY_YRANGE.get(title)
    if yr is not None:
        p.y_range = Range1d(*yr)
        y_lo, y_hi = yr
    else:
        finite = np.concatenate([source.data["min"], source.data["max"]])
        finite = finite[np.isfinite(finite)]
        y_lo, y_hi = (
            (float(finite.min()), float(finite.max())) if finite.size else (-1, 1)
        )
        pad = 0.1 * (y_hi - y_lo or 1)
        p.y_range = Range1d(y_lo - pad, y_hi + pad)
        y_lo, y_hi = p.y_range.start, p.y_range.end

    # usability marker row (matplotlib coloured the tick labels)
    y_marker = y_lo + 0.03 * (y_hi - y_lo)
    for state, color in (("off", "red"), ("ac", "darkorange")):
        mask = usability.isin(
            [state, "false"] if state == "off" else [state]
        ).to_numpy()
        if mask.any():
            p.scatter(
                x=[d for d, m in zip(dets, mask, strict=False) if m],
                y=[y_marker] * int(mask.sum()),
                marker="square", size=7, color=color, legend_label=f"Usability: {state}",
            )  # fmt: skip

    plot_style.string_separators(
        p, dets, list(rows["string"]), y_lo + 0.25 * (y_hi - y_lo)
    )
    p.yaxis.axis_label = info.get("ylabel", unit)
    p.xaxis.major_label_orientation = pi / 2
    p.xaxis.major_label_text_font_size = "8px"
    p.xgrid.grid_line_color = None
    plot_style.finish_legend(p, "top_right", title=f"{period} {run}")
    return p


# ---------------------------------------------------------------------------
# per-detector stability (lmon plots/stability.py)
# ---------------------------------------------------------------------------


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts


def _trace_source(trace, trace_std):
    """ColumnDataSource of a time series with its ±1 sigma band."""
    index = plot_style.utc_naive(trace.index)
    vals = trace.to_numpy(dtype=float)
    if trace_std is not None:
        sig = trace_std.reindex(trace.index).to_numpy(dtype=float)
        sig = np.where(np.isfinite(sig), sig, 0.0)
    else:
        sig = np.zeros(len(vals))
    return ColumnDataSource(
        {"t": index, "v": vals, "lo": vals - sig, "hi": vals + sig, "sig": sig}
    )


def _draw_trace(p, source, color, label, band=True):
    if band:
        p.varea(
            x="t", y1="lo", y2="hi", source=source, fill_color="black",
            fill_alpha=0.2, legend_label="±1 sigma",
        )  # fmt: skip
    return p.line(
        x="t", y="v", source=source, color=color, line_width=1.5, legend_label=label
    )


def _stability_title(period, string, position, detector):
    return (
        f"period: {period} - string: {string} - position: {position} - ged: {detector}"
    )


def _draw_cal_points(p, cal, quadratic=False):
    """FEP/cal-const markers, run boundaries and stepped ±FWHM/2 segments.

    Mirrors lmon ``_draw_cal_points``: markers are shifted -5 h (legacy),
    each cal run's resolution band runs until the next run start (the last
    one for 7 days), annotated with its FWHM value. Returns the run starts.
    """
    if cal is None or cal.empty:
        return []
    cal = cal.sort_values("run_start")
    t0 = [_naive(t) for t in cal["run_start"]]
    n = len(t0)
    nan = np.full(n, np.nan)
    res = cal["res"].astype(float).to_numpy() if "res" in cal else nan
    res_quad = cal["res_quad"].astype(float).to_numpy() if "res_quad" in cal else nan
    shifted = [t - pd.Timedelta(hours=5) for t in t0]
    p.scatter(
        x=shifted, y=cal["fep_diff"].astype(float).to_numpy(), marker="x", size=8,
        color="black", legend_label="FEP gain",
    )  # fmt: skip
    p.scatter(
        x=shifted, y=cal["cal_const_diff"].astype(float).to_numpy(), marker="x",
        size=8, color="red", legend_label="cal. const. diff",
    )  # fmt: skip
    for t in t0:
        p.add_layout(
            Span(
                location=t, dimension="height", line_color="dimgrey", line_dash="dashed"
            )
        )
    ends = [t0[i + 1] if i < n - 1 else t0[i] + pd.Timedelta(days=7) for i in range(n)]
    for values, color, label in (
        (res, "blue", plot_style.QBB_LIN_LABEL),
        (res_quad, "dodgerblue", plot_style.QBB_QUAD_LABEL),
    ):
        if color == "dodgerblue" and not quadratic:
            continue
        ok = np.isfinite(values)
        if not ok.any():
            continue
        for sign in (1, -1):
            p.segment(
                x0=[t0[i] for i in range(n) if ok[i]],
                x1=[ends[i] for i in range(n) if ok[i]],
                y0=[sign * values[i] / 2 for i in range(n) if ok[i]],
                y1=[sign * values[i] / 2 for i in range(n) if ok[i]],
                color=color, line_width=1.5, legend_label=label,
            )  # fmt: skip
        for i in range(n):
            if ok[i]:
                p.add_layout(
                    Label(
                        x=t0[i],
                        y=values[i] / 2 * (1.1 if color == "blue" else 1.5),
                        text=f"{values[i]:.2f}",
                        text_color=color,
                        text_font_size="9px",
                    )  # fmt: skip
                )
    return t0


def gain_shift(
    trace, trace_std, pul, cal, period, detector, string, position, corrected,
    highlight=None, quadratic=False,
):  # fmt: skip
    """Period-to-date gain shift of one detector (lmon ``_build_gain_shift_figure``).

    ``highlight``: optional (start, end) of the selected run, shaded (an
    addition: the pipeline draws the whole period without marking a run).
    """
    p = plot_style.make_figure(
        _stability_title(period, string, position, detector), x_datetime=True
    )
    source = _trace_source(trace, trace_std)
    if corrected:
        if pul is not None and not pul.dropna().empty:
            p.line(
                x=plot_style.utc_naive(pul.index), y=pul.to_numpy(dtype=float),
                color=plot_style.C2, legend_label="PULS01ANA",
            )  # fmt: skip
        line = _draw_trace(
            p, source, plot_style.C4, "GED corrected", trace_std is not None
        )
    else:
        line = _draw_trace(
            p, source, "dodgerblue", "GED uncorrected", trace_std is not None
        )
    t0 = _draw_cal_points(p, cal, quadratic)
    if highlight is not None:
        p.add_layout(
            BoxAnnotation(
                left=_naive(highlight[0]),
                right=_naive(highlight[1]),
                fill_color="gold",
                fill_alpha=0.12,
                level="underlay",
            )  # fmt: skip
        )
    p.yaxis.axis_label = "Energy diff / keV"
    if t0:
        tmax = _naive(trace.dropna().index.max())
        p.x_range = Range1d(
            t0[0] - pd.Timedelta(hours=8), t0[-1] + (tmax - t0[-1]) * 1.5
        )
    p.add_tools(
        HoverTool(
            renderers=[line],
            mode="vline",
            tooltips=[
                ("Time", "@t{%F %H:%M} UTC"),
                ("diff", "@v{0.3f} keV"),
                ("sigma", "@sig{0.3f}"),
            ],
            formatters={"@t": "datetime"},
        )  # fmt: skip
    )
    plot_style.finish_legend(p, "bottom_left")
    return p


def param_stability(
    trace, trace_std, pul, t0, res0, parameter, period, detector, string, position
):
    """One detector's parameter stability over a run (lmon ``_build_param_figure``).

    TrapemaxCtcCal: ±FWHM/2 (``res0``, keV) threshold segments from the run
    start ``t0`` over 7 days, annotated; other parameters: the thresholds
    from mtg-plot-settings. A PULS01ANA trace is drawn for the corrected
    calibrated gain when available.
    """
    info = plot_style.MTG_PLOT_INFO.get(parameter, {})
    colors = info.get("colors", ("dodgerblue", "blue"))
    corrected = parameter == "TrapemaxCtcCal" and pul is not None
    p = plot_style.make_figure(
        _stability_title(period, string, position, detector), x_datetime=True
    )
    source = _trace_source(trace, trace_std)
    if corrected:
        if not pul.dropna().empty:
            p.line(
                x=plot_style.utc_naive(pul.index), y=pul.to_numpy(dtype=float),
                color=plot_style.C2, legend_label="PULS01ANA",
            )  # fmt: skip
        line = _draw_trace(
            p, source, plot_style.C4, "GED corrected", trace_std is not None
        )
    else:
        line = _draw_trace(
            p, source, colors[0], "GED uncorrected", trace_std is not None
        )

    res0 = float("nan") if res0 is None else float(res0)
    lo_lim, hi_lim = (
        (-res0 / 2, res0 / 2)
        if parameter == "TrapemaxCtcCal"  # keV band: calibrated gain only
        else info.get("limits", (None, None))
    )
    if t0 is not None:
        t0 = _naive(t0)
        span = [t0, t0 + pd.Timedelta(days=7)]
        if parameter == "TrapemaxCtcCal":
            if np.isfinite(res0):
                for y in (res0 / 2, -res0 / 2):
                    p.line(
                        span,
                        [y, y],
                        color=colors[1],
                        legend_label=plot_style.QBB_LIN_LABEL,
                    )
                p.add_layout(
                    Label(
                        x=t0,
                        y=res0 / 2 * 1.1,
                        text=f"{res0:.2f}",
                        text_color=colors[1],
                        text_font_size="9px",
                    )  # fmt: skip
                )
            else:
                p.add_layout(
                    Label(
                        x=10,
                        y=-10,
                        x_units="screen",
                        y_units="screen",
                        text="FWHM(Qββ) not in period file",
                        text_font_size="9px",
                        text_color="dimgray",
                    )  # fmt: skip
                )
        else:
            for value, labelled in ((hi_lim, True), (lo_lim, hi_lim is None)):
                if value is not None and np.isfinite(value):
                    p.line(
                        span, [value, value], color=colors[1],
                        legend_label="Threshold" if labelled else "Threshold",
                    )  # fmt: skip
        tmax = _naive(trace.dropna().index.max())
        p.x_range = Range1d(t0 - pd.Timedelta(minutes=30), t0 + (tmax - t0) * 1.1)
    p.yaxis.axis_label = info.get("ylabel", parameter)
    p.add_tools(
        HoverTool(
            renderers=[line],
            mode="vline",
            tooltips=[
                ("Time", "@t{%F %H:%M} UTC"),
                ("value", "@v{0.3f}"),
                ("sigma", "@sig{0.3f}"),
            ],
            formatters={"@t": "datetime"},
        )  # fmt: skip
    )
    plot_style.finish_legend(p, "bottom_left")
    return p


# ---------------------------------------------------------------------------
# forced-trigger summary (lmon plots/summary.py)
# ---------------------------------------------------------------------------


def _steps(p, index, values, color, label, hover=True):
    """steps-mid line (matplotlib ``drawstyle="steps-mid"``)."""
    return p.step(
        x=plot_style.utc_naive(index), y=np.asarray(values, dtype=float), mode="center",
        color=color, line_width=1.2, legend_label=label,
    )  # fmt: skip


def _link_percent_axis(p, avg_total_forced_mhz, label="FT failure fraction (%)"):
    """Secondary axis in percent of the average forced-trigger rate (mHz/kg).

    Mirrors ``secondary_yaxis`` in the pipeline: the percent range follows
    the primary range through a CustomJS so zooming keeps them consistent.
    """
    if (
        avg_total_forced_mhz is None
        or not np.isfinite(avg_total_forced_mhz)
        or avg_total_forced_mhz <= 0
    ):
        return
    scale = 100.0 / avg_total_forced_mhz
    pct = Range1d(start=0, end=1)
    p.extra_y_ranges = {**p.extra_y_ranges, "pct": pct}
    p.add_layout(LinearAxis(y_range_name="pct", axis_label=label), "right")
    p.y_range.js_on_change(
        "start", CustomJS(args={"pct": pct, "s": scale}, code="pct.start = cb_obj.start * s;")
    )  # fmt: skip
    p.y_range.js_on_change(
        "end", CustomJS(args={"pct": pct, "s": scale}, code="pct.end = cb_obj.end * s;")
    )  # fmt: skip
    p.yaxis[1].axis_label_text_font_size = "12px"


def ft_per_string(rates, period, run, string, avg_total_forced_mhz, colors):
    """FT failure rate per detector of one string (lmon ``_ft_string_figure``).

    ``colors`` is an iterator over the shared tab20 cycle (advanced across
    strings, as in the pipeline).
    """
    p = plot_style.make_figure(
        f"{period} - {run} - string {string}", x_datetime=True, height=450,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    for det in rates.columns:
        _steps(p, rates.index, rates[det], next(colors), str(det))
    finite = rates.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    top = float(finite.max()) if finite.size else 1.0
    p.y_range = Range1d(-0.02 * top, top * 1.1 or 1.0)
    _link_percent_axis(p, avg_total_forced_mhz)
    p.yaxis[0].axis_label = "Normalized FT failure rate (mHz/kg)"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "top_left", ncols=2)
    return p


def ft_all_strings(per_string, period, run):
    """Combined FT failure rate, one steps line per string."""
    p = plot_style.make_figure(
        f"{period} - {run} - All strings", x_datetime=True, height=450,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    colors = itertools.cycle(plot_style.TAB20)
    for string in per_string.columns:
        _steps(
            p, per_string.index, per_string[string], next(colors), f"String {string}"
        )
    p.yaxis.axis_label = "Normalized FT failure rate (mHz/kg)"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "top_left", ncols=2)
    return p


def ft_survival(surviving_frac, period):
    """FT surviving-events fraction over all strings (lmon ``_ft_sf_figure``)."""
    p = plot_style.make_figure(
        f"{period} - All strings combined", x_datetime=True, height=350,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    series = (
        surviving_frac.iloc[:, 0]
        if isinstance(surviving_frac, pd.DataFrame)
        else surviving_frac
    )
    _steps(p, series.index, series, "red", "FT surviving events")
    p.yaxis.axis_label = "FT surviving events (%)"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "bottom_left")
    return p


# ---------------------------------------------------------------------------
# quality cuts (lmon plots/qc.py) and event-rate QC (plots/summary.py)
# ---------------------------------------------------------------------------

_THRESHOLDED_FLAGS = ("IsDischarge", "IsSaturated")


def _qc_threshold(p, flag):
    """5 mHz upper threshold with shaded exclusion (IsDischarge/IsSaturated)."""
    if flag not in _THRESHOLDED_FLAGS:
        return
    limit = plot_style.MTG_PLOT_INFO[flag]["limits"][1]
    plot_style.threshold(p, limit, above=True, label=f"{limit} mHz upper threshold")


def qc_rate_series(rates, dets, avg_rates, flag, period, run, string, colors):
    """1 h QC rate per detector of one string (lmon ``_rate_series_figure``).

    ``dets``: (name, position) pairs in string order; ``avg_rates``: name ->
    integrated rate (mHz) for the legend; ``colors``: the shared tab20
    iterator (advanced across flags and strings, as in the pipeline).
    """
    p = plot_style.make_figure(
        f"{period} {run} - String: {string}", x_datetime=True, height=400,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    drawn = 0
    for name, pos in dets:
        if name not in rates.columns:
            continue
        avg = avg_rates.get(name, float("nan"))
        label = f"{name} - pos {pos} - {round(float(avg), 2)} mHz"
        _steps(p, rates.index, rates[name], next(colors), label)
        drawn += 1
    if not drawn:
        return plot_style.empty_figure(
            f"{flag}: no detectors of string {string} in {run}"
        )
    _qc_threshold(p, flag)
    p.yaxis.axis_label = f"{period} {run} - 1h {flag} rate (mHz)"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "top_right")
    return p


def qc_average(rate_by_name, string_groups, flag, dead_time_pct, period, run):
    """Per-detector integrated QC rate on a log axis (lmon ``_average_figure``).

    ``rate_by_name``: name -> mHz; ``string_groups``: {string: [names]} in
    string/position order (detectors without a rate keep their slot).
    """
    factors, strings, xs, ys = [], [], [], []
    for string, names in string_groups.items():
        for name in names:
            if name in factors:
                continue
            factors.append(name)
            strings.append(string)
            rate = rate_by_name.get(name)
            if rate is not None and np.isfinite(rate) and rate > 0:
                xs.append(name)
                ys.append(float(rate))
    title = f"period: {period} - run: {run} - passing {flag}"
    if flag == "IsDischarge":
        title += (
            " - tot dead time unavailable"
            if dead_time_pct is None
            else f" - tot dead time {dead_time_pct:.3f}%"
        )
    if not factors:
        return plot_style.empty_figure(title + " (no detectors)")
    p = plot_style.make_figure(
        title, x_range=FactorRange(*factors), y_axis_type="log", height=400,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    source = ColumnDataSource({"det": xs, "rate": ys})
    glyph = p.scatter(x="det", y="rate", source=source, color="dodgerblue", size=7)
    p.add_tools(
        HoverTool(renderers=[glyph], tooltips=[("Detector", "@det"), ("Rate", "@rate{0.3f} mHz")])
    )  # fmt: skip
    y_lo = min(ys) / 2 if ys else 0.1
    y_hi = max(ys) * 2 if ys else 10
    p.y_range = Range1d(y_lo, y_hi)
    plot_style.string_separators(p, factors, strings, y_lo * (y_hi / y_lo) ** 0.05)
    _qc_threshold(p, flag)
    p.yaxis.axis_label = f"Average rate {flag}=True (mHz)"
    p.xaxis.major_label_orientation = pi / 2
    p.xaxis.major_label_text_font_size = "8px"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "top_right")
    return p


def classifier_grid(edges, counts_by_flag, dets, fracs, par, period, run, string):
    """Per-detector classifier distributions of one string (lmon ``_classifier_figure``).

    ``counts_by_flag``: event-type flag -> {detector: counts}; ``fracs``:
    (detector, flag) -> percent in [-5, 5]. One small log-y figure per
    detector (step outlines per event type), ±5 lines with the outer region
    shaded, x fixed to [-10, 10]; returned as a Bokeh grid.
    """
    figures = []
    all_counts = counts_by_flag.get("All", {})
    for name, pos in dets:
        if name not in all_counts:
            continue
        p = plot_style.make_figure(
            "", width=420, height=260, x_range=Range1d(-10, 10), y_axis_type="log",
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )  # fmt: skip
        p.sizing_mode = "fixed"
        for (flag, label), color in zip(
            plot_style.CLASSIFIER_FLAG_LABELS.items(),
            plot_style.MPL_CYCLE,
            strict=False,
        ):
            counts = counts_by_flag.get(flag, {}).get(name)
            if counts is None:
                counts = np.zeros(len(edges) - 1)
            perc = fracs.get((name, flag), float("nan"))
            y = np.where(np.asarray(counts, dtype=float) > 0, counts, np.nan)
            p.step(
                x=edges[:-1], y=y, mode="after", color=color, line_width=1.2,
                legend_label=f"{label} - {perc:.1f}%",
            )  # fmt: skip
        for x in (-5, 5):
            p.add_layout(
                Span(
                    location=x,
                    dimension="height",
                    line_color="black",
                    line_dash="dashed",
                )
            )
        p.add_layout(
            BoxAnnotation(
                right=-5, fill_color="darkgray", fill_alpha=0.2, level="underlay"
            )
        )
        p.add_layout(
            BoxAnnotation(
                left=5, fill_color="darkgray", fill_alpha=0.2, level="underlay"
            )
        )
        p.xaxis.axis_label = "Classifiers"
        p.yaxis.axis_label = "Counts"
        p.grid.grid_line_color = None
        plot_style.finish_legend(p, "top_right", title=f"{name} (pos {pos})")
        figures.append(p)
    if not figures:
        return plot_style.empty_figure(
            f"{par}: no classifier histograms for string {string}"
        )
    ncols = max(1, int(np.ceil(np.sqrt(len(figures)))))
    grid = gridplot(figures, ncols=ncols, toolbar_location="right")
    return pn.Column(
        pn.pane.Markdown(f"### {period} {run} - string {string} - {par}"),
        pn.pane.Bokeh(grid),
    )


def classifier_fraction_bars(frac, par, period, run, string):
    """In-range fraction per detector and event type (fallback without dist2d).

    ``frac``: qc_classifier_frac rows of one classifier and string.
    """
    rows = frac.sort_values(["string", "detector"]) if "string" in frac else frac
    dets = list(dict.fromkeys(rows["detector"].astype(str)))
    flags = [
        f for f in plot_style.CLASSIFIER_FLAG_LABELS if f in set(rows["event_type"])
    ]
    if not dets or not flags:
        return plot_style.empty_figure(
            f"{par}: no classifier fractions for string {string}"
        )
    p = plot_style.make_figure(
        f"{period} {run} - string {string} - {par}: events within ±5",
        x_range=FactorRange(*dets), height=400, tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    width = 0.8 / len(flags)
    for i, (flag, color) in enumerate(zip(flags, plot_style.MPL_CYCLE, strict=False)):
        sub = rows[rows["event_type"] == flag]
        by_det = dict(
            zip(sub["detector"].astype(str), sub["percent_in_range"], strict=False)
        )
        perc = [float(by_det.get(d, np.nan)) for d in dets]
        offset = -0.4 + width * (i + 0.5)
        p.vbar(
            x=dodge("det", offset, range=p.x_range), top="perc", width=width * 0.9,
            source=ColumnDataSource({"det": dets, "perc": perc}), color=color, alpha=0.8,
            legend_label=plot_style.CLASSIFIER_FLAG_LABELS[flag],
        )  # fmt: skip
    p.y_range = Range1d(0, 105)
    p.yaxis.axis_label = "Events within ±5 (%)"
    p.xaxis.major_label_orientation = pi / 2
    p.xaxis.major_label_text_font_size = "8px"
    plot_style.finish_legend(p, "bottom_left")
    return p


def event_rate_qc(frame, period, run):
    """QC-split hourly event rate (lmon ``_event_rate_figure``)."""
    series = (
        ("all_events", "All events"),
        ("delayed_discharges", "Delayed discharges"),
        ("failing_qc", "Failing QC"),
        ("surviving_qc", "Surviving QC"),
    )
    p = plot_style.make_figure(
        f"{period} {run} - event rate by QC outcome", x_datetime=True, height=380,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )  # fmt: skip
    on_mass = (
        float(frame["on_mass_kg"].iloc[0]) if "on_mass_kg" in frame else float("nan")
    )
    for column, label in series:
        if column not in frame.columns:
            continue
        rate = frame[column].dropna()
        if rate.empty:
            continue
        index = plot_style.utc_naive(rate.index)
        edges = index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)]))
        values = np.append(rate.to_numpy(dtype=float), rate.to_numpy(dtype=float)[-1])
        p.step(
            x=edges, y=values, mode="after", color=plot_style.EVENT_RATE_COLORS[column],
            line_width=1.5, legend_label=label,
        )  # fmt: skip
    p.yaxis.axis_label = "Hourly rate normalized by ON mass (mHz/kg)"
    p.grid.grid_line_color = None
    title = f"ON mass = {on_mass:.1f} kg" if np.isfinite(on_mass) else None
    plot_style.finish_legend(p, "top_right", title=title)
    return p
