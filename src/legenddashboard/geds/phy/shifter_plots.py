"""Bokeh twins of legend-data-monitor's shifter figures.

Pure builders: frames from the period contract in, a figure (or Panel
layout) out. Each function mirrors one figure of the pipeline's
``plots/{summary,stability,qc}.py`` element by element; what cannot be
reproduced in Bokeh (per-tick label colours, the legacy "last cycle"
legend title) is replaced as noted in the docstrings.
"""

from __future__ import annotations

from math import pi

import numpy as np
import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    FactorRange,
    HoverTool,
    Range1d,
    Span,
    Whisker,
)

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
