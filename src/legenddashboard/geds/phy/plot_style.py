"""Shared look and vocabulary for the physics pages.

Mirrors legend-data-monitor's drawing conventions (``monitoring.apply_
monitoring_style`` and ``settings/mtg-plot-settings.yaml``) so the Bokeh
figures read like the pipeline's PDFs: grid on, white background, small
fonts, thresholds as dashed lines with a shaded excluded region, UTC time.
"""

from __future__ import annotations

from math import pi

from bokeh.models import (
    BoxAnnotation,
    DatetimeTickFormatter,
    Label,
    Span,
)
from bokeh.plotting import figure

# matplotlib palettes, by name, so colours match the pipeline figures
MPL_CYCLE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)  # fmt: skip
TAB20 = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728",
    "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2",
    "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
)  # fmt: skip
C2, C4 = MPL_CYCLE[2], MPL_CYCLE[4]  # PULS01ANA / "GED corrected" in lmon

# settings/mtg-plot-settings.yaml (legend-data-monitor), keyed by parameter
MTG_PLOT_INFO = {
    "TrapemaxCtcCal": {
        "ylabel": "Energy diff / keV",
        "colors": ("dodgerblue", "blue"),
        "title": "pulser_stab",
        "unit": "keV",
        "limits": (None, None),
    },
    "Trapemax": {
        "ylabel": "Energy diff / keV",
        "colors": ("dodgerblue", "blue"),
        "title": "pulser_stab_uncalib",
        "unit": "%",
        "limits": (None, None),
    },
    "Baseline": {
        "ylabel": "Baseline % variations",
        "colors": ("red", "firebrick"),
        "title": "baseln_stab",
        "unit": "%",
        "limits": (-10, 10),
    },
    "BlStd": {
        "ylabel": "Baseline std [ADC]",
        "colors": ("peru", "saddlebrown"),
        "title": "baseln_spike",
        "unit": "ADC",
        "limits": (None, 50),
    },
    "FEP_variation": {
        "ylabel": "FEP gain variation [keV]",
        "colors": ("peru", "saddlebrown"),
        "title": "FEP_gain_stab",
        "unit": "keV",
        "limits": (None, None),
    },
    "IsDischarge": {"title": "discharge_rate", "unit": "mHz", "limits": (None, 5)},
    "IsSaturated": {"title": "saturated_rate", "unit": "mHz", "limits": (None, 5)},
    # SiPM channel-health bands, graded on the 60min bins of the spms contract
    # (the contract's own limits attrs are all null for spms keys)
    "IsBsln_WfMode_var": {
        "title": "spms_baseln_stab",
        "unit": "%",
        "limits": (-0.05, 0.05),
    },
    "IsBsln_CurrFwhm_var": {
        "title": "spms_noise_stab",
        "unit": "%",
        "limits": (-5, 5),
    },
    "IsBsln_NPulses": {
        "title": "spms_dark_rate",
        "unit": "per window",
        "rolling": 6,
        "limits": (0.002, 1.0),
    },
    "All_HasAnyNoise": {
        "title": "spms_noisy_frac",
        "unit": "fraction",
        "limits": (None, 0.05),
    },
}
MTG_BY_TITLE = {v["title"]: v for v in MTG_PLOT_INFO.values()}
SPMS_CHECKS = (
    "spms_baseln_stab",
    "spms_noise_stab",
    "spms_dark_rate",
    "spms_noisy_frac",
)

# SiPM explorer vocabulary (display -> contract flag / parameter)
SPMS_TYPES = {
    "Forced trigger": "IsBsln",
    "Physics": "IsPhysics",
    "All events": "All",
}
SPMS_VALUES = {
    "Baseline (mode)": "WfMode",
    "Current noise FWHM": "CurrFwhm",
    "Waveform lower HWHM": "WfLowerHwhm",
    "Pulses per window (dark rate)": "NPulses",
    "Light per event": "PeSum",
    "Largest pulse per event": "PeMax",
    "Noisy waveform fraction": "HasAnyNoise",
}

# legacy pars_to_inspect order: drives the shared tab20 cycle in the QC figures
DEFAULT_QC_FLAGS = (
    "IsHighlyPositivePolarityCandidate",
    "IsValidBlSlope",
    "IsValidBlSlopeRms",
    "IsValidBlPolyRms",
    "IsValidTailRms",
    "IsNotNoiseBurst",
    "IsValidCuspemin",
    "IsValidCuspemax",
    "IsValidTrapTpmax",
    "IsLowCuspemax",
    "IsDischarge",
    "IsSaturated",
)
CLASSIFIER_FLAG_LABELS = {
    "All": "All events",
    "IsPulser": "TP",
    "IsBsln": "FT",
    "IsPhysics": "~TP, ~FT, E>25 keV",
}
QBB_LIN_LABEL = "Qββ ±FWHM/2 lin. (threshold)"
QBB_QUAD_LABEL = "Qββ ±FWHM/2 quad. (threshold)"
EVENT_RATE_COLORS = {
    "all_events": "dimgrey",
    "delayed_discharges": "darkorange",
    "failing_qc": "crimson",
    "surviving_qc": "dodgerblue",
}

UTC_FORMATTER = DatetimeTickFormatter(
    days="%Y/%m/%d", hours="%m/%d %H:%M", minutes="%H:%M"
)


def make_figure(title="", *, x_datetime=False, width=1000, height=450, **kwargs):
    """A figure with the pipeline look; ``kwargs`` go to ``bokeh.plotting.figure``."""
    kwargs.setdefault("tools", "pan,box_zoom,wheel_zoom,hover,reset,save")
    if x_datetime:
        kwargs.setdefault("x_axis_type", "datetime")
    p = figure(width=width, height=height, sizing_mode="scale_width", **kwargs)
    style_figure(p, title)
    if x_datetime:
        p.xaxis.formatter = UTC_FORMATTER
        p.xaxis.axis_label = "Time (UTC)"
    return p


def style_figure(p, title=""):
    """Apply the shared look in place (grid, white background, font sizes)."""
    p.title.text = title
    p.title.align = "center"
    p.title.text_font_size = "14px"
    p.background_fill_color = "white"
    p.grid.grid_line_color = "#d0d0d0"
    p.grid.grid_line_alpha = 0.8
    p.xaxis.axis_label_text_font_size = "12px"
    p.yaxis.axis_label_text_font_size = "12px"
    p.toolbar.logo = None
    return p


def finish_legend(p, location="bottom_left", title=None, ncols=1):
    """Click-to-hide legend in the pipeline's position; no-op without entries."""
    if not p.legend:
        return
    legend = p.legend[0]
    legend.location = location
    legend.click_policy = "hide"
    legend.label_text_font_size = "9px"
    legend.background_fill_alpha = 0.6
    if title:
        legend.title = title
        legend.title_text_font_size = "9px"
    legend.ncols = ncols


def empty_figure(reason: str, width=1000, height=450):
    """A blank figure whose title says why there is nothing to draw."""
    p = figure(width=width, height=height, sizing_mode="scale_width")
    style_figure(p, reason)
    p.title.text_font_size = "18px"
    return p


def threshold(p, value, *, above=True, label=None, color="black", shade=True):
    """Dashed threshold line plus the shaded excluded region beyond it.

    ``above=True`` shades everything above the line (an upper limit);
    ``above=False`` everything below (a lower limit). A ``label`` adds a
    legend entry through an invisible proxy line.
    """
    if value is None:
        return
    p.add_layout(
        Span(location=value, dimension="width", line_color=color, line_dash="dashed")
    )
    if shade:
        box = BoxAnnotation(bottom=value) if above else BoxAnnotation(top=value)
        box.fill_color = "gray"
        box.fill_alpha = 0.25
        box.level = "underlay"
        p.add_layout(box)
    if label:
        legend_proxy(p, label, color=color, line_dash="dashed")


def legend_proxy(p, label, color="black", line_dash="solid", line_width=1.5):
    """An empty line whose only purpose is a legend entry (matplotlib idiom)."""
    return p.line(
        x=[], y=[], color=color, line_dash=line_dash, line_width=line_width,
        legend_label=label,
    )  # fmt: skip


def string_separators(p, factors, strings, y_text):
    """Dashed separators between strings on a categorical axis + "String N" labels.

    ``factors``: the x factors in display order; ``strings``: the string of
    each factor (same length). Labels sit at ``y_text`` in data units.
    """
    start = 0
    for i in range(1, len(factors) + 1):
        if i == len(factors) or strings[i] != strings[start]:
            p.add_layout(
                Label(
                    x=start,
                    y=y_text,
                    x_offset=4,
                    text=f"String {strings[start]}",
                    text_font_size="9px",
                    text_color="dimgray",
                    angle=pi / 2,
                    x_units="data",
                    y_units="data",
                )  # fmt: skip
            )
            if i < len(factors):
                p.add_layout(
                    Span(
                        location=i - 0.5,
                        dimension="height",
                        line_color="gray",
                        line_dash="dashed",
                        line_alpha=0.5,
                    )  # fmt: skip
                )
            start = i


def utc_naive(index):
    """Timestamps as naive UTC datetime64 (bokeh serialises these fastest)."""
    if getattr(index, "tz", None) is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


__all__ = [
    "C2",
    "C4",
    "CLASSIFIER_FLAG_LABELS",
    "DEFAULT_QC_FLAGS",
    "EVENT_RATE_COLORS",
    "MPL_CYCLE",
    "MTG_BY_TITLE",
    "MTG_PLOT_INFO",
    "QBB_LIN_LABEL",
    "QBB_QUAD_LABEL",
    "SPMS_CHECKS",
    "SPMS_TYPES",
    "SPMS_VALUES",
    "TAB20",
    "Label",
    "empty_figure",
    "finish_legend",
    "legend_proxy",
    "make_figure",
    "string_separators",
    "style_figure",
    "threshold",
    "utc_naive",
]
