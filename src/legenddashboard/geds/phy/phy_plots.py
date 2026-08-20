from __future__ import annotations

import dataclasses

import pandas as pd
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    DatetimeTickFormatter,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from seaborn import color_palette

# physics plots
phy_plots_types_dict = {
    "Pulser Events": "IsPulser",
    "Baseline Events": "IsBsln",
}
phy_plots_vals_dict = {
    "Baseline FPGA": "Baseline",
    "Baseline Mean": "BlMean",
    "Noise": "BlStd",
    "Gain": "Trapemax",
    "Cal. Gain": "TrapemaxCtcCal",
    "Gain to Pulser Ratio": "Trapemax_pulser01anaRatio",
    "Gain to Pulser Diff.": "Trapemax_pulser01anaDiff",
    "Rate": "EventRate",
    "PSD Classifier": "AoeCustom",
}
phy_unit_vals = ["Relative", "Absolute"]
# NOTE: these dicts are widget option lists + the v1 fallback vocabulary only;
# on the contract-v2 path labels/units come from the run manifest and per-key
# attrs (single source of truth is the producer's config/settings.py).
phy_plots_sc_vals_dict = {
    "None": False,
    "DAQ Temp. Left 1": "DaqLeft_Temp1",
    "DAQ Temp. Left 2": "DaqLeft_Temp2",
    "DAQ Temp. Right 1": "DaqRight_Temp1",
    "DAQ Temp. Right 2": "DaqRight_Temp2",
    "RREiT": "RREiT",
    "RRNTe": "RRNTe",
    "RRSTe": "RRSTe",
    "ZUL_T_RR": "ZUL_T_RR",
}


def phy_plot_vsTime(
    data_string,
    data_string_mean,
    plot_info,
    plot_type,
    plot_name,
    resample_unit,
    string,
    run,
    period,
    run_dict,
    abs_unit,
    data_sc,
    sc_param,
):
    # add two hours to UTC index (CET display convention); shallow-copy first —
    # mutating a caller-owned (possibly cached) frame in place is not allowed
    if data_string.index[0].utcoffset() != pd.Timedelta(hours=2):
        data_string = data_string.copy(deep=False)
        data_string.index = data_string.index + pd.Timedelta(hours=2)

    data_high_res = data_string.copy()
    data_high_res["datetime"] = data_high_res.index
    data_high_res = data_high_res.reset_index(drop=True)

    # resample data
    if resample_unit != "0min":
        data_resampled = data_string.resample(resample_unit, origin="start").mean()
        data_resampled["datetime"] = data_resampled.index
        data_resampled = data_resampled.reset_index(drop=True)

        source_high_res = ColumnDataSource(data_high_res)
        source_resampled = ColumnDataSource(data_resampled)
    else:
        source_high_res = ColumnDataSource(data_high_res)
        source_resampled = None

    n_channels = len(data_string_mean.columns)
    colors = color_palette("hls", n_channels).as_hex()

    p = figure(
        width=1000,
        height=600,
        x_axis_type="datetime",
        tools="pan,box_zoom,ywheel_zoom,hover,reset,save",
        output_backend="webgl",
        active_scroll="ywheel_zoom",
    )
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. {plot_type} | {plot_name} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.mode = "vline"

    # plot data
    hover_renderers = []
    for i, col in enumerate(data_string_mean.columns):
        time_series_col = f"{col}_val"

        # all timestamp entries
        line_high_res = p.line(
            x="datetime",
            y=time_series_col,
            source=source_high_res,
            color=colors[i],
            line_width=1,
            line_alpha=0.3 if source_resampled is not None else 1,
            legend_label=col,
            name=col,
        )

        # resampled data
        if source_resampled is not None:
            line_resampled = p.line(
                x="datetime",
                y=time_series_col,
                source=source_resampled,
                color=colors[i],
                line_width=2.5,
                legend_label=col,
                name=f"{col}",
            )
            hover_renderers.append(line_resampled)
        else:
            hover_renderers.append(line_high_res)

    p.hover.renderers = hover_renderers
    p.hover.tooltips = [
        ("Time", "$x{%F %H:%M:%S CET}"),
        (
            f"Avg. {plot_info.loc['label'].iloc[0]} ({plot_info.loc['unit'].iloc[0]})",
            f"@{time_series_col}{{0.2f}}",
        ),
        ("Detector", "$name"),
    ]
    p.hover.formatters = {"$x": "datetime", "$source": "printf"}

    # legend
    if p.legend:
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"

    # axis
    if source_resampled is not None:
        data_for_start_time = data_resampled
    else:
        data_for_start_time = data_high_res

    start_time_str = pd.to_datetime(data_for_start_time["datetime"].iloc[0]).strftime(
        "%d/%m/%Y %H:%M:%S"
    )
    p.xaxis.axis_label = f"Time (CET), starting: {start_time_str}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = (
        f"{plot_info.loc['label'].iloc[0]} [{plot_info.loc['unit'].iloc[0]}]"
    )
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d")

    _apply_y_range(p, plot_info.loc["label"].iloc[0], plot_info.loc["unit"].iloc[0])
    _add_sc_overlay(p, data_sc, sc_param, resample_unit)

    return p


def _apply_y_range(p, label, unit):
    """Fixed y-ranges by parameter label (shared by v1 and v2 time plots)."""
    if unit == "%":
        if label == "Noise":
            p.y_range = Range1d(-150, 150)
        elif label in ["FPGA baseline", "Mean Baseline"]:
            p.y_range = Range1d(-10, 10)
        elif label == "Gain to Pulser Difference":
            p.y_range = Range1d(-4, 4)
        elif label == "Event Rate":
            p.y_range = Range1d(-150, 50)
        elif label == "Custom A/E (A_max / cuspEmax)":
            p.y_range = Range1d(-10, 10)
        else:
            p.y_range = Range1d(-1, 1)
    elif label == "Noise":
        p.y_range = Range1d(-150, 150)


def _add_sc_overlay(p, data_sc, sc_param, resample_unit):
    """Overlay a slow-control series on a secondary y-axis (shared v1/v2)."""
    if data_sc is None or data_sc.empty:
        return
    y_range_name = f"{sc_param}_range"
    y_min = data_sc["value"].min() * 0.99
    y_max = data_sc["value"].max() * 1.01
    p.extra_y_ranges = {y_range_name: Range1d(start=y_min, end=y_max)}
    unit = data_sc["unit"][0]
    p.add_layout(
        LinearAxis(
            y_range_name=y_range_name,
            axis_label=f"{sc_param} [{unit}]",
            axis_label_text_font_size="20px",
        ),
        "right",
    )

    sc_data = data_sc.copy()
    sc_data["tstamp"] = pd.to_datetime(sc_data["tstamp"], origin="unix", utc=True)
    sc_data = sc_data.set_index("tstamp")["value"]
    if resample_unit != "0min":
        sc_data_resampled = sc_data.resample(resample_unit).mean()
        p.line(
            sc_data.index,
            sc_data.values,
            color="black",
            alpha=0.2,
            legend_label=sc_param,
            y_range_name=y_range_name,
            line_width=2,
        )
        p.line(
            sc_data_resampled.index,
            sc_data_resampled.values,
            color="black",
            legend_label=sc_param,
            y_range_name=y_range_name,
            line_width=2,
        )
    else:
        p.line(
            sc_data.index,
            sc_data.values,
            color="black",
            legend_label=sc_param,
            y_range_name=y_range_name,
            line_width=2,
        )


@dataclasses.dataclass
class PlotMeta:
    """Display metadata handed from the monitoring view to plot functions."""

    label: str
    unit: str
    abs_unit: str
    flag_display: str  # e.g. "Pulser Events"
    param_display: str  # e.g. "Cal. Gain"
    string: str
    run: str
    period: str
    experiment: str


def phy_plot_binned_vsTime(
    mean_df,
    std_df,
    min_df,
    max_df,
    meta: PlotMeta,
    flagged_ranges=(),
    data_sc=None,
    sc_param=False,
    cadence_label="1min",
):
    """Time view of contract-v2 binned stats: mean line, +-1 sigma band, min/max envelope.

    All frames share a tz-aware UTC DatetimeIndex and detector-name columns;
    no client-side resampling happens here — the cadence was chosen at read
    time and is stated on the x-axis label.
    """
    shift = pd.Timedelta(hours=2)  # CET display convention, as in phy_plot_vsTime
    # drop the tz *after* shifting: bokeh serializes tz-aware indexes as object
    # arrays element by element, which dominates the update round trip
    index = (mean_df.index + shift).tz_localize(None)

    n_channels = len(mean_df.columns)
    colors = color_palette("hls", max(n_channels, 1)).as_hex()

    p = figure(
        width=1000,
        height=600,
        x_axis_type="datetime",
        tools="pan,box_zoom,ywheel_zoom,hover,reset,save",
        output_backend="webgl",
        active_scroll="ywheel_zoom",
    )
    p.title.text = (
        f"{meta.experiment}-{meta.period}-{meta.run} | "
        f"Phy. {meta.flag_display} | {meta.param_display} | {meta.string}"
    )
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.mode = "vline"

    hover_renderers = []
    for i, det in enumerate(mean_df.columns):
        mean = mean_df[det].to_numpy()
        std = std_df[det].to_numpy()
        source = ColumnDataSource(
            {
                "datetime": index,
                "mean": mean,
                "std": std,
                "band_lo": mean - std,
                "band_hi": mean + std,
                "min": min_df[det].to_numpy(),
                "max": max_df[det].to_numpy(),
            }
        )
        # min/max envelope, +-1 sigma band, mean line — one legend entry per detector
        # so click-to-hide toggles the trio together
        p.varea(
            x="datetime",
            y1="min",
            y2="max",
            source=source,
            fill_color=colors[i],
            fill_alpha=0.08,
            legend_label=det,
        )
        p.varea(
            x="datetime",
            y1="band_lo",
            y2="band_hi",
            source=source,
            fill_color=colors[i],
            fill_alpha=0.18,
            legend_label=det,
        )
        line = p.line(
            x="datetime",
            y="mean",
            source=source,
            color=colors[i],
            line_width=2,
            legend_label=det,
            name=det,
        )
        hover_renderers.append(line)

    p.hover.renderers = hover_renderers
    p.hover.tooltips = [
        ("Time", "$x{%F %H:%M:%S CET}"),
        (f"Mean {meta.label} ({meta.unit})", "@mean{0.2f}"),
        ("std dev", "@std{0.2f}"),
        ("min-max", "@min{0.2f} - @max{0.2f}"),
        ("Detector", "$name"),
    ]
    p.hover.formatters = {"$x": "datetime", "$source": "printf"}

    if p.legend:
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"

    if len(index):
        start_time_str = pd.to_datetime(index[0]).strftime("%d/%m/%Y %H:%M:%S")
        p.xaxis.axis_label = (
            f"Time (CET), {cadence_label} bins, starting: {start_time_str}"
        )
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = f"{meta.label} [{meta.unit}]"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d")

    # ignore-keys ranges: kept in the data, displayed shaded
    window_lo, window_hi = (index[0], index[-1]) if len(index) else (None, None)
    shown_flag = False
    for raw_lo, raw_hi, _reason in flagged_ranges:
        shifted_lo = (raw_lo + shift).tz_localize(None)
        shifted_hi = (raw_hi + shift).tz_localize(None)
        if window_lo is not None and (shifted_hi < window_lo or shifted_lo > window_hi):
            continue
        p.add_layout(
            BoxAnnotation(
                left=shifted_lo,
                right=shifted_hi,
                fill_color="orange",
                fill_alpha=0.12,
                level="underlay",
            )
        )
        shown_flag = True
    if shown_flag:
        p.varea(
            x=[index[0]],
            y1=[float("nan")],
            y2=[float("nan")],
            fill_color="orange",
            fill_alpha=0.3,
            legend_label="flagged (ignore-keys)",
        )

    _apply_y_range(p, meta.label, meta.unit)
    _add_sc_overlay(p, data_sc, sc_param, "0min")

    return p


def phy_plot_dist_histogram(dist, meta: PlotMeta):
    """Distribution view from the contract-v2 1-D histogram (all detectors).

    ``dist`` is the (edges, counts, attrs) tuple from
    ``contract_reader.read_dist``. The distribution is filled from every
    detector's samples, so the string selector does not filter it.
    """
    edges, counts, _attrs = dist

    p = figure(
        width=1000,
        height=600,
        tools="pan,box_zoom,ywheel_zoom,hover,reset,save",
        output_backend="webgl",
        active_scroll="ywheel_zoom",
    )
    p.title.text = (
        f"{meta.experiment}-{meta.period}-{meta.run} | "
        f"Phy. {meta.flag_display} | {meta.param_display} | all detectors"
    )
    p.title.align = "center"
    p.title.text_font_size = "25px"

    p.quad(
        top=counts,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="steelblue",
        line_color="white",
        fill_alpha=0.8,
    )
    p.hover.tooltips = [
        (f"{meta.label} ({meta.unit})", "$x{0.2f}"),
        ("Counts", "$y{0}"),
    ]
    p.xaxis.axis_label = f"{meta.label} [{meta.unit}]"
    p.yaxis.axis_label = "Counts"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label_text_font_size = "20px"

    return p
