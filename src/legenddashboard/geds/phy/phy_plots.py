from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    LinearAxis,
    Range1d,
    ZoomInTool,
    ZoomOutTool,
)
from bokeh.palettes import Category20, Turbo256
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
    # add two hours to UTC index 
    if data_string.index[0].utcoffset() != pd.Timedelta(hours=2):
        data_string.index += pd.Timedelta(hours=2)

    data_high_res = data_string.copy()
    data_high_res["datetime"] = data_high_res.index
    data_high_res.reset_index(drop=True, inplace=True)
    
    # resample data 
    if resample_unit != "0min":
        data_resampled = data_string.resample(resample_unit, origin="start").mean()
        data_resampled["datetime"] = data_resampled.index
        data_resampled.reset_index(drop=True, inplace=True)

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
        (f"Avg. {plot_info.loc['label'].iloc[0]} ({plot_info.loc['unit'].iloc[0]})", f"@{time_series_col}{{0.2f}}"),
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
        
    start_time_str = pd.to_datetime(data_for_start_time['datetime'].iloc[0]).strftime('%d/%m/%Y %H:%M:%S')
    p.xaxis.axis_label = f"Time (CET), starting: {start_time_str}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = f"{plot_info.loc['label'].iloc[0]} [{plot_info.loc['unit'].iloc[0]}]"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d")

    # y-range for % unit plots
    label = plot_info.loc["label"].iloc[0]
    unit = plot_info.loc["unit"].iloc[0]
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

    # slow-control data 
    if not data_sc.empty:
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

        # resampling
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

    return p

def phy_plot_histogram(
    data_string,
    plot_info,
    plot_type,
    resample_unit,
    string,
    run,
    period,
    run_dict,
    channels,
    channel_map,
):
    p = figure(
        width=1000,
        height=600,
        x_axis_type="datetime",
        tools="pan,box_zoom,ywheel_zoom,hover,reset,save",
        output_backend="webgl",
        active_scroll="ywheel_zoom",
    )
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. {plot_type} | {plot_info.loc['label'].iloc[0]} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {"$x": "printf", "$snap_y": "printf"}
    p.hover.tooltips = [
        (f"{plot_info.loc['label'].iloc[0]} ({plot_info.loc['unit'].iloc[0]}", "$x{%0.2f}"),
        ("Counts", "$snap_y"),
        ("Detector", "$name"),
    ]

    p.hover.mode = "vline"

    level = 1
    zoom_in = ZoomInTool(
        level=level, dimensions="height", factor=0.5
    )  # set specific zoom factor
    zoom_out = ZoomOutTool(level=level, dimensions="height", factor=0.5)
    p.add_tools(zoom_in, zoom_out)
    # p.toolbar.active_drag = None      use this line to activate only hover and ywheel_zoom as active tool

    len_colours = len(data_string.columns)
    colours = Turbo256[len_colours] if len_colours > 19 else Category20[len_colours]

    for position, data_channel in data_string.groupby("position"):
        # generate histogram
        # needed for cuspEmax because with geant outliers not possible to view normal histo
        hrange = {"keV": [0, 2500]}
        # take full range if not specified
        x_min = (
            hrange[plot_info["unit"]][0]
            if plot_info["unit"] in hrange
            else data_channel[plot_info["parameter"]].min()
        )
        x_max = (
            hrange[plot_info["unit"]][1]
            if plot_info["unit"] in hrange
            else data_channel[plot_info["parameter"]].max()
        )

        # --- bin width
        # bwidth = {"keV": 2.5}  # what to do with binning???
        # bin_width = bwidth[plot_info["unit"]] if plot_info["unit"] in bwidth else None
        # no_bins = int((x_max - x_min) / bin_width) if bin_width else 50
        # counts_ch, bins_ch = np.histogram(data_channel[plot_info["parameter"]], bins=no_bins, range=(x_min, x_max))
        # bins_ch = (bins_ch[:-1] + bins_ch[1:]) / 2

        # --- bin width
        bwidth = {"keV": 2.5}
        bin_width = bwidth.get(plot_info["unit"], 1)

        # Compute number of bins
        if bin_width:
            bin_no = (
                bin_width / 5 if "AoE" not in plot_info["parameter"] else bin_width / 50
            )
            bin_no = bin_no / 2 if "Corrected" in plot_info["parameter"] else bin_no
            bin_no = bin_width if "AoE" not in plot_info["parameter"] else bin_no

            bin_edges = (
                np.arange(x_min, x_max + bin_width, bin_no)
                if plot_info["unit_label"] == "%"
                else np.arange(x_min, x_max + bin_width, bin_no)
            )
        else:
            bin_edges = 50
        counts_ch, bins_ch = np.histogram(
            data_channel[plot_info["parameter"]], bins=bin_edges, range=(x_min, x_max)
        )
        bins_ch = (bins_ch[:-1] + bins_ch[1:]) / 2
        # create plot histo
        histo_df = pd.DataFrame(
            {
                "counts": counts_ch,
                "bins": bins_ch,
                "position": position,
                "cc4_id": data_channel["cc4_id"].unique()[0],
            }
        )
        # plot
        p.line(
            "bins",
            "counts",
            source=histo_df,
            color=colours[position - 1],
            legend_label=f"{data_channel['name'].unique()[0]}",
            name=f"ch {data_channel['channel'].unique()[0]}",
            line_width=2,
        )

    if p.legend:
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"
    p.xaxis.axis_label = f"{plot_info['label']} [{plot_info['unit_label']}]"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Counts"
    p.yaxis.axis_label_text_font_size = "20px"

    return p
