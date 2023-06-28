from bokeh.models import Span, Label, Title, Range1d, HoverTool, Slope
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.plotting import figure, show

import colorcet as cc

import shelve
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle as pkl
import shelve

import panel as pn

def phy_plot_vsTime(data_string, data_string_mean, plot_info, plot_type, resample_unit, string, run, period, run_dict, channel_map, abs_unit):
    # change column names to detector names
    data_string.columns           = ["{}_val".format(channel_map[ch]["name"]) for ch in data_string.columns]
    
    # create plot colours
    len_colours = len(data_string.columns)
    colours = cc.palette['glasbey_category10'][:len_colours]

    
    # add mean values for hover feature
    data_string_mean.columns      = [channel_map[ch]["name"] for ch in data_string_mean.columns]
    for col in data_string_mean.columns:
        data_string[col] = data_string_mean[col][0]
    
    p = figure(width=1000, height=600, x_axis_type='datetime', tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. {plot_type} | {plot_info.loc['label'][0]} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'datetime', '$snap_y': 'printf', "@$name": 'printf'}
    p.hover.tooltips = [( 'Time',   '$x{%F %H:%M:%S}'),
                        (f"{plot_info.loc['label'][0]} ({plot_info.loc['unit'][0]})", '$snap_y{%0.2f}'),
                        (f"Mean {plot_info.loc['label'][0]} ({abs_unit})", '@$name{0.2f}'),
                        ("Detector", "$name")
                        ]

    p.hover.mode = 'vline'


    # plot data
    hover_renderers = []
    if resample_unit == "0min":
        for i, det in enumerate(data_string_mean):
            if "mean" in det: continue
            l = p.line('datetime', f"{det}_val", source=data_string, color=colours[i], legend_label=det, name=det, line_width=2.5)
            hover_renderers.append(l)
    else:
        data_string_resampled = data_string.resample(resample_unit, origin="start").mean()
        
        for i, det in enumerate(data_string_mean):
            if "mean" in det: continue
            l = p.line('datetime', f"{det}_val", source=data_string_resampled, color=colours[i], legend_label=det, name=det, line_width=2.5)
            p.line('datetime', f"{det}_val", source=data_string, color=colours[i], legend_label=det, name=det, line_width=2.5, alpha=0.2)
            hover_renderers.append(l)
            
    
    # draw horizontal line at thresholds from plot info if available
#     if plot_info.loc["lower_lim_var"][0] != 'None' and plot_info.loc["unit"][0] == "%":
#         lower_lim_var = Slope(gradient=0, y_intercept=float(plot_info.loc["lower_lim_var"][0]),
#                 line_color='black', line_dash='dashed', line_width=4)
#         upper_lim_var = Slope(gradient=0, y_intercept=float(plot_info.loc["upper_lim_var"][0]),
#                 line_color='black', line_dash='dashed', line_width=4)

#         p.add_layout(lower_lim_var)
#         p.add_layout(upper_lim_var)
    
    # legend setups etc...
    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = f"Time (UTC), starting: {data_string.index[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = f"{plot_info.loc['label'][0]} [{plot_info.loc['unit'][0]}]"
    p.yaxis.axis_label_text_font_size = "20px"
    
    p.hover.renderers = hover_renderers
    
    if plot_info.loc["unit"][0] == "%":
        if plot_info.loc["label"][0] == 'Noise':
            p.y_range = Range1d(-50, 200)
        # if plot_info.loc["label"][0] == 'FPGA Baseline':
        #     p.y_range = Range1d(-6, 6)
        else:
            p.y_range = Range1d(-6, 6)
    else:
        if plot_info.loc["label"][0] == 'Noise':
            p.y_range = Range1d(0, 50)
    
    return p



def phy_plot_histogram(data_string, plot_info, plot_type, resample_unit, string, run, period, run_dict, channels, channel_map):
    p = figure(width=1000, height=600, x_axis_type='datetime', tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. {plot_type} | {plot_info.loc['label'][0]} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'printf', '$snap_y': 'printf'}
    p.hover.tooltips = [(f"{plot_info.loc['label'][0]} ({plot_info.loc['unit'][0]}", '$x{%0.2f}'),
                        ( 'Counts',   '$snap_y'),
                        ("Detector", "$name")
                        ]

    p.hover.mode = 'vline'

    len_colours = len(data_string.columns)
    if len_colours > 19:
        colours = Turbo256[len_colours]
    else:
        colours = Category20[len_colours]
    
    for position, data_channel in data_string.groupby("position"):
        # generate histogram
        # needed for cuspEmax because with geant outliers not possible to view normal histo
        hrange = {"keV": [0, 2500]}
        # take full range if not specified
        x_min = (hrange[plot_info["unit"]][0] if plot_info["unit"] in hrange else data_channel[plot_info["parameter"]].min())
        x_max = (hrange[plot_info["unit"]][1] if plot_info["unit"] in hrange else data_channel[plot_info["parameter"]].max())

        # --- bin width
        # bwidth = {"keV": 2.5}  # what to do with binning???
        # bin_width = bwidth[plot_info["unit"]] if plot_info["unit"] in bwidth else None
        # no_bins = int((x_max - x_min) / bin_width) if bin_width else 50
        # counts_ch, bins_ch = np.histogram(data_channel[plot_info["parameter"]], bins=no_bins, range=(x_min, x_max))
        # bins_ch = (bins_ch[:-1] + bins_ch[1:]) / 2
        
        # --- bin width
        bwidth = {"keV": 2.5}
        bin_width = bwidth[plot_info["unit"]] if plot_info["unit"] in bwidth else 1

        # Compute number of bins
        if bin_width:
            bin_no = bin_width / 5 if "AoE" not in plot_info["parameter"] else bin_width / 50
            bin_no = bin_no / 2 if "Corrected" in plot_info["parameter"] else bin_no
            bin_no = bin_width if "AoE" not in plot_info["parameter"] else bin_no 
            
            bin_edges = (
                np.arange(x_min, x_max + bin_width, bin_no)
                if plot_info["unit_label"] == "%"
                else np.arange(x_min, x_max + bin_width, bin_no)
            )
        else:
            bin_edges = 50 
        counts_ch, bins_ch = np.histogram(data_channel[plot_info["parameter"]], bins=bin_edges, range=(x_min, x_max))
        bins_ch = (bins_ch[:-1] + bins_ch[1:]) / 2
        # create plot histo
        histo_df = pd.DataFrame({"counts": counts_ch, "bins": bins_ch, "position": position, "cc4_id": data_channel['cc4_id'].unique()[0]})
        # plot    
        p.line("bins", "counts", source=histo_df, color=colours[position-1], 
            legend_label=f"{data_channel['name'].unique()[0]}", name=f"ch {data_channel['channel'].unique()[0]}",
            line_width=2)
    

            

    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = f"{plot_info['label']} [{plot_info['unit_label']}]"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Counts"
    p.yaxis.axis_label_text_font_size = "20px"
    
    return p