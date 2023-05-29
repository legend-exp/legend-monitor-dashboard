from bokeh.models import Span, Label, Title, Range1d, HoverTool
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.plotting import figure, show
import shelve
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import panel as pn


def phy_plot_vsTime(data_string, plot_info, string, run, period, run_dict):
    p = figure(width=1000, height=600, x_axis_type='datetime', tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. | {plot_info['label']} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'datetime', '$y': 'printf', '@{}_mean'.format(plot_info['parameter'].split('_var')[0]): 'printf'}
    p.hover.tooltips = [( 'Time',   '$x{%F %H:%M:%S}'),
                        ( plot_info['label'],  '$y' ), 
                        ( 'Channel', '$name'),
                        ('Position', '@position'),
                        ('CC4', '@cc4_id'),
                        (f"{plot_info['label']} ({plot_info['unit_label']})", '$y'),
                        (f"Mean {plot_info['label']} ({plot_info['unit_label']})", '@{}_mean'.format(plot_info['parameter'].split('_var')[0]))]
    
    p.hover.mode = 'vline'
    
    len_colours = data_string['position'].max()
    if len_colours > 19:
        colours = Turbo256[len_colours]
    else:
        colours = Category20[len_colours]
    
    if plot_info["resampled"] == "only":
        for position, data_channel in data_string.groupby("position"):
            # resample in given time window, as start pick the first timestamp in table
            data_channel_resampled = data_channel.set_index("datetime").resample(plot_info["time_window"], origin="start").mean(numeric_only=True).dropna()
            # convert index to int
            data_channel_resampled = data_channel_resampled.astype({"index": int})
            # get back CC4 and position information for hover tool
            for info_key in ['cc4_id', 'position', 'name', 'index']:
                data_channel_resampled[info_key] = data_channel.reset_index()[info_key][0]
            # reset index
            data_channel_resampled = data_channel_resampled.reset_index()
            # the timestamps in the resampled table will start from the first timestamp, and go with sampling intervals
            # I want to shift them by half sampling window, so that the resampled value is plotted in the middle time window in which it was calculated
            data_channel_resampled["datetime"] = (data_channel_resampled["datetime"] + pd.Timedelta(plot_info["time_window"]) / 2)
            # plot    
            p.line("datetime", plot_info["parameter"], source=data_channel_resampled, color=colours[position-1], 
                legend_label=f"{data_channel['name'].unique()[0]}", name=f"ch {data_channel['channel'].unique()[0]}",
                line_width=2.5)
            # break
    
    if plot_info["resampled"] == "yes":
        line_alpha = 0.2
        window = plot_info["time_window"]
        for position, data_channel in data_string.groupby("position"):
            p.line("datetime", plot_info["parameter"], source=data_channel, color=colours[position-1], 
                legend_label=f"{data_channel['name'].unique()[0]}", name=f"ch {data_channel['channel'].unique()[0]}",
                line_alpha=line_alpha)
            # resample in given time window, as start pick the first timestamp in table
            data_channel_resampled = data_channel.set_index("datetime").resample(plot_info["time_window"], origin="start").mean(numeric_only=True).dropna()
            # convert index to int
            data_channel_resampled = data_channel_resampled.astype({"index": int})
            # get back CC4 and position information for hover tool
            for info_key in ['cc4_id', 'position', 'name', 'index']:
                data_channel_resampled[info_key] = data_channel.reset_index()[info_key][0]
            # reset index
            data_channel_resampled = data_channel_resampled.reset_index()
            # the timestamps in the resampled table will start from the first timestamp, and go with sampling intervals
            # I want to shift them by half sampling window, so that the resampled value is plotted in the middle time window in which it was calculated
            data_channel_resampled["datetime"] = (data_channel_resampled["datetime"] + pd.Timedelta(plot_info["time_window"]) / 2)
            # plot    
            p.line("datetime", plot_info["parameter"], source=data_channel_resampled, color=colours[position-1], 
                legend_label=f"{data_channel['name'].unique()[0]}", name=f"ch {data_channel['channel'].unique()[0]}"+ f" (resampled {window})",
                line_width=2.5)
            
    if plot_info["resampled"] == "no":
        for position, data_channel in data_string.groupby("position"):
            p.line("datetime", plot_info["parameter"], source=data_channel, color=colours[position-1], 
                legend_label=f"{data_channel['name'].unique()[0]}", name=f"ch {data_channel['channel'].unique()[0]}")
            

    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = f"Time (UTC), starting: {data_channel.reset_index()['datetime'][0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = f"{plot_info['label']} [{plot_info['unit_label']}]"
    p.yaxis.axis_label_text_font_size = "20px"
    
    return p



def phy_plot_histogram(data_string, plot_info, string, run, period, run_dict):
    p = figure(width=1000, height=600, y_axis_type="log", tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. | {plot_info['label']} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( f"{plot_info['label']} [{plot_info['unit_label']}]", '$x'),
                        ( 'Counts',  '$y'), 
                        ( 'Channel', '$name'),
                        ('Position', '@position'),
                        ('CC4', '@cc4_id')]
    p.hover.mode = 'vline'
    
    len_colours = data_string['position'].max()
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