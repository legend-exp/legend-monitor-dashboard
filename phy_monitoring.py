from bokeh.models import Span, Label, Title, Range1d, HoverTool
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.plotting import figure, show
import shelve
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle as pkl
import shelve

import panel as pn

@pn.cache(max_items=10, policy='LFU', to_disk=True)
def _get_phy_dataframes(phy_path, phy_plots_vals, period, run):
    data_file = phy_path + f'/generated/plt/phy/{period}/{run}/l200-{period}-{run}-phy-geds'
    phy_data_df_dict, phy_data_resampled_df_dict, phy_plot_info_dict = {}, {}, {}
    print(data_file)
    if not os.path.exists(data_file +'.dat'):
        for phy_plots in phy_plots_vals:
            phy_data_df_dict[phy_plots]   = pd.DataFrame()
            phy_plot_info_dict[phy_plots] = {}
    else:
        with shelve.open(data_file, 'r', protocol=pkl.HIGHEST_PROTOCOL) as file:
            for phy_plots in phy_plots_vals:
                # take df with parameter you want
                phy_data_df   = file['monitoring']['pulser'][phy_plots]['df_geds']

                # take a random plot_info, it should be enough to save only one per time
                phy_plot_info = file['monitoring']['pulser'][phy_plots]['plot_info']
                
                # preselect data
                phy_data_df = phy_data_df[['datetime', 'channel', phy_plot_info["parameter"], f"{phy_plot_info['parameter'].split('_var')[0]}", "{}_mean".format(phy_plot_info['parameter'].split("_var")[0])]].reset_index().set_index(['channel', 'datetime'])
                
                phy_data_resampled_df = phy_data_df.reset_index().set_index("datetime").groupby('channel').resample(phy_plot_info["time_window"], origin="start").mean(numeric_only=True).dropna().drop(columns=['channel'])
                
                # save to dict
                phy_data_df_dict[phy_plots]           = phy_data_df
                phy_data_resampled_df_dict[phy_plots] = phy_data_resampled_df
                phy_plot_info_dict[phy_plots]         = phy_plot_info
                
                
    return phy_data_df_dict, phy_data_resampled_df_dict, phy_plot_info_dict

def phy_plot_vsTime(data_string, data_string_resampled, plot_info, string, run, period, run_dict, channels, channel_map):
    p = figure(width=1000, height=600, x_axis_type='datetime', tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Phy. | {plot_info['label']} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'datetime', '$snap_y': 'printf', '@{}_mean'.format(plot_info['parameter'].split('_var')[0]): 'printf'}
    p.hover.tooltips = [( 'Time',   '$x{%F %H:%M:%S}'),
                        (f"{plot_info['label']} ({plot_info['unit_label']})", '$snap_y{%0.2f}'),
                        (f"Mean {plot_info['label']} ({plot_info['unit_label']})", '@{}_mean{{%d}}'.format(plot_info['parameter'].split('_var')[0])),
                        ("Detector", "$name")
                        ]

    p.hover.mode = 'vline'
    
    len_colours = len(channels)
    if len_colours > 19:
        colours = Turbo256[len_colours]
    else:
        colours = Category20[len_colours]

    window = plot_info["time_window"]

    if plot_info["resampled"] == "only":
        for i, ch in enumerate(channels):
            p.line("datetime", plot_info["parameter"], source=data_string_resampled.loc[ch], color=colours[i], legend_label=channel_map[ch]["name"], name=channel_map[ch]["name"] + f" (resampled {window})")
    
    if plot_info["resampled"] == "yes":
        line_alpha = 0.1
        for i, ch in enumerate(channels):
            p.line("datetime", plot_info["parameter"], source=data_string.loc[ch], color=colours[i], legend_label=channel_map[ch]["name"], name=channel_map[ch]["name"], line_alpha=line_alpha)
            # the timestamps in the resampled table will start from the first timestamp, and go with sampling intervals
            # I want to shift them by half sampling window, so that the resampled value is plotted in the middle time window in which it was calculated
            p.line("datetime", plot_info["parameter"], source=data_string_resampled.loc[ch], color=colours[i], 
                legend_label=channel_map[ch]["name"], name=channel_map[ch]["name"] + f" (resampled {window})",
                line_width=2.5)

    if plot_info["resampled"] == "no":
        for i, ch in enumerate(channels):
            p.line("datetime", plot_info["parameter"], source=data_string.loc[ch], color=colours[i], legend_label=channel_map[ch]["name"], name=channel_map[ch]["name"])

    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = f"Time (UTC), starting: {data_string.reset_index()['datetime'][0].strftime('%d/%m/%Y %H:%M:%S')}"
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