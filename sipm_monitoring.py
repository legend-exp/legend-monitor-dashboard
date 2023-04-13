from bokeh.models import Span, Label, Title, Range1d, HoverTool
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.plotting import figure, show
import shelve
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import panel as pn

import legend_data_monitor as ldm

def sipm_plot_vsTime(data_barrel, barrel, resample_unit):
    p = figure(width=1000, height=600, x_axis_type='datetime', tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = barrel + " (Resampled: " + resample_unit + ")"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.formatters = {'$x': 'datetime', '$y': 'printf'}
    p.hover.tooltips = [( 'Time',   '$x{%F %H:%M:%S}'),
                        ( 'Light Intensity Rate (PE/s)',  '$y' ), 
                        ( 'Channel', '$name')]

    p.hover.mode = 'vline'

    len_colours = len(data_barrel.columns)
    if len_colours > 19:
        colours = Turbo256[len_colours]
    else:
        colours = Category20[len_colours]

    data_barrel_resampled = data_barrel.resample(resample_unit, origin="start").mean()
    for i, col in enumerate(data_barrel_resampled):
        p.line('time', col, source=data_barrel_resampled, color=colours[i], line_width=2.5, legend_label=col, name=col)

    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = f"Time (UTC), starting: {data_barrel_resampled.index[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Light Intensity Rate (PE/s)"
    p.yaxis.axis_label_text_font_size = "20px"
    
    return p



def sipm_plot_histogram(data_barrel, barrel, resample_unit):
    p = figure(width=1000, height=600, y_axis_type="log", x_range = (0, 3), tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = barrel
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.tooltips = [( 'Light Intensity Rate (PE/s)',   '$x'),
                        ( 'Counts',  '$y' ), 
                        ( 'Channel', '$name')]

    p.hover.mode = 'vline'

    len_colours = len(data_barrel.columns)
    if len_colours > 19:
        colours = Turbo256[len_colours]
    else:
        colours = Category20[len_colours]

    for i, col in enumerate(data_barrel):
        data_channel = data_barrel[col]
        counts_ch, bins_ch = np.histogram(data_channel, bins=300, range=(data_channel.min(), 3))
        bins_ch = (bins_ch[:-1] + bins_ch[1:]) / 2
        p.line(bins_ch, counts_ch, color=colours[i], line_width=2.5, legend_label=col, name=col)

    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = "Light Intensity Rate (PE/s)"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Counts"
    p.yaxis.axis_label_text_font_size = "20px"
    
    return p