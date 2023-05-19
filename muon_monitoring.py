import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource

import colorsys
from bokeh.layouts import gridplot

import datetime as dtt
from bokeh.models import DatetimeTickFormatter
from bokeh.transform import log_cmap
from bokeh.models import LogColorMapper
from bokeh.models.tickers import LogTicker
from bokeh.models.axes import LogAxis
from bokeh.models.formatters import LogTickFormatter
from bokeh.models import LogScale
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem
from bokeh.models import Title
import matplotlib as mpl


PMT_ID=['101 Pillbox','704 Pillbox','102 Pillbox','705 Pillbox','708 Pillbox','104 Pillbox','709 Pillbox','105 Pillbox','710 Pillbox','707 Pillbox','201 Floor','202 Floor','203 Floor','706 Floor','206 Floor','208 Floor','701 Floor','703 Floor','301 Floor','302 Floor','303 Floor','304 Floor','305 Floor','306 Floor','307 Floor','308 Floor','309 Floor','310 Floor','311 Floor','312 Floor','401 Wall','402 Wall','403 Wall','404 Wall','409 Wall','410 Wall','501 Wall','502 Wall','503 Wall','504 Wall','507 Wall','508 Wall','509 Wall','510 Wall','602 Wall','603 Wall','605 Wall','606 Wall','607 Wall','608 Wall','609 Wall','610 Wall','702 Wall', 'none']
num_colors = 54
chan_num=55

colors = []
for i in range(num_colors):
    hue = i/num_colors
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    color = (int(r*255), int(g*255), int(b*255))
    colors.append(color)


def muon_plot_spectra(arrays_dict, run, period, run_dict, plot_type):
    x_data = arrays_dict['histo_centers']
    # Extract the Y data column
    y_data = arrays_dict['histo_values']
    
    plots = []
    
    for i in range(9):

        for enum, x in enumerate(range(round(((chan_num - 1) / 9),) * (i + 0), round(((chan_num - 1) / 9),) * (i + 1))):
            p = figure(plot_width=300, plot_height=300, x_axis_label='Pulse height [LSBs]', 
               y_axis_label='counts', y_axis_type="log", x_range=(0, 100), y_range=(1e-0, 2 * np.amax(y_data[i])),
               title="Channel " + str(x) + " (PMT " + str(PMT_ID[x]) + ")")
            p.title.text_font_size = '9pt'
            p.xaxis.axis_label_text_font_size = "10pt"
            p.title.text = f"{run_dict['experiment']}-{period}-{run} | Mu. | Ch. {x} - PMT {PMT_ID[x]}"
            p.title.align = "center"

            # create a ColumnDataSource for each line
            source = ColumnDataSource(data=dict(x=x_data[x], y=y_data[x]))
            step = p.step('x', 'y', source=source, line_width=2, line_join="round", line_cap="round", line_dash="solid", line_color=colors[x], mode='center')
            plots.append(p)

    # create a grid plot with 9 rows and 6 columns
    grid = gridplot([[plots[j*6 + i] for i in range(6)] for j in range(9)])

    return grid


def muon_plot_spp(arrays_dict, run, period, run_dict, plot_type):
    x_data = arrays_dict['mean_LSB']
    # Extract the Y data column
    y_data = arrays_dict['mean_sigma']
    dots=[]
    # Create a figure object
    p = figure(title="SPP gaussian", width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.tooltips = [( 'Pulse Height (LSB)', '$x'),
                        ('σ of SPP', '$y'),
                        ('Channel'  , '$name')]
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | SPP Gaussian"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label_text_font_size = "20px"
    
    # Set the x and y axis labels and limits
    p.xaxis.axis_label = 'Pulse height (LSB)'
    p.yaxis.axis_label = 'σ of SPP'
    
    p.x_range.start = 0
    p.x_range.end = 100
    p.y_range.start = 0
    p.y_range.end = 78
    
    legend_list = []
    for channel in range(53):
        dot = p.scatter(x_data[channel], y_data[channel], color=colors[channel], marker='circle', size=10, name=PMT_ID[channel])
        dots.append(dot)
        legend_list.append((PMT_ID[channel], [dot]))
        
        
    legend1 = Legend(items=legend_list[:18], orientation="vertical", location=(10, 15))
    legend2 = Legend(items=legend_list[18:36], orientation="vertical", location=(10, 15))
    legend3 = Legend(items=legend_list[36:], orientation="vertical", location=(10, 38))
    p.add_layout(legend1, 'right')
    p.add_layout(legend2, 'right')
    p.add_layout(legend3, 'right')
    legend1.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend2.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend3.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    
    return p


def muon_plot_calshift(x_data, y_data, run, period, run_dict, plot_type):
    p = figure(title='Calibration mean shift', x_axis_label='Date', y_axis_label='Mean Shift [LSB]', width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( 'Time',   '$x{%F %H:%M:%S}'),
                        ( 'Mean Shift (LSB)',  '$y' ), 
                        ( 'Channel', '$name')]
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Calibration Mean Shift"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.hover.mode = 'vline'
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.axis_label = f"Time (UTC), starting: {x_data[0][0].strftime('%d/%m/%Y %H:%M:%S')}"
    
    p.xaxis.formatter = DatetimeTickFormatter(
        hours=["%Y-%m-%d %H:%M"],
        days=["%Y-%m-%d"],
        months=["%Y-%m"],
        years=["%Y"]
    )
    
    lines=[]
    legend_list = []
    for channel in range(53):
        dates = [dt.strftime("%Y %m %d %H:%M") for dt in x_data[channel]]
        p.scatter(x_data[channel], y_data[channel], color=colors[channel], name=PMT_ID[channel])
        line=p.line(x_data[channel], y_data[channel], color=colors[channel], name=PMT_ID[channel])
        lines.append(line)
        legend_list.append((PMT_ID[channel], [line]))
        
    legend1 = Legend(items=legend_list[:18], orientation="vertical", location=(10, 15))
    legend2 = Legend(items=legend_list[18:36], orientation="vertical", location=(10, 15))
    legend3 = Legend(items=legend_list[36:], orientation="vertical", location=(10, 38))
    p.add_layout(legend1, 'right')
    p.add_layout(legend2, 'right')
    p.add_layout(legend3, 'right')
    legend1.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend2.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend3.click_policy = "hide" # hide the corresponding line when the legend item is clicked

    return p

# monitoring plots

def muon_plot_intlight(arrays_dict, period, run, run_dict):
    
    x_data = arrays_dict['multiplicity']
    y_data = arrays_dict['int_light']
    fig, ax = plt.subplots()
    ax.hist2d(x_data, y_data, bins=(54, 1000), range=((0,55),(0.1, 1000)), norm=mpl.colors.LogNorm())
    ax.set_yscale("log")
    ax.set_xlabel("PMT Multiplicity",size=15)
    ax.set_ylabel("Integral Light p.e.",size=15)
    ax.set_title(f"{run_dict['experiment']}-{period}-{run} | Muon | Integral Light p.e. vs. Multiplicity")
    
    return fig


def muon_plot_totalRates_hourly(arrays_dict, period, run, run_dict):
    x_data = arrays_dict['duration']
    y_data = arrays_dict['red_rates']
    

    p = figure(x_range=(0, max(x_data)/3600), width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.tooltips = [( 'Time (H)', '$x'),
                        ('Rate (Hz)', '$y'),
                        ('Channel'  , '$name')]
    p.hover.mode = 'vline'
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Red. Hourly Rates for all PMTs"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label = "Time (H)"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Rate (Hz)"
    p.yaxis.axis_label_text_font_size = "20px"
    steps=[]
    legend_list = []
    for chan in range(53):
        #print(np.array(x_data)/3600)
        # step=p.step(np.array(x_data)/3600, np.swapaxes(y_data, 0, 1)[chan], line_color=colors[chan], mode="center", name=PMT_ID[chan])
        step=p.line(np.array(x_data)/3600, np.swapaxes(y_data, 0, 1)[chan], line_color=colors[chan], name=PMT_ID[chan])
        steps.append(step)
        legend_list.append((PMT_ID[chan], [step]))
        
    legend1 = Legend(items=legend_list[:18], orientation="vertical", location=(10, 15))
    legend2 = Legend(items=legend_list[18:36], orientation="vertical", location=(10, 15))
    legend3 = Legend(items=legend_list[36:], orientation="vertical", location=(10, 38))
    p.add_layout(legend1, 'right')
    p.add_layout(legend2, 'right')
    p.add_layout(legend3, 'right')
    legend1.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend2.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend3.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    
    return p

def muon_plot_totalRates_daily(arrays_dict, period, run, run_dict):
    x_data = arrays_dict['times']
    y_data = arrays_dict['red_rates']
    

    p = figure(x_axis_type='datetime', width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( 'Time', '$x'),
                        ('Rate (Hz)', '$snap_y'),
                        ('Channel'  , '$name')]
    p.hover.mode = 'vline'
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Red. Daily Rates for all PMTs"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label = f"Time (UTC), starting: {x_data[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Rate (Hz)"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y/%m/%d')
    
    legend_list = []

    for chan in range(53):
        daily_rates = []
        daily_mean_rates = []
        sum_rate = 0
        count = 0
        current_day = x_data[0].date()

        for i in range(len(x_data)):
            if x_data[i].date() == current_day:
                sum_rate += np.swapaxes(y_data,0,1)[chan][i]
                count += 1
            else:
                daily_rates.append(current_day)
                daily_mean_rates.append(sum_rate/count)
                current_day = x_data[i].date()
                sum_rate = np.swapaxes(y_data,0,1)[chan][i]
                count = 1

        daily_rates.append(current_day)
        daily_mean_rates.append(sum_rate/count)
        daily_rates = [dtt.datetime.combine(date, dtt.datetime.min.time()) for date in daily_rates]

        # Add a glyph renderer for the line
        # step = p.step(daily_rates, daily_mean_rates, line_color=colors[chan], mode="center", name=PMT_ID[chan])
        step = p.line(daily_rates, daily_mean_rates, line_color=colors[chan], name=PMT_ID[chan])
        legend_list.append((PMT_ID[chan], [step]))

    legend1 = Legend(items=legend_list[:18], orientation="vertical", location=(10, 15))
    legend2 = Legend(items=legend_list[18:36], orientation="vertical", location=(10, 15))
    legend3 = Legend(items=legend_list[36:], orientation="vertical", location=(10, 38))
    p.add_layout(legend1, 'right')
    p.add_layout(legend2, 'right')
    p.add_layout(legend3, 'right')
    legend1.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend2.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    legend3.click_policy = "hide" # hide the corresponding line when the legend item is clicked

    return p

def muon_plot_ratesPillBox(arrays_dict, period, run, run_dict):
    
    x_data = arrays_dict['times']
    y_data = arrays_dict['red_rates']

    p = figure(x_axis_type='datetime', width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( 'Time', '$x'),
                        ('Rate (Hz)', '$y'),
                        ('Channel'  , '$name')]
    p.hover.mode = 'vline'
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Red. Daily Rates over Time for Pillbox PMTs"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label = f"Time (UTC), starting: {x_data[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Rate (Hz)"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y/%m/%d')

    renderers = {}
    for chan in range(0,10):
        daily_rates = []
        daily_mean_rates = []
        sum_rate = 0
        count = 0
        current_day = x_data[0].date()

        for i in range(len(x_data)):
            if x_data[i].date() == current_day:
                sum_rate += np.swapaxes(y_data,0,1)[chan][i]
                count += 1
            else:
                daily_rates.append(current_day)
                daily_mean_rates.append(sum_rate/count)
                current_day = x_data[i].date()
                sum_rate = np.swapaxes(y_data,0,1)[chan][i]
                count = 1

        daily_rates.append(current_day)
        daily_mean_rates.append(sum_rate/count)
        daily_rates = [dtt.datetime.combine(date, dtt.datetime.min.time()) for date in daily_rates]

        # Add a glyph renderer for the line
        # step = p.step(daily_rates, daily_mean_rates, line_color=colors[chan], mode="center", name=PMT_ID[chan])
        step = p.line(daily_rates, daily_mean_rates, line_color=colors[chan], name=PMT_ID[chan])
        renderers[PMT_ID[chan]] = step

    # Create a legend with interactive checkboxes
    legend = Legend(items=[(label, [renderer]) for label, renderer in renderers.items()], location='top_left')
    legend.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    p.add_layout(legend, 'right')
    
    return p


def muon_plot_ratesFloor(arrays_dict, period, run, run_dict):
    
    x_data = arrays_dict['times']
    y_data = arrays_dict['red_rates']

    p = figure(x_axis_type='datetime', width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( 'Time', '$x'),
                        ('Rate (Hz)', '$y'),
                        ('Channel'  , '$name')]
    p.hover.mode = 'vline'
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Red. Daily Rates over Time for Floor PMTs"
    
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label = f"Time (UTC), starting: {x_data[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Rate (Hz)"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y/%m/%d')

    renderers = {}
    for chan in range(10,30):
        daily_rates = []
        daily_mean_rates = []
        sum_rate = 0
        count = 0
        current_day = x_data[0].date()

        for i in range(len(x_data)):
            if x_data[i].date() == current_day:
                sum_rate += np.swapaxes(y_data,0,1)[chan][i]
                count += 1
            else:
                daily_rates.append(current_day)
                daily_mean_rates.append(sum_rate/count)
                current_day = x_data[i].date()
                sum_rate = np.swapaxes(y_data,0,1)[chan][i]
                count = 1

        daily_rates.append(current_day)
        daily_mean_rates.append(sum_rate/count)
        daily_rates = [dtt.datetime.combine(date, dtt.datetime.min.time()) for date in daily_rates]

        # Add a glyph renderer for the line
        # step = p.step(daily_rates, daily_mean_rates, line_color=colors[chan], mode="center", name=PMT_ID[chan])
        step = p.line(daily_rates, daily_mean_rates, line_color=colors[chan], name=PMT_ID[chan])
        renderers[PMT_ID[chan]] = step

    # Create a legend with interactive checkboxes
    legend = Legend(items=[(label, [renderer]) for label, renderer in renderers.items()], location='top_left')
    legend.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    p.add_layout(legend, 'right')
    
    return p

def muon_plot_ratesWall(arrays_dict, period, run, run_dict):
    
    x_data = arrays_dict['times']
    y_data = arrays_dict['red_rates']

    p = figure(x_axis_type='datetime', width=1000, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.hover.formatters = {'$x': 'datetime'}
    p.hover.tooltips = [( 'Time', '$x'),
                        ('Rate (Hz)', '$y'),
                        ('Channel'  , '$name')]
    p.hover.mode = 'vline'
    
    p.title.text = f"{run_dict['experiment']}-{period}-{run} | Muon | Red. Daily Rates over Time for Wall PMTs"
    p.title.align = "center"
    p.title.text_font_size = "25px"
    p.xaxis.axis_label = f"Time (UTC), starting: {x_data[0].strftime('%d/%m/%Y %H:%M:%S')}"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "Rate (Hz)"
    p.yaxis.axis_label_text_font_size = "20px"
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y/%m/%d')

    renderers = {}
    for chan in range(30,53):
        daily_rates = []
        daily_mean_rates = []
        sum_rate = 0
        count = 0
        current_day = x_data[0].date()

        for i in range(len(x_data)):
            if x_data[i].date() == current_day:
                sum_rate += np.swapaxes(y_data,0,1)[chan][i]
                count += 1
            else:
                daily_rates.append(current_day)
                daily_mean_rates.append(sum_rate/count)
                current_day = x_data[i].date()
                sum_rate = np.swapaxes(y_data,0,1)[chan][i]
                count = 1

        daily_rates.append(current_day)
        daily_mean_rates.append(sum_rate/count)
        daily_rates = [dtt.datetime.combine(date, dtt.datetime.min.time()) for date in daily_rates]

        # Add a glyph renderer for the line
        # step = p.step(daily_rates, daily_mean_rates, line_color=colors[chan], mode="center", name=PMT_ID[chan])
        step = p.line(daily_rates, daily_mean_rates, line_color=colors[chan], name=PMT_ID[chan])
        renderers[PMT_ID[chan]] = step

    # Create a legend with interactive checkboxes
    legend = Legend(items=[(label, [renderer]) for label, renderer in renderers.items()], location='top_left')
    legend.click_policy = "hide" # hide the corresponding line when the legend item is clicked
    p.add_layout(legend, 'right')
    
    return p