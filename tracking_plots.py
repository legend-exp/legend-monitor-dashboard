import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import param
import pickle as pkl
import shelve
import bisect
import warnings

import numexpr as ne

from bokeh.models import Span, Label, Title, Range1d, Band, ColumnDataSource
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure, show

import colorcet as cc

import datetime as dtt
from  datetime import datetime

from scipy.optimize import minimize

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from src.util import *

def plot_energy(path, run_dict, det, plot, colour, period):
    
    cals= []
    times = []
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    qbb_adc = None
    for run in run_dict:
        
        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            hit_dict = hit_pars_dict[f"ch{channel:07}"]["pars"]["operations"]["cuspEmax_ctc_cal"]
            if qbb_adc is None:
                def find_qbb_adc(val):
                    return ne.evaluate(
                            "abs("+"cuspEmax_ctc*a +b"+"-2039)",
                            local_dict=dict({"cuspEmax_ctc":val}, **{"a":.1, "b":0})
                        ) 
                qbb_adc = minimize(find_qbb_adc, 20000)["x"][0]
            out_data = ne.evaluate(
                f"{hit_dict['expression']}",
                local_dict=dict({"cuspEmax_ctc":qbb_adc}, **hit_dict["parameters"])
            ) 
            cals.append(out_data)
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    if len(cals)>0:
        cals =np.array(cals)
        plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (cals-cals[0]),
               legend_label=det, mode="after", line_width=2, line_color = colour)

        plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (cals-cals[0]),
                legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_energy_res(path, run_dict, det, plot, colour, period, at="Qbb"):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    reses= []
    times = []
    for run in run_dict:
        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            if at == "Qbb":
                reses.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][f"Qbb_fwhm_in_keV"])
            elif at == "2.6":
                reses.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["ecal"]["cuspEmax_ctc_cal"]["pk_fits"]["2614.5"]["fwhm_in_keV"][0])
            times.append(run_dict[run]["timestamp"])
        except:
            pass

    if len(reses)>0:
        plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (reses),
               legend_label=det, mode="after", line_width=2, line_color = colour)

        plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    (reses),
                legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_energy_res_Qbb(path, run_dict, det, plot, colour, period):
    return plot_energy_res(path,run_dict, det, plot, colour, period, at="Qbb")

def plot_energy_res_2614(path, run_dict, det, plot, colour, period):
    return plot_energy_res(path,run_dict, det, plot, colour, period, at="2.6")

def plot_aoe_mean(path, run_dict, det, plot, colour, period):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    means= []
    mean_errs = []
    reses = []
    res_errs = []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            means.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["mean"])
            mean_errs.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["mean_err"])
            reses.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["res"])
            res_errs.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["res_err"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    means=np.array(means)
    reses=np.array(reses)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(means-means[0])/reses,
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(means-means[0])/reses,
            legend_label=det, fill_color="white", size=8, color = colour)
    
    df = pd.DataFrame({"x":[(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                          "lower": [-40 for value in times],
                          "upper":[40 for value in times] })

    source = ColumnDataSource(df)
    band = Band(base="x", lower="lower", upper="upper", source=source,
                fill_alpha=0.01, fill_color="yellow")
    plot.add_layout(band)
        
    df2 = pd.DataFrame({"x":[(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                          "lower": [-20 for value in times],
                          "upper":[20 for value in times] })

    source2 = ColumnDataSource(df2)
    band2 = Band(base="x", lower="lower", upper="upper", source=source2,
                fill_alpha=0.02, fill_color="green")
    plot.add_layout(band2)
    plot.y_range = Range1d(-100, 100)

    return plot

def plot_aoe_sig(path, run_dict, det, plot, colour, period):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    means= []
    mean_errs = []
    reses = []
    res_errs = []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            means.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["mean"])
            mean_errs.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["mean_err"])
            reses.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["res"])
            res_errs.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["1000-1300keV"]["0"]["res_err"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    means=np.array(means)
    reses=np.array(reses)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (reses),
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (reses),
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_aoe_cut(path, run_dict, det, plot, colour, period):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    cuts= []
    times=[]
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cuts.append(hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["aoe"]["low_cut"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    cuts=np.array(cuts)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (cuts),
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (cuts),
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_tau(path, run_dict, det, plot, colour, period):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{period}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(float(dsp_pars_path[f"ch{channel:07}"]["pz"]["tau"][:-3]))
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    values=np.array(values)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(values-values[0])/values[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(values-values[0])/values[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_ctc_const(path, run_dict, det, plot, colour, period):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{period}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(dsp_pars_path[f"ch{channel:07}"]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    values=np.array(values)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (values),
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (values),
            legend_label=det, fill_color="white", size=8, color = colour)


    return plot

def plot_tracking(run_dict, path, plot_func, string, period, plot_type, key="String"):    

    strings_dict, soft_dict, chmap = sorter(path, run_dict[list(run_dict)[0]]["timestamp"], key=key)
    string_dets={}
    for stri in strings_dict:
        dets =[]
        for chan in strings_dict[stri]:
            dets.append(chmap[chan]["name"])
        string_dets[stri] = dets
        
    p = figure(width=1000, height=400, x_axis_type="datetime", tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict[list(run_dict)[0]]['experiment']}-{period} | Cal. Tracking | {plot_type} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "15px"

    colours = cc.palette['glasbey_category10'][:100]
    
    for i, det in enumerate(string_dets[string]):
        try:
            plot_func(path, run_dict, det, p, colours[i], period)
        except:
            pass

    
    for run in run_dict:
        sp = Span(location=datetime.strptime(run_dict[run]["timestamp"], '%Y%m%dT%H%M%SZ'),
                  dimension='height',
                   line_color='black', line_width=1.5)
        p.add_layout(sp)

        label = Label(x=datetime.strptime(run_dict[run]["timestamp"], '%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 200), y=0, 
                     text=run)

        p.add_layout(label)


    p.xaxis.axis_label = "Time"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label_text_font_size = "20px"
    
    if plot_func == plot_energy:
        p.yaxis.axis_label ="Shift of Qbb in keV "
    elif plot_func == plot_energy_res_Qbb:
        p.yaxis.axis_label = "FWHM at Qbb"
    elif plot_func == plot_energy_res_2614:
        p.yaxis.axis_label = "FWHM of 2.6MeV peak"
    elif plot_func == plot_aoe_mean:
        p.yaxis.axis_label = "% Shift of A/E mean"
    elif plot_func == plot_aoe_sig:
        p.yaxis.axis_label = "Shift of A/E Resolution"
    elif plot_func == plot_aoe_sig:
        p.yaxis.axis_label = "Shift of A/E Low Cut"
    elif plot_func == plot_tau:
        p.yaxis.axis_label = "% Shift PZ const"
    elif plot_func == plot_ctc_const:
        p.yaxis.axis_label = "Shift CT constant"
        
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return p

def plot_energy_residuals_period(run_dict, path, period, key="String", download=False):
    strings, soft_dict, channel_map = sorter(path, run_dict[list(run_dict)[0]]["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict[list(run_dict)[0]]["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]

    
    peaks = [2614.5, 583.191, 2103.53]
    
    res = {}
    
    for stri in strings:
        res[stri]={str(peak):[] for peak in peaks}
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]
            res[detector] = {str(peak):[] for peak in peaks}
    
    for run in run_dict:
        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[detector].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
     
        for peak in peaks:
            for stri in strings:
                res[stri][str(peak)].append(np.nan)
                for channel in strings[stri]:
                    detector = channel_map[channel]["name"]
                    try:
                        hit_dict = hit_pars_dict[f"ch{channel:07}"]["pars"]["operations"]["cuspEmax_ctc_cal"]
                        res_dict = hit_pars_dict[f"ch{channel:07}"]["results"]["ecal"]["ecal"]["cuspEmax_ctc_cal"]
                        out_data = ne.evaluate(
                            f"{hit_dict['expression']}",
                            local_dict=dict({"cuspEmax_ctc":res_dict["pk_fits"][str(peak)]["parameters_in_ADC"]["mu"]}, **hit_dict["parameters"])
                        ) - peak
                        res[detector][str(peak)].append(out_data)
                    except:
                        res[detector][str(peak)].append(np.nan)
                
            
            
    p = figure(width=1400, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p.title.text = f"{run_dict[list(run_dict)[0]]['experiment']}-{period} | Cal. | Energy Residuals"
    p.title.align = "center"
    p.title.text_font_size = "25px"

    label_res = [r if 'String' not in r else "" for r in list(res)]

    df_plot = pd.DataFrame()
    df_plot["label_res"]  = label_res
    
    for peak in peaks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_plot, y_plot, y_min, y_max = (np.arange(1, len(list(res))+1, 1), 
                                [np.nanmean(res[det][str(peak)]) for det in res], 
                                [np.nanmin(res[det][str(peak)]) if len(np.array(res[det][str(peak)])[~np.isnan(res[det][str(peak)])])>0 else np.nan for det in res],
                               [np.nanmax(res[det][str(peak)]) if len(np.array(res[det][str(peak)])[~np.isnan(res[det][str(peak)])])>0 else np.nan for det in res])

        err_xs = []
        err_ys = []

        for x, y, yerr_low, yerr_hi in zip(x_plot, y_plot, y_min, y_max):
            err_xs.append((x, x))
            err_ys.append((np.nan_to_num(y - yerr_low), np.nan_to_num(y + yerr_hi)))


        df_plot[f"x_{int(peak)}"]          = np.nan_to_num(x_plot)
        df_plot[f"y_{int(peak)}"]          = np.nan_to_num(y_plot)
        df_plot[f"y_{int(peak)}_min"]      = np.nan_to_num(y_min)
        df_plot[f"y_{int(peak)}_max"]      = np.nan_to_num(y_max)
        df_plot[f"err_xs_{int(peak)}"]     = err_xs
        df_plot[f"err_ys_{int(peak)}"]     = err_ys

    if download:
        return df_plot, f"{run_dict[list(run_dict)[0]]['experiment']}-{period}-energy_residuals.csv"
        
    for peak, peak_color in zip(peaks, ["blue", "green", "red"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if peak == peaks[0]:
                hover_renderer = p.circle(x=f"x_{int(peak)}", y=f"y_{int(peak)}", 
                                          source=df_plot, color=peak_color, size=7, line_alpha=0,
                    legend_label = f'{peak} Average: {np.nanmean([np.nanmean(res[det][f"{peak}"]) for det in res]):.2f}keV', 
                    name = f'{peak} Average: {np.nanmean([np.nanmean(res[det][f"{peak}"]) for det in res]):.2f}keV'
                    )
            else:
                p.circle(x=f"x_{int(peak)}", y=f"y_{int(peak)}", source=df_plot, 
                         color=peak_color, size=7, line_alpha=0,
                    legend_label = f'{peak} Average: {np.nanmean([np.nanmean(res[det][f"{peak}"]) for det in res]):.2f}keV', 
                    name = f'{peak} Average: {np.nanmean([np.nanmean(res[det][f"{peak}"]) for det in res]):.2f}keV'
                    )
            band = Band(base=f"x_{int(peak)}", lower=f"y_{int(peak)}_min", 
                        upper=f"y_{int(peak)}_max", source=ColumnDataSource(df_plot),
                        fill_alpha=0.1, fill_color=peak_color)
            p.add_layout(band)


    p.legend.location = "bottom_right"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = "detector"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = 'peak residuals (keV)'
    p.title.text = f"{run_dict[list(run_dict)[0]]['experiment']}-{period} | Cal. | Energy residuals"
    p.yaxis.axis_label_text_font_size = "20px"

    p.xaxis.major_label_orientation = np.pi/2
    p.xaxis.ticker = np.arange(1, len(list(res))+1, 1)
    p.xaxis.major_label_overrides = {i: label_res[i-1] for i in range(1, len(label_res)+1, 1)}
    p.xaxis.major_label_text_font_style = "bold"

    for stri in strings:
        loc=np.where(np.array(list(res))==stri)[0][0]
        string_span = Span(location=loc+1, dimension='height',
                    line_color='black', line_width=3)
        string_span_label = Label(x=loc+1.5, y=1.1, text=stri, text_font_size='10pt', text_color='blue')
        p.add_layout(string_span_label)
        p.add_layout(string_span)
        
    p.hover.tooltips = [( 'Detector',   '@label_res'),
                        ( '2614',  f"av: @y_2614{{0.00}} min: @y_2614_min{{0.00}} max: @y_2614_max{{0.00}} keV"),
                        ( 'SEP',  f"av: @y_2103{{0.00}} min: @y_2103_min{{0.00}} max: @y_2103_max{{0.00}} keV"),
                        ( '583',  f"av: @y_583{{0.00}} min: @y_583_min{{0.00}} max: @y_583_max{{0.00}} keV")
                        ]
    p.hover.mode = 'vline'
    p.hover.renderers = [hover_renderer]
    
    return p