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

from bokeh.models import Span, Label, Title, Range1d
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure, show

import colorcet as cc

import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from src.util import *

def plot_energy(path, run_dict, det, plot, colour, period):
    
    cals= []
    times = []
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.rawid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{period}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:07}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["a"]*20000 +\
                       hit_pars_dict[f"ch{channel:07}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["b"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    if len(cals)>0:
        cals =np.array(cals)
        plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    100*(cals-cals[0])/cals[0],
               legend_label=det, mode="after", line_width=2, line_color = colour)

        plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                    100*(cals-cals[0])/cals[0],
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
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit_results.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            reses.append(hit_pars_dict[f"ch{channel:07}"]["ecal"]["cuspEmax_ctc_cal"][f"{at}_fwhm"])
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
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit_results.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            means.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["mean"][0])
            mean_errs.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["mean_errs"][0])
            reses.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["res"][0])
            res_errs.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["res_errs"][0])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    means=np.array(means)
    reses=np.array(reses)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (means-means[0])/reses,
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(means-means[0])/reses,
            legend_label=det, fill_color="white", size=8, color = colour)

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
                        f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit_results.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            means.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["mean"][0])
            mean_errs.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["mean_errs"][0])
            reses.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["res"][0])
            res_errs.append(hit_pars_dict[f"ch{channel:07}"]["aoe"]["1000-1300keV"]["res_errs"][0])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    means=np.array(means)
    reses=np.array(reses)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(reses-reses[0])/reses[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(reses-reses[0])/reses[0],
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
                (values-values[0]),
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                (values-values[0]),
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
        p.yaxis.axis_label ="% Shift of keV conversion of 20kADC"
    elif plot_func == plot_energy_res_Qbb:
        p.yaxis.axis_label = "FWHM at Qbb"
    elif plot_func == plot_energy_res_2614:
        p.yaxis.axis_label = "FWHM of 2.6MeV peak"
    elif plot_func == plot_aoe_mean:
        p.yaxis.axis_label = "% Shift of A/E mean"
    elif plot_func == plot_aoe_sig:
        p.yaxis.axis_label = "% Shift of A/E sigma"
    elif plot_func == plot_tau:
        p.yaxis.axis_label = "% Shift PZ const"
    elif plot_func == plot_ctc_const:
        p.yaxis.axis_label = "Shift CT constant"
        
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return pn.pane.Bokeh(p, sizing_mode="stretch_width")
