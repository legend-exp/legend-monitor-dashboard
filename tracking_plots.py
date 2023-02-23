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
import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from util import *

def plot_energy(path, run_dict, det, plot, colour):
    
    cals= []
    times = []
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["a"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["cuspEmax_ctc_cal"]["parameters"]["b"])
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

def plot_energy_res(path, run_dict, det, plot, colour, at="Qbb"):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    reses= []
    times = []
    for run in run_dict:
        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit_results.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            reses.append(hit_pars_dict[f"ch{channel:03}"]["ecal"]["cuspEmax_ctc_cal"][f"{at}_fwhm"])
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

def plot_energy_res_Qbb(path, run_dict, det, plot, colour):
    return plot_energy_res(path,run_dict, det, plot,colour, at="Qbb")

def plot_energy_res_2614(path, run_dict, det, plot, colour):
    return plot_energy_res(path,run_dict, det, plot,colour, at="2.6")

def plot_aoe_mean(path, run_dict, det, plot, colour):

    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    cals= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Corrected"]["parameters"]["a"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Corrected"]["parameters"]["b"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    cals=np.array(cals)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_aoe_sig(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    cals= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        hit_pars_file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict[run]["period"]}/{run}')
        hit_pars_path = os.path.join(hit_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.json')

        with open(hit_pars_path,"r")as r:
            hit_pars_dict = json.load(r)
        try:
            cals.append(hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Classifier"]["parameters"]["c"]*20000 +\
                       hit_pars_dict[f"ch{channel:03}"]["operations"]["AoE_Classifier"]["parameters"]["d"])
            times.append(run_dict[run]["timestamp"])
        except:
            pass
    cals=np.array(cals)
    plot.step([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
           legend_label=det, mode="after", line_width=2, line_color = colour)

    plot.circle([(datetime.strptime(value, '%Y%m%dT%H%M%SZ')) for value in times],
                100*(cals-cals[0])/cals[0],
            legend_label=det, fill_color="white", size=8, color = colour)

    return plot

def plot_tau(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{run_dict[run]["period"]}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(float(dsp_pars_path[f"ch{channel:03}"]["pz"]["tau"][:-3]))
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

def plot_ctc_const(path, run_dict, det, plot, colour):

    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    configs = LegendMetadata(path = prod_config["paths"]["chan_map"])
    
    values= []
    times = []
    for run in run_dict:

        chmap = configs.channelmaps.on(run_dict[run]["timestamp"])
        channel = chmap[det].daq.fcid

        dsp_pars_file_path = os.path.join(prod_config["paths"]["par_dsp"],f'cal/{run_dict[run]["period"]}/{run}')
        dsp_pars_path = os.path.join(dsp_pars_file_path, 
                        f'{run_dict[run]["experiment"]}-{run_dict[run]["period"]}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.json')

        with open(dsp_pars_path,"r")as r:
            dsp_pars_path = json.load(r)
        try:
            values.append(dsp_pars_path[f"ch{channel:03}"]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"])
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

def plot_tracking(run_dict, path, plot_func, string, key="String"):    

    strings_dict, soft_dict, chmap = sorter(path, run_dict[list(run_dict)[0]]["timestamp"], key=key)
    string_dets={}
    for stri in strings_dict:
        dets =[]
        for chan in strings_dict[stri]:
            dets.append(chmap[chan]["name"])
        string_dets[stri] = dets
        
    p = figure(width=700, height=600, x_axis_type="datetime")
    p.title.text = f"String No: {string}"
    p.title.align = "center"
    p.title.text_font_size = "15px"

    colours = Category10[10]
    
    for i, det in enumerate(string_dets[string]):
        try:
            plot_func(path, run_dict, det, p, colours[i])
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


    p.add_layout(Title(text="Time", align="center"), "below")
    
    if plot_func == plot_energy:
        p.add_layout(Title(text="% Shift in keV", align="center"), "left")
    elif plot_func == plot_energy_res_Qbb:
        p.add_layout(Title(text="FWHM at Qbb", align="center"), "left")
    elif plot_func == plot_energy_res_2614:
        p.add_layout(Title(text="FWHM of 2.6MeV peak", align="center"), "left")
    elif plot_func == plot_aoe_mean:
        p.add_layout(Title(text="% Shift of A/E mean", align="center"), "left")
    elif plot_func == plot_aoe_sig:
        p.add_layout(Title(text="% Shift of A/E sigma", align="center"), "left")
    elif plot_func == plot_tau:
        p.add_layout(Title(text="% Shift PZ const", align="center"), "left")
    elif plot_func == plot_ctc_const:
        p.add_layout(Title(text="Shift CT constant", align="center"), "left")
        
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return p