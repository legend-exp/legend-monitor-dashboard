import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import shelve
import bisect
import seaborn as sns
import matplotlib

from bokeh.models import Span, Label, Title, Range1d
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.plotting import figure, show
import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from src.util import *

def build_string_array(chan_map):
    dets = []
    strings = []
    positions = []
    for key,entry in chan_map.items():
        if entry.system == "geds":
            string = entry.location.string
            pos = entry.location.position
            dets.append(key)
            strings.append(string)
            positions.append(int(pos))
            
    return dets, strings, positions

def build_status_map(chan_map, data):
    
    
    dets, strings, positions = build_string_array(chan_map)
    
    string_nos = np.array(sorted(np.unique(strings)))
    pos_nos = np.array(sorted(np.unique(positions)))
    n_strings = len(string_nos)
    max_pos = np.max(positions)
    
    data_array = np.full((max_pos *2+1, n_strings*2+1), np.nan)
    annot_array = np.empty((max_pos *2+1, n_strings*2+1), dtype="object")

    
    for i,det in enumerate(dets):
        index = (2*positions[i]-1, 2*(np.where(strings[i] == string_nos)[0]+1)-1)
        annot_array[index] = det
        proc_status=None
        use_status=None
        proc_status = data[det]["processable"]
        use_status = data[det]["usability"]
        if proc_status == True:
            if use_status == "On":
                data_array[index] =2.
            else :
                data_array[index] =1.
        else:
            data_array[index]=0.
            
    x_axes = np.full(n_strings*2+1, " ",dtype = object)
    for i, s in enumerate(string_nos):
        x_axes[2*(i+1)-1] = f'Str {s}'
    
    y_axes = np.full(max_pos *2+1, " ", dtype = object)
    for i, n in enumerate(pos_nos):
        y_axes[2*(i+1)-1] = f'Pos {n}'
            
    return data_array, x_axes, y_axes, annot_array

def plot_status(run, run_dict, path, key =None):
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    chmap = LegendMetadata(path = prod_config["paths"]["metadata"])
    cfg = LegendMetadata(path = prod_config["paths"]["config"])
    
    config = cfg.on(run_dict["timestamp"], system="phy")
    cmap = chmap.channelmap(run_dict["timestamp"])
    
    status_map = config["analysis"]
    
    data_array, x_labels, y_labels, annotations = build_status_map(cmap, status_map)
    fig = plt.figure(figsize=(10, 6), dpi=1000, facecolor="w", edgecolor="k")
    sns.set(font_scale=1)

    stat_map = sns.heatmap(
        data=data_array,
        yticklabels=y_labels,
        xticklabels=x_labels,
        cmap="Set1",
        fmt='s',
        cbar=False,

        annot=annotations,
        annot_kws={"fontsize":5, 'color':'white'}
    )

    plt.title("Working Detectors")
    plt.tight_layout()
    plt.close()
    return fig

def build_counts_map(chan_map, data):
    dets, strings, positions = build_string_array(chan_map)
    
    string_nos = np.array(sorted(np.unique(strings)))
    pos_nos = np.array(sorted(np.unique(positions)))
    n_strings = len(string_nos)
    max_pos = np.max(positions)
    
    data_array = np.full((max_pos *2+1, n_strings*2+1), np.nan)
    annot_array = np.empty((max_pos *2+1, n_strings*2+1), dtype="object")

    
    for i,det in enumerate(dets):
        index = (2*positions[i]-1, 2*(np.where(strings[i] == string_nos)[0]+1)-1)
        annot_array[index] = data[det]
        data_array[index] =data[det]
        
            
    x_axes = np.full(n_strings*2+1, " ",dtype = object)
    for i, s in enumerate(string_nos):
        x_axes[2*(i+1)-1] = f'Str {s}'
    
    y_axes = np.full(max_pos *2+1, " ", dtype = object)
    for i, n in enumerate(pos_nos):
        y_axes[2*(i+1)-1] = f'Pos {n}'
            
    return data_array, x_axes, y_axes, annot_array

def plot_counts(run, run_dict, path, key =None):
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    chmap = LegendMetadata(path = prod_config["paths"]["metadata"])
    cfg = LegendMetadata(path = prod_config["paths"]["config"])
    
    config = cfg.on(run_dict["timestamp"], system="phy")
    cmap = chmap.channelmap(run_dict["timestamp"])
    
    file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict["period"]}/{run}')
    path = os.path.join(file_path, 
                        f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    with open(path, 'r') as r:
        all_res = json.load(r)
        
    res = {}
    for det in cmap:
        if cmap[det].system == "geds":
            try:
                res[det] = all_res[f"ch{cmap[det].daq.rawid:07}"]["ecal"]["cuspEmax_ctc_cal"]["total_fep"]
            except:
                res[det] = 0 
    
    data_array, x_labels, y_labels, annotations = build_counts_map(cmap, res)
    fig = plt.figure(figsize=(10, 6), dpi=1000, facecolor="w", edgecolor="k")
    sns.set(font_scale=1)

    stat_map = sns.heatmap(
        data=data_array,
        yticklabels=y_labels,
        xticklabels=x_labels,
        cmap="Blues",
        fmt='.0f',
        cbar=False,
        vmin=1000, 
        vmax=3000,
        annot=annotations,
        annot_kws={"fontsize":5, 'color':'white'}
    )

    plt.title("FEP Counts")
    plt.tight_layout()
    plt.close()
    return fig
    
    

def plot_energy_resolutions(run, run_dict, path, key="String", at="Qbb"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"],f'cal/{run_dict["period"]}/{run}')
    path = os.path.join(file_path, 
                            f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    with open(path, 'r') as r:
        all_res = json.load(r)
        
    default = {'cuspEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                  'Qbb_fwhm_err': np.nan, 
                                                  '2.6_fwhm': np.nan, 
                                                  '2.6_fwhm_err': np.nan, 
                                                  'm0': np.nan, 
                                                  'm1': np.nan}, 
                             'zacEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                 'Qbb_fwhm_err': np.nan, 
                                                 '2.6_fwhm': np.nan, 
                                                 '2.6_fwhm_err': np.nan, 
                                                 'm0': np.nan, 
                                                 'm1': np.nan}, 
                             'trapEmax_ctc_cal': {'Qbb_fwhm': np.nan, 
                                                  'Qbb_fwhm_err': np.nan, 
                                                  '2.6_fwhm': np.nan, 
                                                  '2.6_fwhm_err': np.nan, 
                                                  'm0': np.nan, 
                                                  'm1': np.nan}}
    res = {}
    for stri in strings:
        res[stri]=default
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]
            try:
                res[detector] = all_res[f"ch{channel:03}"]["ecal"]
            except:
                res[detector] = default
    
    
            
                
    fig = plt.figure() #
    plt.errorbar(list(res), [res[det]["cuspEmax_ctc_cal"][f"{at}_fwhm"] for det in res],
                yerr=[res[det]["cuspEmax_ctc_cal"][f"{at}_fwhm_err"] for det in res], 
                 marker='o',linestyle = ' ', color='deepskyblue', 
                 label = f'Cusp Average: {np.nanmean([res[det]["cuspEmax_ctc_cal"][f"{at}_fwhm"] for det in res]):.2f}keV')
    plt.errorbar(list(res), [res[det]["zacEmax_ctc_cal"][f"{at}_fwhm"] for det in res],
                yerr=[res[det]["zacEmax_ctc_cal"][f"{at}_fwhm_err"] for det in res], 
                 marker='o',linestyle = ' ', color='green',
                label = f'Zac Average: {np.nanmean([res[det]["zacEmax_ctc_cal"][f"{at}_fwhm"] for det in res]):.2f}keV')
    plt.errorbar(list(res), [res[det]["trapEmax_ctc_cal"][f"{at}_fwhm"] for det in res],
                yerr=[res[det]["trapEmax_ctc_cal"][f"{at}_fwhm_err"] for det in res], marker='o',linestyle = ' ', color='orangered',
                label = f'Trap Average: {np.nanmean([res[det]["trapEmax_ctc_cal"][f"{at}_fwhm"] for det in res]):.2f}keV')
    for stri in strings:
        loc=np.where(np.array(list(res))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    
    for off_det in off_dets:
        loc=np.where(np.array(list(res))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")

    plt.yticks(np.arange(0,11,1))
    plt.xlabel('Detector')
    if at == "Qbb":
        plt.ylabel('FWHM at Qbb (keV)')
    else:
        plt.ylabel('FWHM of 2.6 MeV peak (keV)')
    plt.grid(linestyle='dashed', linewidth=0.5,which="both", axis='both')
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Energy Resolutions")
    plt.legend(loc='upper right')
    plt.ylim([1,5])
    plt.tight_layout()
    plt.close()
    return fig

def plot_energy_resolutions_Qbb(run, run_dict, path, key="String"):
    return plot_energy_resolutions(run, run_dict, path, key=key, at="Qbb")

def plot_energy_resolutions_2614(run, run_dict, path, key="String"):
    return plot_energy_resolutions(run, run_dict, path, key=key, at="2.6")

def plot_no_fitted_energy_peaks(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"])
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [chmap.map("name")[field].daq.rawid for field in soft_dict if soft_dict[field]["processable"]is False]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    res = {}
    with open(file_path, 'r') as r:
        res = json.load(r)

    peaks = [583.191,
        727.330,
        860.564,
        1592.53,
        1620.50,
        2103.53,
        2614.50]
    grid = np.zeros((len(peaks), len(channels)))
    for i,channel in enumerate(channels):
        idxs = []
        try:
            fitted_peaks = res[f"ch{channel:07}"]["ecal"]["cuspEmax_ctc_cal"]["fitted_peaks"]
            if not isinstance(fitted_peaks,list):
                fitted_peaks = [fitted_peaks]
            for j,peak in enumerate(peaks):
                if peak in fitted_peaks:
                    idxs.append(j)
            if len(idxs)>0:
                grid[np.array(idxs),i]=1
                
        except:
            if chmap[channel] in off_dets:
                grid[:,i]=1
            pass
        
    fig=plt.figure()
    plt.imshow(grid, cmap = "brg")
    plt.ylabel("peaks")
    plt.xlabel("channel")

    yticks, ylabels = plt.yticks()
    plt.yticks(ticks = yticks[1:-1], labels = [f"{peak:.1f}" for peak in peaks])

    plt.xticks(ticks = np.arange(0,len(channels),1), labels = [f"{chmap[channel]['name']}" for channel in channels], rotation = 90)
    for off_det in off_dets:
        loc = np.where(np.array(channels)==int(off_det))[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Energy Fits")
    plt.tight_layout()
    plt.show()
    return fig

def plot_no_fitted_aoe_slices(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]
    
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    res = {}
    with open(file_path, 'r') as r:
        res = json.load(r)

    nfits = {}
    for stri in strings:
        res[stri]=np.nan
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]
            try:
                nfits[detector] =res[f"ch{channel:07}"]["aoe"]["correction_fit_results"]["n_of_valid_fits"]
            except:
                nfits[detector] =np.nan
        
    fig=plt.figure()
    plt.scatter(list(nfits), [nfits[channel] for channel in nfits])
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(nfits))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Channel')
    plt.ylabel('# of A/E fits')
    plt.grid(linestyle='dashed', linewidth=0.5,which="both", axis='both')
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} A/E fits")
    plt.tight_layout()
    plt.close()
    return fig

def get_aoe_results(run, run_dict, path, key="String"):

    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]
    
    file_path = os.path.join(prod_config["paths"]["par_hit"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_hit_results.json')
    
    
    with open(file_path, 'r') as r:
        all_res = json.load(r)
    
    default = {'A/E_Energy_param': 'cuspEmax', 
                                 'Cal_energy_param': 'cuspEmax_ctc', 
                                 'dt_param': 'dt_eff', 
                                 'rt_correction': False, 
                                 'Mean_pars': [np.nan, np.nan], 
                                 'Sigma_pars': [np.nan, np.nan], 
                                 'Low_cut': np.nan, 'High_cut': np.nan, 
                                 'Low_side_sfs': {
                                     '1592.5': {
                                         'sf': np.nan, 
                                         'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}, 
                                 '2_side_sfs': {
                                     '1592.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}}
            
        
    aoe_res = {}
    for stri in strings:
        aoe_res[stri]=default
        for channel in strings[stri]:
            detector = channel_map[channel]["name"]

            try:  
                aoe_res[detector] =all_res[f"ch{channel:07}"]["aoe"]
            except:
                aoe_res[detector] = default

            if len(list(aoe_res[detector])) ==10:
                aoe_res[detector].update({
                                 'Low_side_sfs': {
                                     '1592.5': {
                                         'sf': np.nan, 
                                         'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}, 
                                 '2_side_sfs': {
                                     '1592.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '1620.5': {'sf': np.nan, 
                                                'sf_err': np.nan}, 
                                     '2039': {'sf': np.nan, 
                                              'sf_err': np.nan}, 
                                     '2103.53': {'sf': np.nan, 
                                                 'sf_err': np.nan}, 
                                     '2614.5': {'sf': np.nan, 
                                                'sf_err': np.nan}}})  

            elif len(list(aoe_res[detector])) <10:
                aoe_res[detector] = default
                
    fig = plt.figure()
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["1592.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["1592.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ', 
                 label = 'Tl DEP')

    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["1620.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["1620.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Bi FEP')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2039"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2039"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = r'CC @ $Q_{\beta \beta}$')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2103.53"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2103.53"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Tl SEP')
    plt.errorbar(list(aoe_res), [float(aoe_res[det]["Low_side_sfs"]["2614.5"]["sf"]) for det in aoe_res],
                yerr=[float(aoe_res[det]["Low_side_sfs"]["2614.5"]["sf_err"]) for det in aoe_res], 
                 marker='o',linestyle = ' ',  
                 label = f'Tl FEP')

    for stri in strings:
        loc=np.where(np.array(list(aoe_res))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(aoe_res))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.yticks(np.arange(0,110,10))
    plt.xlabel('Detector')
    plt.ylabel('Survival fraction')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} A/E Survival Fractions")
    plt.legend(loc='upper right')
    plt.ylim([0,100])
    plt.tight_layout()
    #plt.savefig("/data1/users/marshall/prod-ref/optim_test/aoe.png")
    plt.close()

    
    return fig


def plot_pz_consts(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]
    
    cal_dict_path = os.path.join(prod_config["paths"]["par_dsp"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_dsp.json')
    
    

    with open(cal_dict_path,'r') as r:
        cal_dict = json.load(r)
    
    taus={}

    for stri in strings:
        taus[stri]=np.nan
        for channel in strings[stri]:
            det = channel_map[channel]["name"]
            try:
                taus[det] = float(cal_dict[f"ch{channel:07}"]["pz"]["tau"][:-3])/1000
            except:
                taus[det] =np.nan
    
    fig = plt.figure()
    plt.errorbar(list(taus),[taus[det] for det in taus] ,yerr=10,
                 marker='o', color='deepskyblue', linestyle = '')
    for stri in strings:
        loc=np.where(np.array(list(taus))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(taus))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Detector')
    plt.ylabel(f'Pz constant ($\mu s$)')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Pole Zero Constants")
    plt.tight_layout()
    plt.close()
    return fig

def plot_alpha(run, run_dict, path, key="String"):
    
    strings, soft_dict, channel_map = sorter(path, run_dict["timestamp"], key=key)
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    cfg_file = prod_config["paths"]["chan_map"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmaps.on(run_dict["timestamp"]).map("daq.rawid")
    channels = [field for field in chmap if chmap[field]["system"]=="geds"]
    
    off_dets = [field for field in soft_dict if soft_dict[field]["processable"]is False]
    
    
    cal_dict_path = os.path.join(prod_config["paths"]["par_dsp"], 
                             f'cal/{run_dict["period"]}/{run}', 
                             f'{run_dict["experiment"]}-{run_dict["period"]}-{run}-cal-{run_dict["timestamp"]}-par_dsp.json')
    
    with open(cal_dict_path,'r') as r:
        cal_dict = json.load(r)
    
    trap_alpha={}
    cusp_alpha={}
    zac_alpha={}

    
    for stri in strings:
        trap_alpha[stri]=np.nan
        cusp_alpha[stri]=np.nan
        zac_alpha[stri]=np.nan
        for channel in strings[stri]:
            det = channel_map[channel]["name"]
            try:
                trap_alpha[det]=(float(cal_dict[f"ch{channel:07}"]["ctc_params"]["trapEmax_ctc"]["parameters"]["a"]))
                cusp_alpha[det]=(float(cal_dict[f"ch{channel:07}"]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"]))
                zac_alpha[det]=(float(cal_dict[f"ch{channel:07}"]["ctc_params"]["zacEmax_ctc"]["parameters"]["a"]))
            except:
                trap_alpha[det]=np.nan
                cusp_alpha[det]=np.nan
                zac_alpha[det]=np.nan

    fig = plt.figure()
    plt.scatter(list(trap_alpha), [trap_alpha[det] for det in trap_alpha],
                 marker='o', color='deepskyblue', label='Trap')
    plt.scatter(list(cusp_alpha), [cusp_alpha[det] for det in cusp_alpha],
                 marker='o', color='orangered', label='Cusp')
    plt.scatter(list(zac_alpha), [zac_alpha[det] for det in zac_alpha],
                 marker='o', color='green', label='Zac')
    for stri in strings:
        loc=np.where(np.array(list(trap_alpha))==stri)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("blue")
        plt.axvline(stri, color='black')
    plt.tick_params(axis='x', labelrotation=90)
    for off_det in off_dets:
        loc = np.where(np.array(list(trap_alpha))==off_det)[0][0]
        plt.gca().get_xticklabels()[loc].set_color("red")
    plt.xlabel('Detector')
    plt.ylabel(f'Alpha Value (1/ns)')
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.title(f"{run_dict['experiment']}-{run_dict['period']}-{run} Charge Trapping Constants")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.close()
    return fig

def plot_bls(plot_dict,chan_dict, channels, 
             string, key="String"):

    p = figure(width=700, height=600, y_axis_type="log")
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    if len(channels) > 19:
        colours = Turbo256[len(channels)]
    else:
        colours = Category20[len(channels)]
    for i,channel in enumerate(channels):
        try:
            plot_dict_chan = plot_dict[f"ch{channel:07}"]

            p.step(plot_dict_chan["baseline_spectrum"]["bins"], 
                     plot_dict_chan["baseline_spectrum"]["bl_array"],
                       legend_label=f'ch{channel:03}: {chan_dict[channel]["name"]}', 
                      mode="after", line_width=2, line_color = colours[i])
        except:
            pass
        
    p.add_layout(Title(text="Wf Baseline Mean - FC Baseline", align="center"), "below")
    p.add_layout(Title(text="Counts", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p
    
def plot_fep_stability_channels2d(plot_dict, chan_dict, channels, yrange, string, 
                                  key="String", energy_param = "cuspEmax_ctc"):
    
    times = None
    p = figure(width=700, height=600, y_axis_type="log", x_axis_type='datetime')
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    if len(channels) > 19:
        colours = Turbo256[len(channels)]
    else:
        colours = Category20[len(channels)]
    with shelve.open(plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
        for i,channel in enumerate(channels):
            try:

                plot_dict_chan = shelf[f"ch{channel:07}"]
                p.line([datetime.fromtimestamp(time) for time in plot_dict_chan[energy_param]["mean_stability"]["time"]], 
                         plot_dict_chan[energy_param]["mean_stability"]["energy"], 
                         legend_label=f'ch{channel:07}: {chan_dict[channel]["name"]}', 
                          line_width=2, line_color = colours[i])
                if times is None:
                    times = [datetime.fromtimestamp(t) for t in plot_dict_chan[energy_param]["mean_stability"]["time"]]      
            except:
                pass

    p.y_range = Range1d(yrange[0], yrange[1])
    p.add_layout(Title(text=f"Time (UTC), starting: {times[0].strftime('%d/%m/%Y %H:%M:%S')}", align="center"), "below")
    p.add_layout(Title(text="Energy (keV)", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p

def plot_energy_spectra(plot_dict, chan_dict, channels, string,  
                        key="String", energy_param = "cuspEmax_ctc"):
    
    p = figure(width=700, height=600, y_axis_type="log")
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    if len(channels) > 19:
        colours = Turbo256[len(channels)]
    else:
        colours = Category20[len(channels)]
    
    for i,channel in enumerate(channels):
        try:

            plot_dict_chan = plot_dict[f"ch{channel:07}"]
            p.step(plot_dict_chan[energy_param]["spectrum"]["bins"][1:], 
                     plot_dict_chan[energy_param]["spectrum"]["counts"], 
                     legend_label=f'ch{channel:07}: {chan_dict[channel]["name"]}', 
                      mode="after", line_width=2, line_color = colours[i])
        except:
            pass
    
    p.add_layout(Title(text=f"Energy (keV)", align="center"), "below")
    p.add_layout(Title(text="Counts", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    return p



def plot_baseline_stability(plot_dict, chan_dict, channels, string,  
                        key="String"):
    
    p = figure(width=700, height=600, x_axis_type='datetime')
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    if len(channels) > 19:
        colours = Turbo256[len(channels)]
    else:
        colours = Category20[len(channels)]
    
    times=None
    for i,channel in enumerate(channels):
        try:
            bl = plot_dict[f'ch{channel:07}']["baseline_stability"]["baseline"]
            bl_spread = plot_dict[f'ch{channel:07}']["baseline_stability"]["spread"]
            mean = np.nanmean(bl[~np.isnan(bl)][:10])
            bl_mean = 100*(bl-mean)/mean
            bl_shift =  100*bl_spread/bl_mean
            
            p.step([datetime.fromtimestamp(time) for time in plot_dict[f'ch{channel:07}']["baseline_stability"]["time"]], 
                     bl_mean, 
                     legend_label=f'ch{channel:07}: {chan_dict[channel]["name"]}', 
                    line_width=2, line_color = colours[i])
            if times is None:
                    times = [datetime.fromtimestamp(t) for t in plot_dict[f'ch{channel:03}']["baseline_stability"]["time"]]      
        except:
            pass
    
    p.add_layout(Title(text=f"Time (UTC), starting: {times[0].strftime('%d/%m/%Y %H:%M:%S')}", align="center"), "below")
    p.add_layout(Title(text="Shift (%)", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p

def plot_stability(plot_dict, chan_dict, channels, string, parameter,
                                  key="String", energy_param = "cuspEmax_ctc"):
    times = None
    p = figure(width=700, height=600, x_axis_type='datetime')
    p.title.text = string
    p.title.align = "center"
    p.title.text_font_size = "15px"
    if len(channels) > 19:
        colours = Turbo256[len(channels)]
    else:
        colours = Category20[len(channels)]
    for i,channel in enumerate(channels):
        try:
            plot_dict_chan = plot_dict[f"ch{channel:07}"]
            
            en = plot_dict_chan[energy_param][parameter]["energy"]
            en_spread = plot_dict_chan[energy_param][parameter]["spread"]
            mean = np.nanmean(en[~np.isnan(en)][:10])
            en_mean = 100*(en-mean)/mean
            en_shift =  100*en_spread/en_mean
            
            p.step([datetime.fromtimestamp(time) for time in plot_dict_chan[energy_param][parameter]["time"]], 
                     en_mean, 
                     legend_label=f'ch{channel:07}: {chan_dict[channel]["name"]}', 
                      line_width=2, line_color = colours[i])
            if times is None:
                times = [datetime.fromtimestamp(t) for t in plot_dict_chan[energy_param][parameter]["time"]]      
        except:
            pass

    p.add_layout(Title(text=f"Time (UTC), starting: {times[0].strftime('%d/%m/%Y %H:%M:%S')}", align="center"), "below")
    p.add_layout(Title(text="Energy Shift (%)", align="center"), "left")
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    return p

def plot_fep_stability_channels2d(plot_dict, chan_dict, channels, string, 
                                  key="String", energy_param = "cuspEmax_ctc"):
    
    return plot_stability(plot_dict, chan_dict, channels, string, "2614_stability",
                                  key="String", energy_param = "cuspEmax_ctc")
    

def plot_pulser_stability_channels2d(plot_dict, chan_dict, channels, string, 
                                  key="String", energy_param = "cuspEmax_ctc"):
    return plot_stability(plot_dict, chan_dict, channels, string, "pulser_stability",
                                  key="String", energy_param = "cuspEmax_ctc")