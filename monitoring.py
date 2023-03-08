import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import panel as pn
import param
import pickle as pkl
import shelve
import bisect

import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

import legend_data_monitor as ldm

from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

from src.util import *
from src.summary_plots import *
from src.tracking_plots import *
from src.detailed_plots import *
from src.phy_monitoring import *

class monitoring(param.Parameterized):
    
    # calibration plots 
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['figure.dpi'] = 100
    
    cal_plots = ['2614_timemap',
                 'peak_fits',
                 'cal_fit',
                 'fwhm_fit',
                 'cut_spectrum',
                 'survival_frac',
                 "spectrum",
                 "logged_spectrum",
                "peak_track"]
    
    aoe_plots = ['dt_deps', 'compt_bands_nocorr', 'band_fits', 'mean_fit', 'sigma_fit', 'compt_bands_corr', 'surv_fracs', 'PSD_spectrum', 'psd_sf']

    baseline_plots= ["baseline_timemap" ]
    
    tau_plots =["slope", "waveforms"]
    
    optimisation_plots = ["trap_kernel", "zac_kernel", "cusp_kernel", "trap_acq", "zac_acq", "cusp_acq"]
    
    _options = {'cuspEmax_ctc': cal_plots , 'zacEmax_ctc': cal_plots,
            'trapEmax_ctc': cal_plots , 'trapTmax': cal_plots, "Baseline": baseline_plots,
            "A/E": aoe_plots, "Tau": tau_plots, "Optimisation": optimisation_plots}
        
    plot_types_summary_dict = {
                        "Detector_Status": plot_status, 
                        "FEP_Counts": plot_counts, 
                        "FWHM Qbb": plot_energy_resolutions_Qbb, 
                        "FWHM FEP": plot_energy_resolutions_2614,
                        "A/E":get_aoe_results, 
                       "Tau":plot_pz_consts, "Alpha": plot_alpha, 
                       "Valid. E": plot_no_fitted_energy_peaks, 
                       "Valid. A/E": plot_no_fitted_aoe_slices,
                       "Baseline_Spectrum": plot_bls, "Energy_Spectra": plot_energy_spectra,
                      "Baseline_Stability": plot_baseline_stability,
                       "2614_Stability":plot_fep_stability_channels2d,
                       "Pulser_Stability":plot_pulser_stability_channels2d
                      }
                      
    
    plot_types_tracking_dict = {"Energy Calib. Const.": plot_energy,"FWHM Qbb": plot_energy_res_Qbb, 
                                "FWHM FEP": plot_energy_res_2614, "A/E Mean": plot_aoe_mean,
                                "A/E Sigma": plot_aoe_sig, "Tau": plot_tau,  "Alpha": plot_ctc_const}
    
    channel = param.Selector(default = 0, objects = [0])
    plot_type_tracking = param.ObjectSelector(default = list(plot_types_tracking_dict)[0], objects= list(plot_types_tracking_dict))
    parameter = param.ObjectSelector(default = list(_options)[0], objects = list(_options))
    plot_type_details = param.ObjectSelector(default = cal_plots[0], objects= cal_plots)#, labels=cal_plots_labels)
    plot_type_summary = param.ObjectSelector(default = list(plot_types_summary_dict)[0], objects= list(plot_types_summary_dict))
    date_range = param.DateRange(default = (datetime.now()-dtt.timedelta(minutes = 10),
                                        datetime.now()+dtt.timedelta(minutes = 10)) , 
                                 bounds=(datetime.now()-dtt.timedelta(minutes = 110),
                                        datetime.now()+dtt.timedelta(minutes = 110)))
    
    # general selectors
    sort_by = param.ObjectSelector(default = list(sort_dict)[0], objects= list(sort_dict))
    string = param.ObjectSelector(default = 0, objects = [0])
    run = param.Selector(default = 0, objects = [0])
    
    # physics plots 
    phy_plots_vals          = ['baseline', 'cuspEmax', 'cuspEmax_ctc_cal', 'bl_std']
    phy_plot_style_dict     = {'Time': phy_plot_vsTime, 'Histogram': phy_plot_histogram}
    phy_resampled_vals      = ['yes', 'no', 'only']
    
    phy_plots           = param.ObjectSelector(default=phy_plots_vals[0], objects=phy_plots_vals, label="Value")
    phy_plot_style      = param.ObjectSelector(default=list(phy_plot_style_dict)[0], objects=list(phy_plot_style_dict), label="Plot Style")
    phy_resampled       = param.ObjectSelector(default=phy_resampled_vals[1], objects=phy_resampled_vals, label="Resampled")
    
    def __init__(self, cal_path, phy_path, name=None):
        super().__init__(name=name)
        self.path=cal_path
        self.phy_path=phy_path
        self.cached_plots ={}
        prod_config = os.path.join(self.path, "config.json")
        self.prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
        
        self.run_dict = gen_run_dict(self.path)
        self.param["run"].objects = list(self.run_dict)
        self.run = list(self.run_dict)[-1]

        self.periods = {}
        for run in self.run_dict: 
            if self.run_dict[run]['period'] not in self.periods:
                self.periods[self.run_dict[run]['period']] = [run]
            else:
                self.periods[self.run_dict[run]['period']].append(run)

        start_period = self.periods[list(self.periods)[0]]

        self.param["date_range"].bounds = (datetime.strptime(self.run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                                 datetime.strptime(self.run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        self.date_range = (datetime.strptime(self.run_dict[sorted(start_period)[0]]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                            datetime.strptime(self.run_dict[sorted(start_period)[-1]]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        
    
        self.update_plot_dict()
        self.update_plot_type_details()
        self.update_strings()
        
        
        self.phy_data_df = pd.DataFrame()
        self.phy_plot_info = {}
        
        self._get_phy_data()
        
        self.meta_df = pd.DataFrame()
        self._get_metadata()
    
    @param.depends("run", "phy_plots", watch=True)
    def _get_phy_data(self):
        data_file = self.phy_path + f'/generated/plt/phy/p02/{self.run}/l200-p02-{self.run}-phy'
        if not os.path.exists(data_file +'.dat'):
            self.phy_data_df = pd.DataFrame()
        else:
            with shelve.open(data_file) as file:

                # take df with parameter you want
                self.phy_data_df = file['monitoring']['pulser'][self.phy_plots]['df_geds']
                
                # take a random plot_info, it should be enough to save only one per time
                self.phy_plot_info = file['monitoring']['pulser'][self.phy_plots]['plot_info']
                    
                # set plotting options
                self.phy_plot_info['plot_style'] = self.phy_plot_style
                self.phy_plot_info['resampled'] = self.phy_resampled
        
    @param.depends("date_range", watch=True)
    def _get_run_dict(self):
        valid_from = [datetime.timestamp(datetime.strptime(self.run_dict[entry]["timestamp"], '%Y%m%dT%H%M%SZ')) for entry in self.run_dict]
        pos1 = bisect.bisect_right(valid_from, datetime.timestamp(self.date_range[0]))
        pos2 = bisect.bisect_left(valid_from, datetime.timestamp(self.date_range[-1]))
        if pos1 < 0:
            pos1 = 0
        if pos2 >= len(self.run_dict):
            pos2 = len(self.run_dict)
        valid_idxs = np.arange(pos1, pos2, 1)
        valid_keys = np.array(list(self.run_dict))[valid_idxs]
        out_dict = {key:self.run_dict[key] for key in valid_keys}
        return out_dict
    
    @param.depends("run")
    def _get_metadata(self):
        chan_dict, channel_map = self.chan_dict, self.channel_map
        df_chan_dict = pd.DataFrame.from_dict(chan_dict).T
        df_chan_dict["Channel"] = df_chan_dict.index
        df_chan_dict.index = df_chan_dict.index.map(lambda x: int(x[2:5]))
        df_channel_map = pd.DataFrame.from_dict(channel_map).T
        df_channel_map["Channel"] = df_channel_map.index.map(lambda x: "ch{:>03d}".format(int(x)))
        df_out = pd.merge(df_channel_map, df_chan_dict, left_on='Channel', right_on='Channel')
        df_out.columns
        df_out = df_out.reset_index().set_index('Channel')[['name', 'hardware_status', 'software_status', 'daq', 'location', 'voltage', 'electronics']]
        df_out['daq'] = df_out['daq'].map(lambda x: "Crate: {}, Card: {}".format(x['crate'], x['card']['id']))
        df_out['location'] = df_out['location'].map(lambda x: "String: {:>02d}, Pos.: {:>02d}".format(x['string'], x['position']))
        df_out['voltage'] = df_out['voltage'].map(lambda x: "Card: {:>02d}, Ch.: {:>02d}".format(x['card']['id'], x['channel']))
        df_out['electronics'] = df_out['electronics'].map(lambda x: "CC4: {}, Ch.: {:>02d}".format(x['cc4']['id'], x['cc4']['channel']))
        df_out['software_status'] =  df_out['software_status'].map(lambda x: True if x == 'On' else False)
        df_out['hardware_status'] =  df_out['hardware_status'].map(lambda x: True if x == 'Fully_biased' else False)
        df_out = df_out.rename({'name': 'Det. Name', 'hardware_status': 'HV Status', 'software_status': 'Software status', 'daq': 'FC card',
            'location': 'Det. Location', 'voltage': 'HV', 'electronics': 'Electronics'}, axis=1)
        self.meta_df = df_out

    @param.depends("date_range", "plot_type_tracking", "string", "sort_by")
    def view_tracking(self):

        figure = plot_tracking(self._get_run_dict(), self.path, self.plot_types_tracking_dict[self.plot_type_tracking], self.string, key=self.sort_by)
        return figure


    @param.depends("sort_by", watch=True)
    def update_strings(self):
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path, self.run_dict[self.run]["timestamp"], key=self.sort_by)

        self.param["string"].objects = list(self.strings_dict)
        self.string = f"{list(self.strings_dict)[0]}"
        
    @param.depends("run", "sort_by", "plot_type_summary", "string")
    def view_summary(self):
        figure=None
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (16, 6)
        plt.rcParams['figure.dpi'] = 100
        if self.plot_type_summary in ["FWHM Qbb", "FWHM FEP","A/E", "Tau", 
                                      "Alpha", "Valid. E", "Valid. A/E", "Detector_Status", "FEP_Counts"]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](self.run, 
                                            self.run_dict[self.run], 
                                            self.path, key=self.sort_by)
            
            
        elif self.plot_type_summary in ["Baseline_Spectrum", "Energy_Spectra", "Baseline_Stability", 
                                        "2614_Stability", "Pulser_Stability"]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](self.common_dict, self.channel_map, 
                            self.strings_dict[self.string],
                            self.string, key=self.sort_by)

        else:
            figure = plt.figure()
            plt.close()
        
        return figure
    
    @param.depends("run", "string", "sort_by", "phy_plots", "phy_plot_style", "phy_resampled")
    def view_phy(self):
        # update plot dict with resampled value
        self.phy_plot_info['resampled'] = self.phy_resampled
        
        # return empty plot if no data exists for run
        if self.phy_data_df.empty:
            p = figure(width=1000, height=600)
            p.title.text = title=f"No data for run {self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p
        
        else:
            # get all data from selected string
            data_string = self.phy_data_df[self.phy_data_df.isin({'channel': self.strings_dict[self.string]})['channel']]
            # plot data
            return self.phy_plot_style_dict[self.phy_plot_style](data_string, self.phy_plot_info, self.string)

    @param.depends("run", watch=True)
    def update_plot_dict(self):
        self.plot_dict = os.path.join(self.prod_config["paths"]["plt"],
                              f'hit/cal/{self.run_dict[self.run]["period"]}/{self.run}',
                            f'{self.run_dict[self.run]["experiment"]}-{self.run_dict[self.run]["period"]}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit')  
    
        with shelve.open(self.plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            channels = list(shelf.keys()) 

        with shelve.open(self.plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.common_dict = shelf["common"]
        channels.remove("common")
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path, self.run_dict[self.run]["timestamp"], "String")
        channel_list = []
        for channel in channels:
            channel_list.append(f"{channel}: {self.channel_map[int(channel[2:])]['name']}")
        
        self.param["channel"].objects = channel_list
        self.channel = channel_list[0]
        
        self.update_strings()
        self.update_channel_plot_dict()

    @param.depends("channel", watch=True)
    def update_channel_plot_dict(self):
        with shelve.open(self.plot_dict, 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.plot_dict_ch = shelf[self.channel[:5]]
        with shelve.open(self.plot_dict.replace("hit","dsp"), 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.dsp_dict = shelf[self.channel[:5]]
    
    
    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        plots = self._options[self.parameter]
        self.param["plot_type_details"].objects = plots
        self.plot_type_details = plots[0]

    @param.depends("run", "channel", "parameter", "plot_type_details")
    def view_details(self):
        if self.parameter in ["A/E", "Baseline"]:
            fig = self.plot_dict_ch[self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        elif self.parameter == "Tau":
            fig = self.dsp_dict[self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        elif self.parameter == "Optimisation":
            fig = self.dsp_dict[f"{self.plot_type_details.split('_')[0]}_optimisation"][f"{self.plot_type_details.split('_')[1]}_space"]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
        else:
            if self.plot_type_details == "spectrum" or self.plot_type_details == "logged_spectrum":
                fig = plot_spectrum(self.plot_dict_ch[self.parameter]["spectrum"], self.channel,
                                    log=False if self.plot_type_details == "spectrum" else True)
            elif self.plot_type_details == "survival_frac":
                fig = plot_survival_frac(self.plot_dict_ch[self.parameter]["survival_frac"])
            elif self.plot_type_details == "cut_spectrum":
                fig = plot_cut_spectra(self.plot_dict_ch[self.parameter]["spectrum"])
            elif self.plot_type_details == "peak_track":
                fig = track_peaks(self.plot_dict_ch[self.parameter])
            else:
                fig = self.plot_dict_ch[self.parameter][self.plot_type_details]
                dummy = plt.figure()
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)

        return fig
    
    @param.depends("run", "channel")
    def get_RunAndChannel(self):
        return pn.pane.Markdown("### Run: {} - String: {}".format(self.run, self.channel))
    
    @param.depends("run")
    def view_meta(self):
        return pn.widgets.Tabulator(self.meta_df, formatters={'Software status': BooleanFormatter(), 'HV Status': BooleanFormatter()})
