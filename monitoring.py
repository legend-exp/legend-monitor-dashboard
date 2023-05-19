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
import time

import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

import legend_data_monitor as ldm

from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LabelSet, LinearColorMapper, BasicTicker, ColorBar, FixedTicker, FuncTickFormatter
from bokeh.palettes import *

from src.util import *
from src.summary_plots import *
from src.tracking_plots import *
from src.detailed_plots import *
from src.phy_monitoring import *
from src.string_visulization import *
from src.sipm_monitoring import *
from src.muon_monitoring import *

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
                        "Detector Status": plot_status, 
                        "FEP Counts": plot_counts, 
                        "FWHM Qbb": plot_energy_resolutions_Qbb, 
                        "FWHM FEP": plot_energy_resolutions_2614,
                        "A/E":get_aoe_results, 
                        "Tau":plot_pz_consts, "Alpha": plot_alpha, 
                        "Valid. E": plot_no_fitted_energy_peaks, 
                        "Valid. A/E": plot_no_fitted_aoe_slices,
                        "Baseline Spectrum": plot_bls, "Energy Spectrum": plot_energy_spectra,
                        "Baseline Stability": plot_baseline_stability,
                        "FEP Stability":plot_fep_stability_channels2d,
                        "Pulser Stability":plot_pulser_stability_channels2d
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
    period = param.Selector(default = 0, objects = [0])
    
    # physics plots 
    phy_plots_vals          = ['baseline', 'cuspEmax', 'cuspEmax_ctc_cal', 'bl_std', 'AoE_Corrected', 'AoE_Classifier']
    phy_plot_style_dict     = {'Time': phy_plot_vsTime, 'Histogram': phy_plot_histogram}
    phy_resampled_vals      = ['yes', 'no', 'only']
    phy_unit_vals           = ['Relative', 'Absolute']
    
    phy_plots           = param.ObjectSelector(default=phy_plots_vals[0], objects=phy_plots_vals, label="Value")
    phy_plot_style      = param.ObjectSelector(default=list(phy_plot_style_dict)[0], objects=list(phy_plot_style_dict), label="Plot Style")
    phy_resampled       = param.ObjectSelector(default=phy_resampled_vals[1], objects=phy_resampled_vals, label="Resampled")
    phy_units           = param.ObjectSelector(default=phy_unit_vals[0], objects=phy_unit_vals, label="Units")
    
    # sipm plots
    # sipm_plots_barrels    = ['InnerBarrel-Top', 'InnerBarrel-Bottom', 'OuterBarrel-Top', 'OuterBarrel-Bottom']
    sipm_plot_style_dict  = {'Time': sipm_plot_vsTime, 'Histogram': sipm_plot_histogram}
    sipm_resampled_vals   = ['1min', '5min', '10min', '30min', '60min']
    
    
    sipm_sort_dict        = ['Barrel']
    sipm_sort_by          = param.ObjectSelector(default = list(sipm_sort_dict)[0], objects= list(sipm_sort_dict))
    
    sipm_barrel          = param.ObjectSelector(default=0, objects=[0])
    sipm_resampled       = param.ObjectSelector(default=sipm_resampled_vals[1], objects=sipm_resampled_vals, label="Resampled")
    sipm_plot_style      = param.ObjectSelector(default=list(sipm_plot_style_dict)[0], objects=list(sipm_plot_style_dict))
    
    # muon plots
    muon_plots_cal_dict     = {"Cal. Spectra": muon_plot_spectra, "Cal. SPP Sigma": muon_plot_spp, "Cal. SPP Shift": muon_plot_calshift}
    muon_plots_cal          = param.ObjectSelector(default = list(muon_plots_cal_dict)[0], objects= list(muon_plots_cal_dict))
    
    muon_plots_mon_dict     = {"Integral Light": muon_plot_intlight, "Total Rates/H": muon_plot_totalRates_hourly, 
                            "Total Rates/D": muon_plot_totalRates_daily, 
                            "Pillbox Rates": muon_plot_ratesPillBox, "Floor Rates": muon_plot_ratesFloor,
                            "Wall Rates": muon_plot_ratesWall}
    muon_plots_mon          = param.ObjectSelector(default = list(muon_plots_mon_dict)[0], objects= list(muon_plots_mon_dict))
    
    
    # visualization
    meta_visu_plots_dict = {"Usability": plot_visu_usability, "Processable": plot_visu_processable, "Processable": plot_visu_processable,
                            "Mass": plot_visu_mass, "Depl. Voltage": plot_visu_depletion, "Oper. Voltage": plot_visu_operation,
                            "Enrichment": plot_visu_enrichment}
    
    meta_visu_plots = param.ObjectSelector(default=list(meta_visu_plots_dict)[0], objects=list(meta_visu_plots_dict))
    
    # downloads
    plot_types_download_dict = ["FWHM Qbb", "FWHM FEP", "A/E", "Tau", "Alpha"]
    plot_types_download = param.Selector(default = plot_types_download_dict[0], objects= plot_types_download_dict)
    
    def __init__(self, cal_path, phy_path, sipm_path, muon_path, tmp_path, name=None):
        super().__init__(name=name)
        self.path=cal_path
        self.phy_path=phy_path
        self.sipm_path=sipm_path
        self.muon_path = muon_path
        self.tmp_path = tmp_path
        self.cached_plots ={}

        prod_config = os.path.join(self.path, "config.json")
        self.prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
        
        self.periods = gen_run_dict(self.path)
        self.param["period"].objects = list(self.periods)
        # self.period = list(self.periods)[-1]
        self.period = 'p03'
        
        self._get_period_data()
        
        self.update_plot_dict()
        self.update_plot_type_details()
        self.update_strings()
        
        self.phy_data_df = pd.DataFrame()
        self.phy_plot_info = {}
        self._get_phy_data()
        
        self.muon_data_dict = {}
        self._get_muon_data()
        
        
        self.sipm_data_df = pd.DataFrame()
        self._get_sipm_data()
        
        self.meta_df = pd.DataFrame()
        self.meta_visu_source      = ColumnDataSource({})
        self.meta_visu_xlabels     = {}
        self.meta_visu_chan_dict   = {}
        self.meta_visu_channel_map = {}
        self._get_metadata()
        
    @param.depends("period", watch=True)
    def _get_period_data(self):
        
        self.run_dict = self.periods[self.period]
        
        self.param["run"].objects = list(self.run_dict)
        self.run = list(self.run_dict)[-1]
        # self.periods = {}
        # for run in self.run_dict: 
        #     if self.run_dict[run]['period'] not in self.periods:
        #         self.periods[self.run_dict[run]['period']] = [run]
        #     else:
        #         self.periods[self.run_dict[run]['period']].append(run)
        
        
        start_period = sorted(list(self.periods))[0]
        start_run    = sorted(list(self.periods[start_period]))[0]
        end_period   = sorted(list(self.periods))[-1]
        end_run      = sorted(list(self.periods[end_period]))[-1]

        self.param["date_range"].bounds = (datetime.strptime(self.periods[start_period][start_run]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), 
                                datetime.strptime(self.periods[end_period][end_run]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        self.date_range = (datetime.strptime(self.periods[start_period][start_run]["timestamp"],'%Y%m%dT%H%M%SZ')-dtt.timedelta(minutes = 100), datetime.strptime(self.periods[end_period][end_run]["timestamp"],'%Y%m%dT%H%M%SZ')+dtt.timedelta(minutes = 110))
        
    
    @param.depends("period", "run", "phy_plots", watch=True)
    def _get_phy_data(self):
        data_file = self.phy_path + f'/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-geds'
        if not os.path.exists(data_file +'.dat'):
            phy_data_df = pd.DataFrame()
            phy_plot_info = {}
        else:
            with shelve.open(data_file, 'r', protocol=pkl.HIGHEST_PROTOCOL) as file:

                # take df with parameter you want
                phy_data_df = file['monitoring']['pulser'][self.phy_plots]['df_geds']
                
                # take a random plot_info, it should be enough to save only one per time
                phy_plot_info = file['monitoring']['pulser'][self.phy_plots]['plot_info']
                
        self.phy_data_df, self.phy_plot_info = phy_data_df, phy_plot_info
    
    @param.depends("period", "run", watch=True)
    def _get_muon_data(self):
        data_file = f"{self.muon_path}/generated/plt/phy/{self.period}/dsp/{self.run}/dashboard_period_{self.period}_run_{self.run}.shelve"
        if not os.path.exists(data_file +'.dat'):
            self.muon_data_dict = {}
        else:
            with shelve.open(data_file, 'r') as f:
                # Create an empty dictionary
                arrays_dict = {}
                
                for key in f.keys():
                    # Add a new key-value pair to the dictionary
                    arrays_dict[key] = np.array(f[key])
                
                self.muon_data_dict = arrays_dict    
            
    @param.depends("period", "date_range", watch=True)
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
    
    @param.depends("period", "run", watch=True)
    def _get_metadata(self):
        try:
            chan_dict, channel_map = self.chan_dict, self.channel_map
            
            df_chan_dict = pd.DataFrame.from_dict(chan_dict).T
            df_chan_dict.index.name = 'name'
            df_chan_dict = df_chan_dict.reset_index()
            
            df_channel_map = pd.DataFrame.from_dict(channel_map).T
            df_channel_map = df_channel_map[df_channel_map['system'] == 'geds']
            
            df_out = pd.merge(df_channel_map, df_chan_dict, left_on='name', right_on='name')
            df_out = df_out.reset_index().set_index('name')[['processable', 'usability', 'daq', 'location', 'voltage', 'electronics', 'characterization', 'production', 'type']]
            df_out['daq'] = df_out['daq'].map(lambda x: "Crate: {}, Card: {}".format(x['crate'], x['card']['id']))
            df_out['location'] = df_out['location'].map(lambda x: "String: {:>02d}, Pos.: {:>02d}".format(x['string'], x['position']))
            df_out['voltage'] = df_out['voltage'].map(lambda x: "Card: {:>02d}, Ch.: {:>02d}".format(x['card']['id'], x['channel']))
            df_out['electronics'] = df_out['electronics'].map(lambda x: "CC4: {}, Ch.: {:>02d}".format(x['cc4']['id'], x['cc4']['channel']))
            df_out['usability'] =  df_out['usability'].map(lambda x: True if x == 'on' else False)
            # df_out['processable'] =  df_out['processable'].map(lambda x: True if x == 'True' else False)
            df_out['Depl. Vol. (kV)'] = df_out['characterization'].map(lambda x: get_characterization(x, 'depletion_voltage_in_V'))/1000
            df_out['Oper. Vol. (kV)'] = df_out['characterization'].map(lambda x: get_characterization(x, 'recommended_voltage_in_V'))/1000
            df_out['Manufacturer'] = df_out['production'].map(lambda x: get_production(x, 'manufacturer'))
            df_out['Mass (kg)'] = df_out['production'].map(lambda x: get_production(x, 'mass_in_g'))/1000
            df_out['Order'] = df_out['production'].map(lambda x: get_production(x, 'order'))
            df_out['Crystal'] = df_out['production'].map(lambda x: get_production(x, 'crystal'))
            df_out['Slice'] = df_out['production'].map(lambda x: get_production(x, 'slice'))
            df_out['Enrichment (%)'] = df_out['production'].map(lambda x: get_production(x, 'enrichment')) * 100
            df_out['Delivery'] = df_out['production'].map(lambda x: get_production(x, 'delivered'))
            df_out = df_out.reset_index().rename({'name': 'Det. Name', 'processable': 'Proc.', 'usability': 'Usabl.', 'daq': 'FC card',
                'location': 'Det. Location', 'voltage': 'HV', 'electronics': 'Electronics', 'type': 'Type'}, axis=1).set_index('Det. Name')
            df_out = df_out.drop(['characterization', 'production'], axis=1)
            df_out = df_out.astype({'Proc.': 'bool', 'Usabl.': 'bool'})
            self.meta_df = df_out
            
            # get metadata visu plot data
            strings_dict, chan_dict, channel_map = sorter(self.path, self.run_dict[self.run]["timestamp"], key="String")
            self.meta_visu_source, self.meta_visu_xlabels = get_plot_source_and_xlabels(chan_dict, channel_map, strings_dict)
            self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
        except:
            pass

    @param.depends("period", "date_range", "plot_type_tracking", "string", "sort_by")
    def view_tracking(self):

        figure = plot_tracking(self._get_run_dict(), self.path, self.plot_types_tracking_dict[self.plot_type_tracking], self.string, self.period, self.plot_type_tracking, key=self.sort_by)
        return figure
    
    @param.depends("period", "run", "muon_plots_cal")
    def view_muon_cal(self):
        if not bool(self.muon_data_dict):
            p = figure(width=1000, height=600)
            p.title.text = title=f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p

        if self.muon_plots_cal == "Cal. SPP Shift":
            data_file = f"{self.muon_path}/generated/plt/phy/{self.period}/dsp/{self.run}/dashboard_period_{self.period}_run_{self.run}.shelve"            
            with shelve.open(data_file, 'r') as f:
                # x_data_str = np.array(list(f['date'].values()))
                x_data_str = np.array(list(f['date'].values()))
                # y_data = np.array(list(f['mean_shift'].values()))
                y_data = np.array(list(f['mean_shift'].values()))

                # Reshape the x_data and y_data arrays
                x_data = np.array([[dtt.datetime.strptime(date_str, '%Y_%m_%d') for date_str in row] for row in x_data_str])
                
                return self.muon_plots_cal_dict[self.muon_plots_cal](x_data, y_data, self.run, self.period, self.run_dict[self.run], self.muon_plots_cal)
        else:
            return self.muon_plots_cal_dict[self.muon_plots_cal](self.muon_data_dict, self.run, self.period, self.run_dict[self.run], self.muon_plots_cal)

    @param.depends("period", "run", "muon_plots_mon")
    def view_muon_mon(self):
        if not bool(self.muon_data_dict):
            p = figure(width=1000, height=600)
            p.title.text = title=f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p

        return self.muon_plots_mon_dict[self.muon_plots_mon](self.muon_data_dict, self.period, self.run, self.run_dict[self.run])

    @param.depends("period", "sort_by", watch=True)
    def update_strings(self):
        self.strings_dict, self.chan_dict, self.channel_map = sorter(self.path, self.run_dict[self.run]["timestamp"], key=self.sort_by)

        self.param["string"].objects = list(self.strings_dict)
        self.string = f"{list(self.strings_dict)[0]}"
        
    @param.depends("period", "sipm_sort_by", watch=True)
    def update_barrels(self):
        self.sipm_out_dict, self.sipm_chmap = sorter(self.path, self.run_dict[self.run]["timestamp"], key=self.sipm_sort_by, spms=True)
        
        self.param["sipm_barrel"].objects = list(self.sipm_out_dict)
        self.sipm_barrel = f"{list(self.sipm_out_dict)[0]}"
        
        
    @param.depends("period", "run", watch=True)
    def _get_sipm_data(self):
        data_file = self.sipm_path + f'{self.period}_{self.run}_spmmon.hdf'
        if not os.path.exists(data_file):
            self.sipm_data_df = pd.DataFrame()
        else:
            self.sipm_data_df = pd.read_hdf(data_file).reset_index().set_index('time').drop(['index'], axis=1)
            self.sipm_data_df.index = pd.to_datetime(self.sipm_data_df.index, unit='s', origin='unix')
        
        self.sipm_out_dict, self.sipm_chmap = sorter(self.path, self.run_dict[self.run]["timestamp"], key=self.sipm_sort_by, spms=True)
        self.sipm_name_dict = {}
        for val in self.sipm_chmap.values():
            self.sipm_name_dict[val['daq']['rawid']] = val['name']
        self.update_barrels()
        
    @param.depends("period", "run", "sort_by", "plot_types_download")
    def download_summary_files(self):
        
        download_file, download_filename = self.plot_types_summary_dict[self.plot_types_download](self.run, 
                                            self.run_dict[self.run], 
                                            self.path, self.period, key=self.sort_by, download=True)
        if not os.path.exists(self.tmp_path + download_filename):
            download_file.to_csv(self.tmp_path + download_filename, index=False)
        return pn.widgets.FileDownload(self.tmp_path + download_filename,
                                button_type='success', embed=False, name="Click to download 'csv'", width=350)
    
    @param.depends("period", "run", "sipm_sort_by", "sipm_resampled", "sipm_barrel", "sipm_plot_style")
    def view_sipm(self):
        if self.sipm_data_df.empty:
            p = figure(width=1000, height=600)
            p.title.text = title=f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p
        else:
            data_barrel = self.sipm_data_df[[f'ch{channel}' for channel in self.sipm_out_dict[self.sipm_barrel] if f'ch{channel}' in self.sipm_data_df.columns]]
            meta_barrel = {}
            return self.sipm_plot_style_dict[self.sipm_plot_style](data_barrel, self.sipm_barrel, self.sipm_resampled, self.sipm_name_dict, self.run, self.period, self.run_dict[self.run])
        
    @param.depends("period", "run", "sort_by", "plot_type_summary", "string")
    def view_summary(self):
        figure=None
        if self.plot_type_summary in ["FWHM Qbb", "FWHM FEP","A/E", "Tau", 
                                        "Alpha", "Valid. E", "Valid. A/E"]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](self.run, 
                                            self.run_dict[self.run], 
                                            self.path, self.period, key=self.sort_by)
            
        elif self.plot_type_summary in ["Detector Status", "FEP Counts"]:
        # elif self.plot_type_summary in ["Detector Status"]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](self.run, 
                                            self.run_dict[self.run], 
                                            self.path, self.meta_visu_source, self.meta_visu_xlabels, self.period, key=self.sort_by)
        elif self.plot_type_summary in ["Baseline Spectrum", "Energy Spectrum", "Baseline Stability", 
                                        "FEP Stability", "Pulser Stability"]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](self.common_dict, self.channel_map, 
                            self.strings_dict[self.string],
                            self.string, self.run, self.period, self.run_dict[self.run], key=self.sort_by)
        else:
            figure = plt.figure()
            plt.close()
        
        return figure
    
    @param.depends("period", "run", "string", "sort_by", "phy_plots", "phy_plot_style", "phy_resampled", "phy_units")
    def view_phy(self):
        # update plot dict with resampled value
        # set plotting options
        # startt = time.time()
        phy_data_df   = self.phy_data_df
        phy_plot_info = self.phy_plot_info
        
        # return empty plot if no data exists for run
        if phy_data_df.empty:
            p = figure(width=1000, height=600)
            p.title.text = title=f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p
        
        if self.phy_units == "Relative":
            phy_plot_info["parameter"] = f"{phy_plot_info['parameter'].split('_var')[0]}_var"
            phy_plot_info["unit_label"] = "%"
        else:
            phy_plot_info["parameter"] = f"{phy_plot_info['parameter'].split('_var')[0]}"
            phy_plot_info["unit_label"] = phy_plot_info["unit"]
        
        # set plotting options
        phy_plot_info['plot_style'] = self.phy_plot_style
        phy_plot_info['resampled'] = self.phy_resampled
        
        # get all data from selected string
        data_string = phy_data_df[phy_data_df.isin({'channel': self.strings_dict[self.string]})['channel']]
        # plot data
        # print("Time to plot: ", time.time()-startt, "s")
        return self.phy_plot_style_dict[self.phy_plot_style](data_string, phy_plot_info, self.string, self.run, self.period, self.run_dict[self.run])

    @param.depends("run", watch=True)
    def update_plot_dict(self):
        self.plot_dict = os.path.join(self.prod_config["paths"]["plt"],
                              f'hit/cal/{self.period}/{self.run}',
                            f'{self.run_dict[self.run]["experiment"]}-{self.period}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit')
        
        # print(self.run_dict)
        # print(self.plot_dict)
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
            self.plot_dict_ch = shelf[self.channel[:9]]
        with shelve.open(self.plot_dict.replace("hit","dsp"), 'r', protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.dsp_dict = shelf[self.channel[:9]]
    
    
    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        plots = self._options[self.parameter]
        self.param["plot_type_details"].objects = plots
        self.plot_type_details = plots[0]

    @param.depends("period", "run", "channel", "parameter", "plot_type_details")
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
    
    @param.depends("period", "run", "channel")
    def get_RunAndChannel(self):
        return pn.pane.Markdown(f"### {self.run_dict[self.run]['experiment']}-{self.period}-{self.run} | Cal. | Details | Channel {self.channel}")

    @param.depends("period", "run")
    def view_meta(self):
        return pn.widgets.Tabulator(self.meta_df, formatters={'Proc.': BooleanFormatter(), 'Usabl.': BooleanFormatter()})
    
    @param.depends("period", "run", "meta_visu_plots")
    def view_meta_visu(self):
        return self.meta_visu_plots_dict[self.meta_visu_plots](self.meta_visu_source, self.meta_visu_chan_dict, self.meta_visu_channel_map, self.meta_visu_xlabels)
