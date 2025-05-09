from __future__ import annotations

import datetime as dtt
import logging
import pickle as pkl
import shelve
import time
from pathlib import Path

import h5py
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param
from legenddashboard.base import Monitoring
import legenddashboard.geds.cal as cal
import legenddashboard.muon.muon_monitoring as muon
import legenddashboard.geds.phy as phy
import legenddashboard.spms.sipm_monitoring as spm
import legenddashboard.string_visulization as visu
from bokeh.models import ColumnDataSource
from bokeh.models.widgets.tables import BooleanFormatter
from bokeh.plotting import figure
from legenddashboard.util import (
    get_characterization,
    get_production,
    sort_dict,
    sorter,
)

log = logging.getLogger(__name__)

# calibration plots
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100


# initiate all the dashboard classes and share the run, period, date parameters between them


class Dashboard(Monitoring):
    # string = param.ObjectSelector(default=0, objects=[0])
    # run = param.Selector(default=0, objects=[0])
    # period = param.Selector(default=0, objects=[0])
    # date_range = param.DateRange(
    #         default=(
    #             datetime.now() - dtt.timedelta(minutes=10),
    #             datetime.now() + dtt.timedelta(minutes=10),
    #         ),
    #         bounds=(
    #             datetime.now() - dtt.timedelta(minutes=110),
    #             datetime.now() + dtt.timedelta(minutes=110),
    #         ),
    #     )
    # channel = param.Selector(default=0, objects=[0])

    plot_type_tracking = param.ObjectSelector(
        default=list(cal.tracking_plots)[1],
        objects=list(cal.tracking_plots),
    )

    parameter = param.ObjectSelector(
        default=next(iter(cal.all_detailed_plots)), objects=list(cal.all_detailed_plots)
    )

    plot_type_details = param.ObjectSelector(
        default=cal.detailed_plots[0], objects=cal.detailed_plots
    )

    plot_type_summary = param.ObjectSelector(
        default=list(cal.summary_plots)[3],
        objects=list(cal.summary_plots),
    )


    # general selectors
    sort_by = param.ObjectSelector(
        default=next(iter(sort_dict)), objects=list(sort_dict)
    )

    # phy_plots_types = param.ObjectSelector(
    #     default=next(iter(phy.phy_plots_types_dict)),
    #     objects=list(phy.phy_plots_types_dict),
    #     label="Type",
    # )
    # phy_plots = param.ObjectSelector(
    #     default=list(phy.phy_plots_vals_dict)[4],
    #     objects=list(phy.phy_plots_vals_dict),
    #     label="Value",
    # )
    # phy_plot_style = param.ObjectSelector(
    #     default=next(iter(phy.phy_plot_style_dict)),
    #     objects=list(phy.phy_plot_style_dict),
    #     label="Plot Style",
    # )
    # phy_resampled = param.Integer(
    #     default=phy.phy_resampled_vals[0],
    #     bounds=(phy.phy_resampled_vals[0], phy.phy_resampled_vals[-1]),
    # )
    # phy_units = param.ObjectSelector(
    #     default=phy.phy_unit_vals[0], objects=phy.phy_unit_vals, label="Units"
    # )
    # # phy_plots_sc        = param.Boolean(default=False, label="SC")
    # phy_plots_sc_vals = param.ObjectSelector(
    #     default=next(iter(phy.phy_plots_sc_vals_dict)),
    #     objects=list(phy.phy_plots_sc_vals_dict),
    #     label="SC Values",
    # )

    # sipm plots
    # sipm_plots_barrels    = ['InnerBarrel-Top', 'InnerBarrel-Bottom', 'OuterBarrel-Top', 'OuterBarrel-Bottom']
    sipm_plot_style_dict = {
        "Time": spm.sipm_plot_vsTime,
        "Histogram": spm.sipm_plot_histogram,
    }
    sipm_resampled_vals = [1, 5, 10, 30, 60]

    sipm_sort_dict = ["Barrel"]
    sipm_sort_by = param.ObjectSelector(
        default=next(iter(sipm_sort_dict)), objects=list(sipm_sort_dict)
    )

    sipm_barrel = param.ObjectSelector(default=0, objects=[0])
    sipm_resampled = param.Integer(
        default=sipm_resampled_vals[0],
        bounds=(sipm_resampled_vals[0], sipm_resampled_vals[-1]),
    )
    sipm_plot_style = param.ObjectSelector(
        default=next(iter(sipm_plot_style_dict)),
        objects=list(sipm_plot_style_dict),
    )

    # muon plots
    muon_plots_cal_dict = {
        "Cal. Spectra": muon.muon_plot_spectra,
        "Cal. SPP Sigma": muon.muon_plot_spp,
        "Cal. SPP Shift": muon.muon_plot_calshift,
    }
    muon_plots_cal = param.ObjectSelector(
        default=next(iter(muon_plots_cal_dict)), objects=list(muon_plots_cal_dict)
    )

    muon_plots_mon_dict = {
        "Integral Light": muon.muon_plot_intlight,
        "Total Rates/H": muon.muon_plot_totalRates_hourly,
        "Total Rates/D": muon.muon_plot_totalRates_daily,
        "Pillbox Rates": muon.muon_plot_ratesPillBox,
        "Floor Rates": muon.muon_plot_ratesFloor,
        "Wall Rates": muon.muon_plot_ratesWall,
    }
    muon_plots_mon = param.ObjectSelector(
        default=next(iter(muon_plots_mon_dict)), objects=list(muon_plots_mon_dict)
    )

    # visualization
    meta_visu_plots_dict = {
        "Usability": visu.plot_visu_usability,
        "Processable": visu.plot_visu_processable,
        "Mass": visu.plot_visu_mass,
        "Depl. Voltage": visu.plot_visu_depletion,
        "Oper. Voltage": visu.plot_visu_operation,
        "Enrichment": visu.plot_visu_enrichment,
    }

    meta_visu_plots = param.ObjectSelector(
        default=next(iter(meta_visu_plots_dict)), objects=list(meta_visu_plots_dict)
    )

    # downloads
    plot_types_download_dict = ["FWHM Qbb", "FWHM FEP", "A/E", "Tau", "Alpha"]
    plot_types_download = param.Selector(
        default=plot_types_download_dict[0], objects=plot_types_download_dict
    )

    def __init__(
        self, cal_path, phy_path, sipm_path, muon_path, llama_path, tmp_path, name=None
    ):
        super().__init__(name=name, path=cal_path)

        # self.path = cal_path
        self.phy_path = phy_path
        self.sipm_path = sipm_path
        self.muon_path = muon_path
        self.llama_path = llama_path
        self.tmp_path = tmp_path
        # self.cached_plots = {}

        # self.startup_bool = True

        self._phy_sc_plotted = False

        # prod_config = Path(self.path) / "dataflow-config.yaml"
        # self.prod_config = Props.read_from(prod_config, subst_pathvar=True)

        # self.periods = gen_run_dict(self.path)
        # self.param["period"].objects = list(self.periods)
        # self.period = list(self.periods)[-1]
        # self.period = 'p04'

        # create initial dataframes
        self.phy_channels = []
        self.phy_data_df = pd.DataFrame()
        self.phy_data_df_mean = pd.DataFrame()
        self.phy_abs_unit = ""
        self.phy_plot_info = None
        self.phy_data_sc = pd.DataFrame()
        self.phy_pane = pn.pane.Bokeh(
            figure(width=1000, height=600), sizing_mode="scale_width"
        )

        self.muon_data_dict = {}

        self.sipm_data_df = pd.DataFrame()

        self.meta_df = pd.DataFrame()
        self.meta_visu_source = ColumnDataSource({})
        self.meta_visu_xlabels = {}
        self.meta_visu_chan_dict = {}
        self.meta_visu_channel_map = {}

        # get available periods and runs
        self._get_period_data()
        self._get_sipm_data()

    # @param.depends("period", watch=True)
    # def _get_period_data(self):
    #     if self.startup_bool:
    #         log.debug("Startup procedure, skip _get_period_data")
    #         self.startup_bool = False
    #     else:
    #         self.run_dict = self.periods[self.period]

    #         self.param["run"].objects = list(self.run_dict)
    #         if self.run == list(self.run_dict)[-1]:
    #             self.run = next(iter(self.run_dict))
    #         else:
    #             self.run = list(self.run_dict)[-1]

    #         start_period = sorted(self.periods)[0]
    #         start_run = sorted(self.periods[start_period])[0]
    #         end_period = sorted(self.periods)[-1]
    #         end_run = sorted(self.periods[end_period])[-1]

    #         self.param["date_range"].bounds = (
    #             datetime.strptime(
    #                 self.periods[start_period][start_run]["timestamp"], "%Y%m%dT%H%M%SZ"
    #             )
    #             - dtt.timedelta(minutes=100),
    #             datetime.strptime(
    #                 self.periods[end_period][end_run]["timestamp"], "%Y%m%dT%H%M%SZ"
    #             )
    #             + dtt.timedelta(minutes=110),
    #         )
    #         self.date_range = (
    #             datetime.strptime(
    #                 self.periods[start_period][start_run]["timestamp"], "%Y%m%dT%H%M%SZ"
    #             )
    #             - dtt.timedelta(minutes=100),
    #             datetime.strptime(
    #                 self.periods[end_period][end_run]["timestamp"], "%Y%m%dT%H%M%SZ"
    #             )
    #             + dtt.timedelta(minutes=110),
    #         )

    @param.depends("run", watch=True)
    def _get_muon_data(self):
        start_time = time.time()
        data_file = f"{self.muon_path}/generated/plt/phy/{self.period}/dsp/{self.run}/dashboard_period_{self.period}_run_{self.run}.shelve"
        if not (Path(data_file) / ".dat").exists():
            self.muon_data_dict = {}
        else:
            with shelve.open(data_file, "r") as f:
                # Create an empty dictionary
                arrays_dict = {}

                for key in f:
                    # Add a new key-value pair to the dictionary
                    arrays_dict[key] = np.array(f[key])

                self.muon_data_dict = arrays_dict
        log.debug("Time to get muon data:", extra={"time": time.time() - start_time})

    # @param.depends("date_range", watch=True)
    # def _get_run_dict(self):
    #     start_time = time.time()
    #     valid_from = [
    #         datetime.timestamp(
    #             datetime.strptime(self.run_dict[entry]["timestamp"], "%Y%m%dT%H%M%SZ")
    #         )
    #         for entry in self.run_dict
    #     ]
    #     if isinstance(self.date_range[0], date):
    #         low_range = datetime.timestamp(
    #             datetime.combine(self.date_range[0], datetime.min.time())
    #         )
    #     else:
    #         low_range = datetime.timestamp(self.date_range[0])
    #     if isinstance(self.date_range[0], date):
    #         high_range = datetime.timestamp(
    #             datetime.combine(self.date_range[1], datetime.max.time())
    #         )
    #     else:
    #         high_range = datetime.timestamp(self.date_range[1])
    #     pos1 = bisect.bisect_right(valid_from, low_range)
    #     pos2 = bisect.bisect_left(valid_from, high_range)
    #     pos1 = max(pos1, 0)
    #     pos2 = min(len(self.run_dict), pos2)
    #     valid_idxs = np.arange(pos1, pos2, 1)
    #     valid_keys = np.array(list(self.run_dict))[valid_idxs]
    #     out_dict = {key: self.run_dict[key] for key in valid_keys}
    #     log.debug("Time to get run dict:", extra={"time": time.time() - start_time})
    #     return out_dict

    @param.depends("run", watch=True)
    def _get_metadata(self):
        start_time = time.time()
        try:
            chan_dict, channel_map = self.chan_dict, self.channel_map

            df_chan_dict = pd.DataFrame.from_dict(chan_dict).T
            df_chan_dict.index.name = "name"
            df_chan_dict = df_chan_dict.reset_index()

            df_channel_map = pd.DataFrame.from_dict(channel_map).T
            df_channel_map = df_channel_map[df_channel_map["system"] == "geds"]

            df_out = df_channel_map.merge(df_chan_dict, left_on="name", right_on="name")
            df_out = df_out.reset_index().set_index("name")[
                [
                    "processable",
                    "usability",
                    "daq",
                    "location",
                    "voltage",
                    "electronics",
                    "characterization",
                    "production",
                    "type",
                ]
            ]
            df_out["daq"] = df_out["daq"].map(
                lambda x: "Crate: {}, Card: {}".format(x["crate"], x["card"]["id"])
            )
            df_out["location"] = df_out["location"].map(
                lambda x: "String: {:>02d}, Pos.: {:>02d}".format(
                    x["string"], x["position"]
                )
            )
            df_out["voltage"] = df_out["voltage"].map(
                lambda x: "Card: {:>02d}, Ch.: {:>02d}".format(
                    x["card"]["id"], x["channel"]
                )
            )
            df_out["electronics"] = df_out["electronics"].map(
                lambda x: "CC4: {}, Ch.: {:>02d}".format(
                    x["cc4"]["id"], x["cc4"]["channel"]
                )
            )
            df_out["usability"] = df_out["usability"].map(lambda x: x == "on")
            # df_out['processable'] =  df_out['processable'].map(lambda x: True if x == 'True' else False)
            df_out["Depl. Vol. (kV)"] = (
                df_out["characterization"].map(
                    lambda x: get_characterization(x, "depletion_voltage_in_V")
                )
                / 1000
            )
            df_out["Oper. Vol. (kV)"] = (
                df_out["characterization"].map(
                    lambda x: get_characterization(x, "recommended_voltage_in_V")
                )
                / 1000
            )
            df_out["Manufacturer"] = df_out["production"].map(
                lambda x: get_production(x, "manufacturer")
            )
            df_out["Mass (kg)"] = (
                df_out["production"].map(lambda x: get_production(x, "mass_in_g"))
                / 1000
            )
            df_out["Order"] = df_out["production"].map(
                lambda x: get_production(x, "order")
            )
            df_out["Crystal"] = df_out["production"].map(
                lambda x: get_production(x, "crystal")
            )
            df_out["Slice"] = df_out["production"].map(
                lambda x: get_production(x, "slice")
            )
            df_out["Enrichment (%)"] = (
                df_out["production"].map(lambda x: get_production(x, "enrichment"))
                * 100
            )
            df_out["Delivery"] = df_out["production"].map(
                lambda x: get_production(x, "delivered")
            )
            df_out = (
                df_out.reset_index()
                .rename(
                    {
                        "name": "Det. Name",
                        "processable": "Proc.",
                        "usability": "Usabl.",
                        "daq": "FC card",
                        "location": "Det. Location",
                        "voltage": "HV",
                        "electronics": "Electronics",
                        "type": "Type",
                    },
                    axis=1,
                )
                .set_index("Det. Name")
            )
            df_out = df_out.drop(["characterization", "production"], axis=1)
            df_out = df_out.astype({"Proc.": "bool", "Usabl.": "bool"})
            self.meta_df = df_out

            # get metadata visu plot data
            # strings_dict, chan_dict, channel_map = sorter(self.path, self.run_dict[self.run]["timestamp"], key="String")
            # self.meta_visu_source, self.meta_visu_xlabels = get_plot_source_and_xlabels(chan_dict, channel_map, strings_dict)
            # self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
        except KeyError:
            pass
        log.debug("Time to get metadata:", extra={"time": time.time() - start_time})

    # @param.depends("period", "date_range", "plot_type_tracking", "string", "sort_by")
    # def view_tracking(self):
    #     figure = None
    #     if self.plot_type_tracking != "Energy Residuals":
    #         figure = trac.plot_tracking(
    #             self._get_run_dict(),
    #             self.path,
    #             self.plot_types_tracking_dict[self.plot_type_tracking],
    #             self.string,
    #             self.period,
    #             self.plot_type_tracking,
    #             key=self.sort_by,
    #         )
    #     else:
    #         figure = trac.plot_energy_residuals_period(
    #             self._get_run_dict(), self.path, self.period, key=self.sort_by
    #         )
    #     return figure

    @param.depends("run", "muon_plots_cal")
    def view_muon_cal(self):
        start_time = time.time()
        if not bool(self.muon_data_dict):
            p = figure(width=1000, height=600)
            p.title.text = f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            log.debug(
                "Time to get muon cal plot:", extra={"time": time.time() - start_time}
            )
            return p

        if self.muon_plots_cal == "Cal. SPP Shift":
            data_file = f"{self.muon_path}/generated/plt/phy/{self.period}/dsp/{self.run}/dashboard_period_{self.period}_run_{self.run}.shelve"
            with shelve.open(data_file, "r") as f:
                # x_data_str = np.array(list(f['date'].values()))
                x_data_str = np.array(list(f["date"].values()))
                # y_data = np.array(list(f['mean_shift'].values()))
                y_data = np.array(list(f["mean_shift"].values()))

                # Reshape the x_data and y_data arrays
                x_data = np.array(
                    [
                        [
                            dtt.datetime.strptime(date_str, "%Y_%m_%d")
                            for date_str in row
                        ]
                        for row in x_data_str
                    ]
                )

                p = self.muon_plots_cal_dict[self.muon_plots_cal](
                    x_data,
                    y_data,
                    self.run,
                    self.period,
                    self.run_dict[self.run],
                    self.muon_plots_cal,
                )
                log.debug(
                    "Time to get muon cal plot:",
                    extra={"time": time.time() - start_time},
                )
                return p
        else:
            p = self.muon_plots_cal_dict[self.muon_plots_cal](
                self.muon_data_dict,
                self.run,
                self.period,
                self.run_dict[self.run],
                self.muon_plots_cal,
            )
            log.debug(
                "Time to get muon cal plot:", extra={"time": time.time() - start_time}
            )
            return p

    @param.depends("run", "muon_plots_mon")
    def view_muon_mon(self):
        start_time = time.time()
        if not bool(self.muon_data_dict):
            p = figure(width=1000, height=600)
            p.title.text = f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            log.debug(
                "Time to get muon mon plot:", extra={"time": time.time() - start_time}
            )
            return p
        if self.muon_plots_mon == "Integral Light":
            p = pn.pane.Matplotlib(
                self.muon_plots_mon_dict[self.muon_plots_mon](
                    self.muon_data_dict, self.period, self.run, self.run_dict[self.run]
                ),
                sizing_mode="scale_width",
            )
            log.debug(
                "Time to get muon mon plot:", extra={"time": time.time() - start_time}
            )
            return p
        p = self.muon_plots_mon_dict[self.muon_plots_mon](
            self.muon_data_dict, self.period, self.run, self.run_dict[self.run]
        )
        log.debug(
            "Time to get muon mon plot:", extra={"time": time.time() - start_time}
        )
        return p

    # @param.depends("sort_by", watch=True)
    # def update_strings(self):
    #     start_time = time.time()
    #     self.strings_dict, self.chan_dict, self.channel_map = sorter(
    #         self.path, self.run_dict[self.run]["timestamp"], key=self.sort_by
    #     )

    #     self.param["string"].objects = list(self.strings_dict)
    #     self.string = f"{next(iter(self.strings_dict))}"
    #     log.debug("Time to update strings:", extra={"time": time.time() - start_time})

    @param.depends("sipm_sort_by", watch=True)
    def update_barrels(self):
        start_time = time.time()
        self.sipm_out_dict, self.sipm_chmap = sorter(
            self.path,
            self.run_dict[self.run]["timestamp"],
            key=self.sipm_sort_by,
            spms=True,
        )

        self.param["sipm_barrel"].objects = list(self.sipm_out_dict)
        self.sipm_barrel = f"{next(iter(self.sipm_out_dict))}"

        log.debug("Time to update barrels:", extra={"time": time.time() - start_time})

    @param.depends("run", watch=True)
    def _get_sipm_data(self):
        start_time = time.time()
        data_file = self.sipm_path + f"{self.period}_{self.run}_spmmon.hdf"
        if not Path(data_file).exists():
            self.sipm_data_df = pd.DataFrame()
        else:
            self.sipm_data_df = (
                pd.read_hdf(data_file)
                .reset_index()
                .set_index("time")
                .drop(["index"], axis=1)
            )
            self.sipm_data_df.index = pd.to_datetime(
                self.sipm_data_df.index, unit="s", origin="unix"
            )

        self.sipm_out_dict, self.sipm_chmap = sorter(
            self.path,
            self.run_dict[self.run]["timestamp"],
            key=self.sipm_sort_by,
            spms=True,
        )
        self.sipm_name_dict = {}
        for val in self.sipm_chmap.values():
            self.sipm_name_dict[val["daq"]["rawid"]] = val["name"]
        self.update_barrels()
        log.debug("Time to get sipm data:", extra={"time": time.time() - start_time})

    # @param.depends("run", "sort_by", "plot_types_download")
    # def download_summary_files(self):
    #     start_time = time.time()
    #     download_file, download_filename = self.plot_types_summary_dict[
    #         self.plot_types_download
    #     ](
    #         self.run,
    #         self.run_dict[self.run],
    #         self.path,
    #         self.period,
    #         key=self.sort_by,
    #         download=True,
    #     )
    #     # log.debug(download_filename)
    #     if not (Path(self.tmp_path) / download_filename).exists():
    #         download_file.to_csv(self.tmp_path + download_filename, index=False)
    #         log.debug(download_file, self.tmp_path)
    #     ret = pn.widgets.FileDownload(
    #         self.tmp_path + download_filename,
    #         filename=download_filename,
    #         button_type="success",
    #         embed=False,
    #         name="Click to download 'csv'",
    #         width=350,
    #     )
    #     log.debug(
    #         "Time to download summary files:", extra={"time": time.time() - start_time}
    #     )
    #     return ret

    @param.depends(
        "run", "sipm_sort_by", "sipm_resampled", "sipm_barrel", "sipm_plot_style"
    )
    def view_sipm(self):
        start_time = time.time()
        if self.sipm_data_df.empty:
            p = figure(width=1000, height=600)
            p.title.text = f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
            p.title.align = "center"
            p.title.text_font_size = "25px"
            log.debug(
                "Time to get sipm plot:", extra={"time": time.time() - start_time}
            )
            return p
        data_barrel = self.sipm_data_df[
            [
                f"ch{channel}"
                for channel in self.sipm_out_dict[self.sipm_barrel]
                if f"ch{channel}" in self.sipm_data_df.columns
            ]
        ]
        p = self.sipm_plot_style_dict[self.sipm_plot_style](
            data_barrel,
            self.sipm_barrel,
            f"{self.sipm_resampled}min",
            self.sipm_name_dict,
            self.run,
            self.period,
            self.run_dict[self.run],
        )
        log.debug("Time to get sipm plot:", extra={"time": time.time() - start_time})
        return p

    # @param.depends("run", "sort_by", "plot_type_summary", "string")
    # def view_summary(self):
    #     start_time = time.time()
    #     figure = None
    #     if self.plot_type_summary in [
    #         "FWHM Qbb",
    #         "FWHM FEP",
    #         "Energy Residuals",
    #         "A/E Status",
    #         "Tau",
    #         "CT Alpha",
    #         "Valid. E",
    #         "Valid. A/E",
    #         "A/E SF",
    #     ]:
    #         figure = self.plot_types_summary_dict[self.plot_type_summary](
    #             self.run,
    #             self.run_dict[self.run],
    #             self.path,
    #             self.period,
    #             key=self.sort_by,
    #         )

    #     elif self.plot_type_summary in ["Detector Status", "FEP Counts"]:
    #         # elif self.plot_type_summary in ["Detector Status"]:
    #         strings_dict, meta_visu_chan_dict, meta_visu_channel_map = sorter(
    #             self.path, self.run_dict[self.run]["timestamp"], key="String"
    #         )
    #         meta_visu_source, meta_visu_xlabels = visu.get_plot_source_and_xlabels(
    #             meta_visu_chan_dict, meta_visu_channel_map, strings_dict
    #         )
    #         # self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
    #         figure = self.plot_types_summary_dict[self.plot_type_summary](
    #             self.run,
    #             self.run_dict[self.run],
    #             self.path,
    #             meta_visu_source,
    #             meta_visu_xlabels,
    #             self.period,
    #             key=self.sort_by,
    #         )
    #     elif self.plot_type_summary in [
    #         "Baseline Spectrum",
    #         "Energy Spectrum",
    #         "Baseline Stability",
    #         "FEP Stability",
    #         "Pulser Stability",
    #     ]:
    #         figure = self.plot_types_summary_dict[self.plot_type_summary](
    #             self.common_dict,
    #             self.channel_map,
    #             self.strings_dict[self.string],
    #             self.string,
    #             self.run,
    #             self.period,
    #             self.run_dict[self.run],
    #             key=self.sort_by,
    #         )
    #     else:
    #         figure = figure()

    #     log.debug("Time to get summary plot:", extra={"time": time.time() - start_time})
    #     return figure

    # @param.depends("run", "string", "sort_by", "phy_plots_types", "phy_plots", "phy_resampled", "phy_units", "phy_plots_sc_vals", watch=True)
    # def _get_phy_data(self):
    #     start_time = time.time()
    #     data_file     = self.phy_path +  f'/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-geds.hdf'
    #     data_file_sc  = self.phy_path +  f'/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-slow_control.hdf'

    #     if not os.path.exists(data_file):
    #         log.debug(f"Time to get phy data: {time.time()-start_time}")
    #         self.phy_data_df = []
    #         return
    #     log.debug(1)
    #     # get filekeys to check if key exists
    #     with h5py.File(data_file, 'r') as f:
    #         filekeys = list(f.keys())
    #     log.debug(2)
    #     # load plot info for current plot value and get all data from selected string
    #     phy_data_key     = f"{self.phy_plots_types_dict[self.phy_plots_types]}_{self.phy_plots_vals_dict[self.phy_plots]}"
    #     if "pulser" in phy_data_key:
    #         if f"{phy_data_key.split('_pulser')[0]}_info" not in filekeys:
    #             self.phy_data_df = pd.DataFrame()
    #             log.debug(f"Time to get phy data: {time.time()-start_time}")
    #             return
    #         self.phy_plot_info           = pd.read_hdf(data_file, key=f"{phy_data_key.split('_pulser')[0]}_info")
    #         if "Diff" in phy_data_key:
    #             self.phy_plot_info.loc["label"][0] = "Gain to Pulser Difference"
    #         else:
    #             self.phy_plot_info.loc["label"][0] = "Gain to Pulser Ratio"
    #     else:
    #         if f"{phy_data_key}_info" not in filekeys:
    #             self.phy_data_df = pd.DataFrame()
    #             log.debug(f"Time to get phy data: {time.time()-start_time}")
    #             return
    #         self.phy_plot_info           = pd.read_hdf(data_file, key=f"{phy_data_key}_info")
    #     log.debug(3)

    #     # self.phy_abs_unit = self.phy_plot_info.loc["unit"][0]

    #     # load dataframe for current plot value and get all data from selected string
    #     if self.phy_units == "Relative":
    #         if f"{phy_data_key}_var" not in filekeys:
    #             self.phy_data_df = pd.DataFrame()
    #             log.debug(f"Time to get phy data: {time.time()-start_time}")
    #             return
    #         self.phy_data_df                    = pd.read_hdf(data_file, key=f"{phy_data_key}_var")
    #         self.phy_plot_info.loc["unit"][0]   = "%"
    #     else:
    #         if phy_data_key not in filekeys:
    #             self.phy_data_df = pd.DataFrame()
    #             log.debug(f"Time to get phy data: {time.time()-start_time}")
    #             return
    #         self.phy_data_df = pd.read_hdf(data_file, key=phy_data_key)
    #     log.debug(4)

    #     # load mean values
    #     if f"{phy_data_key}_mean" not in filekeys:
    #         self.phy_data_df = pd.DataFrame()
    #         log.debug(f"Time to get phy data: {time.time()-start_time}")
    #         return
    #     self.phy_data_df_mean = pd.read_hdf(data_file, key=f"{phy_data_key}_mean")
    #     log.debug(5)

    #     # get sc data if selected
    #     # if self.phy_plots_sc and self.phy_units == "Relative" and os.path.exists(data_file_sc):
    #     if self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals] and os.path.exists(data_file_sc):
    #         self.data_sc = pd.read_hdf(data_file_sc, self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals])
    #         self._phy_sc_plotted = True
    #     else:
    #         self.data_sc = pd.DataFrame()
    #         self._phy_sc_plotted = False
    #     return

    # @param.depends("run", "string", "sort_by", "phy_plots_types", "phy_plots", "phy_plot_style", "phy_resampled", "phy_units", "phy_plots_sc_vals", watch=True)
    # def _get_phy_plot(self):
    #     # Create empty plot inc ase of errors
    #     start_time = time.time()
    #     p = figure(width=1000, height=600)
    #     p.title.text = title=f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
    #     p.title.align = "center"
    #     p.title.text_font_size = "25px"

    #     # return empty plot if no data exists for run
    #     if self.phy_data_df.empty:
    #         log.debug(f"Time to get phy plot: {time.time()-start_time}")
    #         self.phy_pane.object = p
    #     else:
    #         channels = self.strings_dict[self.string]
    #         # check if channel selection actually exists in data
    #         channels            = [ch for ch in channels if ch in self.phy_data_df.columns and ch in self.phy_data_df_mean.columns]
    #         phy_data_df         = self.phy_data_df[channels]
    #         phy_data_df_mean    = self.phy_data_df_mean[channels]

    #         # plot data
    #         p = self.phy_plot_style_dict[self.phy_plot_style](phy_data_df, phy_data_df_mean, self.phy_plot_info, self.phy_plots_types, self.phy_plots,f"{self.phy_resampled}min", self.string, self.run, self.period, self.run_dict[self.run], self.channel_map, self.phy_abs_unit, self.phy_data_sc, self.phy_plots_sc_vals)
    #         log.debug(f"Time to get phy plot: {time.time()-start_time}")
    #         self.phy_pane.object = p

    #     # self.phy_channels = self.strings_dict[self.string]

    # @param.depends("run", "string", "sort_by", "phy_plots_types", "phy_plots", "phy_plot_style", "phy_resampled", "phy_units", "phy_plots_sc_vals")
    # def view_phy(self):
    #     return self.phy_pane

    # @pn.io.profile('clustering', engine='pyinstrument')
    # @param.depends(
    #     "run",
    #     "string",
    #     "sort_by",
    #     "phy_plots_types",
    #     "phy_plots",
    #     "phy_plot_style",
    #     "phy_resampled",
    #     "phy_units",
    #     "phy_plots_sc_vals",
    # )
    # def view_phy(self):
    #     start_time = time.time()
    #     data_file = (
    #         self.phy_path
    #         + f"/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-geds.hdf"
    #     )
    #     data_file_sc = (
    #         self.phy_path
    #         + f"/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-slow_control.hdf"
    #     )

    #     # Create empty plot inc ase of errors
    #     p = figure(width=1000, height=600)
    #     p.title.text = f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
    #     p.title.align = "center"
    #     p.title.text_font_size = "25px"

    #     # return empty plot if no data exists for run
    #     if not Path(data_file).exists():
    #         log.debug("Time to get phy plot:", extra={"time": time.time() - start_time})
    #         return p

    #     # get filekeys to check if key exists
    #     with h5py.File(data_file, "r") as f:
    #         filekeys = list(f.keys())

    #     # load dataframe for current plot value and get all data from selected string
    #     channels = self.strings_dict[self.string]
    #     phy_data_key = f"{self.phy_plots_types_dict[self.phy_plots_types]}_{self.phy_plots_vals_dict[self.phy_plots]}"
    #     if "pulser" in phy_data_key:
    #         if f"{phy_data_key.split('_pulser')[0]}_info" not in filekeys:
    #             return p
    #         phy_plot_info = pd.read_hdf(
    #             data_file, key=f"{phy_data_key.split('_pulser')[0]}_info"
    #         )
    #         if "Diff" in phy_data_key:
    #             phy_plot_info.loc["label"][0] = "Gain to Pulser Difference"
    #         else:
    #             phy_plot_info.loc["label"][0] = "Gain to Pulser Ratio"
    #     else:
    #         if f"{phy_data_key}_info" not in filekeys:
    #             return p
    #         phy_plot_info = pd.read_hdf(data_file, key=f"{phy_data_key}_info")
    #     abs_unit = phy_plot_info.loc["unit"][0]

    #     if self.phy_units == "Relative":
    #         if f"{phy_data_key}_var" not in filekeys:
    #             return p
    #         phy_data_df = pd.read_hdf(data_file, key=f"{phy_data_key}_var")
    #         phy_plot_info.loc["unit"][0] = "%"
    #     else:
    #         if phy_data_key not in filekeys:
    #             return p
    #         phy_data_df = pd.read_hdf(data_file, key=phy_data_key)

    #     # load mean values
    #     if f"{phy_data_key}_mean" not in filekeys:
    #         return p
    #     phy_data_df_mean = pd.read_hdf(data_file, key=f"{phy_data_key}_mean")

    #     # get sc data if selected
    #     # if self.phy_plots_sc and self.phy_units == "Relative" and os.path.exists(data_file_sc):
    #     if (
    #         self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
    #         and Path(data_file_sc).exists()
    #     ):
    #         data_sc = pd.read_hdf(
    #             data_file_sc, self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
    #         )
    #         self._phy_sc_plotted = True
    #     else:
    #         data_sc = pd.DataFrame()
    #         self._phy_sc_plotted = False
    #     # check if channel selection actually exists in data
    #     channels = [
    #         ch
    #         for ch in channels
    #         if ch in phy_data_df.columns and ch in phy_data_df_mean.columns
    #     ]
    #     phy_data_df = phy_data_df[channels]
    #     phy_data_df_mean = phy_data_df_mean[channels]

    #     # plot data
    #     p = self.phy_plot_style_dict[self.phy_plot_style](
    #         phy_data_df,
    #         phy_data_df_mean,
    #         phy_plot_info,
    #         self.phy_plots_types,
    #         self.phy_plots,
    #         f"{self.phy_resampled}min",
    #         self.string,
    #         self.run,
    #         self.period,
    #         self.run_dict[self.run],
    #         self.channel_map,
    #         abs_unit,
    #         data_sc,
    #         self.phy_plots_sc_vals,
    #     )
    #     log.debug("Time to get phy plot:", extra={"time": time.time() - start_time})
    #     # self.bokeh_pane.object = p
    #     return p

    # @param.depends("run", watch=True)
    # def update_plot_dict(self):
    #     start_time = time.time()
    #     self.plot_dict = (
    #         Path(self.prod_config["paths"]["plt"])
    #         / f"hit/cal/{self.period}/{self.run}"
    #         / f'{self.run_dict[self.run]["experiment"]}-{self.period}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit'
    #     )

    #     # log.debug(self.run_dict)
    #     # log.debug(self.plot_dict)
    #     with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
    #         channels = list(shelf.keys())

    #     with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
    #         self.common_dict = shelf["common"]
    #     channels.remove("common")
    #     self.strings_dict, self.chan_dict, self.channel_map = sorter(
    #         self.path, self.run_dict[self.run]["timestamp"], "String"
    #     )
    #     channel_list = []
    #     for channel in channels:
    #         channel_list.append(
    #             f"{channel}: {self.channel_map[int(channel[2:])]['name']}"
    #         )

    #     self.param["channel"].objects = channel_list
    #     self.channel = channel_list[0]

    #     self.update_strings()
    #     self.update_channel_plot_dict()
    #     log.debug("Time to update plot dict:", extra={"time": time.time() - start_time})

    # @param.depends("channel", watch=True)
    # def update_channel_plot_dict(self):
    #     start_time = time.time()
    #     log.debug(self.channel)
    #     with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
    #         self.plot_dict_ch = shelf[self.channel[:9]]
    #     with shelve.open(
    #         self.plot_dict.replace("hit", "dsp"), "r", protocol=pkl.HIGHEST_PROTOCOL
    #     ) as shelf:
    #         self.dsp_dict = shelf[self.channel[:9]]
    #     log.debug(
    #         "Time to update channel plot dict:",
    #         extra={"time": time.time() - start_time},
    #     )

    # @param.depends("parameter", watch=True)
    # def update_plot_type_details(self):
    #     start_time = time.time()
    #     plots = self._options[self.parameter]
    #     self.param["plot_type_details"].objects = plots
    #     self.plot_type_details = plots[0]
    #     log.debug(
    #         "Time to update plot type details:",
    #         extra={"time": time.time() - start_time},
    #     )

    # @param.depends("run", "channel", "parameter", "plot_type_details")
    # def view_details(self):
    #     if self.parameter == "A/E":
    #         fig = self.plot_dict_ch["aoe"][self.plot_type_details]
    #         dummy = plt.figure()
    #         new_manager = dummy.canvas.manager
    #         new_manager.canvas.figure = fig
    #         fig.set_canvas(new_manager.canvas)
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.parameter == "Baseline":
    #         fig = self.plot_dict_ch["ecal"][self.plot_type_details]
    #         dummy = plt.figure()
    #         new_manager = dummy.canvas.manager
    #         new_manager.canvas.figure = fig
    #         fig.set_canvas(new_manager.canvas)
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.parameter == "Tau":
    #         fig = self.dsp_dict["tau"][self.plot_type_details]
    #         dummy = plt.figure()
    #         new_manager = dummy.canvas.manager
    #         new_manager.canvas.figure = fig
    #         fig.set_canvas(new_manager.canvas)
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.parameter == "Optimisation":
    #         fig = self.dsp_dict[f"{self.plot_type_details.split('_')[0]}_optimisation"][
    #             f"{self.plot_type_details.split('_')[1]}_space"
    #         ]
    #         dummy = plt.figure()
    #         new_manager = dummy.canvas.manager
    #         new_manager.canvas.figure = fig
    #         fig.set_canvas(new_manager.canvas)
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.plot_type_details in {"spectrum", "logged_spectrum"}:
    #         fig = depl.plot_spectrum(
    #             self.plot_dict_ch["ecal"][self.parameter]["spectrum"],
    #             self.channel,
    #             log=self.plot_type_details != "spectrum",
    #         )
    #         fig_pane = fig
    #     elif self.plot_type_details == "survival_frac":
    #         fig = depl.plot_survival_frac(
    #             self.plot_dict_ch["ecal"][self.parameter]["survival_frac"]
    #         )
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.plot_type_details == "cut_spectrum":
    #         fig = depl.plot_cut_spectra(
    #             self.plot_dict_ch["ecal"][self.parameter]["spectrum"]
    #         )
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     elif self.plot_type_details == "peak_track":
    #         fig = depl.track_peaks(self.plot_dict_ch["ecal"][self.parameter])
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     else:
    #         fig = self.plot_dict_ch["ecal"][self.parameter][self.plot_type_details]
    #         dummy = plt.figure()
    #         new_manager = dummy.canvas.manager
    #         new_manager.canvas.figure = fig
    #         fig.set_canvas(new_manager.canvas)
    #         fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
    #     return fig_pane

    # @param.depends("run", "channel")
    # def get_RunAndChannel(self):
    #     start_time = time.time()
    #     ret = pn.pane.Markdown(
    #         f"### {self.run_dict[self.run]['experiment']}-{self.period}-{self.run} | Cal. Details | Channel {self.channel}"
    #     )
    #     log.debug(
    #         "Time to get run and channel:", extra={"time": time.time() - start_time}
    #     )
    #     return ret

    @param.depends("run")
    def view_meta(self):
        start_time = time.time()
        ret = pn.widgets.Tabulator(
            self.meta_df,
            formatters={"Proc.": BooleanFormatter(), "Usabl.": BooleanFormatter()},
            frozen_columns=[0],
        )
        log.debug("Time to get meta:", extra={"time": time.time() - start_time})
        return ret

    @param.depends("run", "meta_visu_plots")
    def view_meta_visu(self):
        start_time = time.time()
        strings_dict, meta_visu_chan_dict, meta_visu_channel_map = sorter(
            self.path, self.run_dict[self.run]["timestamp"], key="String"
        )
        meta_visu_source, meta_visu_xlabels = visu.get_plot_source_and_xlabels(
            meta_visu_chan_dict, meta_visu_channel_map, strings_dict
        )
        figure = None
        figure = self.meta_visu_plots_dict[self.meta_visu_plots](
            meta_visu_source,
            meta_visu_chan_dict,
            meta_visu_channel_map,
            meta_visu_xlabels,
        )
        log.debug("Time to get meta visu:", extra={"time": time.time() - start_time})
        return figure

    # def view_llama(self):
    #     start_time = time.time()
    #     try:
    #         llama_data = pd.read_csv(
    #             self.llama_path + "monivalues.txt",
    #             sep=r"\s+",
    #             dtype={"timestamp": np.int64},
    #             parse_dates=[1],
    #         )
    #         llama_data["timestamp"] = pd.to_datetime(
    #             llama_data["timestamp"], origin="unix", unit="s"
    #         )
    #         llama_data = llama_data.rename(
    #             columns={
    #                 "#run_no": "#Run",
    #                 "timestamp": "Timestamp",
    #                 "triplet_val": "Triplet Lifetime (µs)",
    #                 "triplet_err": "Error Triplet Lifetime (µs)",
    #                 "ly_val": "Light Yield",
    #                 "ly_err": "Error Light Yield",
    #             }
    #         )
    #     except KeyError:
    #         p = figure(width=1000, height=600)
    #         p.title.text = "No current Llama data available."
    #         p.title.align = "center"
    #         p.title.text_font_size = "25px"
    #         log.debug(
    #             "Time to get llama plot:", extra={"time": time.time() - start_time}
    #         )
    #         return p
    #     llama_width, llama_height = 1200, 400
    #     # add two hours to x values to convert from UTC to CET if values still in UTC
    #     if llama_data["Timestamp"][0].utcoffset() is None:
    #         llama_data["Timestamp"] += pd.Timedelta(hours=2)
    #     triplet_plot = hv.Scatter(
    #         llama_data, ["Timestamp", "Triplet Lifetime (µs)"], label="Triplet LT"
    #     )
    #     triplet_plot_error = hv.ErrorBars(
    #         llama_data,
    #         vdims=["Triplet Lifetime (µs)", "Error Triplet Lifetime (µs)"],
    #         kdims=["Timestamp"],
    #         label="Triplet LT Error",
    #     )
    #     triplet_plot.opts(
    #         xlabel="Time (CET)",
    #         ylabel="Triplet Lifetime (µs)",
    #         tools=["hover"],
    #         line_width=1.5,
    #         color="blue",
    #         width=llama_width,
    #         height=llama_height,
    #     )
    #     triplet_plot_error.opts(
    #         line_width=0.2, width=llama_width, height=llama_height, show_grid=True
    #     )

    #     lightyield_plot = hv.Scatter(
    #         llama_data, ["Timestamp", "Light Yield"], label="Light Yield"
    #     )
    #     lightyield_plot_error = hv.ErrorBars(
    #         llama_data,
    #         vdims=["Light Yield", "Error Light Yield"],
    #         kdims=["Timestamp"],
    #         label="Light Yield Error",
    #     )
    #     lightyield_plot.opts(
    #         xlabel="Time (CET)",
    #         ylabel="Light yield (a.u.)",
    #         tools=["hover"],
    #         line_width=1.5,
    #         color="orange",
    #         width=llama_width,
    #         height=llama_height,
    #         show_grid=True,
    #     )
    #     lightyield_plot_error.opts(
    #         line_width=0.2, width=llama_width, height=llama_height
    #     )

    #     layout = (
    #         triplet_plot * triplet_plot_error + lightyield_plot * lightyield_plot_error
    #     )
    #     layout.opts(width=llama_width, height=llama_height).cols(1)
    #     log.debug("Time to get llama plot:", extra={"time": time.time() - start_time})
    #     return layout

    # def get_llama_lastUpdate(self):
    #     start_time = time.time()
    #     llama_pathlib = Path(self.llama_path + "monivalues.txt")
    #     ret = "Last modified: {}".format(
    #         pd.to_datetime(llama_pathlib.stat().st_mtime, origin="unix", unit="s")
    #     )
    #     log.debug(
    #         "Time to get llama last update:", extra={"time": time.time() - start_time}
    #     )
    #     return ret
