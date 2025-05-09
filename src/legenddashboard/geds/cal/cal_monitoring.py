from __future__ import annotations

import logging
import pickle as pkl
import shelve
import time
from pathlib import Path

import matplotlib.pyplot as plt
import panel as pn
import param
from legenddashboard.base import Monitoring
import legenddashboard.geds.cal as cal
import legenddashboard.string_visulization as visu
from legenddashboard.util import (
    sort_dict,
    sorter,
)


log = logging.getLogger(__name__)

# calibration plots
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100


class CalMonitoring(Monitoring):

    channel = param.Selector(default=0, objects=[0])
    string = param.ObjectSelector(default=1, objects=[i+1 for i in range(11)], allow_refs=True, nested_refs=True)
    # general selectors
    sort_by = param.ObjectSelector(
        default=next(iter(sort_dict)), objects=list(sort_dict)
    )

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
    plot_types_download_dict = ["FWHM Qbb", "FWHM FEP", "A/E", "PZ", "Alpha"]
    plot_types_download = param.Selector(
        default=plot_types_download_dict[0], objects=plot_types_download_dict
    )

    @param.depends("run", "channel")
    def get_run_and_channel(self):
        start_time = time.time()
        ret = pn.pane.Markdown(
            f"### {self.run_dict[self.run]['experiment']}-{self.period}-{self.run} | Cal. Details | Channel {self.channel}"
        )
        log.debug(
            "Time to get run and channel:", extra={"time": time.time() - start_time}
        )
        return ret

    @param.depends("sort_by", watch=True)
    def update_strings(self):
        start_time = time.time()
        self.strings_dict, self.chan_dict, self.channel_map = sorter(
            self.base_path, self.run_dict[self.run]["timestamp"], key=self.sort_by
        )

        self.param["string"].objects = list(self.strings_dict)
        self.string = f"{next(iter(self.strings_dict))}"
        log.debug("Time to update strings:", extra={"time": time.time() - start_time})

    @param.depends("run", "sort_by", "plot_types_download")
    def download_summary_files(self):
        start_time = time.time()
        download_file, download_filename = self.plot_types_summary_dict[
            self.plot_types_download
        ](
            self.run,
            self.run_dict[self.run],
            self.base_path,
            self.period,
            key=self.sort_by,
            download=True,
        )
        # log.debug(download_filename)
        if not (Path(self.tmp_path) / download_filename).exists():
            download_file.to_csv(self.tmp_path + download_filename, index=False)
            log.debug(download_file, self.tmp_path)
        ret = pn.widgets.FileDownload(
            self.tmp_path + download_filename,
            filename=download_filename,
            button_type="success",
            embed=False,
            name="Click to download 'csv'",
            width=350,
        )
        log.debug(
            "Time to download summary files:", extra={"time": time.time() - start_time}
        )
        return ret
    
    @param.depends("run", "sort_by", "plot_type_summary", "string")
    def view_summary(self):
        start_time = time.time()
        figure = None
        if self.plot_type_summary in [
            "FWHM Qbb",
            "FWHM FEP",
            "Energy Residuals",
            "A/E Status",
            "PZ",
            "CT Alpha",
            "Valid. E",
            "Valid. A/E",
            "A/E SF",
        ]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](
                self.run,
                self.run_dict[self.run],
                self.base_path,
                self.period,
                key=self.sort_by,
            )

        elif self.plot_type_summary in ["Detector Status", "FEP Counts"]:
            # elif self.plot_type_summary in ["Detector Status"]:
            strings_dict, meta_visu_chan_dict, meta_visu_channel_map = sorter(
                self.base_path, self.run_dict[self.run]["timestamp"], key="String"
            )
            meta_visu_source, meta_visu_xlabels = visu.get_plot_source_and_xlabels(
                meta_visu_chan_dict, meta_visu_channel_map, strings_dict
            )
            # self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
            figure = self.plot_types_summary_dict[self.plot_type_summary](
                self.run,
                self.run_dict[self.run],
                self.base_path,
                meta_visu_source,
                meta_visu_xlabels,
                self.period,
                key=self.sort_by,
            )
        elif self.plot_type_summary in [
            "Baseline Spectrum",
            "Energy Spectrum",
            "Baseline Stability",
            "FEP Stability",
            "Pulser Stability",
        ]:
            figure = self.plot_types_summary_dict[self.plot_type_summary](
                self.common_dict,
                self.channel_map,
                self.strings_dict[self.string],
                self.string,
                self.run,
                self.period,
                self.run_dict[self.run],
                key=self.sort_by,
            )
        else:
            figure = figure()

        log.debug("Time to get summary plot:", extra={"time": time.time() - start_time})
        return figure


    @param.depends("period", "date_range", "plot_type_tracking", "string", "sort_by")
    def view_tracking(self):
        figure = None
        if self.plot_type_tracking != "Energy Residuals":
            figure = cal.plot_tracking(
                self._get_run_dict(),
                self.base_path,
                self.plot_types_tracking_dict[self.plot_type_tracking],
                self.string,
                self.period,
                self.plot_type_tracking,
                key=self.sort_by,
            )
        else:
            figure = cal.plot_energy_residuals_period(
                self._get_run_dict(), self.base_path, self.period, key=self.sort_by
            )
        return figure

    @param.depends("run", watch=True)
    def update_plot_dict(self):
        start_time = time.time()
        self.plot_dict = (
            Path(self.prod_config["paths"]["plt"])
            / f"hit/cal/{self.period}/{self.run}"
            / f'{self.run_dict[self.run]["experiment"]}-{self.period}-{self.run}-cal-{self.run_dict[self.run]["timestamp"]}-plt_hit'
        )

        # log.debug(self.run_dict)
        # log.debug(self.plot_dict)
        with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            channels = list(shelf.keys())

        with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.common_dict = shelf["common"]
        channels.remove("common")
        self.strings_dict, self.chan_dict, self.channel_map = sorter(
            self.base_path, self.run_dict[self.run]["timestamp"], "String"
        )

        self.param["channel"].objects = channels
        self.channel = channels[0]

        self.update_strings()
        self.update_channel_plot_dict()
        log.debug("Time to update plot dict:", extra={"time": time.time() - start_time})

    @param.depends("channel", watch=True)
    def update_channel_plot_dict(self):
        start_time = time.time()
        log.debug(self.channel)
        with shelve.open(self.plot_dict, "r", protocol=pkl.HIGHEST_PROTOCOL) as shelf:
            self.plot_dict_ch = shelf[self.channel[:9]]
        with shelve.open(
            str(self.plot_dict).replace("hit", "dsp"), "r", protocol=pkl.HIGHEST_PROTOCOL
        ) as shelf:
            self.dsp_dict = shelf[self.channel[:9]]
        log.debug(
            "Time to update channel plot dict:",
            extra={"time": time.time() - start_time},
        )

    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        start_time = time.time()
        plots = self._options[self.parameter]
        self.param["plot_type_details"].objects = plots
        self.plot_type_details = plots[0]
        log.debug(
            "Time to update plot type details:",
            extra={"time": time.time() - start_time},
        )

    @param.depends("run", "channel", "parameter", "plot_type_details")
    def view_details(self):
        if self.parameter == "A/E":
            fig = self.plot_dict_ch["aoe"][self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.parameter == "Baseline":
            fig = self.plot_dict_ch["ecal"][self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.parameter == "PZ":
            fig = self.dsp_dict["pz"][self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.parameter == "Optimisation":
            fig = self.dsp_dict[f"{self.plot_type_details.split('_')[0]}_optimisation"][
                f"{self.plot_type_details.split('_')[1]}_space"
            ]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.plot_type_details in {"spectrum", "logged_spectrum"}:
            fig = cal.plot_spectrum(
                self.plot_dict_ch["ecal"][self.parameter]["spectrum"],
                self.channel,
                log=self.plot_type_details != "spectrum",
            )
            fig_pane = fig
        elif self.plot_type_details == "survival_frac":
            fig = cal.plot_survival_frac(
                self.plot_dict_ch["ecal"][self.parameter]["survival_frac"]
            )
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.plot_type_details == "cut_spectrum":
            fig = cal.plot_cut_spectra(
                self.plot_dict_ch["ecal"][self.parameter]["spectrum"]
            )
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        elif self.plot_type_details == "peak_track":
            fig = cal.track_peaks(self.plot_dict_ch["ecal"][self.parameter])
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        else:
            fig = self.plot_dict_ch["ecal"][self.parameter][self.plot_type_details]
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig_pane = pn.pane.Matplotlib(fig, sizing_mode="scale_width")
        return fig_pane