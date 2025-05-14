from __future__ import annotations

import argparse
import logging
import pickle as pkl
import shelve
import time
from pathlib import Path

import matplotlib.pyplot as plt
import panel as pn
import param

import legenddashboard.geds.string_visulization as visu
from legenddashboard.geds import cal
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.util import logo_path, read_config, sorter

log = logging.getLogger(__name__)

# calibration plots
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100


class CalMonitoring(GedMonitoring):
    cached_data = param.Dict(default={"hit": {}, "dsp": {}})
    tmp_path = param.String("/tmp/")
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
    plot_types_download = param.Selector(
        objects=["FWHM Qbb", "FWHM FEP", "A/E", "PZ", "Alpha"],
        default="FWHM Qbb",
    )

    @param.depends("run", "sort_by", "plot_types_download")
    def download_summary_files(self):
        start_time = time.time()
        download_file, download_filename = cal.summary_plots[self.plot_types_download](
            self.prod_config,
            self.run,
            self.run_dict[self.run],
            self.base_path,
            self.period,
            key=self.sort_by,
            download=True,
            sort_dets_obj=self.sort_obj,
            cache_data=self.cached_data,
        )
        # log.debug(download_filename)
        tmp_path = Path(self.tmp_path)
        if not (tmp_path / download_filename).exists():
            download_file.to_csv(tmp_path / download_filename, index=False)
            log.debug(download_file, tmp_path)
        ret = pn.widgets.FileDownload(
            tmp_path / download_filename,
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
        if self.cached_data == {}:
            self.cached_data = {
                "hit": {},
                "dsp": {},
            }
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
            figure = cal.summary_plots[self.plot_type_summary](
                self.prod_config,
                self.run,
                self.run_dict[self.run],
                self.base_path,
                self.period,
                key=self.sort_by,
                sort_dets_obj=self.sort_obj,
                cache_data=self.cached_data,
            )

        elif self.plot_type_summary in ["Detector Status", "FEP Counts"]:
            # elif self.plot_type_summary in ["Detector Status"]:
            strings_dict, meta_visu_chan_dict, meta_visu_channel_map = sorter(
                self.base_path,
                self.run_dict[self.run]["timestamp"],
                key="String",
                sort_dets_obj=self.sort_obj,
            )
            meta_visu_source, meta_visu_xlabels = visu.get_plot_source_and_xlabels(
                meta_visu_chan_dict, meta_visu_channel_map, strings_dict
            )
            # self.meta_visu_chan_dict, self.meta_visu_channel_map = chan_dict, channel_map
            figure = cal.summary_plots[self.plot_type_summary](
                self.prod_config,
                self.run,
                self.run_dict[self.run],
                self.base_path,
                meta_visu_source,
                meta_visu_xlabels,
                self.period,
                key=self.sort_by,
                sort_dets_obj=self.sort_obj,
                cache_data=self.cached_data,
            )
        elif self.plot_type_summary in [
            "Baseline Spectrum",
            "Energy Spectrum",
            "Baseline Stability",
            "FEP Stability",
            "Pulser Stability",
        ]:
            figure = cal.summary_plots[self.plot_type_summary](
                self.prod_config,
                self.common_dict,
                self.channel_map,
                self.strings_dict[self.string],
                self.string,
                self.run,
                self.period,
                self.run_dict[self.run],
                key=self.sort_by,
                sort_dets_obj=self.sort_obj,
                cache_data=self.cached_data,
            )
        else:
            figure = plt.figure()

        log.debug("Time to get summary plot:", extra={"time": time.time() - start_time})
        return figure

    @param.depends("period")
    def _clear_cached_data(self):
        """
        Clear the cached data if the period changes.
        """
        self.cached_data = {}

    @param.depends("period", "date_range", "plot_type_tracking", "string", "sort_by")
    def view_tracking(self):
        figure = None
        if self.plot_type_tracking != "Energy Residuals":
            figure = cal.plot_tracking(
                self._get_run_dict(),
                self.base_path,
                cal.tracking_plots[self.plot_type_tracking],
                self.string,
                self.period,
                self.plot_type_tracking,
                key=self.sort_by,
                cache_data=self.cached_data,
                sort_dets_obj=self.sort_obj,
            )
        else:
            figure = cal.plot_energy_residuals_period(
                self._get_run_dict(),
                self.base_path,
                self.period,
                key=self.sort_by,
                cache_data=self.cached_data,
                sort_dets_obj=self.sort_obj,
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
            self.base_path,
            self.run_dict[self.run]["timestamp"],
            "String",
            sort_dets_obj=self.sort_obj,
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
            str(self.plot_dict).replace("hit", "dsp"),
            "r",
            protocol=pkl.HIGHEST_PROTOCOL,
        ) as shelf:
            self.dsp_dict = shelf[self.channel[:9]]
        log.debug(
            "Time to update channel plot dict:",
            extra={"time": time.time() - start_time},
        )

    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        start_time = time.time()
        plots = cal.all_detailed_plots[self.parameter]
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

    def build_detailed_pane(self, widget_widths: int = 140):
        details_ch_param = pn.Param(
            self.param,
            widgets={
                "channel": {"widget_type": pn.widgets.Select, "width": widget_widths}
            },
            parameters=["channel"],
            show_labels=False,
            show_name=False,
            sort=True,
        )

        details_type_param = pn.Param(
            self.param,
            widgets={
                "plot_type_details": {
                    "widget_type": pn.widgets.Select,
                    "width": widget_widths,
                }
            },
            # 'plot_type_details': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success',
            #         'orientation':"vertical", 'width': widget_widths}},
            parameters=["plot_type_details"],
            show_labels=False,
            show_name=False,
            sort=True,
        )

        details_param_currentValue = pn.pane.Markdown(f"## {self.parameter}")
        details_param = pn.widgets.MenuButton(
            name="Detailed Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.parameter.objects,
        )

        def update_details_plots(event):
            self.parameter = event.new
            details_param_currentValue.object = f"## {event.new}"

        details_param.on_click(update_details_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                details_param,
            ),
            pn.Row("## Current Plot:", details_param_currentValue),
            pn.Row("Channel:", details_ch_param, "Plot type:", details_type_param),
            self.get_run_and_channel,
            self.view_details,
            name="Cal. Details",
            sizing_mode="scale_both",
        )

    def build_summary_pane(self, widget_widths: int = 140):
        summary_param_currentValue = pn.pane.Markdown(f"## {self.plot_type_summary}")
        summary_param = pn.widgets.MenuButton(
            name="Summary Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.plot_type_summary.objects,
        )

        def update_summary_plots(event):
            self.plot_type_summary = event.new
            summary_param_currentValue.object = f"## {event.new}"

        summary_param.on_click(update_summary_plots)

        summary_param_download = pn.Param(
            self.param,
            widgets={
                "plot_types_download": {
                    "widget_type": pn.widgets.Select,
                    "width": widget_widths,
                }
            },
            parameters=["plot_types_download"],
            show_labels=False,
            show_name=False,
        )
        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                summary_param,
            ),
            pn.Row("## Current Plot:", summary_param_currentValue),
            "Download Raw Data",
            pn.Row(summary_param_download, self.download_summary_files),
            pn.pane.Bokeh(self.view_summary, sizing_mode="scale_both"),
            name="Cal. Summary",
            sizing_mode="scale_both",
        )

    def build_tracking_pane(self, widget_widths: int = 140):
        # tracking_range_param = pn.Param(
        #     self.param,
        #     widgets={
        #         "date_range": {
        #             "widget_type": pn.widgets.DatetimeRangePicker,
        #             "width": widget_widths,
        #             "enable_time": False,
        #             "enable_seconds": False,
        #         }
        #     },
        #     parameters=["date_range"],
        #     show_labels=False,
        #     show_name=False,
        # )

        tracking_param_currentValue = pn.pane.Markdown(f"## {self.plot_type_tracking}")
        tracking_param = pn.widgets.MenuButton(
            name="Tracking Plots",
            button_type="primary",
            width=widget_widths,
            items=self.param.plot_type_tracking.objects,
        )

        def update_tracking_plots(event):
            self.plot_type_tracking = event.new
            tracking_param_currentValue.object = f"## {event.new}"

        tracking_param.on_click(update_tracking_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Calibration.svg",
                    height=25,
                ),
                tracking_param,
            ),
            pn.Row("## Current Plot:", tracking_param_currentValue),
            # pn.Row("Selected time range:", tracking_range_param),
            self.view_tracking,
            name="Cal. Tracking",
            sizing_mode="scale_both",
        )

    def build_cal_panes(self, widget_widths: int = 140):
        self.update_plot_dict()
        return {
            "Cal. Summary": self.build_summary_pane(widget_widths),
            "Cal. Details": self.build_detailed_pane(widget_widths),
            "Cal. Tracking": self.build_tracking_pane(widget_widths),
        }

    @classmethod
    def init_cal_panes(
        cls,
        base_path,
        widget_widths: int = 140,
    ):
        """
        Initialize the calibration panes.

        Args:
            widget_widths (int): Width of the widgets.

        Returns:
            dict: Dictionary containing the calibration panes.
        """
        cal_monitor = cls(base_path=base_path)
        return cal_monitor.build_cal_panes(widget_widths)

    @classmethod
    def display_cal_panes(
        cls,
        base_path,
        notebook=False,
        widget_widths: int = 140,
    ):
        """
        View the calibration panes.

        Args:
            widget_widths (int): Width of the widgets.

        Returns:
            dict: Dictionary containing the calibration panes.
        """
        cal_monitor = cls(base_path=base_path, notebook=notebook)
        sidebar = cal_monitor.build_sidebar()
        return pn.Row(
            sidebar, pn.Tabs(*cal_monitor.build_cal_panes(widget_widths).values())
        )


def run_dashboard_cal() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    args = argparser.parse_args()

    config = read_config(args.config_file)
    cal_panes = CalMonitoring.display_cal_panes(config.base, args.widget_widths)
    print("Starting Cal. Monitoring on port ", args.port)  # noqa: T201
    pn.serve(cal_panes, port=args.port, show=False)
