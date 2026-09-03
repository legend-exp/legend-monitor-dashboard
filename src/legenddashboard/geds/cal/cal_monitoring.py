from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import panel as pn
import param
from matplotlib.figure import Figure

import legenddashboard.geds.string_visulization as visu
from legenddashboard.geds import cal
from legenddashboard.geds.cal.shelf_cache import (
    _stat_key,
    render_png,
    shelf_entry,
    shelf_keys,
)
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.util import get_par_cache, logo_path, read_config, sorter

log = logging.getLogger(__name__)

# calibration plots
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100


class CalMonitoring(GedMonitoring):
    cached_data = param.Dict(default=None)
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
    plot_type_details_objects = param.List(default=cal.detailed_plots)
    channel_objects = param.List(default=[])

    plot_type_summary = param.ObjectSelector(
        default=list(cal.summary_plots)[3],
        objects=list(cal.summary_plots),
    )
    # Options must be keys of cal.summary_plots (functions that support
    # ``download=True``).
    plot_types_download = param.Selector(
        objects=["FWHM Qbb", "FWHM FEP", "A/E SF", "PZ", "CT Alpha"],
        default="FWHM Qbb",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Shared, bounded cache of parsed parameter files; reused across all
        # user sessions so the high-latency filesystem is read at most once per
        # run and memory stays bounded (see legenddashboard.util.get_par_cache).
        self.cached_data = get_par_cache()
        # The view_*/download_* methods are re-rendered by Panel through their
        # @param.depends decorators (they are placed directly in the panes);
        # additional param.watch registrations would compute every plot twice
        # per interaction. Only genuine state updaters are watchers; they are
        # registered before any view pane (same precedence -> registration
        # order -> shelves are loaded before the views that show them render)
        # and before the initial update_plot_dict call.
        self.param.watch(self.update_plot_dict, ["run_dict", "run"])
        self.update_plot_dict()

    @param.depends("run_dict", "run", "sort_by", "plot_types_download")
    def download_summary_files(self, event=None):  # noqa: ARG002
        start_time = time.time()
        try:
            download_file, download_filename = cal.summary_plots[
                self.plot_types_download
            ](
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
            # Serialise on click, in memory. The filename is derived only from
            # experiment/period/run/plot type, so every session viewing the
            # same run resolved to the *same* path on disk: concurrent renders
            # truncated and rewrote the file while another session's download
            # was streaming it. Keeping the CSV per-session also means it can
            # never be served stale.
            ret = pn.widgets.FileDownload(
                callback=lambda df=download_file: io.StringIO(df.to_csv(index=False)),
                filename=download_filename,
                button_type="success",
                embed=False,
                name="Click to download 'csv'",
                width=350,
            )
        except Exception:
            log.exception("Failed to build summary download for %s", self.run)
            ret = pn.widgets.FileDownload(
                None,
                filename="temp",
                button_type="success",
                embed=False,
                name="Click to download 'csv'",
                width=350,
            )
        log.debug("Time to download summary files: %.3fs", time.time() - start_time)
        return ret

    @param.depends("run_dict", "run", "sort_by", "string", "plot_type_summary")
    def view_summary(self, event=None):  # noqa: ARG002
        start_time = time.time()
        figure = None
        try:
            if not self.cached_data:
                self.cached_data = get_par_cache()
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
                figure = Figure()
        except Exception:
            log.exception(
                "Failed to build summary plot '%s' for %s",
                self.plot_type_summary,
                self.run,
            )
        log.debug("Time to get summary plot: %.3fs", time.time() - start_time)
        return figure

    @param.depends("run_dict", "date_range", "sort_by", "string", "plot_type_tracking")
    def view_tracking(self, event=None):  # noqa: ARG002
        figure = None
        try:
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
        except Exception:
            log.exception(
                "Failed to build tracking plot '%s' for %s %s",
                self.plot_type_tracking,
                self.period,
                self.string,
            )
        return figure

    def update_plot_dict(self, *events):  # noqa: ARG002
        start_time = time.time()
        run_info = self.run_dict[self.run]
        file_stem = (
            f"{run_info['experiment']}-{self.period}-{self.run}-cal-"
            f"{run_info['timestamp']}"
        )
        plt_base = Path(self.prod_config["paths"]["plt"])
        self.plot_dict = (
            plt_base / f"hit/cal/{self.period}/{self.run}" / f"{file_stem}-plt_hit"
        )
        # Build the dsp path explicitly rather than str.replace("hit", "dsp"),
        # which would corrupt any other "hit" substring in the deployment path.
        self.dsp_plot_dict = (
            plt_base / f"dsp/cal/{self.period}/{self.run}" / f"{file_stem}-plt_dsp"
        )

        channels = shelf_keys(self.plot_dict)
        if "common" in channels:
            channels.remove("common")
        if not channels:
            msg = f"No channels found in plot file {self.plot_dict}"
            raise RuntimeError(msg)
        self.strings_dict, self.chan_dict, self.channel_map = sorter(
            self.base_path,
            run_info["timestamp"],
            "String",
            sort_dets_obj=self.sort_obj,
        )

        self.channel_objects = channels
        if self.channel not in channels:  # keep the user's channel when valid
            self.channel = channels[0]

        self.update_strings()
        log.debug("Time to update plot dict: %.3fs", time.time() - start_time)

    # Shelve contents are loaded lazily through the process-wide cache, so a
    # run switch costs a key listing and only the views actually shown pay
    # for unpickling (once per file version, shared by all sessions).
    @property
    def common_dict(self):
        return shelf_entry(self.plot_dict, "common")

    @property
    def plot_dict_ch(self):
        return shelf_entry(self.plot_dict, self.channel[:9])

    @property
    def dsp_dict(self):
        return shelf_entry(self.dsp_plot_dict, self.channel[:9])

    def _png_pane(self, get_figure):
        """Rasterise a cached (shared) figure once; serve PNG bytes after."""
        # fingerprint the shelf (path+mtime+size) so a regenerated shelf drops
        # the stale PNG, and key on the channel id the shelve lookup uses
        key = (
            *_stat_key(self.plot_dict),
            self.channel[:9],
            self.parameter,
            self.plot_type_details,
        )
        return pn.pane.PNG(
            io.BytesIO(render_png(key, get_figure)), sizing_mode="scale_width"
        )

    @param.depends("parameter", watch=True)
    def update_plot_type_details(self):
        start_time = time.time()
        plots = cal.all_detailed_plots[self.parameter]
        self.plot_type_details_objects = plots
        self.plot_type_details = plots[0]
        log.debug("Time to update plot type details: %.3fs", time.time() - start_time)

    @param.depends("run_dict", "run", "channel", "parameter", "plot_type_details")
    def view_details(self, event=None):  # noqa: ARG002
        fig_pane = pn.pane.Matplotlib(Figure(), sizing_mode="scale_width")
        try:
            if self.parameter == "A/E":
                fig_pane = self._png_pane(
                    lambda: self.plot_dict_ch["aoe"][self.plot_type_details]
                )
            elif self.parameter == "Baseline":
                fig_pane = self._png_pane(
                    lambda: self.plot_dict_ch["ecal"][self.plot_type_details]
                )
            elif self.parameter == "PZ":
                fig_pane = self._png_pane(
                    lambda: self.dsp_dict["pz"][self.plot_type_details]
                )
            elif self.parameter == "Optimisation":
                fig_pane = self._png_pane(
                    lambda: self.dsp_dict[
                        f"{self.plot_type_details.split('_')[0]}_optimisation"
                    ][f"{self.plot_type_details.split('_')[1]}_space"]
                )
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
                fig_pane = self._png_pane(
                    lambda: self.plot_dict_ch["ecal"][self.parameter][
                        self.plot_type_details
                    ]
                )
        except Exception:
            log.exception(
                "Failed to build detailed plot '%s'/'%s' for channel %s",
                self.parameter,
                self.plot_type_details,
                self.channel,
            )
        return fig_pane

    def build_detailed_pane(self, widget_widths: int = 140):
        details_ch_param = pn.widgets.Select(
            value=self.param.channel,
            options=self.param.channel_objects,
            width=widget_widths,
        )

        details_type_param = pn.widgets.Select(
            # 'plot_type_details': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success',
            #         'orientation':"vertical", 'width': widget_widths}},
            value=self.param.plot_type_details,
            options=self.param.plot_type_details_objects,
            width=widget_widths,
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
            pn.param.ParamMethod(self.get_run_and_channel, lazy=True),
            pn.param.ParamMethod(
                self.view_details,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
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
            pn.Row(
                summary_param_download,
                pn.param.ParamMethod(self.download_summary_files, lazy=True),
            ),
            pn.param.ParamMethod(
                self.view_summary,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
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
            pn.param.ParamMethod(
                self.view_tracking,
                lazy=True,
                loading_indicator=True,
                sizing_mode="stretch_width",
            ),
            name="Cal. Tracking",
            sizing_mode="scale_both",
        )

    def build_cal_panes(self, widget_widths: int = 140):
        # update_plot_dict already ran in __init__ (and re-runs via its
        # period/run watcher); no need to redo the shelve reads here.
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
            pn.Row: Row containing the calibration panes.
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
    cal_panes = CalMonitoring.display_cal_panes(
        config.base, widget_widths=args.widget_widths
    )
    print("Starting Cal. Monitoring on port ", args.port)  # noqa: T201
    pn.serve(cal_panes, port=args.port, show=False)
