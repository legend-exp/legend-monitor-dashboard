from __future__ import annotations

import argparse
import datetime as dtt
import logging
import shelve
import time
from pathlib import Path

import numpy as np
import panel as pn
import param
from bokeh.models.widgets.tables import BooleanFormatter
from bokeh.plotting import figure

from legenddashboard import muon
from legenddashboard.base import Monitoring
from legenddashboard.util import logo_path, read_config

log = logging.getLogger(__name__)


class MuonMonitoring(Monitoring):
    muon_path = param.String("")

    # muon plots
    muon_plots_cal = param.ObjectSelector(
        default=next(iter(muon.muon_plots_cal_dict)),
        objects=list(muon.muon_plots_cal_dict),
    )

    muon_plots_mon = param.ObjectSelector(
        default=next(iter(muon.muon_plots_mon_dict)),
        objects=list(muon.muon_plots_mon_dict),
    )
    data_dict = param.Dict({})

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

    def build_muon_cal_pane(self, widget_widths=140):
        muon_cal_param_currentValue = pn.pane.Markdown(f"## {self.muon_plots_cal}")
        muon_cal_param = pn.widgets.MenuButton(
            name="Calibration",
            button_type="primary",
            width=widget_widths,
            items=self.param.muon_plots_cal.objects,
        )

        def update_muon_cal_plots(event):
            self.muon_plots_cal = event.new
            muon_cal_param_currentValue.object = f"## {event.new}"

        muon_cal_param.on_click(update_muon_cal_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Muon.svg",
                    height=25,
                ),
                muon_cal_param,
            ),
            pn.Row("## Current Plot:", muon_cal_param_currentValue),
            pn.Row(self.view_muon_cal, sizing_mode="scale_both"),
            name="Muon Cal. Plots",
            sizing_mode="scale_both",
        )

    def build_muon_mon_pane(self, widget_widths=140):
        muon_mon_param_currentValue = pn.pane.Markdown(f"## {self.muon_plots_mon}")
        muon_mon_param = pn.widgets.MenuButton(
            name="Monitoring",
            button_type="primary",
            width=widget_widths,
            items=self.param.muon_plots_mon.objects,
        )

        def update_muon_mon_plots(event):
            self.muon_plots_mon = event.new
            muon_mon_param_currentValue.object = f"## {event.new}"

        muon_mon_param.on_click(update_muon_mon_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Muon.svg",
                    height=25,
                ),
                muon_mon_param,
            ),
            pn.Row("## Current Plot:", muon_mon_param_currentValue),
            pn.Row(self.view_muon_mon, sizing_mode="scale_both"),
            name="Muon Mon. Plots",
            sizing_mode="scale_both",
        )

    def build_muon_panes(
        self,
        widget_widths=140,
    ):
        return {
            "Muon Cal. Plots": self.build_muon_cal_pane(widget_widths),
            "Muon Mon. Plots": self.build_muon_mon_pane(widget_widths),
        }

    @classmethod
    def init_muon_panes(
        cls,
        base_path,
        muon_path,
        widget_widths=140,
    ):
        """
        Initialize the MuonMonitoring class and create the muon panes.

        Args:
            base_path (Path): The base path for the monitoring.
            muon_path (Path): The path to the muon data.

        Returns:
            dict: A dictionary containing the muon panes.
        """
        return cls(
            base_path=base_path,
            muon_path=muon_path,
        ).build_muon_panes(base_path, widget_widths)

    @classmethod
    def display_muon_panes(
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
        muon_monitor = cls(base_path=base_path, notebook=notebook)
        sidebar = muon_monitor.build_sidebar()
        return pn.Row(
            sidebar, pn.Tabs(*muon_monitor.build_muon_panes(widget_widths).values())
        )


def run_dashboard_muon() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    args = argparser.parse_args()

    config = read_config(args.config_file)
    muon_panes = MuonMonitoring.display_cal_panes(config.base, args.widget_widths)
    print("Starting Muon Monitoring on port ", args.port)  # noqa: T201
    pn.serve(muon_panes, port=args.port, show=False)
