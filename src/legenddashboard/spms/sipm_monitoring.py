from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import panel as pn
import param
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.plotting import figure

import legenddashboard.spms.sipm_plots as spm
from legenddashboard.base import Monitoring
from legenddashboard.util import logo_path, sorter

log = logging.getLogger(__name__)

sipm_plot_style_dict = {
    "Time": spm.sipm_plot_vsTime,
    "Histogram": spm.sipm_plot_histogram,
}


class SiPMMonitoring(Monitoring):
    sipm_path = param.String("")

    # sipm plots
    # sipm_plots_barrels    = ['InnerBarrel-Top', 'InnerBarrel-Bottom', 'OuterBarrel-Top', 'OuterBarrel-Bottom']
    # sipm_resampled_vals = [1, 5, 10, 30, 60]

    sipm_sort_by = param.ObjectSelector(objects=["Barrel"], default="Barrel")

    sipm_barrel = param.ObjectSelector(default=0, objects=[0])
    sipm_resampled = param.Integer(
        default=1,
        bounds=(1, 60),
    )
    sipm_plot_style = param.ObjectSelector(
        default=next(iter(sipm_plot_style_dict)),
        objects=list(sipm_plot_style_dict),
    )
    sipm_plot_style_dict = param.Dict(sipm_plot_style_dict)

    # self._get_sipm_data()

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

    def build_spm_pane(
        self,
        widget_widths=140,
    ):
        sipm_param_currentValue = pn.pane.Markdown(f"## {self.sipm_barrel}")
        sipm_param = pn.widgets.MenuButton(
            name="SiPM",
            button_type="primary",
            width=widget_widths,
            items=self.param.sipm_barrel.objects,
        )

        def update_sipm_plots(event):
            self.sipm_barrel = event.new
            sipm_param_currentValue.object = f"## {event.new}"

        sipm_param.on_click(update_sipm_plots)

        sipm_plot_style_param = pn.Param(
            self.param,
            widgets={
                "sipm_plot_style": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "button_type": "light",
                    "orientation": "vertical",
                    "width": widget_widths,
                }
            },
            parameters=["sipm_plot_style"],
            show_labels=False,
            show_name=False,
            sort=False,
        )

        sipm_plot_param = pn.Param(
            self.param,
            widgets={
                "sipm_resampled": {
                    "widget_type": pn.widgets.IntSlider,
                    "width": widget_widths,
                    "format": PrintfTickFormatter(format="%d min"),
                }
            },
            parameters=["sipm_resampled"],
            show_labels=False,
            show_name=False,
            sort=False,
        )

        return pn.Column(
            pn.Row(pn.pane.SVG(logo_path / "Physics.svg", height=25), sipm_param),
            pn.Row("## Current Plot:", sipm_param_currentValue),
            pn.Row(sipm_plot_style_param, pn.Column("Resampled", sipm_plot_param)),
            pn.pane.Bokeh(self.view_sipm, sizing_mode="scale_both"),
            name="SiPM Monitoring",
            sizing_mode="scale_both",
        )

    @classmethod
    def init_spm_pane(
        cls,
        base_path,
        spm_path,
        widget_widths=140,
    ):
        """
        Initialize the SiPM monitoring pane.
        """
        return cls(
            path=base_path,
            sipm_path=spm_path,
        ).build_spm_pane(widget_widths)
