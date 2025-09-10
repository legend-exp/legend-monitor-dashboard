from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import pandas as pd
import panel as pn
import param
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.plotting import figure

from legenddashboard.geds import phy
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.util import logo_path, read_config

log = logging.getLogger(__name__)


class PhyMonitoring(GedMonitoring):
    phy_path = param.String("")
    phy_plots_types = param.ObjectSelector(
        default=next(iter(phy.phy_plots_types_dict)),
        objects=list(phy.phy_plots_types_dict),
        label="Type",
    )
    phy_plots = param.ObjectSelector(
        default=list(phy.phy_plots_vals_dict)[4],
        objects=list(phy.phy_plots_vals_dict),
        label="Value",
    )
    phy_plot_style = param.ObjectSelector(
        default=next(iter(phy.phy_plot_style_dict)),
        objects=list(phy.phy_plot_style_dict),
        label="Plot Style",
    )
    phy_resampled = param.Integer(
        default=phy.phy_resampled_vals[0],
        bounds=(phy.phy_resampled_vals[0], phy.phy_resampled_vals[-1]),
    )
    phy_units = param.ObjectSelector(
        default=phy.phy_unit_vals[0], objects=phy.phy_unit_vals, label="Units"
    )
    # phy_plots_sc        = param.Boolean(default=False, label="SC")
    phy_plots_sc_vals = param.ObjectSelector(
        default=next(iter(phy.phy_plots_sc_vals_dict)),
        objects=list(phy.phy_plots_sc_vals_dict),
        label="SC Values",
    )

    # create initial dataframes
    phy_data_df = pd.DataFrame()
    phy_data_df_mean = pd.DataFrame()
    phy_abs_unit = ""
    phy_plot_info = None
    phy_data_sc = pd.DataFrame()
    phy_pane = pn.pane.Bokeh(figure(width=1000, height=600), sizing_mode="scale_width")
    _phy_sc_plotted = False

    @param.depends(
        "run",
        "string",
        "sort_by",
        "phy_plots_types",
        "phy_plots",
        "phy_plot_style",
        "phy_resampled",
        "phy_units",
        "phy_plots_sc_vals",
    )
    def update_plots(self):
        start_time = time.time()
        data_file = (
            self.phy_path
            + f"/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-geds.hdf"
        )
        data_file_sc = (
            self.phy_path
            + f"/generated/plt/phy/{self.period}/{self.run}/l200-{self.period}-{self.run}-phy-slow_control.hdf"
        )

        # Create empty plot inc ase of errors
        p = figure(width=1000, height=600)
        p.title.text = f"No data for run {self.run_dict[self.run]['experiment']}-{self.period}-{self.run}"
        p.title.align = "center"
        p.title.text_font_size = "25px"

        # return empty plot if no data exists for run
        if not Path(data_file).exists():
            log.debug("Time to get phy plot:", extra={"time": time.time() - start_time})
            return p

        # get filekeys to check if key exists
        with h5py.File(data_file, "r") as f:
            filekeys = list(f.keys())

        # load dataframe for current plot value and get all data from selected string
        channels = self.strings_dict[self.string]
        phy_data_key = f"{self.phy_plots_types_dict[self.phy_plots_types]}_{self.phy_plots_vals_dict[self.phy_plots]}"
        if "pulser" in phy_data_key:
            if f"{phy_data_key.split('_pulser')[0]}_info" not in filekeys:
                return p
            phy_plot_info = pd.read_hdf(
                data_file, key=f"{phy_data_key.split('_pulser')[0]}_info"
            )
            if "Diff" in phy_data_key:
                phy_plot_info.loc["label"][0] = "Gain to Pulser Difference"
            else:
                phy_plot_info.loc["label"][0] = "Gain to Pulser Ratio"
        else:
            if f"{phy_data_key}_info" not in filekeys:
                return p
            phy_plot_info = pd.read_hdf(data_file, key=f"{phy_data_key}_info")
        abs_unit = phy_plot_info.loc["unit"][0]

        if self.phy_units == "Relative":
            if f"{phy_data_key}_var" not in filekeys:
                return p
            phy_data_df = pd.read_hdf(data_file, key=f"{phy_data_key}_var")
            phy_plot_info.loc["unit"][0] = "%"
        else:
            if phy_data_key not in filekeys:
                return p
            phy_data_df = pd.read_hdf(data_file, key=phy_data_key)

        # load mean values
        if f"{phy_data_key}_mean" not in filekeys:
            return p
        phy_data_df_mean = pd.read_hdf(data_file, key=f"{phy_data_key}_mean")

        # get sc data if selected
        # if self.phy_plots_sc and self.phy_units == "Relative" and os.path.exists(data_file_sc):
        if (
            self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
            and Path(data_file_sc).exists()
        ):
            data_sc = pd.read_hdf(
                data_file_sc, self.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
            )
            self._phy_sc_plotted = True
        else:
            data_sc = pd.DataFrame()
            self._phy_sc_plotted = False
        # check if channel selection actually exists in data
        channels = [
            ch
            for ch in channels
            if ch in phy_data_df.columns and ch in phy_data_df_mean.columns
        ]
        phy_data_df = phy_data_df[channels]
        phy_data_df_mean = phy_data_df_mean[channels]

        # plot data
        p = self.phy_plot_style_dict[self.phy_plot_style](
            phy_data_df,
            phy_data_df_mean,
            phy_plot_info,
            self.phy_plots_types,
            self.phy_plots,
            f"{self.phy_resampled}min",
            self.string,
            self.run,
            self.period,
            self.run_dict[self.run],
            self.channel_map,
            abs_unit,
            data_sc,
            self.phy_plots_sc_vals,
        )
        log.debug("Time to get phy plot:", extra={"time": time.time() - start_time})
        # self.bokeh_pane.object = p
        return p

    def build_phy_pane(self, widget_widths=140):
        """
        Build the phy pane with all widgets and plots
        """

        # physics_style_param = pn.Param(
        #     self.param,
        #     widgets={
        #         "phy_plot_style": {
        #             "widget_type": pn.widgets.RadioButtonGroup,
        #             "button_type": "light",
        #             "orientation": "vertical",
        #             "width": widget_widths,
        #         }
        #     },
        #     parameters=["phy_plot_style"],
        #     show_labels=False,
        #     show_name=False,
        #     sort=False,
        # )
        physics_param_resampled = pn.Param(
            self.param,
            widgets={
                "phy_resampled": {
                    "widget_type": pn.widgets.IntSlider,
                    "width": widget_widths,
                    "format": PrintfTickFormatter(format="%d min"),
                    "value_throttled": True,
                },
            },
            parameters=["phy_resampled"],
            show_labels=False,
            show_name=False,
            sort=False,
        )
        physics_param_units = pn.Param(
            self.param,
            widgets={
                "phy_units": {"widget_type": pn.widgets.RadioBoxGroup, "inline": True}
            },
            parameters=["phy_units"],
            show_labels=False,
            show_name=False,
            sort=False,
        )
        physics_param_types = pn.Param(
            self.param,
            widgets={
                "phy_plots_types": {
                    "widget_type": pn.widgets.RadioButtonGroup,
                    "orientation": "vertical",
                    "button_type": "primary",
                    "button_style": "outline",
                    "width": widget_widths,
                }
            },
            parameters=["phy_plots_types"],
            show_labels=False,
            show_name=False,
            sort=False,
        )

        physics_param_currentValue = pn.pane.Markdown(f"## {self.phy_plots}")
        physics_param = pn.widgets.MenuButton(
            name="HPGe Detectors",
            button_type="primary",
            width=widget_widths,
            items=self.param.phy_plots.objects,
        )

        def update_phy_plots(event):
            self.phy_plots = event.new
            physics_param_currentValue.object = f"## {event.new}"

        physics_param.on_click(update_phy_plots)

        # SC
        # sc_param_currentValue = pn.pane.Markdown(f"## Not selected or no data available")
        sc_param = pn.widgets.MenuButton(
            name="Slow Control",
            button_type="warning",
            width=widget_widths,
            items=self.param.phy_plots_sc_vals.objects,
        )

        def update_sc_plots(event):
            self.phy_plots_sc_vals = event.new
            # if self.phy_plots_sc_vals == "None" or not self._phy_sc_plotted:
            #         sc_param_currentValue.object = f"## Not selected or no data available"
            # else:
            #         sc_param_currentValue.object = f"## {event.new}"

        sc_param.on_click(update_sc_plots)

        phy_gspec = pn.GridSpec(width=3 * widget_widths + 10, max_height=800)
        phy_gspec[:, 0] = physics_param_types
        phy_gspec[:, 1] = pn.Spacer(width=5)
        phy_gspec[0, 2] = pn.widgets.Button(
            name="Units", button_type="primary", width=widget_widths, disabled=True
        )
        phy_gspec[1, 2] = physics_param_units
        phy_gspec[:, 3] = pn.Spacer(width=5)
        phy_gspec[0, 4] = pn.widgets.Button(
            name="Resampled", button_type="primary", width=widget_widths, disabled=True
        )
        phy_gspec[1, 4] = physics_param_resampled
        # phy_gspec[:, 5] = pn.Spacer(width=5)
        # phy_gspec[0, 6] = pn.widgets.Button(name="Show Slow Control", button_type='danger', width=widget_widths, disabled=True)
        # phy_gspec[1, 6] = sc_param_selected
        # pn.Row(physics_param_types, pn.Column("Units", physics_param_units), pn.Column("Resampled", physics_param_resampled), pn.Column("Show Slow Control", sc_param_selected)),
        # pn.Row(phy_gspec)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Physics.svg",
                    height=25,
                ),
                physics_param,
                sc_param,
            ),
            pn.Row("## Current Plot:", physics_param_currentValue),
            # pn.Row("## Current SC Plot:", sc_param_currentValue),
            pn.Row(phy_gspec),
            pn.pane.Bokeh(self.update_plots(), sizing_mode="scale_width"),
            name="Phy. Monitoring",
            sizing_mode="stretch_width",
        )

    @classmethod
    def init_phy_panes(
        cls,
        base_path,
        phy_path,
        widget_widths: int = 140,
    ):
        phy_monitor = cls(
            base_path=base_path,
            phy_path=phy_path,
        )
        return phy_monitor.build_phy_pane(widget_widths)

    @classmethod
    def display_phy(
        cls,
        base_path,
        notebook=False,
        widget_widths: int = 140,
    ):
        """
        View the Physics panes.

        Args:
            widget_widths (int): Width of the widgets.

        Returns:
            pn.Row: Row containing the sidebar and the phy pane.
        """
        phy_monitor = cls(base_path=base_path, notebook=notebook)
        sidebar = phy_monitor.build_sidebar()
        return pn.Row(sidebar, phy_monitor.build_phy_pane(widget_widths))


def run_dashboard_phy() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    args = argparser.parse_args()

    config = read_config(args.config_file)
    phy_pane = GedMonitoring.display_phy(config.base, args.widget_widths)
    print("Starting Phy. Monitoring on port ", args.port)  # noqa: T201
    pn.serve(phy_pane, port=args.port, show=False)
