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
from legenddashboard.geds.phy import contract_reader
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
        default=60,
        bounds=(0, 60),
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Per-instance state. As class attributes these would be shared by
        # every session in the process -- the Bokeh pane in particular can
        # belong to only one document, so a shared instance would break
        # multi-user serving.
        self.phy_data_df = pd.DataFrame()
        self.phy_data_df_mean = pd.DataFrame()
        self.phy_abs_unit = ""
        self.phy_plot_info = None
        self.phy_data_sc = pd.DataFrame()
        self.phy_pane = pn.pane.Bokeh(
            figure(width=1000, height=600), sizing_mode="scale_width"
        )
        self._phy_sc_plotted = False

    @param.depends(
        "period",
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
        """Render the phy view: contract-v2 when a run manifest exists, else v1."""
        start_time = time.time()
        experiment = self.run_dict[self.run]["experiment"]
        run_dir = Path(self.phy_path) / "generated/plt/hit/phy" / self.period / self.run

        manifest = contract_reader.find_manifest(
            self.phy_path, self.period, self.run, experiment
        )
        if manifest is not None:
            p = self._update_plots_v2(manifest, run_dir, experiment)
        else:
            p = self._update_plots_v1(run_dir, experiment)

        msg = f"Time to get phy plot: {time.time() - start_time}"
        log.debug(msg)
        return p

    def _empty_figure(self, reason="No data"):
        p = figure(width=1000, height=600)
        experiment = self.run_dict[self.run]["experiment"]
        p.title.text = f"{reason} for run {experiment}-{self.period}-{self.run}"
        p.title.align = "center"
        p.title.text_font_size = "25px"
        return p

    def _read_sc(self, data_file_sc):
        """Slow-control frame for the current selection (empty when off)."""
        if (
            phy.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
            and Path(data_file_sc).exists()
        ):
            data_sc = pd.read_hdf(
                data_file_sc, phy.phy_plots_sc_vals_dict[self.phy_plots_sc_vals]
            )
            self._phy_sc_plotted = True
        else:
            data_sc = pd.DataFrame()
            self._phy_sc_plotted = False
        return data_sc

    # ------------------------------------------------------------------
    # contract-v2 path
    # ------------------------------------------------------------------

    def _update_plots_v2(self, manifest, run_dir, experiment):
        data_file = contract_reader.geds_file_from_manifest(manifest, run_dir)
        if data_file is None or not data_file.exists():
            return self._empty_figure("Manifest names no geds file")

        flag = phy.phy_plots_types_dict[self.phy_plots_types]
        param_name = phy.phy_plots_vals_dict[self.phy_plots]
        relative = self.phy_units == "Relative"
        key_param = param_name + ("_var" if relative else "")

        keys = contract_reader.available_keys(manifest)
        if f"{flag}_{key_param}" not in keys:
            return self._empty_figure(f"Key {flag}_{key_param} missing")

        series = contract_reader.read_binned(
            data_file,
            flag,
            key_param,
            contract_reader.snap_cadence(
                self.phy_resampled, manifest.get("cadences", ["1min"])
            ),
        )

        label, unit = contract_reader.label_and_unit(
            manifest, series.attrs, param_name, relative
        )
        if "pulser01ana" in param_name:
            label = (
                "Gain to Pulser Difference"
                if "Diff" in param_name
                else "Gain to Pulser Ratio"
            )
        base_series_attrs = series.attrs if not relative else {}
        abs_unit = base_series_attrs.get("unit", unit)

        # detector-name columns straight from the contract
        names = [
            n for n in self.strings_dict.get(self.string, []) if n in series.detectors
        ]
        if not names:
            return self._empty_figure(f"No detectors of string {self.string} in data")

        meta = phy.PlotMeta(
            label=label,
            unit=unit,
            abs_unit=abs_unit,
            flag_display=self.phy_plots_types,
            param_display=self.phy_plots,
            string=str(self.string),
            run=self.run,
            period=self.period,
            experiment=experiment,
        )

        if self.phy_plot_style == "Histogram":
            if f"{flag}_{key_param}_dist" not in keys:
                return self._empty_figure("No distribution stored")
            return phy.phy_plot_dist_histogram(
                contract_reader.read_dist(data_file, flag, key_param), meta
            )

        cadence = contract_reader.snap_cadence(
            self.phy_resampled, manifest.get("cadences", ["1min"])
        )
        data_file_sc = (
            run_dir / f"{experiment}-{self.period}-{self.run}-phy-slow_control.hdf"
        )
        return phy.phy_plot_binned_vsTime(
            mean_df=series.to_frame("mean")[names],
            std_df=series.to_frame("std")[names],
            min_df=series.to_frame("min")[names],
            max_df=series.to_frame("max")[names],
            meta=meta,
            flagged_ranges=contract_reader.flagged_ranges(manifest),
            data_sc=self._read_sc(data_file_sc),
            sc_param=self.phy_plots_sc_vals,
            cadence_label=cadence,
        )

    # ------------------------------------------------------------------
    # v1 fallback (pre-manifest files)
    # ------------------------------------------------------------------

    def _update_plots_v1(self, run_dir, experiment):
        data_file = run_dir / f"{experiment}-{self.period}-{self.run}-phy-geds.hdf"
        data_file_sc = (
            run_dir / f"{experiment}-{self.period}-{self.run}-phy-slow_control.hdf"
        )

        # return empty plot if no data exists for run
        if not Path(data_file).exists():
            return self._empty_figure("No data")

        if self.phy_plot_style == "Histogram":
            return self._empty_figure("Histogram view requires v2 monitoring files")

        # get filekeys to check if key exists
        with h5py.File(data_file, "r") as f:
            filekeys = list(f.keys())

        # load dataframe for current plot value and get all data from selected string
        channel_names = self.strings_dict.get(self.string, [])
        if not channel_names:
            msg = f"No channel_names found for string {self.string}"
            log.error(msg)
            return self._empty_figure(f"No channels for string {self.string}")

        channels = [
            self.name_to_rawid[name]
            for name in channel_names
            if name in self.name_to_rawid
        ]
        phy_data_key = f"{phy.phy_plots_types_dict[self.phy_plots_types]}_{phy.phy_plots_vals_dict[self.phy_plots]}"
        if "pulser" in phy_data_key:
            if f"{phy_data_key.split('_pulser')[0]}_info" not in filekeys:
                return self._empty_figure("Info key missing")
            phy_plot_info = pd.read_hdf(
                data_file, key=f"{phy_data_key.split('_pulser')[0]}_info"
            )
            if "Diff" in phy_data_key:
                phy_plot_info.loc["label"].iloc[0] = "Gain to Pulser Difference"
            else:
                phy_plot_info.loc["label"].iloc[0] = "Gain to Pulser Ratio"
        else:
            if f"{phy_data_key}_info" not in filekeys:
                return self._empty_figure("Info key missing")
            phy_plot_info = pd.read_hdf(data_file, key=f"{phy_data_key}_info")
        abs_unit = phy_plot_info.loc["unit"].iloc[0]

        if self.phy_units == "Relative":
            if f"{phy_data_key}_var" not in filekeys:
                return self._empty_figure(f"Key {phy_data_key}_var missing")
            phy_data_df = pd.read_hdf(data_file, key=f"{phy_data_key}_var")
            phy_plot_info.loc["unit", phy_plot_info.columns[0]] = "%"
        else:
            if phy_data_key not in filekeys:
                return self._empty_figure(f"Key {phy_data_key} missing")
            phy_data_df = pd.read_hdf(data_file, key=phy_data_key)

        # load mean values
        if f"{phy_data_key}_mean" not in filekeys:
            return self._empty_figure(f"Key {phy_data_key}_mean missing")
        phy_data_df_mean = pd.read_hdf(data_file, key=f"{phy_data_key}_mean")

        # get sc data if selected
        data_sc = self._read_sc(data_file_sc)

        # check if channel selection actually exists in data
        channels = [
            ch
            for ch in channels
            if ch in phy_data_df.columns and ch in phy_data_df_mean.columns
        ]
        phy_data_df = phy_data_df[channels]
        phy_data_df_mean = phy_data_df_mean[channels]

        # map rawids to detector names
        phy_data_df = phy_data_df.rename(
            columns={
                rawid: f"{self.rawid_to_name[rawid]}_val"
                for rawid in phy_data_df.columns
            }
        )
        phy_data_df_mean = phy_data_df_mean.rename(columns=self.rawid_to_name)

        # plot data
        return phy.phy_plot_vsTime(
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
            abs_unit,
            data_sc,
            self.phy_plots_sc_vals,
        )

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
            # loading_indicator greys the stale figure during the re-render
            # round trip instead of leaving it frozen
            pn.param.ParamMethod(self.update_plots, loading_indicator=True),
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
    msg = f"Starting Phy. Monitoring on port {args.port}"
    log.info(msg)
    pn.serve(phy_pane, port=args.port, show=False)
