from __future__ import annotations

import logging
import time

import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource
from bokeh.models.widgets.tables import BooleanFormatter

import legenddashboard.geds.string_visulization as visu
from legenddashboard.base import Monitoring
from legenddashboard.util import (
    get_characterization,
    get_production,
    logo_path,
    sort_dict,
    sorter,
)

log = logging.getLogger(__name__)

meta_visu_plots_dict = {
    "Usability": visu.plot_visu_usability,
    "Processable": visu.plot_visu_processable,
    "Mass": visu.plot_visu_mass,
    "Depl. Voltage": visu.plot_visu_depletion,
    "Oper. Voltage": visu.plot_visu_operation,
    "Enrichment": visu.plot_visu_enrichment,
}


class GedMonitoring(Monitoring):
    channel = param.Selector(default=0, objects=[0], allow_refs=True, nested_refs=True)
    string = param.ObjectSelector(
        default="01",
        objects=[f"{i + 1:02}" for i in range(11)],
        allow_refs=True,
        nested_refs=True,
    )
    # general selectors
    sort_by = param.ObjectSelector(
        default=next(iter(sort_dict)),
        objects=list(sort_dict),
        allow_refs=True,
        nested_refs=True,
    )

    meta_visu_plots = param.ObjectSelector(
        default=next(iter(meta_visu_plots_dict)), objects=list(meta_visu_plots_dict)
    )
    meta_visu_plots_dict = param.Dict(meta_visu_plots_dict)
    meta_df = pd.DataFrame()
    meta_visu_source = ColumnDataSource({})
    meta_visu_xlabels = param.Dict({})
    meta_visu_chan_dict = param.Dict({})
    meta_visu_channel_map = param.Dict({})

    chan_dict = param.Dict({})
    strings_dict = param.Dict({})
    channel_map = param.Dict({})

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
        strings_dict, self.chan_dict, self.channel_map = sorter(
            self.base_path,
            self.run_dict[self.run]["timestamp"],
            key=self.sort_by,
            sort_dets_obj=self.sort_obj,
        )

        self.param["string"].objects = list(strings_dict)
        self.string = f"{next(iter(strings_dict))}"
        self.strings_dict = strings_dict
        log.debug("Time to update strings:", extra={"time": time.time() - start_time})

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
                df_out["production"].map(
                    lambda x: get_production(x, "enrichment")["val"]
                )
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

        except KeyError:
            pass
        log.debug("Time to get metadata:", extra={"time": time.time() - start_time})

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
            self.base_path,
            self.run_dict[self.run]["timestamp"],
            key="String",
            sort_dets_obj=self.sort_obj,
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

    def build_sidebar(self, sidebar_instance=None):
        string_param = pn.widgets.MenuButton(
            name=f"{self.string}",
            button_type="primary",
            sizing_mode="stretch_width",
            items=self.param.string.objects,
        )

        def update_string(event):
            self.string = event.new
            string_param.name = f"{self.string}"

        string_param.on_click(update_string)

        sort_by_param = pn.widgets.MenuButton(
            name=f"Sorted by {self.sort_by}",
            button_type="primary",
            sizing_mode="stretch_width",
            items=self.param.sort_by.objects,
        )

        def update_sort_by(event):
            self.sort_by = event.new
            string_param.items = self.param.string.objects
            string_param.name = f"{self.string}"
            sort_by_param.name = f"Sorted by {self.sort_by}"

        sort_by_param.on_click(update_sort_by)
        # sort_by_param    = pn.Param(cal.param, widgets={'sort_by': {'widget_type': pn.widgets.Select, 'width': 100}}, parameters=['sort_by'], show_labels=False, show_name=False, design=Bootstrap)

        if sidebar_instance is not None:
            sidebar = sidebar_instance
        else:
            sidebar = super().build_sidebar()
        sidebar.append(
            pn.pane.SVG(
                logo_path / "Sort_by.svg",
                height=25,
            )
        )
        sidebar.append(sort_by_param)
        sidebar.append(pn.pane.SVG(logo_path / "String.svg", height=25))
        sidebar.append(string_param)
        return sidebar

    def build_meta_pane(self, widget_widths=130):
        self.update_strings()
        self._get_metadata()
        meta_param_currentValue = pn.pane.Markdown(f"## {self.meta_visu_plots}")
        meta_param = pn.widgets.MenuButton(
            name="Visualization",
            button_type="primary",
            width=widget_widths,
            items=self.param.meta_visu_plots.objects,
        )

        def update_meta_plots(event):
            self.meta_visu_plots = event.new
            meta_param_currentValue.object = f"## {event.new}"

        meta_param.on_click(update_meta_plots)

        return pn.Column(
            pn.Row(
                pn.pane.SVG(
                    logo_path / "Metadata.svg",
                    height=25,
                ),
                meta_param,
            ),
            pn.Row("## Current Plot:", meta_param_currentValue),
            pn.Tabs(
                (
                    "Detector Visu",
                    pn.pane.Bokeh(self.view_meta_visu, sizing_mode="scale_both"),
                ),
                ("Detector Map", self.view_meta),
                sizing_mode="scale_both",
            ),
            name="MetaData",
            sizing_mode="scale_both",
        )

    @classmethod
    def build_ged_base(
        cls,
        path,
        notebook=False,
        widget_widths=130,
        monitoring_instance=None,
        sidebar_instance=None,
    ):
        if monitoring_instance is not None:
            ged_monitoring = cls(
                base_path=path,
                notebook=notebook,
                run_dict=monitoring_instance.params.run_dict,
                period=monitoring_instance.params.period,
                prod_config=monitoring_instance.params.prod_config,
            )
        else:
            ged_monitoring = cls(
                base_path=path,
                notebook=notebook,
            )
        sidebar = ged_monitoring.build_sidebar(sidebar_instance)
        meta_pane = ged_monitoring.build_meta_pane(widget_widths)
        return sidebar, meta_pane, ged_monitoring

    @classmethod
    def display_meta(
        cls,
        path,
        notebook=False,
        widget_widths=130,
    ):
        sidebar, meta_pane, _ = cls.build_ged_base(
            path=path,
            notebook=notebook,
            widget_widths=widget_widths,
        )
        return pn.Column(
            sidebar,
            meta_pane,
            sizing_mode="scale_both",
        )
