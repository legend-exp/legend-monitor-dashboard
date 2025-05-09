import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import holoviews as hv
from bokeh.plotting import figure

from legenddashboard.base import Monitor

log = logging.getLogger(__name__)

class LlamaMonitor(Monitor):

    def __init__(
        self, base_path, llama_path, name=None
    ):
        super().__init__(name=name, path=base_path)
        self.llama_path = llama_path

    def view_llama(self):
        start_time = time.time()
        try:
            llama_data = pd.read_csv(
                self.llama_path + "monivalues.txt",
                sep=r"\s+",
                dtype={"timestamp": np.int64},
                parse_dates=[1],
            )
            llama_data["timestamp"] = pd.to_datetime(
                llama_data["timestamp"], origin="unix", unit="s"
            )
            llama_data = llama_data.rename(
                columns={
                    "#run_no": "#Run",
                    "timestamp": "Timestamp",
                    "triplet_val": "Triplet Lifetime (µs)",
                    "triplet_err": "Error Triplet Lifetime (µs)",
                    "ly_val": "Light Yield",
                    "ly_err": "Error Light Yield",
                }
            )
        except KeyError:
            p = figure(width=1000, height=600)
            p.title.text = "No current Llama data available."
            p.title.align = "center"
            p.title.text_font_size = "25px"
            log.debug(
                "Time to get llama plot:", extra={"time": time.time() - start_time}
            )
            return p
        llama_width, llama_height = 1200, 400
        # add two hours to x values to convert from UTC to CET if values still in UTC
        if llama_data["Timestamp"][0].utcoffset() is None:
            llama_data["Timestamp"] += pd.Timedelta(hours=2)
        triplet_plot = hv.Scatter(
            llama_data, ["Timestamp", "Triplet Lifetime (µs)"], label="Triplet LT"
        )
        triplet_plot_error = hv.ErrorBars(
            llama_data,
            vdims=["Triplet Lifetime (µs)", "Error Triplet Lifetime (µs)"],
            kdims=["Timestamp"],
            label="Triplet LT Error",
        )
        triplet_plot.opts(
            xlabel="Time (CET)",
            ylabel="Triplet Lifetime (µs)",
            tools=["hover"],
            line_width=1.5,
            color="blue",
            width=llama_width,
            height=llama_height,
        )
        triplet_plot_error.opts(
            line_width=0.2, width=llama_width, height=llama_height, show_grid=True
        )

        lightyield_plot = hv.Scatter(
            llama_data, ["Timestamp", "Light Yield"], label="Light Yield"
        )
        lightyield_plot_error = hv.ErrorBars(
            llama_data,
            vdims=["Light Yield", "Error Light Yield"],
            kdims=["Timestamp"],
            label="Light Yield Error",
        )
        lightyield_plot.opts(
            xlabel="Time (CET)",
            ylabel="Light yield (a.u.)",
            tools=["hover"],
            line_width=1.5,
            color="orange",
            width=llama_width,
            height=llama_height,
            show_grid=True,
        )
        lightyield_plot_error.opts(
            line_width=0.2, width=llama_width, height=llama_height
        )

        layout = (
            triplet_plot * triplet_plot_error + lightyield_plot * lightyield_plot_error
        )
        layout.opts(width=llama_width, height=llama_height).cols(1)
        log.debug("Time to get llama plot:", extra={"time": time.time() - start_time})
        return layout

    def get_llama_lastUpdate(self):
        start_time = time.time()
        llama_pathlib = Path(self.llama_path + "monivalues.txt")
        ret = "Last modified: {}".format(
            pd.to_datetime(llama_pathlib.stat().st_mtime, origin="unix", unit="s")
        )
        log.debug(
            "Time to get llama last update:", extra={"time": time.time() - start_time}
        )
        return ret
    
    def update(self):
        pass