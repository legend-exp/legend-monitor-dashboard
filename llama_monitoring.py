import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import param
import pickle as pkl
import shelve
import bisect
from pathlib import Path


import datetime as dtt
from  datetime import datetime

from legendmeta import LegendMetadata
from legendmeta.catalog import Props

from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter
import holoviews as hv
from bokeh.plotting import figure, show

hv.extension('bokeh')

class llama_monitoring(param.Parameterized):
    def __init__(self, path, name=None):
        super().__init__(name=name)
        self.llama_path = path
        
        
    def view_llama(self):
        try:
            llama_data = pd.read_csv(self.llama_path + 'monivalues.txt', sep='\s+', dtype={'timestamp': np.int64}, parse_dates=[1])
            llama_data["timestamp"] = pd.to_datetime(llama_data["timestamp"], origin='unix', unit='s')
            llama_data = llama_data.rename(columns={"#run_no": "#Run", "timestamp": "Timestamp", "triplet_val" : "Triplet Lifetime (µs)", "triplet_err": "Error Triplet Lifetime (µs)",	"ly_val": "Light Yield", "ly_err": "Error Light Yield"})
        except:
            p = figure(width=1000, height=600)
            p.title.text = title=f"No current Llama data available."
            p.title.align = "center"
            p.title.text_font_size = "25px"
            return p
        llama_width, llama_height = 1200, 400
        triplet_plot = hv.Scatter(llama_data, ["Timestamp", "Triplet Lifetime (µs)"], label="Triplet LT")
        triplet_plot_error = hv.ErrorBars(llama_data, vdims=['Triplet Lifetime (µs)', 'Error Triplet Lifetime (µs)'], kdims=['Timestamp'], label="Triplet LT Error")
        triplet_plot.opts(xlabel="Time", ylabel="Triplet Lifetime (µs)", tools=['hover'], line_width=1.5, color='blue', width=llama_width, height=llama_height)
        triplet_plot_error.opts(line_width=0.2, width=llama_width, height=llama_height, show_grid=True)

        lightyield_plot = hv.Scatter(llama_data, ["Timestamp", "Light Yield"], label="Light Yield")
        lightyield_plot_error = hv.ErrorBars(llama_data, vdims=['Light Yield', 'Error Light Yield'], kdims=['Timestamp'], label="Light Yield Error")
        lightyield_plot.opts(xlabel="Time", ylabel="Light yield (a.u.)", tools=['hover'], line_width=1.5, color='orange', width=llama_width, height=llama_height, show_grid=True)
        lightyield_plot_error.opts(line_width=0.2, width=llama_width, height=llama_height)

        layout = triplet_plot * triplet_plot_error + lightyield_plot * lightyield_plot_error
        layout.opts(width=llama_width, height=llama_height).cols(1)
        
        return layout
    
    def get_lastUpdate(self):
        llama_pathlib = Path(self.llama_path + 'monivalues.txt')
        return "Last modified: {}".format(pd.to_datetime(llama_pathlib.stat().st_mtime, origin='unix', unit='s'))