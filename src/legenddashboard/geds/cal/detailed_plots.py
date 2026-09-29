from __future__ import annotations

import plotly.graph_objects as go

detailed_plots = [
    "2614_timemap",
    "pulser_timemap",
    "peak_fits",
    "cal_fit",
    "fwhm_fit",
    "cut_spectrum",
    "survival_frac",
    "spectrum",
    "logged_spectrum",
    "peak_track",
]

aoe_plots = [
    "mean_time",
    "plot_dt_dep",
    "compt_bands_uncorrected",
    "mean_fit",
    "sigma_fit",
    "compt_bands_corrected",
    "cut_fit",
    "classifier",
    "survival_fractions",
    "spectrum",
    "sf_v_energy",
]

lq_plots = [
    "stability",
    "spectrum",
    "sf_v_energy",
    "survival_fractions",
    "cut_fit",
    "classifier",
    "drift_time",
]

baseline_plots = ["baseline_timemap"]

tau_plots = ["slope", "waveforms"]

optimisation_plots = [
    "trap_kernel",
    "zac_kernel",
    "cusp_kernel",
    "trap_acq",
    "zac_acq",
    "cusp_acq",
]

all_detailed_plots = {
    "cuspEmax_ctc_cal": detailed_plots,
    "zacEmax_ctc_cal": detailed_plots,
    "trapEmax_ctc_cal": detailed_plots,
    "trapEftp_ctc_cal": detailed_plots,
    "dplmsEmax_ctc_cal": detailed_plots,
    "Baseline": baseline_plots,
    "A/E": aoe_plots,
    "LQ": lq_plots,
    "PZ": tau_plots,
    "Optimisation": optimisation_plots,
}


def plot_spectrum(plot_dict, channel, log=False):
    fig = go.Figure()
    bins = plot_dict["bins"]
    counts = plot_dict["counts"]

    fig.add_trace(
        go.Scatter(x=bins, y=counts, name=channel, line_shape="hvh", line={"width": 1})
    )

    fig.update_traces(mode="lines")

    fig.update_layout(
        xaxis={
            "showline": True,
            "showgrid": True,
            "showticklabels": True,
            "linecolor": "grey",
            "linewidth": 2,
            "ticks": "outside",
            "tickfont": {
                "family": "Arial",
                "size": 12,
                "color": "rgb(82, 82, 82)",
            },
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "linecolor": "grey",
            "linewidth": 2,
            "showticklabels": True,
            "tickfont": {
                "family": "Arial",
                "size": 12,
                "color": "rgb(82, 82, 82)",
            },
        },
        autosize=False,
        margin={
            "autoexpand": False,
            "l": 100,
            "r": 20,
            "t": 110,
        },
        showlegend=False,
        plot_bgcolor="white",
    )
    annotations = []
    # Title
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.2,
            "y": 1.05,
            "xanchor": "left",
            "yanchor": "bottom",
            "text": channel,
            "font": {
                "family": "Arial",
                "size": 20,
                "color": "rgb(82, 82, 82)",
            },
            "showarrow": False,
        }
    )
    # X label
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.1,
            "xanchor": "center",
            "yanchor": "top",
            "text": "Energy (keV)",
            "font": {
                "family": "Arial",
                "size": 12,
                "color": "rgb(82, 82, 82)",
            },
            "showarrow": False,
        }
    )

    # Y label
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": -0.1,
            "y": 0.5,
            "xanchor": "left",
            "yanchor": "middle",
            "text": "Counts",
            "textangle": 270,
            "font": {
                "family": "Arial",
                "size": 12,
                "color": "rgb(82, 82, 82)",
            },
            "showarrow": False,
        }
    )

    fig.update_layout(yaxis={"showexponent": "all", "exponentformat": "none"})
    if log is True:
        fig.update_yaxes(
            type="log",
        )
        fig.update_layout(yaxis={"showexponent": "all", "exponentformat": "power"})
    fig.update_layout(annotations=annotations)
    return fig
