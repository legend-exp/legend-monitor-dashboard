from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

_BERLIN_TZ = ZoneInfo("Europe/Berlin")


def _percent_shift(counts, spread):
    """Return (% shift, % error) w.r.t. the mean of the first 10 finite values.

    Returns ``(None, None)`` when there is no finite data or the mean is zero,
    so callers can skip the trace instead of dividing by zero.
    """
    finite = counts[~np.isnan(counts)]
    if len(finite) == 0:
        return None, None
    mean = np.mean(finite[:10])
    if mean == 0:
        return None, None
    return 100 * (counts - mean) / mean, 100 * spread / mean


detailed_plots = [
    "2614_timemap",
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
    "trapTmax_cal": detailed_plots,
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


def plot_survival_frac(plot_dict):
    fig = plt.figure()
    plt.step(plot_dict["bins"], plot_dict["sf"], where="mid")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Survival Fraction (%)")
    plt.close()
    return fig


def plot_cut_spectra(plot_dict):
    fig = plt.figure()
    plt.step(plot_dict["bins"], plot_dict["counts"], where="mid", label="After Cuts")
    plt.step(
        plot_dict["bins"], plot_dict["cut_counts"], where="mid", label="Cut Spectrum"
    )
    if np.isnan(plot_dict["pulser_counts"]).all():
        pass
    else:
        plt.step(
            plot_dict["bins"], plot_dict["pulser_counts"], where="mid", label="Pulser"
        )

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.close()
    return fig


def track_peaks(plot_dict):
    time_bins = plot_dict["2614_stability"]["time"]

    fig = plt.figure(figsize=(8, 6))
    if len(time_bins) == 0:
        plt.close()
        return fig

    th_shift, th_shift_err = _percent_shift(
        plot_dict["2614_stability"]["energy"], plot_dict["2614_stability"]["spread"]
    )
    if th_shift is not None:
        plt.step(time_bins, th_shift, where="mid", label="2.6 MeV peak", color="blue")
        plt.fill_between(
            time_bins,
            th_shift - th_shift_err,
            th_shift + th_shift_err,
            step="mid",
            alpha=0.1,
            color="blue",
        )

    lep_shift, lep_shift_err = _percent_shift(
        plot_dict["583_stability"]["energy"], plot_dict["583_stability"]["spread"]
    )
    if lep_shift is not None:
        plt.step(
            time_bins, lep_shift, where="mid", label="583 keV peak", color="orange"
        )
        plt.fill_between(
            time_bins,
            lep_shift - lep_shift_err,
            lep_shift + lep_shift_err,
            step="mid",
            alpha=0.1,
            color="orange",
        )

    pulser_counts = plot_dict["pulser_stability"]["energy"]
    pulser_spread = plot_dict["pulser_stability"]["spread"]
    pulser_shift, pulser_shift_err = _percent_shift(pulser_counts, pulser_spread)
    if pulser_shift is not None:
        pulser_mean = np.mean(pulser_counts[~np.isnan(pulser_counts)][:10])
        plt.step(
            time_bins,
            pulser_shift,
            where="mid",
            color="red",
            label=f"{pulser_mean:.0f} keV Pulser",
        )
        plt.fill_between(
            time_bins,
            pulser_shift - pulser_shift_err,
            pulser_shift + pulser_shift_err,
            step="mid",
            alpha=0.1,
            color="red",
        )

    # Label ticks from the data timestamps, not the matplotlib-generated tick
    # locations, which can extend beyond the valid timestamp range.
    start = datetime.fromtimestamp(time_bins[0], tz=_BERLIN_TZ)
    plt.xlabel(f"Time (CET/CEST) starting : {start.strftime('%d/%m/%y %H:%M')}")
    plt.ylabel("% Shift")
    ticks, _ = plt.xticks()
    in_range = [t for t in ticks if time_bins[0] <= t <= time_bins[-1]]
    plt.xticks(
        in_range,
        [
            datetime.fromtimestamp(tick, tz=_BERLIN_TZ).strftime("%H:%M")
            for tick in in_range
        ],
    )
    plt.xlim([time_bins[0] - 10, time_bins[-1] + 10])

    plt.grid(which="both")
    plt.legend()
    plt.ylim([-1, 1])
    plt.close()
    return fig
