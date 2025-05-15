from __future__ import annotations

import contextlib
import datetime as dtt
import warnings
from datetime import datetime
from pathlib import Path

import colorcet as cc
import numexpr as ne
import numpy as np
import pandas as pd
from bokeh.models import (
    Band,
    ColumnDataSource,
    Label,
    Range1d,
    Span,
    ZoomInTool,
    ZoomOutTool,
)
from bokeh.plotting import figure
from dbetto import Props
from scipy.optimize import minimize

from legenddashboard.util import sorter


def plot_energy(path, run_dict, det, plot, colour, period, cache_data=None):
    cals = []
    times = []
    qbb_adc = None
    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                Path(hit_pars_file_path)
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict

        try:
            hit_dict = hit_pars_dict[det]["pars"]["operations"]["cuspEmax_ctc_cal"]
            if qbb_adc is None:

                def find_qbb_adc(val):
                    return ne.evaluate(
                        "abs(" + "cuspEmax_ctc*a +b" + "-2039)",
                        local_dict=dict({"cuspEmax_ctc": val}, a=0.1, b=0),
                    )

                qbb_adc = minimize(find_qbb_adc, 20000)["x"][0]
            out_data = ne.evaluate(
                f"{hit_dict['expression']}",
                local_dict=dict({"cuspEmax_ctc": qbb_adc}, **hit_dict["parameters"]),
            )
            cals.append(out_data)
            times.append(run_dict[run]["timestamp"])
        except KeyError:
            pass
    if len(cals) > 0:
        cals = np.array(cals)
        plot.step(
            [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            (cals - cals[0]),
            legend_label=det,
            mode="after",
            line_width=2,
            line_color=colour,
        )

        plot.scatter(
            [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            (cals - cals[0]),
            legend_label=det,
            fill_color="white",
            size=8,
            color=colour,
        )

    return plot


def plot_energy_res(
    path, run_dict, det, plot, colour, period, at="Qbb", cache_data=None
):
    reses = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)

            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                hit_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict
        try:
            if at == "Qbb":
                reses.append(
                    hit_pars_dict[det]["results"]["ecal"]["cuspEmax_ctc_cal"][
                        "eres_linear"
                    ]["Qbb_fwhm_in_kev"]
                )
            elif at == "2.6":
                reses.append(
                    hit_pars_dict[det]["results"]["ecal"]["cuspEmax_ctc_cal"][
                        "pk_fits"
                    ][2614.511]["fwhm_in_kev"]
                )
            times.append(run_dict[run]["timestamp"])
        except KeyError:
            pass

    if len(reses) > 0:
        plot.step(
            [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            (reses),
            legend_label=det,
            mode="after",
            line_width=2,
            line_color=colour,
        )

        plot.scatter(
            [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            (reses),
            legend_label=det,
            fill_color="white",
            size=8,
            color=colour,
        )

    return plot


def plot_energy_res_Qbb(path, run_dict, det, plot, colour, period, cache_data=None):
    return plot_energy_res(
        path, run_dict, det, plot, colour, period, at="Qbb", cache_data=cache_data
    )


def plot_energy_res_2614(path, run_dict, det, plot, colour, period, cache_data=None):
    return plot_energy_res(
        path, run_dict, det, plot, colour, period, at="2.6", cache_data=cache_data
    )


def plot_aoe_mean(path, run_dict, det, plot, colour, period, cache_data=None):
    means = []
    mean_errs = []
    reses = []
    res_errs = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                hit_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict
        try:
            det_dict = hit_pars_dict[det]

            means.append(det_dict["results"]["aoe"]["1000-1300keV"][0]["mean"])
            mean_errs.append(det_dict["results"]["aoe"]["1000-1300keV"][0]["mean_err"])
            reses.append(det_dict["results"]["aoe"]["1000-1300keV"][0]["res"])
            res_errs.append(det_dict["results"]["aoe"]["1000-1300keV"][0]["res_err"])
            times.append(run_dict[run]["timestamp"])
        except KeyError:
            continue
    means = np.array(means)
    reses = np.array(reses)
    plot.step(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        100 * (means - means[0]) / reses,
        legend_label=det,
        mode="after",
        line_width=2,
        line_color=colour,
    )

    plot.scatter(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        100 * (means - means[0]) / reses,
        legend_label=det,
        fill_color="white",
        size=8,
        color=colour,
    )

    mean_df = pd.DataFrame(
        {
            "x": [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            "lower": [-40 for value in times],
            "upper": [40 for value in times],
        }
    )

    source = ColumnDataSource(mean_df)
    band = Band(
        base="x",
        lower="lower",
        upper="upper",
        source=source,
        fill_alpha=0.01,
        fill_color="yellow",
    )
    plot.add_layout(band)

    mean_df2 = pd.DataFrame(
        {
            "x": [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
            "lower": [-20 for value in times],
            "upper": [20 for value in times],
        }
    )

    source2 = ColumnDataSource(mean_df2)
    band2 = Band(
        base="x",
        lower="lower",
        upper="upper",
        source=source2,
        fill_alpha=0.02,
        fill_color="green",
    )
    plot.add_layout(band2)
    plot.y_range = Range1d(-100, 100)

    return plot


def plot_aoe_sig(path, run_dict, det, plot, colour, period, cache_data=None):
    means = []
    mean_errs = []
    reses = []
    res_errs = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                hit_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict

        try:
            means.append(
                hit_pars_dict[det]["results"]["aoe"]["1000-1300keV"][0]["mean"]
            )
            mean_errs.append(
                hit_pars_dict[det]["results"]["aoe"]["1000-1300keV"][0]["mean_err"]
            )
            reses.append(hit_pars_dict[det]["results"]["aoe"]["1000-1300keV"][0]["res"])
            res_errs.append(
                hit_pars_dict[det]["results"]["aoe"]["1000-1300keV"][0]["res_err"]
            )
            times.append(run_dict[run]["timestamp"])
        except KeyError:
            pass
    means = np.array(means)
    reses = np.array(reses)
    plot.step(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (reses),
        legend_label=det,
        mode="after",
        line_width=2,
        line_color=colour,
    )

    plot.scatter(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (reses),
        legend_label=det,
        fill_color="white",
        size=8,
        color=colour,
    )

    return plot


def plot_aoe_cut(path, run_dict, det, plot, colour, period, cache_data=None):
    cuts = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                hit_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict

        with contextlib.suppress(KeyError):
            cuts.append(hit_pars_dict[det]["results"]["aoe"]["low_cut"])
            times.append(run_dict[run]["timestamp"])

    cuts = np.array(cuts)
    plot.step(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (cuts),
        legend_label=det,
        mode="after",
        line_width=2,
        line_color=colour,
    )

    plot.scatter(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (cuts),
        legend_label=det,
        fill_color="white",
        size=8,
        color=colour,
    )

    return plot


def plot_tau(path, run_dict, det, plot, colour, period, cache_data=None):
    values = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["dsp"]:
            dsp_pars_dict = cache_data["dsp"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            dsp_pars_file_path = (
                Path(prod_config["paths"]["par_dsp"]) / f"cal/{period}/{run}"
            )

            dsp_pars_path = (
                dsp_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.yaml'
            )

            dsp_pars_dict = Props.read_from(dsp_pars_path)
            cache_data["dsp"][run] = dsp_pars_dict

        with contextlib.suppress(KeyError):
            values.append(float(dsp_pars_dict[det]["pz"]["tau1"][:-3]))
            times.append(run_dict[run]["timestamp"])

    values = np.array(values)
    plot.step(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        100 * (values - values[0]) / values[0],
        legend_label=det,
        mode="after",
        line_width=2,
        line_color=colour,
    )

    plot.scatter(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        100 * (values - values[0]) / values[0],
        legend_label=det,
        fill_color="white",
        size=8,
        color=colour,
    )

    return plot


def plot_ctc_const(path, run_dict, det, plot, colour, period, cache_data=None):
    values = []
    times = []
    for run in run_dict:
        if cache_data is not None and run in cache_data["dsp"]:
            dsp_pars_dict = cache_data["dsp"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            dsp_pars_file_path = (
                Path(prod_config["paths"]["par_dsp"]) / f"cal/{period}/{run}"
            )
            dsp_pars_path = (
                dsp_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_dsp.yaml'
            )

            dsp_pars_dict = Props.read_from(dsp_pars_path)
            cache_data["dsp"][run] = dsp_pars_dict

        with contextlib.suppress(KeyError):
            values.append(
                dsp_pars_dict[det]["ctc_params"]["cuspEmax_ctc"]["parameters"]["a"]
            )
            times.append(run_dict[run]["timestamp"])

    values = np.array(values)
    plot.step(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (values),
        legend_label=det,
        mode="after",
        line_width=2,
        line_color=colour,
    )

    plot.scatter(
        [(datetime.strptime(value, "%Y%m%dT%H%M%SZ")) for value in times],
        (values),
        legend_label=det,
        fill_color="white",
        size=8,
        color=colour,
    )

    return plot


def plot_tracking(
    run_dict,
    path,
    plot_func,
    string,
    period,
    plot_type,
    key="String",
    cache_data=None,
    sort_dets_obj=None,
):
    strings_dict, _, chmap = sorter(
        path,
        run_dict[next(iter(run_dict))]["timestamp"],
        key=key,
        sort_dets_obj=sort_dets_obj,
    )
    string_dets = {}
    for stri in strings_dict:
        dets = []
        for chan in strings_dict[stri]:
            dets.append(chmap[chan]["name"])
        string_dets[stri] = dets

    p = figure(
        width=1000,
        height=400,
        x_axis_type="datetime",
        tools="pan, box_zoom, ywheel_zoom, hover,reset,save",
        active_scroll="ywheel_zoom",
    )
    p.title.text = f"{run_dict[next(iter(run_dict))]['experiment']}-{period} | Cal. Tracking | {plot_type} | {string}"
    p.title.align = "center"
    p.title.text_font_size = "15px"

    level = 1
    zoom_in = ZoomInTool(
        level=level, dimensions="height", factor=0.5
    )  # set specific zoom factor
    zoom_out = ZoomOutTool(level=level, dimensions="height", factor=0.5)
    p.add_tools(zoom_in, zoom_out)
    # p.toolbar.active_drag = None      use this line to activate only hover and ywheel_zoom as active tool

    colours = cc.palette["glasbey_category10"][:100]

    if cache_data is None or len(cache_data) == 0:
        cache_data = {"hit": {}, "dsp": {}}

    for i, det in enumerate(string_dets[string]):
        # with contextlib.suppress(KeyError):
        plot_func(path, run_dict, det, p, colours[i], period, cache_data)

    for run in run_dict:
        sp = Span(
            location=datetime.strptime(run_dict[run]["timestamp"], "%Y%m%dT%H%M%SZ"),
            dimension="height",
            line_color="black",
            line_width=1.5,
        )
        p.add_layout(sp)

        label = Label(
            x=datetime.strptime(run_dict[run]["timestamp"], "%Y%m%dT%H%M%SZ")
            + dtt.timedelta(minutes=200),
            y=0,
            text=run,
        )

        p.add_layout(label)

    p.xaxis.axis_label = "Time"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label_text_font_size = "20px"

    if plot_func == plot_energy:
        p.yaxis.axis_label = "Shift of Qbb in keV "
    elif plot_func == plot_energy_res_Qbb:
        p.yaxis.axis_label = "FWHM at Qbb"
    elif plot_func == plot_energy_res_2614:
        p.yaxis.axis_label = "FWHM of 2.6MeV peak"
    elif plot_func == plot_aoe_mean:
        p.yaxis.axis_label = "% Shift of A/E mean"
    elif plot_func == plot_aoe_sig:
        p.yaxis.axis_label = "Shift of A/E Resolution"
    elif plot_func == plot_aoe_sig:
        p.yaxis.axis_label = "Shift of A/E Low Cut"
    elif plot_func == plot_tau:
        p.yaxis.axis_label = "% Shift PZ const"
    elif plot_func == plot_ctc_const:
        p.yaxis.axis_label = "Shift CT constant"

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def plot_energy_residuals_period(
    run_dict,
    path,
    period,
    key="String",
    download=False,
    cache_data=None,
    sort_dets_obj=None,
):
    strings, _, _ = sorter(
        path,
        run_dict[next(iter(run_dict))]["timestamp"],
        key=key,
        sort_dets_obj=sort_dets_obj,
    )

    peaks = [2614.511, 583.191, 2103.511]

    res = {}

    if cache_data is None or len(cache_data) == 0:
        cache_data = {"hit": {}, "dsp": {}}

    for stri in strings:
        res[stri] = {peak: [] for peak in peaks}
        for detector in strings[stri]:
            res[detector] = {peak: [] for peak in peaks}

    for run in run_dict:
        if cache_data is not None and run in cache_data["hit"]:
            hit_pars_dict = cache_data["hit"][run]
        else:
            prod_config = Path(path) / "dataflow-config.yaml"
            prod_config = Props.read_from(prod_config, subst_pathvar=True)
            hit_pars_file_path = (
                Path(prod_config["paths"]["par_hit"]) / f"cal/{period}/{run}"
            )

            hit_pars_path = (
                hit_pars_file_path
                / f'{run_dict[run]["experiment"]}-{period}-{run}-cal-{run_dict[run]["timestamp"]}-par_hit.yaml'
            )

            hit_pars_dict = Props.read_from(hit_pars_path)
            cache_data["hit"][run] = hit_pars_dict

        for peak in peaks:
            for stri in strings:
                res[stri][peak].append(np.nan)
                for detector in strings[stri]:
                    try:
                        hit_dict = hit_pars_dict[detector]["pars"]["operations"][
                            "cuspEmax_ctc_cal"
                        ]
                        res_dict = hit_pars_dict[detector]["results"]["ecal"][
                            "cuspEmax_ctc_cal"
                        ]
                        out_data = (
                            ne.evaluate(
                                f"{hit_dict['expression']}",
                                local_dict=dict(
                                    {
                                        "cuspEmax_ctc": res_dict["pk_fits"][peak][
                                            "parameters"
                                        ]["mu"]
                                    },
                                    **hit_dict["parameters"],
                                ),
                            )
                            - peak
                        )
                        res[detector][peak].append(out_data)
                    except KeyError:
                        continue

    # p = figure(width=1400, height=600, tools="pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out,hover,reset,save")
    p = figure(
        width=1400,
        height=600,
        tools="pan, box_zoom, ywheel_zoom, hover,reset,save",
        active_scroll="ywheel_zoom",
    )
    p.title.text = f"{run_dict[next(iter(run_dict))]['experiment']}-{period} | Cal. | Energy Residuals"
    p.title.align = "center"
    p.title.text_font_size = "25px"

    level = 1
    zoom_in = ZoomInTool(
        level=level, dimensions="height", factor=0.5
    )  # set specific zoom factor
    zoom_out = ZoomOutTool(level=level, dimensions="height", factor=0.5)
    p.add_tools(zoom_in, zoom_out)
    # p.toolbar.active_drag = None      use this line to activate only hover and ywheel_zoom as active tool

    label_res = [r if "String" not in r else "" for r in list(res)]

    df_plot = pd.DataFrame()
    df_plot["label_res"] = label_res

    for peak in peaks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_plot, y_plot, y_min, y_max = (
                np.arange(1, len(list(res)) + 1, 1),
                [np.nanmean(res[det][peak]) for det in res],
                [
                    np.nanmin(res[det][peak])
                    if len(np.array(res[det][peak])[~np.isnan(res[det][peak])]) > 0
                    else np.nan
                    for det in res
                ],
                [
                    np.nanmax(res[det][peak])
                    if len(np.array(res[det][peak])[~np.isnan(res[det][peak])]) > 0
                    else np.nan
                    for det in res
                ],
            )

        err_xs = []
        err_ys = []

        for x, y, yerr_low, yerr_hi in zip(x_plot, y_plot, y_min, y_max, strict=False):
            err_xs.append((x, x))
            err_ys.append((np.nan_to_num(y - yerr_low), np.nan_to_num(y + yerr_hi)))

        df_plot[f"x_{int(peak)}"] = np.nan_to_num(x_plot)
        df_plot[f"y_{int(peak)}"] = np.nan_to_num(y_plot)
        df_plot[f"y_{int(peak)}_min"] = np.nan_to_num(y_min)
        df_plot[f"y_{int(peak)}_max"] = np.nan_to_num(y_max)
        df_plot[f"err_xs_{int(peak)}"] = err_xs
        df_plot[f"err_ys_{int(peak)}"] = err_ys

    if download:
        return (
            df_plot,
            f"{run_dict[next(iter(run_dict))]['experiment']}-{period}-energy_residuals.csv",
        )

    for peak, peak_color in zip(peaks, ["blue", "green", "red"], strict=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if peak == peaks[0]:
                hover_renderer = p.scatter(
                    x=f"x_{int(peak)}",
                    y=f"y_{int(peak)}",
                    source=df_plot,
                    color=peak_color,
                    size=7,
                    line_alpha=0,
                    legend_label=f"{peak} Average: {np.nanmean([np.nanmean(res[det][peak]) for det in res]):.2f}keV",
                    name=f"{peak} Average: {np.nanmean([np.nanmean(res[det][peak]) for det in res]):.2f}keV",
                )
            else:
                p.scatter(
                    x=f"x_{int(peak)}",
                    y=f"y_{int(peak)}",
                    source=df_plot,
                    color=peak_color,
                    size=7,
                    line_alpha=0,
                    legend_label=f"{peak} Average: {np.nanmean([np.nanmean(res[det][peak]) for det in res]):.2f}keV",
                    name=f"{peak} Average: {np.nanmean([np.nanmean(res[det][peak]) for det in res]):.2f}keV",
                )
            band = Band(
                base=f"x_{int(peak)}",
                lower=f"y_{int(peak)}_min",
                upper=f"y_{int(peak)}_max",
                source=ColumnDataSource(df_plot),
                fill_alpha=0.1,
                fill_color=peak_color,
            )
            p.add_layout(band)

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "detector"
    p.xaxis.axis_label_text_font_size = "20px"
    p.yaxis.axis_label = "peak residuals (keV)"
    p.title.text = f"{run_dict[next(iter(run_dict))]['experiment']}-{period} | Cal. | Energy residuals"
    p.yaxis.axis_label_text_font_size = "20px"

    p.xaxis.major_label_orientation = np.pi / 2
    p.xaxis.ticker = np.arange(1, len(list(res)) + 1, 1)
    p.xaxis.major_label_overrides = {
        i: label_res[i - 1] for i in range(1, len(label_res) + 1, 1)
    }
    p.xaxis.major_label_text_font_style = "bold"

    for stri in strings:
        loc = np.where(np.array(list(res)) == stri)[0][0]
        string_span = Span(
            location=loc + 1, dimension="height", line_color="black", line_width=3
        )
        string_span_label = Label(
            x=loc + 1.5, y=1.1, text=stri, text_font_size="10pt", text_color="blue"
        )
        p.add_layout(string_span_label)
        p.add_layout(string_span)

    p.hover.tooltips = [
        ("Detector", "@label_res"),
        ("2614", "av: @y_2614{0.00} min: @y_2614_min{0.00} max: @y_2614_max{0.00} keV"),
        ("SEP", "av: @y_2103{0.00} min: @y_2103_min{0.00} max: @y_2103_max{0.00} keV"),
        ("583", "av: @y_583{0.00} min: @y_583_min{0.00} max: @y_583_max{0.00} keV"),
    ]
    p.hover.mode = "vline"
    p.hover.renderers = [hover_renderer]

    return p
