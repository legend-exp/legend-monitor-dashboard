"""Bokeh DSP-parameter detail plots (PZ, filter optimisation, noise, DPLMS).

Drawn from the dataflow's ``-plt_dsp.lh5`` plot data: ``pz/<plot>_data``,
``<filter>_optimisation/data``, ``noise_optimisation/nopt/...`` and ``dplms/...``.
"""

from __future__ import annotations

import numpy as np
from bokeh.models import ColumnDataSource, Span, Whisker

from legenddashboard.geds.cal.native_plots import _grid, _small_figure
from legenddashboard.geds.cal.native_psd_plots import _centres, _image, _log_counts
from legenddashboard.geds.phy.plot_style import (
    MPL_CYCLE,
    TAB20,
    empty_figure,
    finish_legend,
    make_figure,
)

_WF_ALPHA = 0.4


def plot_waveforms(data, title, ylabel="ADU"):
    """
    Overlay a sample of waveforms.

    Parameters
    ----------
    data : dict
        ``{"waveforms" (n, n_samples), "samples"?, "xlim"?, "ylim"?}``.
    title, ylabel : str
        Plot title and y-axis label.

    Returns
    -------
    bokeh.plotting.figure
        One line per waveform.
    """
    if not data or "waveforms" not in data:
        return empty_figure(f"{title} | no data")
    wfs = np.asarray(data["waveforms"], dtype=float)
    x = np.asarray(data.get("samples", np.arange(wfs.shape[1])), dtype=float)
    p = make_figure(title, x_axis_label="Samples", y_axis_label=ylabel)
    p.multi_line([x] * len(wfs), list(wfs), line_alpha=_WF_ALPHA,
                 color=[TAB20[i % len(TAB20)] for i in range(len(wfs))])  # fmt: skip
    for rng, key in ((p.x_range, "xlim"), (p.y_range, "ylim")):
        lim = np.asarray(data.get(key, [np.nan, np.nan]), dtype=float)
        if np.isfinite(lim).all():
            rng.start, rng.end = lim
    p.hover.tooltips = [("sample", "$x{0}"), ("value", "$y{0.0000}")]
    return p


def plot_slopes(data, title):
    """Tail-slope histogram with the fitted mode, and the zoom around it."""
    if not data:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, width=600, x_axis_label="Slope", y_axis_label="Counts")
    p.step(_centres(data["edges"]), data["counts"], mode="center", color=MPL_CYCLE[0])
    if "mode" not in data:
        return p
    p.add_layout(Span(location=data["mode"], dimension="height", line_color="red"))
    inset = data.get("inset")
    if not inset:
        return p
    z = _small_figure("mode ± 4 sigma", x_axis_label="Slope")
    z.step(_centres(inset["edges"]), inset["counts"], mode="center", color=MPL_CYCLE[0])
    z.add_layout(Span(location=data["mode"], dimension="height", line_color="red"))
    return _grid([[p, z]])


def _optimiser_markers(p, data, dim_x, dim_y=None):
    """Samples, failed samples, initial samples and the optimum."""

    def xy(points, values):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if dim_y is None:
            return points[:, dim_x], np.asarray(values, dtype=float)
        return points[:, dim_x], points[:, dim_y]

    x, y = xy(data["samples_x"], data["samples_y"])
    failed = np.asarray(data.get("failed", np.zeros(len(x), bool)), dtype=bool)
    p.scatter(
        x[~failed], y[~failed], size=7, color=MPL_CYCLE[0], legend_label="samples"
    )
    if failed.any():
        p.scatter(x[failed], y[failed], size=7, color="green", legend_label="failed")
    if "init_x" in data:
        ix, iy = xy(data["init_x"], data.get("init_y", []))
        p.scatter(ix, iy, size=8, color="red", legend_label="initial")
    ox, oy = xy([data["optimal_x"]], [data["y_min"]])
    p.scatter(ox, oy, size=11, marker="star", color="orange", legend_label="optimum")


def plot_optimiser(data, title, which="kernel"):
    """
    Gaussian-process prediction or acquisition function of a filter optimisation.

    Parameters
    ----------
    data : dict
        ``<filter>_optimisation/data`` (see pygama ``BayesianOptimizer.get_plot_data``).
    title : str
        Plot title.
    which : {"kernel", "acq"}
        GP mean (± std in 1D) or the acquisition function.

    Returns
    -------
    bokeh.plotting.figure
        Line (1D) or image (2D) with the sample points.
    """
    if not data:
        return empty_figure(f"{title} | no data")
    labels = data.get("labels", {})
    values = data["mean"] if which == "kernel" else data["acq"]
    if "grid" in data:  # 1D
        x = np.asarray(data["grid"], dtype=float)
        ylabel = "Kernel value" if which == "kernel" else "Acquisition value"
        p = make_figure(title, x_axis_label=labels.get("0", ""), y_axis_label=ylabel)
        if which == "kernel":
            std = np.asarray(data["std"], dtype=float)
            p.varea(x, values - std, values + std, alpha=0.15, color=MPL_CYCLE[0])
        p.line(x, values, color=MPL_CYCLE[1], line_width=2)
        _optimiser_markers(p, data, 0)
    else:  # 2D: dim 1 along x, dim 0 along y, as in the pygama plot
        hist = {"counts": np.asarray(values, dtype=float).T,
                "x_edges": _edges(data["grid_1"]), "y_edges": _edges(data["grid_0"])}  # fmt: skip
        p = make_figure(title, x_axis_label=labels.get("1", ""),
                        y_axis_label=labels.get("0", ""))  # fmt: skip
        _image(p, hist, title_bar="Kernel" if which == "kernel" else "Acquisition",
               log=False)  # fmt: skip
        _optimiser_markers(p, data, 1, 0)
    finish_legend(p, "top_right")
    return p


def _edges(centres):
    c = np.asarray(centres, dtype=float)
    step = c[1] - c[0] if len(c) > 1 else 0.1
    return np.append(c - step / 2, c[-1] + step / 2)


def plot_nopt_optimization(data, title, filter_name):
    """Noise FOM against filter parameter with the spline and the chosen value."""
    if not data:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label=f"{filter_name} parameter (µs)",
                    y_axis_label="FOM (ADC)")  # fmt: skip
    x, y, err = (np.asarray(data[k], dtype=float) for k in ("par", "fom", "fom_err"))
    src = ColumnDataSource({"x": x, "y": y, "lo": y - err, "hi": y + err})
    p.scatter(
        "x",
        "y",
        source=src,
        marker="x",
        size=8,
        color=MPL_CYCLE[0],
        legend_label="samples",
    )
    p.add_layout(Whisker(source=src, base="x", upper="hi", lower="lo"))
    p.line(data["spline_x"], data["spline_y"], color="black", line_dash="dotted",
           legend_label="fit")  # fmt: skip
    best = ColumnDataSource({"y": [data["best_val"]], "lo": [data["best_par"] - data["best_par_err"]],
                             "hi": [data["best_par"] + data["best_par_err"]], "x": [data["best_par"]]})  # fmt: skip
    p.scatter("x", "y", source=best, size=9, color="red",
              legend_label=f"best: {data['best_par']:.2f} ± {data['best_par_err']:.2f} µs")  # fmt: skip
    p.add_layout(Whisker(source=best, base="y", upper="hi", lower="lo", dimension="width",
                         line_color="red"))  # fmt: skip
    finish_legend(p, "top_right")
    return p


def plot_nopt_distributions(data, title):
    """Energy distribution for each grid point of the noise optimisation."""
    if not data:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Energy (ADC)", y_axis_label="Counts")
    for i, (par, hist) in enumerate(sorted(data.items(), key=lambda kv: float(kv[0]))):
        p.step(_centres(hist["edges"]), hist["counts"], mode="center",
               color=TAB20[i % len(TAB20)], legend_label=f"{float(par):.1f} µs")  # fmt: skip
    finish_legend(p, "top_right", ncols=2)
    return p


def plot_fft(fft, title):
    """Mean noise power spectral density of one channel."""
    if not fft:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_type="log", y_axis_type="log",
                    x_axis_label="Frequency (MHz)", y_axis_label="Power spectral density")  # fmt: skip
    p.line(
        np.asarray(fft["frequency"])[1:], np.asarray(fft["psd"])[1:], color=MPL_CYCLE[0]
    )
    p.hover.tooltips = [("Freq (MHz)", "$x{0.000}"), ("PSD", "$y{0.00e}")]
    return p


def plot_dplms_selection(data, title):
    """Rough-energy spectra before/after selection and the waveform-shape cuts."""
    if not data:
        return empty_figure(f"{title} | no data")
    figs = []
    rough = data.get("rough_energy")
    if rough:
        p = _small_figure("rough energy", y_axis_type="log", x_axis_label="ADC")
        x = _centres(rough["edges"])
        p.step(
            x,
            _log_counts(rough["initial"]),
            mode="center",
            color="blue",
            legend_label="initial",
        )
        p.step(
            x,
            _log_counts(rough["selected"]),
            mode="center",
            color="red",
            legend_label="selected",
        )
        finish_legend(p, "top_right")
        figs.append(p)
    for par in ("centroid", "peak_pos", "risetime"):
        hist = data.get(par)
        if not hist:
            continue
        p = _small_figure(par, y_axis_type="log", x_axis_label=par)
        p.step(_centres(hist["edges"]), _log_counts(hist["counts"]), mode="center",
               color=MPL_CYCLE[0])  # fmt: skip
        for cut in np.asarray(hist["cut"], dtype=float):
            p.add_layout(Span(location=cut, dimension="height", line_color="black",
                              line_dash="dotted"))  # fmt: skip
        figs.append(p)
    return _grid([figs[i : i + 2] for i in range(0, len(figs), 2)])


def plot_dplms_filter(dplms, title):
    """DPLMS filter coefficients."""
    coeffs = (dplms or {}).get("coefficients")
    if coeffs is None or not np.size(coeffs):
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Sample", y_axis_label="Coefficient")
    p.line(np.arange(len(coeffs)), coeffs, color="red", legend_label="filter")
    p.add_layout(
        Span(location=0, dimension="width", line_color="black", line_dash="dotted")
    )
    finish_legend(p, "top_right")
    return p
