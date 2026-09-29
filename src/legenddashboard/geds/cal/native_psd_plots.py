"""Bokeh A/E and LQ detail plots drawn from saved data and par-file fits.

Histograms and sweep points come from the plot-data LH5 (``aoe/<set>/<plot>_data``,
``lq/<set>/<plot>_data``); fitted curves are re-evaluated with pygama from the
``results.aoe.<set>`` / ``results.lq.<set>`` entries of the par file.
"""

from __future__ import annotations

import numpy as np
import pygama.math.distributions as pgd
from bokeh.models import (
    BoxAnnotation,
    ColorBar,
    ColumnDataSource,
    LogColorMapper,
    Span,
    Whisker,
)
from bokeh.palettes import Viridis256

from legenddashboard.geds.cal.fit_funcs import eval_fit
from legenddashboard.geds.cal.native_plots import _grid, _small_figure
from legenddashboard.geds.phy.plot_style import (
    MPL_CYCLE,
    TAB20,
    empty_figure,
    finish_legend,
    make_figure,
)

DEP = 1592.5


def _centres(edges):
    edges = np.asarray(edges, dtype=float)
    return (edges[1:] + edges[:-1]) / 2


def _log_counts(counts):
    counts = np.asarray(counts, dtype=float)
    return np.where(counts > 0, counts, np.nan)  # log axes drop empty bins


def _image(p, hist, *, title_bar="Counts"):
    """Draw a ``{"counts", "x_edges", "y_edges"}`` 2D histogram with log colours."""
    counts = np.asarray(hist["counts"], dtype=float).T.copy()
    counts[counts <= 0] = np.nan
    x, y = np.asarray(hist["x_edges"]), np.asarray(hist["y_edges"])
    high = np.nanmax(counts) if np.isfinite(counts).any() else 1.0
    mapper = LogColorMapper(
        Viridis256, low=1, high=max(high, 1.0), nan_color=(0, 0, 0, 0)
    )
    p.image(image=[counts], x=x[0], y=y[0], dw=x[-1] - x[0], dh=y[-1] - y[0],
            color_mapper=mapper)  # fmt: skip
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.add_layout(ColorBar(color_mapper=mapper, title=title_bar), "right")
    p.hover.tooltips = [("x", "$x{0.000}"), ("y", "$y{0.000}"), ("counts", "@image")]
    return p


def _errorbars(p, x, y, err, color, label=None, marker="circle"):
    src = ColumnDataSource({"x": x, "y": y, "lo": y - err, "hi": y + err})
    kwargs = {"legend_label": label} if label else {}
    p.scatter("x", "y", source=src, marker=marker, size=6, color=color, **kwargs)
    p.add_layout(Whisker(source=src, base="x", upper="hi", lower="lo",
                         line_color=color))  # fmt: skip


def plot_spectra(spec, title, labels):
    """
    Energy spectra before and after a PSD cut.

    Parameters
    ----------
    spec : dict
        ``spectrum_data``: ``edges`` plus one counts array per key of *labels*.
    title : str
        Plot title.
    labels : dict
        ``{key: legend label}`` of the histograms to draw.

    Returns
    -------
    bokeh.plotting.figure
        Log-y spectra.
    """
    if not spec:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, y_axis_type="log", x_axis_label="Energy (keV)",
                    y_axis_label="Counts")  # fmt: skip
    x = _centres(spec["edges"])
    for i, (key, label) in enumerate(labels.items()):
        p.step(x, _log_counts(spec[key]), mode="center", color=MPL_CYCLE[i],
               legend_label=label)  # fmt: skip
    p.hover.tooltips = [("E (keV)", "$x{0.0}"), ("counts", "$y{0}")]
    finish_legend(p, "top_right")
    return p


AOE_SPECTRA = {"before": "before PSD", "low_cut": "low side cut",
               "double_cut": "double sided cut", "rejected": "rejected"}  # fmt: skip
LQ_SPECTRA = {
    "before": "before PSD",
    "after_cut": "after LQ cut",
    "rejected": "rejected",
}


def plot_sf_vs_energy(sf, title):
    """Survival fraction of a PSD cut against energy."""
    if not sf:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Energy (keV)", y_axis_label="Survival (%)",
                    y_range=(0, 100))  # fmt: skip
    p.step(_centres(sf["edges"]), sf["sf"], mode="center", color=MPL_CYCLE[0])
    p.hover.tooltips = [("E (keV)", "$x{0.0}"), ("SF (%)", "$y{0.0}")]
    return p


def plot_classifier(hist, title, ylabel):
    """2D energy vs PSD classifier histogram."""
    if not hist:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Energy (keV)", y_axis_label=ylabel)
    return _image(p, hist)


def plot_dt_dep(maps, title):
    """A/E vs drift time for each energy range, in a grid."""
    if not maps:
        return empty_figure(f"{title} | no data")
    figs = []
    for name, hist in maps.items():
        p = _small_figure(name, x_axis_label="A/E", y_axis_label="Drift time (ns)")
        figs.append(_image(p, hist))
    return _grid([figs[i : i + 2] for i in range(0, len(figs), 2)])


def plot_compt_bands(bands, title, xlabel):
    """Density-normalised A/E histograms of Compton bands, overlaid."""
    if not bands:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label=xlabel, y_axis_label="Density")
    for i, (name, band) in enumerate(bands.items()):
        p.step(_centres(band["edges"]), band["counts"], mode="center",
               color=TAB20[i % len(TAB20)], legend_label=name)  # fmt: skip
    finish_legend(p, "top_left", ncols=2)
    return p


def plot_energy_corr(points, fit_results, title, which):
    """
    Compton-band A/E mean or width against energy with the fitted curve.

    Parameters
    ----------
    points : dict
        ``mean_fit_data`` / ``sigma_fit_data`` (``energy``, ``mean``, ``sigma``, errors).
    fit_results : dict
        ``results.aoe.<set>.correction_fit_results`` from the par file.
    title : str
        Plot title.
    which : {"mean", "sigma"}
        Quantity to plot.

    Returns
    -------
    bokeh.models.GridPlot or bokeh.plotting.figure
        Points and curve above the percentage residuals.
    """
    if not points:
        return empty_figure(f"{title} | no data")
    fit = (fit_results or {}).get("mean_fits" if which == "mean" else "SigmaFits")
    x = np.asarray(points["energy"], dtype=float)
    y = np.asarray(points[which], dtype=float)
    err = np.asarray(points[f"{which}_err"], dtype=float)
    top = make_figure(title, width=900, height=380,
                      y_axis_label="A/E mean" if which == "mean" else "A/E sigma")  # fmt: skip
    _errorbars(top, x, y, err, MPL_CYCLE[0], "bands")
    dep = (fit_results or {}).get("dep_fit") or {}
    key = "mu" if which == "mean" else "sigma"
    if key in dep.get("pars", {}):
        _errorbars(top, np.array([DEP]), np.array([dep["pars"][key]]),
                   np.array([dep.get("errs", {}).get(key, 0.0)]), MPL_CYCLE[2], "DEP")  # fmt: skip
    bottom = make_figure(width=900, height=200, x_range=top.x_range,
                         x_axis_label="Energy (keV)", y_axis_label="Residual (%)")  # fmt: skip
    model = eval_fit(fit, x) if fit else None
    if model is not None:
        xs = np.linspace(x.min(), x.max(), 200)
        top.line(xs, eval_fit(fit, xs), color=MPL_CYCLE[1], line_width=2,
                 legend_label=fit.get("func") or fit.get("function"))  # fmt: skip
        bottom.scatter(x, 100 * (y - model) / model, size=6, color=MPL_CYCLE[0])
        bottom.add_layout(Span(location=0, dimension="width", line_color="gray"))
    finish_legend(top, "top_right")
    return _grid([[top], [bottom]])


def plot_cut_fit(cut, title):
    """DEP survival fraction against A/E cut value with the fitted sigmoid."""
    if not cut:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Cut value", y_axis_label="DEP survival (%)")
    x = np.asarray(cut["cut_vals"], dtype=float)
    _errorbars(
        p, x, np.asarray(cut["sf"]), np.asarray(cut["sf_err"]), MPL_CYCLE[0], "sweep"
    )
    sigmoid = {
        "module": "pygama.pargen.AoE_cal",
        "function": cut["function"],
        "pars": cut["pars"],
    }
    xs = np.linspace(x.min(), x.max(), 300)
    ys = eval_fit(sigmoid, xs)
    if ys is not None:
        p.line(xs, ys, color=MPL_CYCLE[1], line_width=2, legend_label="sigmoid fit")
    p.add_layout(Span(location=cut["low_cut"], dimension="height", line_color="red",
                      line_dash="dashed"))  # fmt: skip
    p.add_layout(Span(location=100 * cut["dep_acc"], dimension="width", line_color="red",
                      line_dash="dashed"))  # fmt: skip
    finish_legend(p, "bottom_right")
    return p


def plot_survival_curves(curves, title, cut_key="low_cut"):
    """Survival fraction against cut value for each peak, with the chosen cut."""
    if not curves or not curves.get("peaks"):
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Cut value", y_axis_label="Survival (%)",
                    y_range=(0, 105))  # fmt: skip
    for i, (peak, pts) in enumerate(
        sorted(curves["peaks"].items(), key=lambda kv: float(kv[0]))
    ):
        x = np.asarray(pts["cut_vals"], dtype=float)
        y = np.asarray(pts["sf"], dtype=float)
        p.line(x, y, color=MPL_CYCLE[i % 10], legend_label=f"{float(peak):.1f} keV")
        _errorbars(p, x, y, np.asarray(pts["sf_err"], dtype=float), MPL_CYCLE[i % 10])
    p.add_layout(Span(location=curves[cut_key], dimension="height", line_color="black"))
    finish_legend(p, "top_right")
    return p


def plot_time_series(series, title, value, ylabel):
    """Per-run value ± error from a par-file time series (``{tstamp: {...}}``)."""
    rows = [(k, v) for k, v in (series or {}).items() if v.get(value) is not None]
    if not rows:
        return empty_figure(f"{title} | no data")
    labels = [k for k, _ in rows]
    y = np.array([v[value] for _, v in rows], dtype=float)
    err = np.array([v.get(f"{value}_err") or 0.0 for _, v in rows], dtype=float)
    p = make_figure(title, x_range=labels, y_axis_label=ylabel)
    src = ColumnDataSource({"x": labels, "y": y, "lo": y - err, "hi": y + err})
    p.scatter("x", "y", source=src, size=8, color=MPL_CYCLE[0])
    p.add_layout(Whisker(source=src, base="x", upper="hi", lower="lo"))
    return p


def plot_lq_drift_time(hist, rt_correction, title):
    """LQ vs drift time in the DEP with the correction line and selection box."""
    if not hist:
        return empty_figure(f"{title} | no data")
    p = make_figure(title, x_axis_label="Drift time (ns)", y_axis_label="LQ")
    _image(p, hist)
    if rt_correction is not None and len(rt_correction) >= 2:
        xs = np.linspace(hist["x_edges"][0], hist["x_edges"][-1], 100)
        p.line(xs, rt_correction[0] * xs + rt_correction[1], color="red", line_width=2)
    dt, lq = hist.get("dt_range"), hist.get("lq_range")
    if dt is not None and lq is not None:
        p.add_layout(BoxAnnotation(left=dt[0], right=dt[1], bottom=lq[0], top=lq[1],
                                   fill_alpha=0, line_color="black"))  # fmt: skip
    return p


def plot_lq_cut_fit(hist, cut_fit_pars, title):
    """Sideband-subtracted DEP LQ histogram with the Gaussian cut fit."""
    if not hist:
        return empty_figure(f"{title} | no data")
    edges = np.asarray(hist["edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    x = _centres(edges)
    top = _small_figure(title, width=900, height=380, y_axis_label="Counts")
    top.step(x, counts, mode="center", color=MPL_CYCLE[0], legend_label="data")
    bottom = _small_figure(
        height=200, width=900, x_range=top.x_range, x_axis_label="LQ"
    )
    bottom.yaxis.axis_label = "Residual"
    if cut_fit_pars:
        mu, sigma = cut_fit_pars.get("mu"), cut_fit_pars.get("sigma")
        model = pgd.gaussian.pdf_norm(x, edges[0], edges[-1], mu, sigma)
        model = model * np.diff(edges) * counts.sum()
        top.line(
            x, model, color=MPL_CYCLE[1], line_width=2, legend_label="Gaussian fit"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            bottom.scatter(x, np.where(model > 0, (counts - model) / model, np.nan),
                           marker="square", size=5, color=MPL_CYCLE[0])  # fmt: skip
    finish_legend(top, "top_right")
    return _grid([[top], [bottom]])
