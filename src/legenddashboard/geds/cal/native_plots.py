"""Bokeh calibration detail plots drawn from saved data and par-file fits.

Histograms come from the plot-data LH5 (see ``plot_data``); fit curves are
re-evaluated from the ``results`` and ``pars`` sections of the par files
(see ``fit_funcs``), so these replace the pickled matplotlib figures.
"""

from __future__ import annotations

import re

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColorBar, ColumnDataSource, LogColorMapper, Span, Whisker
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from legenddashboard.geds.cal.fit_funcs import eval_expression, eval_fit, peak_counts
from legenddashboard.geds.phy.plot_style import (
    MPL_CYCLE,
    empty_figure,
    finish_legend,
    make_figure,
    style_figure,
)

QBB = 2039.0
EXCLUDED_FROM_ERES = (511.0, 1592.53, 2103.53)  # as pygama's plot_eres_fit
_IDENT = re.compile(r"[A-Za-z_]\w*")
_TOOLS = "pan,box_zoom,wheel_zoom,hover,reset,save"


def _small_figure(title="", width=420, height=280, **kwargs):
    p = figure(width=width, height=height, tools=_TOOLS, **kwargs)
    return style_figure(p, title)


def _grid(rows):
    return gridplot(rows, toolbar_location="right", sizing_mode="scale_width")


def calibrate(cal_op, x):
    """Apply a ``pars.operations`` calibration (``a + b * cuspEmax_ctc``) to ``x``."""
    names = set(_IDENT.findall(cal_op["expression"])) - set(cal_op["parameters"])
    return eval_expression(
        cal_op["expression"], cal_op["parameters"], sorted(names)[0], x
    )


def _valid_fits(pk_fits):
    """``(peak_kev, fit)`` pairs of valid fits, sorted by energy."""
    out = []
    for peak, fit in (pk_fits or {}).items():
        pars = fit.get("parameters") or {}
        if fit.get("validity", True) and all(
            v is not None and np.isfinite(v) for v in pars.values()
        ):
            out.append((float(peak), fit))
    return sorted(out, key=lambda pf: pf[0])


def _match_fit(pk_fits, peak):
    for key, fit in (pk_fits or {}).items():
        if abs(float(key) - peak) < 0.01:
            return fit
    return None


def plot_peak_fits(peak_hists, pk_fits, cal_op, title, ncols=3):
    """
    Per-peak histograms with the stored fit and its pulls.

    Parameters
    ----------
    peak_hists : dict
        ``{peak_kev: {"edges", "counts"}}`` in uncalibrated units.
    pk_fits : dict
        ``results.ecal.<param>.pk_fits`` from the par file.
    cal_op : dict
        The ``pars.operations`` entry of the calibrated parameter.
    title : str
        Title prefix.
    ncols : int
        Peaks per row.

    Returns
    -------
    bokeh.models.GridPlot
        One histogram-plus-pulls column per peak.
    """
    if not peak_hists:
        return empty_figure(f"{title} | no peak histograms")
    tops, bottoms = [], []
    for key in sorted(peak_hists, key=float):
        peak = float(key)
        edges = np.asarray(peak_hists[key]["edges"], dtype=float)
        counts = np.asarray(peak_hists[key]["counts"], dtype=float)
        centres = (edges[1:] + edges[:-1]) / 2
        x_kev = calibrate(cal_op, centres)
        fit = _match_fit(pk_fits, peak)
        label = f"{peak:.1f} keV"
        if fit is not None and fit.get("p_value") is not None:
            label += f" | p = {fit['p_value']:.3f}"
        top = _small_figure(label, y_axis_label="Counts")
        top.step(x_kev, counts, mode="center", color=MPL_CYCLE[0], legend_label="data")
        bottom = _small_figure(
            height=130, x_range=top.x_range, x_axis_label="Energy (keV)"
        )
        bottom.yaxis.axis_label = "Pull"
        model = None
        if fit is not None and fit.get("parameters"):
            model = peak_counts(fit, centres, edges[1] - edges[0])
        if model is not None:
            valid = fit.get("validity", True)
            top.line(x_kev, model, color=MPL_CYCLE[1], line_width=2,
                     line_dash="solid" if valid else "dashed",
                     legend_label="fit" if valid else "fit (invalid)")  # fmt: skip
            with np.errstate(divide="ignore", invalid="ignore"):
                pulls = np.where(model > 0, (counts - model) / np.sqrt(model), np.nan)
            bottom.scatter(x_kev, pulls, size=3, color=MPL_CYCLE[0])
            bottom.add_layout(Span(location=0, dimension="width", line_color="gray"))
        top.hover.tooltips = [("E (keV)", "$x{0.00}"), ("counts", "$y{0}")]
        finish_legend(top, "top_right")
        tops.append(top)
        bottoms.append(bottom)
    rows = []
    for i in range(0, len(tops), ncols):
        rows += [tops[i : i + ncols], bottoms[i : i + ncols]]
    return _grid(rows)


def plot_cal_fit(pk_fits, cal_op, title):
    """
    Fitted peak positions against the calibration curve, with keV residuals.

    Parameters
    ----------
    pk_fits : dict
        ``results.ecal.<param>.pk_fits`` from the par file.
    cal_op : dict
        The ``pars.operations`` entry of the calibrated parameter.
    title : str
        Plot title.

    Returns
    -------
    bokeh.models.GridPlot
        Calibration panel above a residual panel.
    """
    fits = [(pk, f) for pk, f in _valid_fits(pk_fits) if f.get("position") is not None]
    if not fits:
        return empty_figure(f"{title} | no valid peak fits")
    kev = np.array([pk for pk, _ in fits])
    pos = np.array([f["position"] for _, f in fits])
    pos_err = np.array([f.get("position_uncertainty") or 0.0 for _, f in fits])
    cal_pos = calibrate(cal_op, pos)
    err_kev = np.abs(calibrate(cal_op, pos + pos_err) - cal_pos)
    resid = cal_pos - kev

    top = make_figure(title, width=900, height=380, y_axis_label="Energy (ADC)")
    adc = np.linspace(0, pos.max() * 1.05, 50)
    top.line(
        calibrate(cal_op, adc), adc, color=MPL_CYCLE[2], legend_label="calibration"
    )
    top.scatter(kev, pos, marker="x", size=10, color=MPL_CYCLE[0], legend_label="peaks")
    top.hover.tooltips = [("E (keV)", "$x{0.0}"), ("ADC", "$y{0.0}")]
    finish_legend(top, "top_left")

    src = ColumnDataSource(
        {"kev": kev, "resid": resid, "lo": resid - err_kev, "hi": resid + err_kev}
    )
    bottom = make_figure(
        width=900,
        height=200,
        x_range=top.x_range,
        x_axis_label="Energy (keV)",
        y_axis_label="Residual (keV)",
    )
    bottom.scatter("kev", "resid", source=src, size=7, color=MPL_CYCLE[0])
    bottom.add_layout(Whisker(source=src, base="kev", upper="hi", lower="lo"))
    bottom.add_layout(Span(location=0, dimension="width", line_color="gray"))
    bottom.hover.tooltips = [("E (keV)", "@kev{0.0}"), ("residual", "@resid{0.000}")]
    return _grid([[top], [bottom]])


def plot_fwhm_fit(pk_fits, eres_linear, eres_quadratic, title):
    """
    Peak FWHMs with the stored linear and quadratic resolution curves.

    Parameters
    ----------
    pk_fits : dict
        ``results.ecal.<param>.pk_fits`` from the par file.
    eres_linear, eres_quadratic : dict or None
        ``results.ecal.<param>.eres_linear`` / ``eres_quadratic``.
    title : str
        Plot title.

    Returns
    -------
    bokeh.models.GridPlot
        Resolution panel above the normalised residuals of the linear curve.
    """
    fits = [
        (pk, f)
        for pk, f in _valid_fits(pk_fits)
        if f.get("fwhm_in_kev") is not None and np.isfinite(f["fwhm_in_kev"])
    ]
    if not fits:
        return empty_figure(f"{title} | no valid peak fits")
    kev = np.array([pk for pk, _ in fits])
    fwhm = np.array([f["fwhm_in_kev"] for _, f in fits])
    err = np.array([f.get("fwhm_err_in_kev") or 0.0 for _, f in fits])
    used = ~np.isclose(kev[:, None], EXCLUDED_FROM_ERES, atol=1).any(axis=1)

    top = make_figure(title, width=900, height=380, y_axis_label="FWHM (keV)")
    xs = np.arange(200, 2700, 10.0)
    curves = {}
    for name, eres, color in (
        ("Linear", eres_linear, MPL_CYCLE[1]),
        ("Quadratic", eres_quadratic, MPL_CYCLE[2]),
    ):
        ys = eval_fit(eres, xs) if eres else None
        if ys is None:
            continue
        curves[name] = eres
        at_qbb = eval_fit(eres, [QBB])[0]
        top.line(xs, ys, color=color, line_width=2,
                 legend_label=f"{name}: {float(at_qbb):.2f} keV at Qbb")  # fmt: skip
    for mask, marker, label in ((used, "circle", "fitted"), (~used, "x", "not fitted")):
        if mask.any():
            src = ColumnDataSource({"kev": kev[mask], "fwhm": fwhm[mask],
                                    "lo": fwhm[mask] - err[mask], "hi": fwhm[mask] + err[mask]})  # fmt: skip
            top.scatter("kev", "fwhm", source=src, marker=marker, size=8,
                        color=MPL_CYCLE[0], legend_label=label)  # fmt: skip
            top.add_layout(Whisker(source=src, base="kev", upper="hi", lower="lo"))
    top.add_layout(Span(location=QBB, dimension="height", line_color="gray", line_dash="dashed"))  # fmt: skip
    top.hover.tooltips = [("E (keV)", "$x{0.0}"), ("FWHM (keV)", "$y{0.000}")]
    finish_legend(top, "top_left")

    bottom = make_figure(
        width=900,
        height=200,
        x_range=top.x_range,
        x_axis_label="Energy (keV)",
        y_axis_label="Norm. residual",
    )
    if "Linear" in curves:
        lin = eval_fit(curves["Linear"], kev)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_res = np.where(err > 0, (fwhm - lin) / err, np.nan)
        bottom.scatter(kev[used], norm_res[used], size=7, color=MPL_CYCLE[0])
        bottom.add_layout(Span(location=0, dimension="width", line_color="gray"))
    return _grid([[top], [bottom]])


def plot_timemap(hist, title, ylabel):
    """
    Log-scaled 2D time-vs-value histogram (energy or baseline).

    Parameters
    ----------
    hist : dict
        ``{"counts" (n_time, n_value), "time_edges" (unix s), "value_edges"}``.
    title, ylabel : str
        Plot title and y-axis label.

    Returns
    -------
    bokeh.plotting.figure
        Image plot with a colour bar.
    """
    if not hist or "counts" not in hist:
        return empty_figure(f"{title} | no data")
    counts = np.asarray(hist["counts"], dtype=float).T.copy()
    counts[counts <= 0] = np.nan  # empty bins transparent, as LogNorm
    t_ms = np.asarray(hist["time_edges"], dtype=float) * 1e3
    v = np.asarray(hist["value_edges"], dtype=float)
    p = make_figure(title, x_datetime=True, y_axis_label=ylabel)
    high = np.nanmax(counts) if np.isfinite(counts).any() else 1.0
    mapper = LogColorMapper(
        Viridis256, low=1, high=max(high, 1.0), nan_color=(0, 0, 0, 0)
    )
    p.image(image=[counts], x=t_ms[0], y=v[0], dw=t_ms[-1] - t_ms[0], dh=v[-1] - v[0],
            color_mapper=mapper)  # fmt: skip
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.add_layout(ColorBar(color_mapper=mapper, title="Counts"), "right")
    p.hover.tooltips = [
        ("time", "$x{%F %H:%M}"),
        (ylabel, "$y{0.00}"),
        ("counts", "@image"),
    ]
    p.hover.formatters = {"$x": "datetime"}
    return p


def plot_survival_frac(sf, title):
    """Survival fraction (%) of the quality cuts against energy."""
    p = make_figure(
        title, x_axis_label="Energy (keV)", y_axis_label="Survival fraction (%)"
    )
    p.step(sf["bins"], sf["sf"], mode="center", color=MPL_CYCLE[0])
    p.hover.tooltips = [("E (keV)", "$x{0.0}"), ("SF (%)", "$y{0.00}")]
    return p


def plot_cut_spectra(spectrum, title):
    """Spectra passing and failing the quality cuts, plus pulser events."""
    p = make_figure(title, y_axis_type="log", x_axis_label="Energy (keV)",
                    y_axis_label="Counts")  # fmt: skip
    bins = np.asarray(spectrum["bins"])
    for key, label, color in (("counts", "After cuts", MPL_CYCLE[0]),
                              ("cut_counts", "Cut", MPL_CYCLE[1]),
                              ("pulser_counts", "Pulser", MPL_CYCLE[3])):  # fmt: skip
        y = np.asarray(spectrum.get(key, []), dtype=float)
        if len(y) != len(bins) or not np.isfinite(y).any():
            continue
        y = np.where(y > 0, y, np.nan)  # log axis
        p.step(bins, y, mode="center", color=color, legend_label=label)
    p.hover.tooltips = [("E (keV)", "$x{0.0}"), ("counts", "$y{0}")]
    finish_legend(p, "top_right")
    return p


def _percent_shift(values, spread):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or np.mean(finite[:10]) == 0:
        return None, None
    mean = np.mean(finite[:10])
    return 100 * (values - mean) / mean, 100 * np.asarray(spread, dtype=float) / mean


def plot_peak_track(ecal_param, title):
    """Percent shift of the 2614 keV, 583 keV and pulser peaks over the run."""
    p = make_figure(title, x_datetime=True, y_axis_label="% shift")
    for key, label, color in (("2614_stability", "2614 keV", MPL_CYCLE[0]),
                              ("583_stability", "583 keV", MPL_CYCLE[1]),
                              ("pulser_stability", "pulser", MPL_CYCLE[3])):  # fmt: skip
        stab = ecal_param.get(key)
        if not stab or len(stab["time"]) == 0:
            continue
        shift, err = _percent_shift(stab["energy"], stab["spread"])
        if shift is None:
            continue
        t = np.asarray(stab["time"], dtype=float) * 1e3
        p.varea(t, shift - err, shift + err, color=color, alpha=0.15)
        p.step(t, shift, mode="center", color=color, line_width=2, legend_label=label)
    p.hover.tooltips = [("time", "$x{%F %H:%M}"), ("shift (%)", "$y{0.000}")]
    p.hover.formatters = {"$x": "datetime"}
    finish_legend(p, "bottom_left")
    return p
