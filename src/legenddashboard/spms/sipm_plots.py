"""SiPM figure builders (p.e. spectra for PE-calibration validation).

The spms contract carries ``hist/<flag>_EnergyInPe_dist2d``: per-SiPM
histograms of the per-pulse ``energy_in_pe`` over a fixed 0-5 p.e. range.
Since ``energy_in_pe = pe_a + pe_m * energy``, a valid calibration puts the
single-photoelectron peaks at exactly 1, 2, 3 ... p.e., so the measured
1 p.e. centroid minus 1.0 is the gain drift in p.e. units.
"""

from __future__ import annotations

import numpy as np
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import Range1d, Span

from legenddashboard.geds.phy import plot_style

PE_MARKS = (1, 2, 3, 4, 5)
#: |centroid - 1| beyond which a channel is called drifting / badly drifting
WARN_DEVIATION = 0.05
BAD_DEVIATION = 0.10
#: band the dominant low-energy feature must fall in to be read as the 1 p.e.
#: peak: a few p22 SiPMs peak near 0.45-0.5 p.e. with only a weak shoulder
#: near 0.9, which is not a gain drift to be quoted but a channel to look at
EXPECTED_PEAK = (0.6, 1.5)
_OK, _WARN, _BAD = "seagreen", "darkorange", "red"


def spe_centroid(
    edges, counts, floor=None, window=(0.4, 1.8), smooth=5, half=0.15, valley_max=1.0
):
    """Position of the 1 p.e. peak, or None when it cannot be located.

    The histogram is filled from unmasked pulses, so below the valid-hit
    threshold it is dominated by noise which on some channels is taller than
    the 1 p.e. peak and reaches past that threshold. The search therefore
    walks right from ``floor`` while the (smoothed) spectrum still falls --
    off the noise tail into its valley -- and only then takes the maximum;
    the returned centroid is the counts-weighted mean within ``half`` p.e.
    of that maximum, which resolves below the bin width. The walk stops at
    ``valley_max`` so a tail that outweighs the peak everywhere cannot carry
    it past the peak: the valley preceding the 1 p.e. peak lies below 1 p.e.
    even on channels whose gain has drifted high.

    Parameters
    ----------
    edges : numpy.ndarray
        Bin edges of the spectrum, length ``len(counts) + 1``.
    counts : numpy.ndarray
        Bin contents.
    floor : float, optional
        Lower bound of the search, normally the channel's ``threshold_a``.
    window : tuple
        (low, high) p.e. bounds the 1 p.e. peak is looked for in.
    smooth : int
        Width, in bins, of the moving average used to find the peak.
    half : float
        Half-width, in p.e., of the interval averaged around the peak.
    valley_max : float
        Upper bound, in p.e., of the walk off the noise tail.

    Returns
    -------
    float or None
        The 1 p.e. centroid in p.e.
    """
    centres = 0.5 * (np.asarray(edges[:-1], float) + np.asarray(edges[1:], float))
    counts = np.asarray(counts, dtype=float)
    low = window[0] if floor is None else max(window[0], float(floor))
    inside = np.flatnonzero((centres >= low) & (centres <= window[1]))
    if inside.size == 0 or counts[inside].sum() <= 0:
        return None
    smoothed = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    start = inside[0]
    last = inside[np.searchsorted(centres[inside], valley_max, side="right") - 1]
    while start + 1 <= last and smoothed[start + 1] < smoothed[start]:
        start += 1
    rest = np.arange(start, inside[-1] + 1)
    if counts[rest].sum() <= 0:
        return None
    peak = centres[rest[np.argmax(smoothed[rest])]]
    near = (centres >= max(peak - half, centres[start])) & (centres <= peak + half)
    weights = counts[near]
    if weights.sum() <= 0:
        return float(peak)
    return float((centres[near] * weights).sum() / weights.sum())


def spe_peak(edges, counts, floor=None, expected=EXPECTED_PEAK, **kwargs):
    """(centroid, dominant) of the low-energy peak of one SiPM.

    ``dominant`` is where the search actually landed; ``centroid`` repeats it
    only when it falls inside ``expected``, so a channel whose dominant
    feature sits far from 1 p.e. is reported as *no 1 p.e. peak here*
    instead of as a large gain drift.

    Returns
    -------
    tuple
        (centroid or None, dominant or None), both in p.e.
    """
    dominant = spe_centroid(edges, counts, floor, **kwargs)
    if dominant is None:
        return None, None
    inside = expected[0] <= dominant <= expected[1]
    return (dominant if inside else None), dominant


def spe_verdict(edges, counts, floor=None, expected=EXPECTED_PEAK, **kwargs):
    """(centroid, dominant, below) for one SiPM's spectrum.

    Searching from the valid-hit threshold keeps the noise turn-on out of the
    way, but it also hides a channel whose *largest* low-energy feature sits
    below that threshold -- which is what a few p22 SiPMs do (feature near
    0.45-0.55 p.e., only a weak shoulder near 1). ``below`` reports such a
    feature so the panel can say so instead of quoting a tidy centroid found
    above it.

    Returns
    -------
    tuple
        (centroid or None, dominant or None, below or None), in p.e.
    """
    centroid, dominant = spe_peak(edges, counts, floor, expected, **kwargs)
    below = None
    if floor is not None and np.isfinite(floor):
        _, unfloored = spe_peak(edges, counts, None, expected, **kwargs)
        if (
            unfloored is not None
            and unfloored < float(floor)
            and not expected[0] <= unfloored <= expected[1]
        ):
            below = unfloored
    return centroid, dominant, below


def centroid_color(centroid, dominant=None, below=None):
    """Verdict colour for a 1 p.e. centroid.

    Grey when nothing was found at all, red when something was found but not
    where a 1 p.e. peak belongs.
    """
    if centroid is None:
        return _BAD if dominant is not None else "dimgray"
    deviation = abs(centroid - 1.0)
    if deviation > BAD_DEVIATION:
        return _BAD
    if deviation > WARN_DEVIATION or below is not None:
        return _WARN
    return _OK


def _label(det, centroid, dominant=None, below=None):
    note = f", larger feature at {below:.2f}" if below is not None else ""
    if centroid is not None:
        drift = (centroid - 1) * 100
        return f"{det} - 1 p.e. at {centroid:.2f} ({drift:+.0f}%){note}"
    if dominant is not None:
        low, high = EXPECTED_PEAK
        return f"{det} - no peak in {low}-{high} p.e. (largest at {dominant:.2f})"
    return f"{det} - 1 p.e. not found{note}"


def _summary(results):
    """One line over a group: median centroid, drifting and unreadable channels."""
    found = [r[0] for r in results.values() if r[0] is not None]
    off_band = sum(r[0] is None and r[1] is not None for r in results.values())
    sub = sum(r[2] is not None for r in results.values())
    note = f", {off_band} with no peak in band" if off_band else ""
    note += f", {sub} with a larger sub-threshold feature" if sub else ""
    if not found:
        return f"no 1 p.e. peak located{note}"
    bad = sum(abs(c - 1) > BAD_DEVIATION for c in found)
    warn = sum(WARN_DEVIATION < abs(c - 1) <= BAD_DEVIATION for c in found)
    return (
        f"median {np.median(found):.3f} p.e., "
        f"{bad}/{len(found)} off by >{BAD_DEVIATION:.0%}, "
        f"{warn} by >{WARN_DEVIATION:.0%}{note}"
    )


def _marks(p, threshold=None):
    """Integer p.e. markers, plus the valid-hit threshold when known."""
    for mark in PE_MARKS:
        p.add_layout(
            Span(
                location=mark,
                dimension="height",
                line_color="black",
                line_dash="dashed",
                line_alpha=0.5,
            )  # fmt: skip
        )
    if threshold is not None and np.isfinite(threshold):
        p.add_layout(
            Span(
                location=float(threshold),
                dimension="height",
                line_color="dimgray",
                line_dash="dotted",
                line_alpha=0.7,
            )  # fmt: skip
        )


def _spectrum_source(edges, counts):
    """(x, y) of a step outline with zero bins dropped for the log axis."""
    counts = np.asarray(counts, dtype=float)
    return np.asarray(edges[:-1], float), np.where(counts > 0, counts, np.nan)


def pe_spectrum_grid(
    edges, counts_by_det, dets, thresholds, group, period, run, sample
):
    """One small log-y p.e. spectrum per SiPM, with its 1 p.e. centroid marked."""
    figures, results = [], {}
    for det in dets:
        counts = counts_by_det.get(det)
        if counts is None:
            continue
        threshold = thresholds.get(det)
        centroid, dominant, below = spe_verdict(edges, counts, threshold)
        results[det] = (centroid, dominant, below)
        color = centroid_color(centroid, dominant, below)
        p = plot_style.make_figure(
            "", width=420, height=250, x_range=Range1d(0, 5), y_axis_type="log",
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )  # fmt: skip
        p.sizing_mode = "fixed"
        x, y = _spectrum_source(edges, counts)
        p.step(x=x, y=y, mode="after", color="steelblue", line_width=1.2)
        _marks(p, threshold)
        if dominant is not None:
            p.add_layout(
                Span(
                    location=dominant,
                    dimension="height",
                    line_color=color,
                    line_width=2,
                    # dashed when it is not where a 1 p.e. peak belongs
                    line_dash="solid" if centroid is not None else "dashed",
                )
            )
        if below is not None:
            p.add_layout(
                Span(
                    location=below,
                    dimension="height",
                    line_color=_WARN,
                    line_dash="dotted",
                    line_width=2,
                )
            )
        p.title.text = _label(det, centroid, dominant, below)
        p.title.text_color = color
        p.xaxis.axis_label = "Pulse energy [p.e.]"
        p.yaxis.axis_label = "Pulses"
        p.grid.grid_line_color = None
        figures.append(p)
    if not figures:
        return plot_style.empty_figure(f"No p.e. spectra for {group} in {period} {run}")
    ncols = max(1, int(np.ceil(np.sqrt(len(figures)))))
    return pn.Column(
        pn.pane.Markdown(
            f"### {period} {run} - {sample} - {group}: {_summary(results)}"
        ),
        pn.pane.Bokeh(gridplot(figures, ncols=ncols, toolbar_location="right")),
    )


def pe_spectrum_overlay(
    edges, counts_by_det, dets, thresholds, group, period, run, sample
):
    """All SiPMs of a group overlaid; the legend carries each 1 p.e. centroid."""
    drawn = [det for det in dets if counts_by_det.get(det) is not None]
    if not drawn:
        return plot_style.empty_figure(f"No p.e. spectra for {group} in {period} {run}")
    results = {
        det: spe_verdict(edges, counts_by_det[det], thresholds.get(det))
        for det in drawn
    }
    p = plot_style.make_figure(
        f"{period} {run} - {sample} - {group}: {_summary(results)}",
        x_range=Range1d(0, 5), y_axis_type="log", height=600,
        tools="pan,box_zoom,wheel_zoom,hover,reset,save",
    )  # fmt: skip
    colors = plot_style.TAB20
    for i, det in enumerate(drawn):
        x, y = _spectrum_source(edges, counts_by_det[det])
        p.step(
            x=x, y=y, mode="after", color=colors[i % len(colors)], line_width=1.2,
            legend_label=_label(det, *results[det]),
        )  # fmt: skip
    _marks(p)
    p.xaxis.axis_label = "Pulse energy [p.e.]"
    p.yaxis.axis_label = "Pulses"
    p.grid.grid_line_color = None
    plot_style.finish_legend(p, "top_right", ncols=2)
    return p
