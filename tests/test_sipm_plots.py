"""SiPM p.e. spectrum builders and the 1 p.e. centroid estimator."""

from __future__ import annotations

import numpy as np
import pytest
from bokeh.models import Span

from legenddashboard.spms import sipm_plots

EDGES = np.linspace(0.0, 5.0, 251)  # the contract's 250 bins over 0-5 p.e.
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])


def spectrum(peak=1.0, width=0.12, noise_amp=0.0, noise_scale=0.08):
    """A gaussian p.e. peak on an optional falling sub-threshold noise tail."""
    counts = 500 * np.exp(-0.5 * ((CENTRES - peak) / width) ** 2)
    if noise_amp:
        counts = counts + noise_amp * np.exp(-CENTRES / noise_scale)
    return counts


@pytest.mark.parametrize("peak", [0.90, 1.00, 1.15])
def test_centroid_finds_planted_peak(peak):
    got = sipm_plots.spe_centroid(EDGES, spectrum(peak))
    assert got == pytest.approx(peak, abs=0.02)


def test_centroid_survives_a_noise_tail_taller_than_the_peak():
    # the trap: unmasked pulses put sub-threshold noise above the 1 p.e. peak,
    # and on some channels it reaches past that channel's threshold
    # a tail that only reaches past the threshold: the walk finds the valley
    counts = spectrum(0.95, noise_amp=20_000, noise_scale=0.12)
    assert counts[:10].max() >= counts.max()  # noise dominates the low end
    assert sipm_plots.spe_centroid(EDGES, counts, floor=0.66) == pytest.approx(
        0.95, abs=0.05
    )
    # a tail that outweighs the peak everywhere cannot drag the estimate away
    # from the 1 p.e. region: the walk stops at valley_max
    swamped = spectrum(0.95, noise_amp=50_000, noise_scale=0.18)
    assert sipm_plots.spe_centroid(EDGES, swamped, floor=0.66) == pytest.approx(
        1.0, abs=0.1
    )


def test_centroid_none_when_empty():
    assert sipm_plots.spe_centroid(EDGES, np.zeros(250)) is None
    assert sipm_plots.spe_centroid(EDGES, spectrum(1.0), floor=4.9) is None


def test_centroid_color_bands():
    assert sipm_plots.centroid_color(1.00) == "seagreen"
    assert sipm_plots.centroid_color(0.93) == "darkorange"
    assert sipm_plots.centroid_color(1.20) == "red"
    assert sipm_plots.centroid_color(None) == "dimgray"


def _counts(dets):
    return {det: spectrum(peak) for det, peak in dets.items()}


def _vspans(p):
    return sorted(
        s.location for s in p.center if isinstance(s, Span) and s.dimension == "height"
    )


def test_grid_panels_marks_and_verdicts():
    counts = _counts({"S1": 1.0, "S2": 1.25})
    layout = sipm_plots.pe_spectrum_grid(
        EDGES,
        counts,
        ["S1", "S2", "S3"],
        {"S1": 0.5},
        "IB top",
        "p20",
        "r001",
        "Forced",
    )
    figures = [child[0] for child in layout[1].object.children]
    assert len(figures) == 2  # S3 has no spectrum -> no panel, like the classifier grid
    assert figures[0].title.text_color == "seagreen"
    assert figures[1].title.text_color == "red"
    assert "1 p.e. at 1.25 (+25%)" in figures[1].title.text
    # 1-5 p.e. markers, the threshold line and the measured centroid
    assert set(_vspans(figures[0])) >= {1.0, 2.0, 3.0, 4.0, 5.0, 0.5}
    assert max(_vspans(figures[0])) == 5.0
    assert _vspans(figures[1])[1] == pytest.approx(1.25, abs=0.01)  # centroid
    assert 0.5 not in _vspans(figures[1])  # no threshold known for S2
    assert "1/2 off by >10%" in layout[0].object
    assert figures[0].y_scale.__class__.__name__ == "LogScale"


def test_overlay_legend_carries_centroids():
    counts = _counts({"S1": 1.0, "S2": 0.85})
    p = sipm_plots.pe_spectrum_overlay(
        EDGES, counts, ["S1", "S2"], {}, "IB top", "p20", "r001", "Physics"
    )
    labels = [item.label["value"] for item in p.legend[0].items]
    assert labels[0].startswith("S1 - 1 p.e. at 0.99")
    assert "-15%" in labels[1]
    assert _vspans(p) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert (p.x_range.start, p.x_range.end) == (0, 5)


def test_builders_report_missing_group():
    empty = sipm_plots.pe_spectrum_grid(
        EDGES, {}, ["S1"], {}, "IB top", "p20", "r001", "Forced"
    )
    assert "No p.e. spectra" in empty.title.text


def test_off_band_peak_is_flagged_not_quoted_as_drift():
    """A dominant feature far below 1 p.e. is reported, not fitted through.

    Some p22 SiPMs peak near 0.45-0.5 p.e. with only a weak shoulder near
    0.9; calling that a -55 % gain drift would assert an interpretation the
    spectrum does not support.
    """
    counts = spectrum(0.47) + 0.15 * spectrum(0.95)
    centroid, dominant = sipm_plots.spe_peak(EDGES, counts)
    assert centroid is None
    assert 0.45 <= dominant <= 0.55  # the shoulder pulls the weighted mean up
    assert sipm_plots.centroid_color(centroid, dominant) == "red"
    label = sipm_plots._label("S070", centroid, dominant)
    assert "no peak in 0.6-1.5 p.e." in label
    assert f"{dominant:.2f}" in label
    assert "%" not in label  # no drift percentage is claimed


def test_in_band_peak_still_reports_a_centroid():
    centroid, dominant = sipm_plots.spe_peak(EDGES, spectrum(1.06))
    assert centroid == pytest.approx(1.06, abs=0.03)
    assert centroid == dominant
    assert sipm_plots.centroid_color(centroid, dominant) == "darkorange"
    assert "1 p.e. at 1.05 (+5%)" in sipm_plots._label("S060", centroid, dominant)


def test_summary_counts_off_band_channels():
    results = {
        "A": (1.00, 1.00, None),
        "B": (0.88, 0.88, None),
        "C": (None, 0.47, None),
        "D": (None, None, None),
        "E": (1.01, 1.01, 0.5),
    }
    summary = sipm_plots._summary(results)
    assert "median 1.000 p.e." in summary
    assert "1/3 off by >10%" in summary
    assert "1 with no peak in band" in summary
    assert "1 with a larger sub-threshold feature" in summary


def test_sub_threshold_feature_is_reported_with_the_centroid():
    """A peak found above threshold is quoted, but a larger one below it is named.

    Searching from the valid-hit threshold otherwise returns a tidy centroid
    for a channel whose spectrum is dominated by something else entirely.
    """
    counts = spectrum(0.50) + 0.2 * spectrum(1.02)
    centroid, dominant, below = sipm_plots.spe_verdict(EDGES, counts, floor=0.75)
    assert centroid == pytest.approx(1.02, abs=0.03)
    assert dominant == centroid
    assert below == pytest.approx(0.50, abs=0.05)
    assert sipm_plots.centroid_color(centroid, dominant, below) == "darkorange"
    label = sipm_plots._label("S096", centroid, dominant, below)
    assert f"1 p.e. at {centroid:.2f}" in label
    assert f"larger feature at {below:.2f}" in label
    # without a floor the dominant feature is the sub-threshold one
    assert sipm_plots.spe_verdict(EDGES, counts)[0] is None
