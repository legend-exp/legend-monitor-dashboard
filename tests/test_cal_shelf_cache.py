from __future__ import annotations

import dbm.dumb
import pickle as pkl
import shelve

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from legenddashboard.geds.cal import shelf_cache
from legenddashboard.geds.cal.summary_plots import plot_fft_spectra

FFT_KEYS = ("noise_optimisation", "nopt", "fft")
FREQ = np.linspace(0, 3.90625, 65)


@pytest.fixture
def dsp_shelf(tmp_path):
    path = tmp_path / "l200-p00-r000-cal-20250101T000000Z-plt_dsp"
    fig = plt.figure()
    fig.add_subplot().plot([1, 2], [3, 4])
    with shelve.Shelf(
        dbm.dumb.open(str(path), "c"), pkl.HIGHEST_PROTOCOL
    ) as sh:  # as dataflow
        for i, det in enumerate(["V00000A", "V00000B"]):
            fft = {"frequency": FREQ, "psd": FREQ * (i + 1), "fig": fig}
            sh[det] = {"noise_optimisation": {"nopt": {"fft": fft}}, "pz": {"f": fig}}
        sh["V00000C"] = {"pz": {"f": fig}}  # no nopt run
    plt.close(fig)
    return path


def test_shelf_data_skips_figures(dsp_shelf):
    fft = shelf_cache.shelf_data(dsp_shelf, "V00000B", FFT_KEYS)
    assert set(fft) == {"frequency", "psd"}
    np.testing.assert_array_equal(fft["psd"], FREQ * 2)
    assert shelf_cache.shelf_data(dsp_shelf, "V00000B", FFT_KEYS) is fft  # cached


def test_shelf_data_missing(dsp_shelf):
    assert shelf_cache.shelf_data(dsp_shelf, "V00000C", FFT_KEYS) is None
    assert shelf_cache.shelf_data(dsp_shelf, "nope", FFT_KEYS) is None


def test_plot_fft_spectra(dsp_shelf):
    dets = ["V00000A", "V00000B", "V00000C"]
    ffts = shelf_cache.shelf_data_many(dsp_shelf, dets, FFT_KEYS)
    assert set(ffts) == {"V00000A", "V00000B"}
    chan_dict = {d: {"name": d} for d in dets}
    p = plot_fft_spectra(
        None, ffts, chan_dict, dets, "String:01", "r000", "p00", {"experiment": "l200"}
    )
    assert len(p.renderers) == 2
    assert len(p.renderers[0].data_source.data["x"]) == len(FREQ) - 1  # f=0 dropped
