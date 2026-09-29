"""Cal plot-data LH5 reader, fit re-evaluation and native Bokeh builders."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from bokeh.models import GridPlot
from bokeh.plotting import figure

from legenddashboard.geds.cal import fit_funcs, native_plots
from legenddashboard.geds.cal.plot_data import (
    CommonData,
    data_keys,
    decode_key,
    encode_key,
    plt_data_path,
    read_group,
)

CAL_OP = {"expression": "a + b * cuspEmax_ctc", "parameters": {"a": 0.0, "b": 0.1376}}
FIT = {
    "function": "gauss_on_step",
    "validity": True,
    "parameters": {
        "x_lo": 18700.0,
        "x_hi": 19300.0,
        "n_sig": 8000.0,
        "mu": 19000.0,
        "sigma": 8.0,
        "n_bkg": 400.0,
        "hstep": -0.6,
    },  # fmt: skip
    "p_value": 0.5,
    "position": 19000.0,
    "position_uncertainty": 0.1,
    "fwhm_in_kev": 2.6,
    "fwhm_err_in_kev": 0.02,
}
PK_FITS = {
    2614.511: FIT,
    583.191: {**FIT, "position": 4238.3, "fwhm_in_kev": 1.6},
    1592.511: {**FIT, "position": 11573.0, "fwhm_in_kev": 2.2},
}
ERES = {
    "function": "FWHMLinear",
    "module": "pygama.pargen.energy_cal",
    "expression": "(a+b*x)**(0.5)",
    "parameters": {"a": 1.56, "b": 0.0021},
}


@pytest.fixture
def plt_file(tmp_path):
    path = tmp_path / "l200-p00-r000-cal-20250101T000000Z-plt_hit.lh5"
    edges = np.arange(18700.0, 19300.0, 0.7)
    x = (edges[1:] + edges[:-1]) / 2
    counts = np.random.default_rng(1).poisson(
        fit_funcs.peak_counts(FIT, x, edges[1] - edges[0])
    )
    with h5py.File(path, "w") as f:
        g = f.create_group("V00000A/ecal/cuspEmax_ctc_cal/peak_hists/2614p511")
        g["edges"] = edges
        g["counts"] = counts
        f["V00000A/ecal/cuspEmax_ctc_cal/func"] = b"gauss_on_step"
        f["V00000A/ecal/cuspEmax_ctc_cal/best"] = 4.5
        spec = f.create_group("common/V00000A/cuspEmax_ctc_cal/spectrum")
        spec["bins"] = np.arange(10.0)
        spec["counts"] = np.arange(10)
    return path


def test_keys():
    assert encode_key(2614.511) == "2614p511"
    assert decode_key("2614p511") == "2614.511"
    assert decode_key("2614_stability") == "2614_stability"
    assert decode_key("peak_hists") == "peak_hists"
    assert str(plt_data_path("/a/x-plt_hit")) == "/a/x-plt_hit.lh5"


def test_read_group(plt_file):
    assert data_keys(plt_file) == ["V00000A", "common"]
    ecal = read_group(plt_file, "V00000A", "ecal", "cuspEmax_ctc_cal")
    assert set(ecal["peak_hists"]) == {"2614.511"}
    assert ecal["func"] == "gauss_on_step"
    assert ecal["best"] == 4.5
    counts = read_group(plt_file, "V00000A", "ecal", "cuspEmax_ctc_cal", "peak_hists", 2614.511)["counts"]  # fmt: skip
    assert not counts.flags.writeable
    assert read_group(plt_file, "V00000B") is None
    assert read_group(plt_file.with_name("missing.lh5"), "V00000A") is None


def test_common_data(plt_file):
    common = CommonData(plt_file)
    assert list(common) == ["V00000A"]
    assert len(common["V00000A"]["cuspEmax_ctc_cal"]["spectrum"]["bins"]) == 10
    with pytest.raises(KeyError):
        common["V00000B"]


@pytest.mark.parametrize("name", ["gauss_on_step", "hpge_peak"])
def test_peak_counts_matches_pygama(name):
    import pygama.math.distributions as pgd

    pars = dict(FIT["parameters"])
    if name == "hpge_peak":
        pars.update(htail=0.2, tau=20.0)
    func = getattr(pgd, name)
    x = np.linspace(pars["x_lo"], pars["x_hi"], 101)
    ref = func.get_pdf(x, *[pars[k] for k in func.required_args()]) * 0.5
    got = fit_funcs.peak_counts({"function": name, "parameters": pars}, x, 0.5)
    np.testing.assert_allclose(got, ref)
    assert fit_funcs.peak_counts({"function": "unknown"}, x, 1.0) is None


def test_eval_fit():
    x = np.array([200.0, 2039.0])
    np.testing.assert_allclose(fit_funcs.eval_fit(ERES, x), np.sqrt(1.56 + 0.0021 * x))
    sigmoid = {"func": "SigmoidFit", "module": "pygama.pargen.AoE_cal",
               "pars": {"a": 90.0, "b": 1.0, "c": 5.0, "d": -2.0}}  # fmt: skip
    from pygama.pargen.AoE_cal import SigmoidFit

    np.testing.assert_allclose(
        fit_funcs.eval_fit(sigmoid, x), SigmoidFit.func(x, **sigmoid["pars"])
    )
    assert fit_funcs.eval_fit({"module": "nope", "function": "x"}, x) is None


def test_eval_expression():
    x = np.array([0.0, 100.0])
    np.testing.assert_allclose(
        fit_funcs.eval_expression(ERES["expression"], ERES["parameters"], "x", x),
        np.sqrt(1.56 + 0.0021 * x),
    )
    assert native_plots.calibrate(CAL_OP, 19000.0) == pytest.approx(2614.4)


def test_peak_fits(plt_file):
    hists = read_group(plt_file, "V00000A", "ecal", "cuspEmax_ctc_cal", "peak_hists")
    grid = native_plots.plot_peak_fits(hists, PK_FITS, CAL_OP, "V00000A")
    assert isinstance(grid, GridPlot)
    top = grid.children[0][0]
    assert len(top.renderers) == 2  # data + fit
    assert isinstance(native_plots.plot_peak_fits({}, PK_FITS, CAL_OP, "x"), figure)


def test_cal_and_fwhm_fit():
    assert isinstance(native_plots.plot_cal_fit(PK_FITS, CAL_OP, "x"), GridPlot)
    grid = native_plots.plot_fwhm_fit(PK_FITS, ERES, ERES, "x")
    assert isinstance(grid, GridPlot)
    legend = [item.label.value for item in grid.children[0][0].legend[0].items]
    assert "fitted" in legend
    assert "not fitted" in legend  # 1592.5 keV (DEP) is excluded from the fit
    assert isinstance(native_plots.plot_cal_fit({}, CAL_OP, "x"), figure)


def test_timemap_and_spectra():
    hist = {
        "counts": np.array([[0, 3], [5, 0]]),
        "time_edges": np.array([1.7e9, 1.7e9 + 180, 1.7e9 + 360]),
        "value_edges": np.array([2580.0, 2581.0, 2582.0]),
    }
    p = native_plots.plot_timemap(hist, "x", "Energy (keV)")
    assert p.renderers[0].glyph.__class__.__name__ == "Image"
    assert isinstance(native_plots.plot_timemap({}, "x", "E"), figure)

    bins = np.arange(5.0)
    spec = {"bins": bins, "counts": bins + 1, "cut_counts": bins, "pulser_counts": np.full(5, np.nan)}  # fmt: skip
    assert len(native_plots.plot_cut_spectra(spec, "x").renderers) == 2  # no pulser
    assert native_plots.plot_survival_frac({"bins": bins, "sf": bins}, "x").renderers

    stab = {"time": 1.7e9 + np.arange(12) * 180.0, "energy": np.full(12, 2614.5), "spread": np.full(12, 0.1)}  # fmt: skip
    ecal = {"2614_stability": stab, "583_stability": stab, "pulser_stability": stab}
    assert (
        len(native_plots.plot_peak_track(ecal, "x").renderers) == 6
    )  # band + line each


def _hist2d(nx=4, ny=3):
    return {"counts": np.arange(nx * ny).reshape(nx, ny),
            "x_edges": np.linspace(0, 1, nx + 1), "y_edges": np.linspace(0, 2, ny + 1)}  # fmt: skip


def test_psd_builders():
    from legenddashboard.geds.cal import native_psd_plots as psd

    edges = np.linspace(900, 3000, 11)
    ones = np.ones(10)
    aoe_spec = {"edges": edges, "before": 3 * ones, "low_cut": 2 * ones,
                "double_cut": ones, "rejected": 2 * ones}  # fmt: skip
    assert len(psd.plot_spectra(aoe_spec, "x", psd.AOE_SPECTRA).renderers) == 4
    lq_spec = {
        "edges": edges,
        "before": 3 * ones,
        "after_cut": ones,
        "rejected": 2 * ones,
    }
    assert len(psd.plot_spectra(lq_spec, "x", psd.LQ_SPECTRA).renderers) == 3
    assert psd.plot_sf_vs_energy({"edges": edges, "sf": 50 * ones}, "x").renderers
    assert psd.plot_classifier(_hist2d(), "x", "A/E").renderers
    grid = psd.plot_dt_dep({"a": _hist2d(), "b": _hist2d(), "c": _hist2d()}, "x")
    assert isinstance(grid, GridPlot)
    bands = {"900-920": {"edges": np.linspace(0.9, 1.1, 5), "counts": np.ones(4)}}
    assert len(psd.plot_compt_bands(bands, "x", "A/E").renderers) == 1

    e = np.linspace(900, 2300, 8)
    pts = {"energy": e, "mean": 1 - 1e-6 * e, "mean_err": 1e-4 * np.ones(8),
           "sigma": 0.01 * np.ones(8), "sigma_err": 1e-4 * np.ones(8), "band_width": 20.0}  # fmt: skip
    fits = {
        "mean_fits": {"func": "Pol1", "module": "pygama.pargen.AoE_cal",
                      "pars": {"a": -1e-6, "b": 1.0}},
        "SigmaFits": {"func": "SigmaFit", "module": "pygama.pargen.AoE_cal",
                      "pars": {"a": 1e-5, "b": 10.0, "c": 2.0}},
        "dep_fit": {"pars": {"mu": 0.998, "sigma": 0.004}, "errs": {"mu": 1e-4, "sigma": 1e-4}},
    }  # fmt: skip
    for which in ("mean", "sigma"):
        grid = psd.plot_energy_corr(pts, fits, "x", which)
        top, bottom = grid.children[0][0], grid.children[1][0]
        assert len(top.renderers) == 3  # bands, DEP, curve
        assert bottom.renderers

    cut = {"cut_vals": np.linspace(-8, 0, 20), "sf": np.linspace(20, 100, 20),
           "sf_err": np.ones(20), "function": "SigmoidFit",
           "pars": {"a": 90.0, "b": 1.0, "c": 1.0, "d": 2.0}, "low_cut": -1.8, "dep_acc": 0.9}  # fmt: skip
    assert len(psd.plot_cut_fit(cut, "x").renderers) == 2  # sweep + sigmoid
    curves = {"low_cut": -1.8, "peaks": {"1592.5": {"cut_vals": cut["cut_vals"],
              "sf": cut["sf"], "sf_err": cut["sf_err"]}}}  # fmt: skip
    assert psd.plot_survival_curves(curves, "x").renderers
    series = {"20250101T000000Z": {"mean": 0.1, "mean_err": 0.01}}
    assert psd.plot_time_series(series, "x", "mean", "A/E").renderers

    dt = {
        **_hist2d(),
        "dt_range": np.array([0.2, 0.8]),
        "lq_range": np.array([0.5, 1.5]),
    }
    assert (
        len(psd.plot_lq_drift_time(dt, [0.5, 0.1], "x").renderers) == 2
    )  # image + line
    hist = {
        "counts": np.array([1.0, 5.0, 9.0, 5.0, 1.0]),
        "edges": np.linspace(-1, 1, 6),
    }
    grid = psd.plot_lq_cut_fit(hist, {"mu": 0.0, "sigma": 0.4}, "x")
    assert len(grid.children[0][0].renderers) == 2  # data + Gaussian

    for empty in (psd.plot_spectra({}, "x", psd.AOE_SPECTRA), psd.plot_cut_fit({}, "x"),
                  psd.plot_time_series({}, "x", "mean", "y")):  # fmt: skip
        assert isinstance(empty, figure)


def test_read_psd_set(tmp_path):
    path = tmp_path / "plt.lh5"
    with h5py.File(path, "w") as f:
        g = f.create_group("V00000A/aoe/dplms/cut_fit_data")
        g["cut_vals"] = np.arange(3.0)
        g["function"] = b"SigmoidFit"
        g.create_group("pars")["a"] = 90.0
    data = read_group(path, "V00000A", "aoe", "dplms", "cut_fit_data")
    assert data["function"] == "SigmoidFit"
    assert data["pars"]["a"] == 90.0
