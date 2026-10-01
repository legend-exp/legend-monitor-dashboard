"""Re-evaluate calibration fits stored in the par files with pygama.

The par files keep each fit as a function name plus named parameters: peak
fits as ``pk_fits[peak] = {"function": "hpge_peak", "parameters": {...}}``
(a ``pygama.math.distributions`` name) and resolution / correction curves as
``{"module": "pygama.pargen.energy_cal", "function": "FWHMLinear", ...}``.
Calibrations (``pars.operations``) are numexpr ``expression`` strings.
"""

from __future__ import annotations

import importlib

import numexpr as ne
import numpy as np
import pygama.math.distributions as pgd
import pygama.pargen.data_cleaning as dc


def peak_counts(pk_fit, x, bin_width):
    """
    Expected counts per bin of a stored peak fit.

    Parameters
    ----------
    pk_fit : dict
        One ``pk_fits`` entry: ``function`` name and ``parameters`` mapping.
    x : array_like
        Bin centres.
    bin_width : float
        Width of the bins in the same units as ``x``.

    Returns
    -------
    ndarray or None
        Counts per bin, or None if the function is unknown.
    """
    func = getattr(pgd, str(pk_fit.get("function")), None)
    if func is None or not hasattr(func, "get_pdf"):
        return None
    pars = [pk_fit["parameters"][k] for k in func.required_args()]
    return func.get_pdf(np.asarray(x, dtype=float), *pars) * bin_width


def qc_fit_counts(fit, x, bin_width):
    """
    Expected counts of a qc classifier fit (``qc/<cut>_data/fit``).

    Parameters
    ----------
    fit : dict
        ``function`` name (in ``pygama.pargen.data_cleaning`` or
        ``pygama.math.distributions``) and ``pars`` array.
    x : array_like
        Bin centres.
    bin_width : float
        Width of the bins.

    Returns
    -------
    ndarray or None
        Counts per bin, or None if the function is unknown.
    """
    name = str(fit.get("function"))
    x = np.asarray(x, dtype=float)
    pars = np.asarray(fit["pars"], dtype=float)
    if name == "skewed_fit":
        return dc.skewed_fit(x, *pars)[1] * bin_width
    func = getattr(dc, name, None) or getattr(pgd, name, None)
    if func is None or not hasattr(func, "pdf_ext"):
        return None
    return func.pdf_ext(x, *pars)[1] * bin_width


def fit_class(entry):
    """The pygama class named by a par-file fit entry, or None if unknown."""
    name = entry.get("function") or entry.get("func")
    try:
        return getattr(importlib.import_module(entry["module"]), name)
    except (KeyError, ImportError, AttributeError, TypeError):
        return None


def eval_fit(entry, x):
    """
    Evaluate a par-file fit entry (e.g. ``eres_linear``, ``mean_fits``) at ``x``.

    Parameters
    ----------
    entry : dict
        Fit entry with ``module``, ``function`` (or ``func``) and ``parameters``
        (or ``pars``).
    x : array_like
        Values to evaluate at.

    Returns
    -------
    ndarray or None
        The fitted curve, or None if the class cannot be resolved.
    """
    cls = fit_class(entry)
    if cls is None:
        return None
    pars = entry.get("parameters") or entry.get("pars")
    return np.asarray(cls.func(np.asarray(x, dtype=float), **pars), dtype=float)


def eval_expression(expression, parameters, variable, x):
    """
    Evaluate a par-file ``expression`` (e.g. ``"a + b * cuspEmax_ctc"``).

    Parameters
    ----------
    expression : str
        numexpr expression string.
    parameters : dict
        Named constants used by the expression.
    variable : str
        Name of the independent variable in the expression.
    x : array_like
        Values to evaluate at.

    Returns
    -------
    ndarray
        Expression evaluated at ``x``.
    """
    x = np.asarray(x, dtype=float)
    local = {k: float(v) for k, v in parameters.items()}
    local[variable] = x
    return np.broadcast_to(ne.evaluate(expression, local_dict=local), x.shape)


def prewarm() -> None:
    """Compile the numba peak shapes once, so no viewer pays for it."""
    x = np.linspace(-5.0, 5.0, 11)
    pars = {"x_lo": -5.0, "x_hi": 5.0, "n_sig": 1.0, "mu": 0.0, "sigma": 1.0,
            "htail": 0.1, "tau": 1.0, "n_bkg": 1.0, "hstep": 0.1}  # fmt: skip
    for name in ("gauss_on_step", "hpge_peak"):
        peak_counts({"function": name, "parameters": pars}, x, 1.0)
    pgd.gaussian.pdf_norm(x, -5.0, 5.0, 0.0, 1.0)
