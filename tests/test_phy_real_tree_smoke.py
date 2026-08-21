"""Render every shifter family and expert combination on a real lmon tree.

Opt-in: set ``LEGEND_PHY_TREE`` to an lmon output root (the folder holding
``generated/``) and ``LEGEND_PRODENV`` to the production cycle root the
dashboard's run discovery needs, e.g.::

    LEGEND_PHY_TREE=/data1/users/marshall/lmon-v2-p22/auto/v2.0.0 \\
    LEGEND_PRODENV=/data2/public/prodenv/prod-blind/auto/v2.0.0 \\
    .venv/bin/python -m pytest tests/test_phy_real_tree_smoke.py -q
"""

from __future__ import annotations

import os

import pytest

PHY_TREE = os.environ.get("LEGEND_PHY_TREE")
PRODENV = os.environ.get("LEGEND_PRODENV")
pytestmark = pytest.mark.skipif(
    not (PHY_TREE and PRODENV), reason="LEGEND_PHY_TREE / LEGEND_PRODENV not set"
)


@pytest.fixture(scope="module")
def monitors():
    from legenddashboard.geds.phy.phy_monitoring import PhyMonitoring
    from legenddashboard.geds.phy.phy_shifter import PhyShifterMonitoring

    expert = PhyMonitoring(base_path=PRODENV, phy_path=PHY_TREE, name="expert")
    shifter = PhyShifterMonitoring(base_path=PRODENV, phy_path=PHY_TREE, name="shifter")
    return expert, shifter


def _has_content(obj):
    if hasattr(obj, "renderers"):
        return bool(obj.renderers)
    return True  # Panel layout (grid / fallback column)


def test_shifter_every_family_and_metric(monitors):
    _, mon = monitors
    period = mon.period
    runs = list(mon.run_dict)[-2:]
    built, missing = [], []
    for run in runs:
        mon.run = run
        for family in mon.param.shifter_family.objects:
            mon.shifter_family = family
            metrics = list(mon.param.shifter_metric.objects) or [None]
            for metric in metrics:
                if metric is not None:
                    mon.shifter_metric = metric
                obj = mon.update_shifter_plot()
                title = getattr(getattr(obj, "title", None), "text", "")
                tag = f"{period}/{run} {family}/{metric}"
                if "not in" in title or "not implemented" in title:
                    missing.append(tag)
                else:
                    assert _has_content(obj), tag
                    built.append(tag)
    assert built, "nothing rendered"
    print(f"\nshifter built {len(built)}, missing {len(missing)}: {missing[:6]}")


def test_expert_every_offered_combination(monitors):
    mon, _ = monitors
    from legenddashboard.geds.phy import (
        contract_reader,
        phy_plot_style_dict,
        phy_unit_vals,
    )

    with_manifest = [
        run
        for run in mon.run_dict
        if contract_reader.find_manifest(PHY_TREE, mon.period, run) is not None
    ]
    assert with_manifest, "no run with a v2 manifest in the latest period"
    mon.run = with_manifest[-1]
    n_ok = 0
    for flag in list(mon.param.phy_plots_types.objects):
        mon.phy_plots_types = flag
        for value in list(mon.param.phy_plots.objects):
            mon.phy_plots = value
            for corr in list(mon.param.phy_pulser_corr.objects):
                mon.phy_pulser_corr = corr
                for style in phy_plot_style_dict:
                    mon.phy_plot_style = style
                    for units in phy_unit_vals:
                        mon.phy_units = units
                        p = mon.update_plots()
                        assert p.renderers, f"{flag}/{value}/{corr}/{style}/{units}"
                        n_ok += 1
    assert n_ok > 0
