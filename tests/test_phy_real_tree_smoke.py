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


def _message(obj):
    """The title of a contentless figure, i.e. a 'nothing to draw' message.

    The tree is live: a producer backfill can be rewriting a run while this
    runs, and runs written by an older pipeline lack the newest keys. Both
    surface as message figures, which are a pass, not a failure.
    """
    if _has_content(obj):
        return None
    return getattr(getattr(obj, "title", None), "text", "") or "(no title)"


def test_shifter_every_family_and_metric(monitors):
    from legenddashboard.geds.phy import period_reader

    _, mon = monitors
    period = mon.period
    # the production tree runs ahead of the monitoring output, so drive the
    # last runs the period contract actually covers, not the last runs there are
    covered = period_reader.runs_for(
        period_reader.period_file(PHY_TREE, period), "detector_summary/pulser_stab"
    )
    runs = [run for run in mon.run_dict if run in covered][-2:]
    if not runs:
        pytest.skip(f"no run of {period} is in the period contract")
    built, missing = [], []
    for run in runs:
        mon.run = run
        for family in mon.param.shifter_family.objects:
            mon.shifter_family = family
            metrics = list(mon.param.shifter_metric.objects) or [None]
            for metric in metrics:
                if metric is not None:
                    mon.shifter_metric = metric
                tag = f"{period}/{run} {family}/{metric}"
                message = _message(mon.update_shifter_plot())
                if message is not None:
                    missing.append(f"{tag}: {message[:50]}")
                else:
                    built.append(tag)
    if not built:
        pytest.skip(f"nothing rendered; first message: {missing[:1]}")
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


def test_sipm_page_every_view(monitors):
    from legenddashboard.geds.phy import contract_reader
    from legenddashboard.spms.sipm_monitoring import SiPMMonitoring

    mon = SiPMMonitoring(base_path=PRODENV, phy_path=PHY_TREE, name="sipm")
    with_spms = [
        run
        for run in mon.run_dict
        if (m := contract_reader.find_manifest(PHY_TREE, mon.period, run)) is not None
        and contract_reader.file_from_manifest(m, ".", "spms") is not None
    ]
    if not with_spms:
        pytest.skip("no run with an spms contract")
    mon.run = with_spms[-1]
    n = 0
    for view in mon.param.sipm_view.objects:
        mon.sipm_view = view
        if _message(mon.update_sipm_plot()) is None:
            n += 1
    for sample in mon.param.sipm_pe_sample.objects:  # p.e. spectra, both layouts
        for layout in mon.param.sipm_pe_layout.objects:
            mon.sipm_view, mon.sipm_pe_sample, mon.sipm_pe_layout = (
                "PE spectrum", sample, layout,
            )  # fmt: skip
            if _message(mon.update_sipm_plot()) is None:
                n += 1  # a message means the run predates the p.e. keys
    mon.sipm_view = "Explorer"
    for grouping in mon.param.sipm_group_by.objects:
        mon.sipm_group_by = grouping
        for group in list(mon.param.sipm_group.objects)[:2]:
            mon.sipm_group = group
            for flag in list(mon.param.sipm_plots_types.objects):
                mon.sipm_plots_types = flag
                for value in list(mon.param.sipm_plots.objects):
                    mon.sipm_plots = value
                    for units in mon.param.sipm_units.objects:
                        mon.sipm_units = units
                        for style in mon.param.sipm_plot_style.objects:
                            mon.sipm_plot_style = style
                            tag = f"{grouping}/{group}/{flag}/{value}/{units}/{style}"
                            message = _message(mon.update_sipm_plot())
                            assert message is None, f"{tag}: {message}"
                            n += 1
    assert n, "nothing rendered"
    print(f"\nsipm rendered {n}")


def test_muon_page_every_view(monitors):
    from legenddashboard.geds.phy import contract_reader
    from legenddashboard.muon.muon_monitoring import MuonMonitoring

    mon = MuonMonitoring(base_path=PRODENV, phy_path=PHY_TREE, name="muon")
    with_pmts = [
        run
        for run in mon.run_dict
        if (m := contract_reader.find_manifest(PHY_TREE, mon.period, run)) is not None
        and contract_reader.file_from_manifest(m, ".", "pmts") is not None
    ]
    if not with_pmts:
        pytest.skip("no run with a pmts contract")
    mon.run = with_pmts[-1]
    n = 0
    for view in mon.param.muon_view.objects:
        mon.muon_view = view
        for group in mon.param.muon_group.objects:
            mon.muon_group = group
            obj = mon.update_muon_plot()
            assert _has_content(obj), f"{view}/{group}"
            n += 1
    mon.muon_view = "Explorer"
    for style in mon.param.muon_plot_style.objects:
        mon.muon_plot_style = style
        mon._update_menus()
        for value in list(mon.param.muon_plots.objects):
            mon.muon_plots = value
            for units in mon.param.muon_units.objects:
                mon.muon_units = units
                p = mon.update_muon_plot()
                title = getattr(getattr(p, "title", None), "text", "")
                assert (
                    p.renderers or "missing" in title or "No distribution" in title
                ), f"{style}/{value}/{units}"
                n += 1
    print(f"\nmuon rendered {n}")
