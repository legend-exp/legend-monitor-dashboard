from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import panel as pn
from dbetto import AttrsDict, Props

from legenddashboard.base import Monitoring
from legenddashboard.geds.cal.cal_monitoring import CalMonitoring
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.geds.phy.phy_monitoring import PhyMonitoring
from legenddashboard.llama.llama_monitoring import LlamaMonitoring
from legenddashboard.muons.muon_monitoring import MuonMonitoring
from legenddashboard.spms.sipm_monitoring import SiPMMonitoring

# somehow TUM server needs Agg -> needs fix in the future
mpl.use("Agg")

# use terminal and tabulator extensions, sizing mode stretch enables nicer layout
pn.extension("terminal")
pn.extension("tabulator")
pn.extension("plotly")
pn.extension("katex", "mathjax")


def build_dashboard(
    config: str | dict,
    logo_path,
    widget_widths: int = 140,
):
    if isinstance(config, str | Path):
        config = AttrsDict(Props.read_from(config))
    else:
        config = AttrsDict(config)

    # path to period data
    data_path = config.base
    # path to calibration data
    cal_path = config.cal
    # path to physics data
    phy_path = config.phy
    # path to sipm data
    sipm_path = config.sipm
    # path to muon data
    muon_path = config.muon
    # tmp path for caching
    tmp_cal_path = config.tmp
    # llama data path
    llama_path = config.llama

    # use Golden layout template with header and LEGEND logo from LEGEND webpage
    l200_monitoring = pn.template.GoldenTemplate(
        header_background="#f8f8fa",
        header_color="#1A2A5B",
        title="L200 Monitoring Dashboard",
        sidebar_width=7,
        site="",
        logo="https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Images/Logo/logo_legend_tag_next.svg",
        favicon="https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Favicons/android-chrome-96x96.png",
        site_url="https://legend.edm.nat.tum.de/l200_monitoring_auto/",
    )
    # needed to set header title color
    custom_header_title_css = """
    #header {padding: 0}
    .title {
        font-weight: bold;
        font-family: bradley hand;, cursive
        padding-left: 10px;
        color: #1A2A5B;
    }
    """
    l200_monitoring.config.raw_css.append(custom_header_title_css)

    base_monitor = Monitoring(
        base_path=data_path,
        name="L200 Monitoring",
    )
    ged_monitor = GedMonitoring(
        path=cal_path,
        base_path=base_monitor.params.base_path,
        run_dict=base_monitor.params.run_dict,
        periods=base_monitor.params.periods,
        period=base_monitor.params.period,
        run=base_monitor.params.run,
        date_range=base_monitor.param.date_range,
        name="L200 Ged Monitoring",
    )

    cal_monitor = CalMonitoring(
        path=ged_monitor.cal_path,
        tmp_path=tmp_cal_path,
        base_path=base_monitor.params.base_path,
        run_dict=base_monitor.params.run_dict,
        periods=base_monitor.params.periods,
        period=base_monitor.params.period,
        run=base_monitor.params.run,
        date_range=base_monitor.param.date_range,
        channel=ged_monitor.param.channel,
        string=ged_monitor.param.string,
        sort_by=ged_monitor.param.sort_by,
        name="L200 Cal Monitoring",
    )
    phy_monitor = PhyMonitoring(
        phy_path=phy_path,
        base_path=base_monitor.params.base_path,
        run_dict=base_monitor.params.run_dict,
        periods=base_monitor.params.periods,
        period=base_monitor.params.period,
        run=base_monitor.params.run,
        date_range=base_monitor.param.date_range,
        channel=ged_monitor.param.channel,
        string=ged_monitor.param.string,
        sort_by=ged_monitor.param.sort_by,
        name="L200 Phy Monitoring",
    )

    sipm_monitor = SiPMMonitoring(
        sipm_path=sipm_path,
        base_path=base_monitor.params.base_path,
        run_dict=base_monitor.params.run_dict,
        periods=base_monitor.params.periods,
        period=base_monitor.params.period,
        run=base_monitor.params.run,
        date_range=base_monitor.param.date_range,
        name="L200 SiPM Monitoring",
    )

    muon_monitor = MuonMonitoring(
        muon_path=muon_path,
        base_path=base_monitor.params.base_path,
        run_dict=base_monitor.params.run_dict,
        periods=base_monitor.params.periods,
        period=base_monitor.params.period,
        run=base_monitor.params.run,
        date_range=base_monitor.param.date_range,
        name="L200 Muon Monitoring",
    )
    llama_monitor = LlamaMonitoring(
        llama_path=llama_path,
        name="L200 Llama Monitoring",
    )

    sidebar = base_monitor.build_sidebar(
        logo_path=logo_path,
    )
    l200_monitoring.sidebar.append(
        ged_monitor.build_sidebar(logo_path, sidebar_instance=sidebar)
    )

    cal_panes = cal_monitor.build_cal_panes(
        logo_path=logo_path,
        widget_widths=widget_widths,
    )
    # cal
    for pane in cal_panes.values:
        l200_monitoring.main.append(pane)

    # phy
    l200_monitoring.main.append(
        phy_monitor.build_phy_pane(
            logo_path=logo_path,
            widget_widths=widget_widths,
        )
    )

    # spms
    l200_monitoring.main.append(
        sipm_monitor.build_phy_pane(
            logo_path=logo_path,
            widget_widths=widget_widths,
        )
    )

    # muon
    muon_panes = muon_monitor.build_muon_panes(
        logo_path=logo_path,
        widget_widths=widget_widths,
    )
    for pane in muon_panes.values:
        l200_monitoring.main.append(pane)

    # metadata
    l200_monitoring.main.append(
        ged_monitor.build_meta_pane(logo_path=logo_path, widget_widths=widget_widths)
    )

    # llama
    l200_monitoring.main.append(
        llama_monitor.build_llama_pane(logo_path=logo_path, widget_widths=widget_widths)
    )

    return l200_monitoring


def build_header_logos():
    # Header
    return pn.Row(
        pn.pane.Image(
            "https://legend.edm.nat.tum.de/logos/github-mark.png",
            link_url="https://github.com/legend-exp/",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            "https://legend.edm.nat.tum.de/logos/logo_indico.png",
            link_url="https://indico.legend-exp.org",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            "https://legend.edm.nat.tum.de/logos/confluence.png",
            link_url="https://legend-exp.atlassian.net/wiki/spaces/LEGEND/overview",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            "https://legend.edm.nat.tum.de/logos/elog.png",
            link_url="https://elog.legend-exp.org/ELOG/",
            fixed_aspect=True,
            width=30,
        ),
        width=1800,
        align="center",
    )


def build_info_pane(info_path):
    with Path(info_path).open() as f:
        general_information = f.read()
    return pn.pane.Markdown(general_information)


def run_dashboard() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config-file", type=str, required=True)
    argparser.add_argument("--port", type=int, default=9000)
    argparser.add_argument("--widget-widths", type=int, default=140, required=False)
    args = argparser.parse_args()

    logo_path = Path(__file__).parent.parent.parent / "logos"
    info_path = Path(__file__).parent.parent.parent / "information" / "general.md"

    l200_monitoring = build_dashboard(args.config_file, logo_path, args.widget_widths)

    l200_monitoring.header.append(
        pn.Row(pn.Spacer(width=120), build_header_logos(), sizing_mode="stretch_width")
    )

    l200_monitoring.main.append(pn.Tabs(("Information", build_info_pane(info_path))))
    l200_monitoring.servable(port=args.port)
