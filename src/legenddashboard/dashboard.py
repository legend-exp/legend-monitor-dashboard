from __future__ import annotations

import argparse
from pathlib import Path

import panel as pn

from legenddashboard.base import Monitoring
from legenddashboard.geds.cal.cal_monitoring import CalMonitoring
from legenddashboard.geds.ged_monitoring import GedMonitoring
from legenddashboard.geds.phy.phy_monitoring import PhyMonitoring
from legenddashboard.llama.llama_monitoring import LlamaMonitoring
from legenddashboard.muon.muon_monitoring import MuonMonitoring
from legenddashboard.spms.sipm_monitoring import SiPMMonitoring
from legenddashboard.util import read_config


def build_dashboard(
    config: str | dict,
    widget_widths: int = 140,
    disable_page: list[str] | None = None,
):
    config = read_config(config)

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
        base_path=cal_path,
        run_dict=base_monitor.param.run_dict,
        periods=base_monitor.param.periods,
        period=base_monitor.param.period,
        run=base_monitor.param.run,
        date_range=base_monitor.param.date_range,
        name="L200 Ged Monitoring",
    )
    sidebar = base_monitor.build_sidebar()
    l200_monitoring.sidebar.append(ged_monitor.build_sidebar(sidebar_instance=sidebar))

    if "cal" not in disable_page:
        cal_monitor = CalMonitoring(
            base_path=cal_path,
            tmp_path=tmp_cal_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            channel=ged_monitor.param.channel,
            string=ged_monitor.param.string,
            sort_by=ged_monitor.param.sort_by,
            name="L200 Cal Monitoring",
        )
        cal_panes = cal_monitor.build_cal_panes(
            widget_widths=widget_widths,
        )
        # cal
        for pane in cal_panes.values():
            l200_monitoring.main.append(pane)
    if "phy" not in disable_page:
        if "cal" not in disable_page:
            phy_monitor = PhyMonitoring(
                base_path=cal_path,
                phy_path=phy_path,
                run_dict=base_monitor.param.run_dict,
                periods=base_monitor.param.periods,
                period=base_monitor.param.period,
                run=base_monitor.param.run,
                date_range=base_monitor.param.date_range,
                channel=ged_monitor.param.channel,
                string=ged_monitor.param.string,
                sort_by=ged_monitor.param.sort_by,
                name="L200 Phy Monitoring",
            )
        else:
            phy_monitor = PhyMonitoring(
                phy_path=phy_path,
                base_path=cal_path,
                run_dict=base_monitor.param.run_dict,
                periods=base_monitor.param.periods,
                period=base_monitor.param.period,
                run=base_monitor.param.run,
                date_range=base_monitor.param.date_range,
                name="L200 Phy Monitoring",
            )
        l200_monitoring.main.append(
            phy_monitor.build_phy_pane(
                widget_widths=widget_widths,
            )
        )
    if "spm" not in disable_page:
        sipm_monitor = SiPMMonitoring(
            sipm_path=sipm_path,
            base_path=cal_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            name="L200 SiPM Monitoring",
        )
        l200_monitoring.main.append(
            sipm_monitor.build_spm_pane(
                widget_widths=widget_widths,
            )
        )

    if "muon" not in disable_page:
        muon_monitor = MuonMonitoring(
            muon_path=muon_path,
            base_path=cal_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            name="L200 Muon Monitoring",
        )
        muon_panes = muon_monitor.build_muon_panes(
            widget_widths=widget_widths,
        )
        for pane in muon_panes.values():
            l200_monitoring.main.append(pane)
    if "meta" not in disable_page:
        l200_monitoring.main.append(
            ged_monitor.build_meta_pane(widget_widths=widget_widths)
        )
    if "llama" not in disable_page:
        llama_monitor = LlamaMonitoring(
            llama_path=llama_path,
            base_path=cal_path,
            name="L200 Llama Monitoring",
        )
        l200_monitoring.main.append(
            llama_monitor.build_llama_pane(widget_widths=widget_widths)
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
    argparser.add_argument("config_file", type=str)
    argparser.add_argument("-p", "--port", type=int, default=9000)
    argparser.add_argument(
        "-w", "--widget_widths", type=int, default=140, required=False
    )
    argparser.add_argument(
        "-d", "--disable_page", nargs="*", required=False, default=[]
    )
    args = argparser.parse_args()

    info_path = Path(__file__).parent.parent.parent / "information" / "general.md"

    l200_monitoring = build_dashboard(
        args.config_file, args.widget_widths, args.disable_page
    )

    l200_monitoring.header.append(
        pn.Row(pn.Spacer(width=120), build_header_logos(), sizing_mode="stretch_width")
    )

    l200_monitoring.main.append(pn.Tabs(("Information", build_info_pane(info_path))))
    print("Starting Monitoring Dashboard on port ", args.port)  # noqa: T201
    pn.serve(l200_monitoring, port=args.port, show=False)
