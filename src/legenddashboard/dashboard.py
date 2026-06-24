from __future__ import annotations

import argparse
import datetime as dt
import importlib.resources
import os
import secrets
from pathlib import Path

import panel as pn


def get_paths():
    # This finds the directory where dashboard.py lives
    curr_dir = Path(__file__).parent.resolve()
    img_dir = curr_dir / "information" / "img"
    logo_dir = curr_dir / "logos"

    # Verify it exists before passing it to pn.serve
    if not img_dir.exists():
        print(f"Warning: Logo directory not found at {img_dir}")  # noqa: T201

    # Verify it exists before passing it to pn.serve
    if not logo_dir.exists():
        print(f"Warning: Logo directory not found at {logo_dir}")  # noqa: T201

    return img_dir, logo_dir


def build_dashboard(
    config: str | dict,
    widget_widths: int = 140,
    disable_page: list[str] | None = None,
):
    from legenddashboard.base import Monitoring
    from legenddashboard.geds.cal.cal_monitoring import CalMonitoring
    from legenddashboard.geds.ged_monitoring import GedMonitoring
    from legenddashboard.geds.phy.phy_monitoring import PhyMonitoring
    from legenddashboard.llama.llama_monitoring import LlamaMonitoring
    from legenddashboard.muon.muon_monitoring import MuonMonitoring
    from legenddashboard.spms.sipm_monitoring import SiPMMonitoring
    from legenddashboard.util import period_refresh_registry, read_config

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
    # Register this session for the single, server-wide periodic scan for new
    # periods/runs (one filesystem scan for all users, pushed into every live
    # session), and make sure that scan is scheduled exactly once per hour.
    period_refresh_registry.register(base_monitor, data_path)
    period_refresh_registry.ensure_scheduled(dt.timedelta(hours=1))

    # Manual refresh button, placed in the top-right corner of the header.
    refresh_button = pn.widgets.Button(
        name="Refresh",
        icon="refresh",
        button_type="primary",
        width=120,
        description="Check now for new periods and runs",
    )

    def _on_refresh(event):
        # Trigger the shared scan; results are pushed to every session
        # (including this one) via its document.
        refresh_button.loading = True
        try:
            period_refresh_registry.scan_and_push()
        finally:
            refresh_button.loading = False

    refresh_button.on_click(_on_refresh)
    l200_monitoring.header.append(
        pn.Row(
            pn.Spacer(width=120),
            build_header_logos(),
            pn.HSpacer(),
            refresh_button,
            sizing_mode="stretch_width",
        )
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
            sort_by=ged_monitor.param.sort_by,
            name="L200 Cal Monitoring",
        )
        ged_monitor.param.watch(
            lambda e: setattr(cal_monitor, "string", e.new), "string"
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
                sort_by=ged_monitor.param.sort_by,
                name="L200 Phy Monitoring",
            )
            ged_monitor.param.watch(
                lambda e: setattr(phy_monitor, "string", e.new), "string"
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
    _, logo_dir = get_paths()
    # Header
    return pn.Row(
        pn.pane.Image(
            logo_dir / "github-mark.png",
            link_url="https://github.com/legend-exp/",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            logo_dir / "logo_indico.png",
            link_url="https://indico.legend-exp.org",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            logo_dir / "confluence.png",
            link_url="https://legend-exp.atlassian.net/wiki/spaces/LEGEND/overview",
            fixed_aspect=True,
            width=24,
        ),
        pn.pane.Image(
            logo_dir / "elog.png",
            link_url="https://elog.legend-exp.org/ELOG/",
            fixed_aspect=True,
            width=30,
        ),
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
        "-w", "--widget-widths", type=int, default=140, required=False
    )
    argparser.add_argument(
        "-d", "--disable-page", nargs="*", required=False, default=[]
    )
    # num_procs is fixed to 1: warm=True (pre-warmed sessions) requires a single
    # process, and the expensive read-only metadata is shared across sessions
    # in-process anyway (see legenddashboard.util.get_sort_dets / get_par_cache).
    argparser.add_argument("--num-threads", type=int, default=4)
    argparser.add_argument(
        "--websocket-origin",
        nargs="*",
        default=None,
        help=(
            "Allowed websocket origin host(s), e.g. the public NERSC spin "
            "hostname. Required when serving behind a reverse proxy, otherwise "
            "Bokeh rejects the websocket connection and the dashboard never "
            "updates."
        ),
    )

    args = argparser.parse_args()

    info_path = (
        importlib.resources.files("legenddashboard") / "information" / "general.md"
    )
    img_dir, logo_dir = get_paths()

    def _build_dash():
        # Build a fresh dashboard per session so each user gets independent
        # widget state. The heavy, read-only data (metadata catalogs and parsed
        # parameter files) is cached and shared across sessions, so this stays
        # cheap despite running on every connection.
        l200_monitoring = build_dashboard(
            args.config_file, args.widget_widths, args.disable_page
        )
        l200_monitoring.main.append(
            pn.Tabs(("Information", build_info_pane(info_path)))
        )
        return l200_monitoring

    print(  # noqa: T201
        f"Starting Monitoring Dashboard on port: {args.port} with {args.num_threads} threads"
    )
    serve_kwargs = {
        "port": args.port,
        "show": False,
        "enable_xsrf_cookies": True,
        "warm": True,
        "use_xheaders": True,
        "address": "0.0.0.0",
        "num_procs": 1,
        "num_threads": args.num_threads,
        "static_dirs": {"img": img_dir, "logos": logo_dir},
        "global_loading_spinner": True,
    }
    if args.websocket_origin:
        serve_kwargs["websocket_origin"] = args.websocket_origin

    # Optional authentication. On NERSC spin the username, password and cookie
    # secret are injected as environment variables from spin secrets, so they
    # never appear in the image, the command line or the repo. When
    # DASHBOARD_PASSWORD is set, Panel shows a login page. If DASHBOARD_USERNAME
    # is also set, only that username/password pair is accepted; otherwise any
    # username with the matching password works.
    password = os.environ.get("DASHBOARD_PASSWORD")
    username = os.environ.get("DASHBOARD_USERNAME")
    basic_auth = {username: password} if (password and username) else password
    if basic_auth:
        cookie_secret = os.environ.get("DASHBOARD_COOKIE_SECRET")
        if not cookie_secret:
            # A cookie secret is required to sign the login cookie. Generate an
            # ephemeral one if none is provided, but warn: it changes on every
            # restart (invalidating logins) and differs across replicas, so set
            # it as a spin secret for stable sessions.
            cookie_secret = secrets.token_urlsafe(32)
            print(  # noqa: T201
                "DASHBOARD_COOKIE_SECRET not set; generated an ephemeral one. "
                "Logins will be invalidated on restart -- set it as a spin "
                "secret for stable sessions."
            )
        serve_kwargs["basic_auth"] = basic_auth
        serve_kwargs["cookie_secret"] = cookie_secret
        print("Shared-password authentication enabled.")  # noqa: T201
    else:
        print(  # noqa: T201
            "No DASHBOARD_PASSWORD set; serving without authentication."
        )

    pn.serve(_build_dash, **serve_kwargs)
