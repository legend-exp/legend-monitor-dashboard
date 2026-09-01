from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.resources
import os
import secrets
import sys
import threading
from pathlib import Path

import panel as pn
from panel.auth import BasicAuthProvider, BasicLoginHandler
from panel.io.resources import CDN_DIST


def get_paths():
    # This finds the directory where dashboard.py lives
    curr_dir = Path(__file__).parent.resolve()
    img_dir = curr_dir / "information" / "img"
    logo_dir = curr_dir / "logos"

    # Verify it exists before passing it to pn.serve
    if not img_dir.exists():
        print(f"Warning: Image directory not found at {img_dir}")  # noqa: T201

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
    from legenddashboard.geds.phy.phy_shifter import PhyShifterMonitoring
    from legenddashboard.llama.llama_monitoring import LlamaMonitoring
    from legenddashboard.muon.muon_monitoring import MuonMonitoring
    from legenddashboard.spms.sipm_monitoring import SiPMMonitoring
    from legenddashboard.util import period_refresh_registry, read_config

    if disable_page is None:
        disable_page = ()
    config = read_config(config)

    # path to period data
    data_path = config.base
    # path to calibration data
    cal_path = config.cal
    # path to physics data
    phy_path = config.phy
    # path to muon data
    muon_path = config.muon
    # tmp path for caching
    tmp_cal_path = config.tmp
    # llama data path
    llama_path = config.llama

    # FastListTemplate with header and LEGEND logo from the LEGEND webpage. The
    # main panes are wrapped in a single pn.Tabs (see below) instead of the old
    # GoldenTemplate windows: GoldenLayout builds its panes with JavaScript
    # after page load, and in Firefox the Bokeh embed runs before those target
    # divs exist ("could not find ... HTML tag"), leaving the page blank.
    l200_monitoring = pn.template.FastListTemplate(
        header_background="#f8f8fa",
        header_color="#1A2A5B",
        title="L200 Monitoring Dashboard",
        sidebar_width=300,
        main_layout=None,
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
        font-family: "Bradley Hand", cursive;
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

    async def _on_refresh(event):
        # Trigger the shared scan; results are pushed to every session
        # (including this one) via its document. Run the filesystem scan on a
        # worker thread so the event loop (and the loading indicator) stays
        # responsive while the high-latency scan runs.
        refresh_button.loading = True
        try:
            await asyncio.to_thread(period_refresh_registry.scan_and_push)
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

    # Collect every main pane as a (title, pane) tab; they are placed into a
    # single pn.Tabs at the end so all target divs exist in the DOM at embed
    # time (deterministic rendering across browsers, incl. Firefox).
    main_tabs: list = []

    if "cal" not in disable_page:
        cal_monitor = CalMonitoring(
            base_path=cal_path,
            tmp_path=tmp_cal_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
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
        for title, pane in cal_panes.items():
            main_tabs.append((title, pane))
    if "phy" not in disable_page:
        # the ged sidebar exists whether or not the cal pages do, so both phy
        # pages always follow its channel/sorting/string selections
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
        shifter_monitor = PhyShifterMonitoring(
            base_path=cal_path,
            phy_path=phy_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            sort_by=ged_monitor.param.sort_by,
            name="L200 Phy Shifter",
        )
        ged_monitor.param.watch(
            lambda e: setattr(shifter_monitor, "string", e.new), "string"
        )
        main_tabs.append(
            ("Phy. Shifter", shifter_monitor.build_shifter_pane(widget_widths))
        )
        main_tabs.append(
            (
                "Phy. Expert",
                phy_monitor.build_phy_pane(
                    widget_widths=widget_widths,
                ),
            )
        )
    if "spm" not in disable_page:
        sipm_monitor = SiPMMonitoring(
            base_path=cal_path,
            phy_path=phy_path,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            name="L200 SiPM Monitoring",
        )
        main_tabs.append(("SiPM", sipm_monitor.build_sipm_pane(widget_widths)))

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
        for title, pane in muon_panes.items():
            main_tabs.append((title, pane))
    if "meta" not in disable_page:
        main_tabs.append(
            ("MetaData", ged_monitor.build_meta_pane(widget_widths=widget_widths))
        )
    if "metaedit" not in disable_page and "metadata_edit" in config:
        from legenddashboard.metadata.meta_monitoring import MetaMonitoring

        meta_monitor = MetaMonitoring(
            base_path=cal_path,
            meta_path=config.metadata_edit,
            run_dict=base_monitor.param.run_dict,
            periods=base_monitor.param.periods,
            period=base_monitor.param.period,
            run=base_monitor.param.run,
            date_range=base_monitor.param.date_range,
            name="L200 Metadata Editor",
        )
        main_tabs.append(
            ("Metadata Editor", meta_monitor.build_metadata_pane(widget_widths))
        )
    if "llama" not in disable_page:
        llama_monitor = LlamaMonitoring(
            llama_path=llama_path,
            base_path=cal_path,
            name="L200 Llama Monitoring",
        )
        main_tabs.append(
            ("Llama", llama_monitor.build_llama_pane(widget_widths=widget_widths))
        )

    # Information tab (was previously appended by the per-session factory).
    info_path = (
        importlib.resources.files("legenddashboard") / "information" / "general.md"
    )
    main_tabs.append(("Information", build_info_pane(info_path)))

    # Single Tabs holds every pane (the GoldenTemplate/Firefox embed fix:
    # Panel owns the tab divs). dynamic=True renders a tab's content when it
    # is first opened, so a run switch only re-renders the visible tab
    # instead of every page.
    l200_monitoring.main.append(
        pn.Tabs(*main_tabs, sizing_mode="stretch_both", dynamic=True)
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


class _XSRFBasicLoginHandler(BasicLoginHandler):
    """Panel's basic login handler, extended to carry the Tornado XSRF token.

    The stock handler renders a form without the ``_xsrf`` field, so serving
    with ``xsrf_cookies=True`` would reject every login attempt with a 403.
    """

    def get(self):
        try:
            errormessage = self.get_argument("error")
        except Exception:
            errormessage = ""
        next_url = self.get_argument("next", pn.state.base_url)
        if next_url:
            if pn.state.base_url and not next_url.startswith(pn.state.base_url):
                next_url = next_url.replace("/", pn.state.base_url, 1)
            self.set_cookie("next_url", next_url)
        html = self._login_template.render(
            login_endpoint=self._login_endpoint,
            errormessage=errormessage,
            PANEL_CDN=CDN_DIST,
            # Rendering the hidden form field also sets the _xsrf cookie.
            xsrf_input=self.xsrf_form_html(),
        )
        self.write(html)


class _XSRFBasicAuthProvider(BasicAuthProvider):
    """BasicAuthProvider whose login form includes the XSRF token."""

    @property
    def login_handler(self):
        _XSRFBasicLoginHandler._login_endpoint = self._login_endpoint
        _XSRFBasicLoginHandler._login_template = self._login_template
        return _XSRFBasicLoginHandler


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

    img_dir, logo_dir = get_paths()

    # Clone-or-update the editable metadata checkout for the Metadata editor
    # page before serving (also prunes per-user workspace worktrees: clean
    # ones are removed, ones with un-pushed edits are kept). Python-side (not
    # an entrypoint script) so local runs and the container behave
    # identically; failures only disable the editor page, never the dashboard.
    if "metaedit" not in args.disable_page:
        from legenddashboard.util import read_config

        _paths = read_config(args.config_file)
        if "metadata_edit" in _paths:
            from legenddashboard.metadata import meta_git

            meta_git.ensure_clone(
                _paths.metadata_edit,
                os.environ.get("METADATA_EDIT_URL", meta_git.DEFAULT_URL),
            )

    # Parsed par files are cached on disk under paths.tmp (restart-proof) and
    # parsed up front, so a session's first clicks do not pay the
    # multi-second yaml parse.
    if "cal" not in args.disable_page:
        from legenddashboard.util import (
            configure_par_disk_cache,
            prewarm_run_pars,
            read_config,
        )

        _paths = read_config(args.config_file)
        configure_par_disk_cache(_paths.get("tmp"))
        prewarm_run_pars(_paths.cal, n_periods=1)  # latest period: before serving
        threading.Thread(  # the rest: in the background
            target=prewarm_run_pars, args=(_paths.cal,), daemon=True
        ).start()

    def _build_dash():
        # Build a fresh dashboard per session so each user gets independent
        # widget state. The heavy, read-only data (metadata catalogs and parsed
        # parameter files) is cached and shared across sessions, so this stays
        # cheap despite running on every connection. The Information tab is now
        # added inside build_dashboard's main Tabs.
        return build_dashboard(args.config_file, args.widget_widths, args.disable_page)

    print(  # noqa: T201
        f"Starting Monitoring Dashboard on port: {args.port} with {args.num_threads} threads"
    )

    # These are pn.config options, not pn.serve kwargs: passed as kwargs they
    # would be silently swallowed as Tornado settings, leaving the server
    # single-threaded and without a loading indicator.
    pn.config.nthreads = args.num_threads
    pn.config.global_loading_spinner = True

    serve_kwargs = {
        "port": args.port,
        "show": False,
        # Tornado cross-site request forgery protection. The Bokeh/Tornado
        # kwarg is ``xsrf_cookies`` (``enable_xsrf_cookies`` is only the CLI
        # flag name and would be silently ignored here). The login form ships
        # its own template carrying the token (see _XSRFBasicLoginHandler).
        "xsrf_cookies": True,
        "warm": True,
        "use_xheaders": True,
        "address": "0.0.0.0",
        "num_procs": 1,
        "static_dirs": {"img": img_dir, "logos": logo_dir},
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
        # Passing ``basic_auth`` to pn.serve would install Panel's stock login
        # form, which lacks the XSRF field. Install our provider instead and
        # expose the credentials via pn.config.basic_auth, which the login
        # handler's validation falls back to.
        login_template = (
            importlib.resources.files("legenddashboard")
            / "templates"
            / "basic_login.html"
        )
        pn.config.basic_auth = basic_auth
        serve_kwargs["auth_provider"] = _XSRFBasicAuthProvider(
            login_template=str(login_template)
        )
        serve_kwargs["cookie_secret"] = cookie_secret
        print("Shared-password authentication enabled.")  # noqa: T201
    else:
        if username:
            print(  # noqa: T201
                "=" * 70 + "\nWARNING: DASHBOARD_USERNAME is set but DASHBOARD_PASSWORD"
                " is missing\nor empty -- the configured credentials are NOT"
                " in effect.\n" + "=" * 70,
                file=sys.stderr,
            )
        print(  # noqa: T201
            "=" * 70 + "\nWARNING: no DASHBOARD_PASSWORD set; the dashboard is served"
            " WITHOUT\nauthentication and is publicly accessible.\n" + "=" * 70,
            file=sys.stderr,
        )

    pn.serve(_build_dash, **serve_kwargs)
