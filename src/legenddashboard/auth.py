"""Login gate for the hosted dashboard.

Three modes, decided by :func:`configure_auth` from the environment:

- **LDAP** (``$DASHBOARD_LDAP_SERVER`` set) -- users sign in with their own
  directory credentials. Two bind strategies:
  *direct bind* (``$DASHBOARD_LDAP_USER_DN_TEMPLATE`` turns the login name
  into a DN, bound with the user's own password) or *search-then-bind* (a
  service account looks the user up under ``$DASHBOARD_LDAP_SEARCH_BASE``
  and the found DN is re-bound with the user's password).
  ``$DASHBOARD_LDAP_GROUP_DN`` optionally restricts access to one group.
- **shared password** (``$DASHBOARD_PASSWORD`` set) -- the previous
  behaviour, kept as the fallback.
- **open** -- no authentication, with a loud warning.

Every mode serves the login form through :class:`_XSRFBasicLoginHandler`,
which adds the Tornado XSRF field the stock Panel form lacks: the server runs
with ``xsrf_cookies=True``, so a form without it is rejected with a 403.

This plugs into Panel's basic-auth machinery by subclassing the
(semi-internal) ``BasicLoginHandler``/``BasicAuthProvider`` pair -- the same
extension pattern Panel itself uses for PAM auth. ``tests/test_auth.py`` has
a guard test over the surfaces we override; re-check them on a Panel upgrade.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import ssl
import sys
from urllib.parse import urlparse

import certifi
import ldap3
import panel as pn
import tornado.escape
from ldap3.core.exceptions import LDAPException
from ldap3.utils.conv import escape_filter_chars
from ldap3.utils.dn import escape_rdn
from panel.auth import BasicAuthProvider, BasicLoginHandler
from panel.io.resources import CDN_DIST
from panel.io.state import state

_TIMEOUT = 5  # seconds; _validate blocks the IOLoop, so keep LDAP calls short


# ---------------------------------------------------------------------------
# XSRF-carrying basic auth (used on its own, and as the LDAP base class)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LDAP
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LDAPConfig:
    """LDAP connection settings, normally read from the environment."""

    server_url: str
    user_dn_template: str | None = None
    bind_dn: str | None = None
    bind_password: str | None = None
    search_base: str | None = None
    user_filter: str = "(uid={username})"
    group_dn: str | None = None
    starttls: bool = False
    ca_file: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LDAPConfig | None:
        """Build a config from ``$DASHBOARD_LDAP_*``, or None if LDAP is off.

        Raises ``ValueError`` on an incomplete or contradictory configuration
        so a misconfigured server fails at startup, not at first login.
        """
        if env is None:
            env = dict(os.environ)
        server_url = env.get("DASHBOARD_LDAP_SERVER")
        if not server_url:
            return None
        cfg = cls(
            server_url=server_url,
            user_dn_template=env.get("DASHBOARD_LDAP_USER_DN_TEMPLATE") or None,
            bind_dn=env.get("DASHBOARD_LDAP_BIND_DN") or None,
            bind_password=env.get("DASHBOARD_LDAP_BIND_PASSWORD") or None,
            search_base=env.get("DASHBOARD_LDAP_SEARCH_BASE") or None,
            user_filter=env.get("DASHBOARD_LDAP_USER_FILTER")
            or "(uid={username})",
            group_dn=env.get("DASHBOARD_LDAP_GROUP_DN") or None,
            starttls=env.get("DASHBOARD_LDAP_STARTTLS", "").lower()
            in ("1", "true", "yes"),
            ca_file=env.get("DASHBOARD_LDAP_CA_FILE") or None,
        )
        cfg._check()
        return cfg

    def _check(self) -> None:
        if self.starttls and self.server_url.lower().startswith("ldaps://"):
            msg = (
                "DASHBOARD_LDAP_STARTTLS only applies to ldap:// URLs; "
                "ldaps:// is already TLS"
            )
            raise ValueError(msg)
        if self.user_dn_template is not None:
            if "{username}" not in self.user_dn_template:
                msg = "DASHBOARD_LDAP_USER_DN_TEMPLATE must contain '{username}'"
                raise ValueError(msg)
            return  # direct-bind mode; service account not required
        missing = [
            name
            for name, value in (
                ("DASHBOARD_LDAP_BIND_DN", self.bind_dn),
                ("DASHBOARD_LDAP_BIND_PASSWORD", self.bind_password),
                ("DASHBOARD_LDAP_SEARCH_BASE", self.search_base),
            )
            if not value
        ]
        if missing:
            msg = (
                "incomplete LDAP configuration: set "
                "DASHBOARD_LDAP_USER_DN_TEMPLATE (direct bind) or all of "
                "DASHBOARD_LDAP_BIND_DN, DASHBOARD_LDAP_BIND_PASSWORD and "
                "DASHBOARD_LDAP_SEARCH_BASE (search-then-bind); missing "
                f"{', '.join(missing)}"
            )
            raise ValueError(msg)
        if "{username}" not in self.user_filter:
            msg = "DASHBOARD_LDAP_USER_FILTER must contain '{username}'"
            raise ValueError(msg)


def _safe_next_url(url: str | None) -> str:
    """A same-origin relative redirect target, else the app root.

    Panel >=1.9 ships ``_validate_next_url`` for this, but it is private and
    absent from the versions this package also supports, so we do the check
    ourselves: anything with a scheme or host, or not under ``base_url``, is
    an open-redirect attempt and falls back to the root.
    """
    base = state.base_url or "/"
    if not url:
        return base
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc or not url.startswith("/"):
        return base
    return url if url.startswith(base) else base


class LDAPLoginHandler(_XSRFBasicLoginHandler):
    """Validates the login form against an LDAP directory."""

    _ldap_config: LDAPConfig  # set by LDAPAuthProvider.login_handler

    _AUTH_UNAVAILABLE = "Authentication service unavailable; try again later."

    def _validate(self, username: str, password: str) -> bool:
        self._auth_error: str | None = None
        # An empty password would be an LDAP *unauthenticated bind*, which
        # many servers report as success. Reject before touching the network.
        if not username or not username.strip() or not password:
            return False
        try:
            return self._ldap_check(self._ldap_config, username, password)
        except LDAPException as exc:
            # Server unreachable, TLS failure, service-account rejected, ...
            # Details go to the operator only; the user gets a generic notice.
            print(  # noqa: T201 (intentional operator-facing log)
                f"dashboard: LDAP error during login for {username!r}: {exc!r}",
                file=sys.stderr,
            )
            self._auth_error = self._AUTH_UNAVAILABLE
            return False

    def post(self) -> None:
        # Panel's BasicLoginHandler.post, but with our error message so
        # infrastructure failures read differently from bad credentials
        # (without leaking which of user/password/group failed). Written
        # against public handler API only -- Panel's own implementation
        # differs across the versions this package supports.
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        if self._validate(username, password):
            self.set_current_user(username)
            self.redirect(_safe_next_url(self.get_cookie("next_url", state.base_url)))
        else:
            error = getattr(self, "_auth_error", None) or "Invalid username or password!"
            self.redirect(
                self.request.uri + "?error=" + tornado.escape.url_escape(error)
            )

    def _ldap_check(self, cfg: LDAPConfig, username: str, password: str) -> bool:
        server = self._server(cfg)

        if cfg.user_dn_template:
            # Direct bind: derive the DN and try the user's own credentials.
            user_dn = cfg.user_dn_template.format(username=escape_rdn(username))
            conn = self._connect(server, cfg, user_dn, password)
            if conn is None:
                return False
            ok = self._in_group(conn, cfg, user_dn, username)
            conn.unbind()
            return ok

        # Search-then-bind: locate the entry as the service account, then
        # verify the password by binding as the found DN.
        svc = self._connect(server, cfg, cfg.bind_dn, cfg.bind_password)
        if svc is None:
            msg = (
                "LDAP service-account bind failed (check "
                "DASHBOARD_LDAP_BIND_DN/_PASSWORD)"
            )
            raise LDAPException(msg)
        try:
            filt = cfg.user_filter.format(username=escape_filter_chars(username))
            svc.search(cfg.search_base, filt, attributes=[])
            if len(svc.entries) != 1:  # unknown or ambiguous user
                return False
            user_dn = svc.entries[0].entry_dn
            if not self._in_group(svc, cfg, user_dn, username):
                return False
        finally:
            svc.unbind()
        conn = self._connect(server, cfg, user_dn, password)
        if conn is None:
            return False
        conn.unbind()
        return True

    @staticmethod
    def _server(cfg: LDAPConfig) -> ldap3.Server:
        tls = ldap3.Tls(
            validate=ssl.CERT_REQUIRED,
            ca_certs_file=cfg.ca_file or certifi.where(),
        )
        return ldap3.Server(cfg.server_url, tls=tls, connect_timeout=_TIMEOUT)

    @staticmethod
    def _connect(
        server: ldap3.Server,
        cfg: LDAPConfig,
        user_dn: str | None,
        password: str | None,
    ) -> ldap3.Connection | None:
        """Bind as ``user_dn``; ``None`` means the credentials were rejected."""
        conn = ldap3.Connection(
            server, user=user_dn, password=password, receive_timeout=_TIMEOUT
        )
        if cfg.starttls:
            conn.start_tls()
        if not conn.bind():
            conn.unbind()
            return None
        return conn

    @staticmethod
    def _in_group(
        conn: ldap3.Connection, cfg: LDAPConfig, user_dn: str, username: str
    ) -> bool:
        """Check group membership, if a group is configured at all.

        One BASE-scope read of the group entry, matching groupOfNames
        (``member``), groupOfUniqueNames (``uniqueMember``) and posixGroup
        (``memberUid``) in a single filter.
        """
        if not cfg.group_dn:
            return True
        dn, uid = escape_filter_chars(user_dn), escape_filter_chars(username)
        filt = f"(|(member={dn})(uniqueMember={dn})(memberUid={uid}))"
        conn.search(cfg.group_dn, filt, search_scope=ldap3.BASE, attributes=[])
        return bool(conn.entries)


class LDAPAuthProvider(_XSRFBasicAuthProvider):
    """Auth provider whose login form checks LDAP, not a password list."""

    def __init__(self, ldap_config: LDAPConfig, **kwargs: object):
        self._ldap_config = ldap_config
        super().__init__(**kwargs)

    @property
    def login_handler(self) -> type[LDAPLoginHandler]:
        # Same class-attribute wiring as the base property (one provider per
        # process), just targeting our handler subclass.
        LDAPLoginHandler._login_endpoint = self._login_endpoint
        LDAPLoginHandler._login_template = self._login_template
        LDAPLoginHandler._ldap_config = self._ldap_config
        return LDAPLoginHandler


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def _template(name: str) -> str:
    import importlib.resources  # noqa: PLC0415

    return str(importlib.resources.files("legenddashboard") / "templates" / name)


def _cookie_secret(env: dict[str, str]) -> str:
    """The signing secret for the login cookie, warning on an ephemeral one."""
    secret = env.get("DASHBOARD_COOKIE_SECRET")
    if secret:
        return secret
    # A cookie secret is required to sign the login cookie. Generate an
    # ephemeral one if none is provided, but warn: it changes on every
    # restart (invalidating logins) and differs across replicas, so set it
    # as a spin secret for stable sessions.
    print(  # noqa: T201
        "DASHBOARD_COOKIE_SECRET not set; generated an ephemeral one. "
        "Logins will be invalidated on restart -- set it as a spin "
        "secret for stable sessions."
    )
    return secrets.token_urlsafe(32)


def configure_auth(serve_kwargs: dict, env: dict[str, str] | None = None) -> str:
    """Install the login gate into ``serve_kwargs``; return a status banner.

    LDAP wins when configured, else the shared password, else the dashboard
    is served open. Mutates ``serve_kwargs`` in place.

    Note that ``auth_provider`` is never passed alongside ``basic_auth`` or
    ``login_template``: Panel would then build its own provider and silently
    discard ours, losing both the XSRF field and the LDAP check.
    """
    if env is None:
        env = dict(os.environ)

    ldap_config = LDAPConfig.from_env(env)  # raises on a half-configured server
    if ldap_config is not None:
        if env.get("DASHBOARD_PASSWORD"):
            print(  # noqa: T201
                "DASHBOARD_LDAP_SERVER is set; ignoring DASHBOARD_PASSWORD.",
                file=sys.stderr,
            )
        serve_kwargs["auth_provider"] = LDAPAuthProvider(
            ldap_config,
            login_template=_template("basic_login.html"),
            logout_template=_template("logout.html"),
        )
        serve_kwargs["cookie_secret"] = _cookie_secret(env)
        group = " (restricted to one group)" if ldap_config.group_dn else ""
        return f"LDAP authentication enabled against {ldap_config.server_url}{group}."

    password = env.get("DASHBOARD_PASSWORD")
    username = env.get("DASHBOARD_USERNAME")
    basic_auth = {username: password} if (password and username) else password
    if basic_auth:
        # Passing ``basic_auth`` to pn.serve would install Panel's stock login
        # form, which lacks the XSRF field. Install our provider instead and
        # expose the credentials via pn.config.basic_auth, which the login
        # handler's validation falls back to.
        pn.config.basic_auth = basic_auth
        serve_kwargs["auth_provider"] = _XSRFBasicAuthProvider(
            login_template=_template("basic_login.html"),
            logout_template=_template("logout.html"),
        )
        serve_kwargs["cookie_secret"] = _cookie_secret(env)
        return "Shared-password authentication enabled."

    if username:
        print(  # noqa: T201
            "=" * 70 + "\nWARNING: DASHBOARD_USERNAME is set but DASHBOARD_PASSWORD"
            " is missing\nor empty -- the configured credentials are NOT"
            " in effect.\n" + "=" * 70,
            file=sys.stderr,
        )
    print(  # noqa: T201
        "=" * 70 + "\nWARNING: no DASHBOARD_PASSWORD or DASHBOARD_LDAP_SERVER set;"
        " the dashboard\nis served WITHOUT authentication and is publicly"
        " accessible.\n" + "=" * 70,
        file=sys.stderr,
    )
    return "No authentication."
