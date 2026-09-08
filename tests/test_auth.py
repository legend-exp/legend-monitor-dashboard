"""Tests for the login gate: LDAP validation and the auth-mode wiring.

No real (or containerised) LDAP server: ``ldap3`` is monkeypatched at module
level, so every bind/search is driven by the ``FakeConnection`` class attrs.
"""

from __future__ import annotations

from typing import ClassVar

import ldap3
import panel as pn
import pytest
from ldap3.core.exceptions import LDAPException
from panel.auth import BasicAuthProvider, BasicLoginHandler

from legenddashboard.auth import (
    LOGIN_HINT,
    LDAPAuthProvider,
    LDAPConfig,
    LDAPLoginHandler,
    _safe_next_url,
    _XSRFBasicAuthProvider,
    _XSRFBasicLoginHandler,
    configure_auth,
)

DIRECT_ENV = {
    "DASHBOARD_LDAP_SERVER": "ldaps://ldap.example:636",
    "DASHBOARD_LDAP_USER_DN_TEMPLATE": "uid={username},ou=people,dc=example,dc=org",
}
SEARCH_ENV = {
    "DASHBOARD_LDAP_SERVER": "ldaps://ldap.example:636",
    "DASHBOARD_LDAP_BIND_DN": "cn=svc,dc=example,dc=org",
    "DASHBOARD_LDAP_BIND_PASSWORD": "svcpw",
    "DASHBOARD_LDAP_SEARCH_BASE": "ou=people,dc=example,dc=org",
}


# ---------------------------------------------------------------- config


def test_from_env_disabled_without_server():
    assert LDAPConfig.from_env({}) is None
    assert (
        LDAPConfig.from_env({"DASHBOARD_LDAP_USER_DN_TEMPLATE": "uid={username}"})
        is None
    )


def test_from_env_direct_bind():
    cfg = LDAPConfig.from_env(DIRECT_ENV)
    assert cfg.user_dn_template.startswith("uid={username}")
    assert cfg.bind_dn is None


def test_from_env_search_bind():
    cfg = LDAPConfig.from_env(SEARCH_ENV)
    assert cfg.user_dn_template is None
    assert cfg.search_base == "ou=people,dc=example,dc=org"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"DASHBOARD_LDAP_USER_DN_TEMPLATE": "uid=fixed,ou=people"}, "username"),
        (
            {"DASHBOARD_LDAP_SERVER": "ldap://x", "DASHBOARD_LDAP_STARTTLS": "0"},
            "incomplete",
        ),
        (
            {
                **SEARCH_ENV,
                "DASHBOARD_LDAP_SERVER": "ldap://x",
                "DASHBOARD_LDAP_BIND_DN": "",
            },
            "DASHBOARD_LDAP_BIND_DN",
        ),
        ({**DIRECT_ENV, "DASHBOARD_LDAP_STARTTLS": "1"}, "ldaps"),
        ({**SEARCH_ENV, "DASHBOARD_LDAP_USER_FILTER": "(uid=fixed)"}, "username"),
    ],
)
def test_from_env_invalid(overrides, match):
    env = {"DASHBOARD_LDAP_SERVER": "ldaps://ldap.example:636", **overrides}
    with pytest.raises(ValueError, match=match):
        LDAPConfig.from_env(env)


# ---------------------------------------------------------------- fakes


class FakeConnection:
    """Stands in for ldap3.Connection; behaviour driven by class attrs."""

    passwords: ClassVar[dict] = {}  # dn -> password accepted by bind()
    search_results: ClassVar[dict] = {}  # search base -> list of entry DNs
    instances: ClassVar[list] = []

    def __init__(self, server, user=None, password=None, **kwargs):
        self.server, self.user, self.password = server, user, password
        self.kwargs = kwargs
        self.entries = []
        self.bound = False
        self.unbound = False
        self.started_tls = False
        self.searches = []
        FakeConnection.instances.append(self)

    def start_tls(self):
        self.started_tls = True

    def bind(self):
        self.bound = self.passwords.get(self.user) == self.password
        return self.bound

    def unbind(self):
        self.unbound = True

    def search(self, base, filt, **kwargs):
        self.searches.append((base, filt, kwargs))
        self.entries = [FakeEntry(dn) for dn in self.search_results.get(base, [])]
        return bool(self.entries)


class FakeEntry:
    def __init__(self, dn):
        self.entry_dn = dn


@pytest.fixture
def fake_ldap(monkeypatch):
    FakeConnection.passwords = {}
    FakeConnection.search_results = {}
    FakeConnection.instances = []
    monkeypatch.setattr(ldap3, "Connection", FakeConnection)
    monkeypatch.setattr(ldap3, "Server", lambda url, **_kw: url)
    monkeypatch.setattr(ldap3, "Tls", lambda **_kw: None)
    return FakeConnection


def make_handler(env):
    """A handler with only the attributes _validate needs (no tornado app)."""
    handler = LDAPLoginHandler.__new__(LDAPLoginHandler)
    handler._ldap_config = LDAPConfig.from_env(env)
    return handler


# ---------------------------------------------------------------- _validate


@pytest.mark.parametrize(
    ("username", "password"), [("", "pw"), ("  ", "pw"), ("alice", ""), ("", "")]
)
def test_empty_credentials_rejected_without_network(fake_ldap, username, password):
    # An empty password must never reach the server: it would be an LDAP
    # unauthenticated bind, which many servers report as success.
    handler = make_handler(DIRECT_ENV)
    assert handler._validate(username, password) is False
    assert fake_ldap.instances == []


def test_direct_bind_success_and_dn_escaping(fake_ldap):
    dn = "uid=alice\\,ou\\=x,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {dn: "pw"}
    handler = make_handler(DIRECT_ENV)
    # the comma in the username must be escaped into the DN, not split it
    assert handler._validate("alice,ou=x", "pw") is True
    (conn,) = fake_ldap.instances
    assert conn.user == dn
    assert conn.unbound


def test_direct_bind_bad_password(fake_ldap):
    fake_ldap.passwords = {"uid=alice,ou=people,dc=example,dc=org": "right"}
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "wrong") is False
    assert handler._auth_error is None  # generic "invalid username or password"


def test_search_bind_success_and_filter_escaping(fake_ldap):
    user_dn = "uid=alice,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {"cn=svc,dc=example,dc=org": "svcpw", user_dn: "pw"}
    fake_ldap.search_results = {"ou=people,dc=example,dc=org": [user_dn]}
    handler = make_handler(SEARCH_ENV)

    assert handler._validate("alice", "pw") is True
    svc, user = fake_ldap.instances
    assert svc.unbound
    assert user.user == user_dn

    # a filter-injection attempt reaches the server fully escaped
    fake_ldap.instances.clear()
    fake_ldap.search_results = {}  # nothing matches the (escaped) filter
    assert handler._validate("*)(uid=*", "pw") is False
    (svc,) = fake_ldap.instances
    assert svc.searches[0][1] == "(uid=\\2a\\29\\28uid=\\2a)"


@pytest.mark.parametrize("hits", [[], ["uid=a,ou=p", "uid=b,ou=p"]])
def test_search_bind_requires_exactly_one_entry(fake_ldap, hits):
    fake_ldap.passwords = {"cn=svc,dc=example,dc=org": "svcpw"}
    fake_ldap.search_results = {"ou=people,dc=example,dc=org": hits}
    handler = make_handler(SEARCH_ENV)
    assert handler._validate("alice", "pw") is False


def test_search_bind_bad_service_account_is_unavailable(fake_ldap, capsys):
    fake_ldap.passwords = {}  # service bind fails
    handler = make_handler(SEARCH_ENV)
    assert handler._validate("alice", "pw") is False
    assert handler._auth_error == LDAPLoginHandler._AUTH_UNAVAILABLE
    assert "LDAP error" in capsys.readouterr().err


# ---------------------------------------------------------------- group check


def test_group_membership_required(fake_ldap):
    env = {
        **DIRECT_ENV,
        "DASHBOARD_LDAP_GROUP_DN": "cn=legend,ou=groups,dc=example,dc=org",
    }
    user_dn = "uid=alice,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {user_dn: "pw"}

    # not a member: the group entry does not match the membership filter
    handler = make_handler(env)
    assert handler._validate("alice", "pw") is False
    (conn,) = fake_ldap.instances
    base, filt, kwargs = conn.searches[0]
    assert base == "cn=legend,ou=groups,dc=example,dc=org"
    for atom in (f"member={user_dn}", f"uniqueMember={user_dn}", "memberUid=alice"):
        assert atom in filt
    assert kwargs["search_scope"] == ldap3.BASE

    # member: same search now yields the group entry
    fake_ldap.instances.clear()
    group_dn = env["DASHBOARD_LDAP_GROUP_DN"]
    fake_ldap.search_results = {group_dn: [group_dn]}
    assert handler._validate("alice", "pw") is True


def test_no_group_configured_skips_search(fake_ldap):
    fake_ldap.passwords = {"uid=alice,ou=people,dc=example,dc=org": "pw"}
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "pw") is True
    (conn,) = fake_ldap.instances
    assert conn.searches == []


# ---------------------------------------------------------------- errors


@pytest.mark.usefixtures("fake_ldap")
def test_server_error_reports_unavailable(monkeypatch, capsys):
    def boom(*args, **kwargs):
        msg = "socket connection error"
        raise LDAPException(msg)

    monkeypatch.setattr(ldap3, "Connection", boom)
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "pw") is False
    assert handler._auth_error == LDAPLoginHandler._AUTH_UNAVAILABLE
    err = capsys.readouterr().err
    assert "LDAP error" in err
    assert "pw" not in err  # never log the password


# ---------------------------------------------------------------- redirects


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/steal",
        "//evil.example/steal",
        "http://evil.example",
        "not-a-path",
    ],
)
def test_safe_next_url_rejects_offsite(url):
    assert _safe_next_url(url) == (pn.state.base_url or "/")


def test_safe_next_url_keeps_local_path():
    assert _safe_next_url("/") == "/"
    assert _safe_next_url(None) == (pn.state.base_url or "/")


# ---------------------------------------------------------------- wiring


def test_configure_auth_ldap_wins(monkeypatch):
    # basic_auth alongside auth_provider would make Panel build its own
    # provider and silently discard ours (losing XSRF and the LDAP check).
    monkeypatch.setattr(pn.config, "basic_auth", None, raising=False)
    kwargs: dict = {}
    env = {
        **DIRECT_ENV,
        "DASHBOARD_PASSWORD": "ignored",
        "DASHBOARD_COOKIE_SECRET": "s",
    }
    banner = configure_auth(kwargs, env)

    assert isinstance(kwargs["auth_provider"], LDAPAuthProvider)
    assert "basic_auth" not in kwargs
    assert "login_template" not in kwargs
    assert kwargs["cookie_secret"] == "s"
    assert pn.config.basic_auth in (None, {})
    assert "LDAP" in banner


def test_configure_auth_falls_back_to_password(monkeypatch):
    monkeypatch.setattr(pn.config, "basic_auth", None, raising=False)
    kwargs: dict = {}
    env = {
        "DASHBOARD_PASSWORD": "pw",
        "DASHBOARD_USERNAME": "shifter",
        "DASHBOARD_COOKIE_SECRET": "s",
    }
    banner = configure_auth(kwargs, env)

    provider = kwargs["auth_provider"]
    assert isinstance(provider, _XSRFBasicAuthProvider)
    assert not isinstance(provider, LDAPAuthProvider)
    assert "basic_auth" not in kwargs
    assert pn.config.basic_auth == {"shifter": "pw"}
    assert "Shared-password" in banner


def test_configure_auth_open_when_unset(capsys):
    kwargs: dict = {}
    banner = configure_auth(kwargs, {})
    assert kwargs == {}
    assert "No authentication" in banner
    assert "WITHOUT" in capsys.readouterr().err


def test_configure_auth_raises_on_half_configured_ldap():
    with pytest.raises(ValueError, match="incomplete"):
        configure_auth({}, {"DASHBOARD_LDAP_SERVER": "ldaps://ldap.example:636"})


def test_ldap_provider_serves_our_handler():
    provider = LDAPAuthProvider(
        LDAPConfig.from_env(DIRECT_ENV), login_template=None, logout_template=None
    )
    handler = provider.login_handler
    assert handler is LDAPLoginHandler
    assert handler._ldap_config.server_url == DIRECT_ENV["DASHBOARD_LDAP_SERVER"]
    # the XSRF-carrying get() must survive the subclassing, or every login
    # POST is rejected with a 403 by xsrf_cookies=True
    assert handler.get is _XSRFBasicLoginHandler.get
    assert handler.get is not BasicLoginHandler.get


def test_panel_internals_we_subclass_still_exist():
    """Guard: these are semi-private Panel surfaces and we allow panel>=1.5."""
    assert hasattr(BasicLoginHandler, "_validate")
    assert hasattr(BasicLoginHandler, "post")
    assert isinstance(BasicAuthProvider.login_handler, property)


# ---------------------------------------------------------------- user chip


@pytest.mark.parametrize("user", [None, "", "guest"])
def test_user_chip_hidden_when_unauthenticated(monkeypatch, user):
    # Panel reports "guest" when no auth provider is configured.
    from legenddashboard import dashboard

    monkeypatch.setattr(pn.state, "_current_user", user, raising=False)
    monkeypatch.setattr(type(pn.state), "user", property(lambda _self: user))
    assert dashboard.build_user_chip() is None


def test_user_chip_shows_name_and_logout(monkeypatch):
    from legenddashboard import dashboard

    monkeypatch.setattr(type(pn.state), "user", property(lambda _self: "a<b>lice"))
    chip = dashboard.build_user_chip()
    assert chip is not None
    assert "a&lt;b&gt;lice" in chip.object  # escaped, never raw HTML
    assert './logout"' in chip.object


# ---------------------------------------------------------------- login hint


def _login_template():
    """The login template as the running provider would hold it.

    Built through a real provider so the test breaks if the shipped template
    stops being the one the LDAP gate serves.
    """
    provider = LDAPAuthProvider(
        LDAPConfig.from_env(DIRECT_ENV),
        login_template=_template_path("basic_login.html"),
        logout_template=_template_path("logout.html"),
    )
    return provider._login_template


def _template_path(name):
    import importlib.resources

    return str(importlib.resources.files("legenddashboard") / "templates" / name)


def test_login_hint_rendered_only_in_ldap_mode():
    template = _login_template()

    # LDAP mode: the handler passes the hint, so the page carries it
    html = template.render(
        login_endpoint="/login",
        errormessage="",
        login_hint=LOGIN_HINT,
        xsrf_input="",
        PANEL_CDN="",
    )
    assert LOGIN_HINT in html
    assert '<p class="login-hint">' in html

    # shared-password mode renders the same template without the variable
    # (the .login-hint CSS rule is always present; the element is not)
    plain = template.render(
        login_endpoint="/login", errormessage="", xsrf_input="", PANEL_CDN=""
    )
    assert '<p class="login-hint">' not in plain
    assert LOGIN_HINT not in plain


def test_only_the_ldap_handler_carries_the_hint():
    # the hint reaches the page through the shared XSRF get(), so the two
    # handlers must differ only in this class attribute
    assert LDAPLoginHandler._login_hint == LOGIN_HINT
    assert _XSRFBasicLoginHandler._login_hint == ""
