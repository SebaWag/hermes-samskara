"""The trust boundary — where a transcript is allowed to travel.

These tests exist because of a review finding, and they pin the property that
finding was about: **inside the agent, Samskara has no model of its own.** The
only model it can use is the one the host passes in as `ctx.llm`; if the host
does not pass one, extraction is skipped rather than redirected to whatever key
happens to be in the environment.

The second half is the credential path: `SAMSKARA_PASSWORD` is read through
`agent.secret_scope.get_secret`, declared in the config schema with `secret` +
`env_var`, and never written to `samskara.json`.

Nothing here needs Hermes, a network or a graph.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.llm import (  # noqa: E402
    CLI_API_KEY_VAR,
    CLI_BASE_URL_VAR,
    HermesLLM,
    OpenAICompatLLM,
    make_cli_llm,
    make_llm,
)

_OTHER_PROVIDER_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")


# ---------------------------------------------------------------------------
# 1. inside the plugin: the host model, or nothing
# ---------------------------------------------------------------------------
class _HostLLM:
    """A stand-in for `ctx.llm`, recording what it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, instructions, input, json_schema):  # noqa: A002 - host API
        self.calls += 1
        return {"triples": []}


class _Ctx:
    def __init__(self, llm=None) -> None:
        if llm is not None:
            self.llm = llm


@pytest.fixture
def other_provider_keys(monkeypatch):
    """The dangerous environment: keys for *other* providers are exported."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-never-be-sent")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-should-never-be-sent")
    monkeypatch.delenv(CLI_BASE_URL_VAR, raising=False)
    monkeypatch.delenv(CLI_API_KEY_VAR, raising=False)
    return monkeypatch


def test_no_ctx_means_no_model():
    llm = make_llm(None)
    assert llm.available is False
    assert "ctx.llm" in llm.unavailable_reason


def test_host_without_llm_does_not_fall_back_to_environment(other_provider_keys):
    """The review's finding, as an assertion.

    Hermes' in-process memory-provider path does not expose `ctx.llm`. With
    `OPENAI_API_KEY` — or `DEEPSEEK_API_KEY` — exported, the old code would
    build an OpenAI-compatible client pointed at api.deepseek.com. Now it
    returns an unavailable adapter and makes no call at all.
    """
    llm = make_llm(_Ctx())
    assert llm.available is False
    assert isinstance(llm, HermesLLM)


def test_unavailable_host_model_refuses_to_run(other_provider_keys):
    llm = make_llm(_Ctx())
    with pytest.raises(RuntimeError):
        llm("extract", "a transcript", {"type": "object"})


def test_host_model_is_used_when_present():
    host = _HostLLM()
    llm = make_llm(_Ctx(host))
    assert llm.available is True
    llm("extract", "a transcript", {"type": "object"})
    assert host.calls == 1


# ---------------------------------------------------------------------------
# 2. the CLI client: explicit endpoint, explicit key, no inherited host
# ---------------------------------------------------------------------------
def test_cli_client_needs_both_variables(other_provider_keys):
    assert OpenAICompatLLM().available is False
    assert make_cli_llm().available is False


def test_cli_client_never_reuses_another_providers_key(other_provider_keys):
    """A key written for OpenAI must never be presented to a DeepSeek host."""
    llm = OpenAICompatLLM()
    assert llm.api_key == ""
    assert llm.base_url == ""
    assert "deepseek.com" not in (llm.base_url or "")


def test_cli_client_requires_an_explicit_base_url(monkeypatch):
    monkeypatch.setenv(CLI_API_KEY_VAR, "sk-samskara")
    monkeypatch.delenv(CLI_BASE_URL_VAR, raising=False)
    llm = OpenAICompatLLM()
    assert llm.available is False
    assert CLI_BASE_URL_VAR in llm.unavailable_reason


def test_cli_client_with_both_variables_is_available(monkeypatch):
    monkeypatch.setenv(CLI_BASE_URL_VAR, "https://llm.example.com/v1")
    monkeypatch.setenv(CLI_API_KEY_VAR, "sk-samskara")
    llm = OpenAICompatLLM()
    assert llm.available is True
    assert llm.base_url == "https://llm.example.com/v1"
    assert llm.api_key == "sk-samskara"


# ---------------------------------------------------------------------------
# 3. session end: no network when the host offers no model
# ---------------------------------------------------------------------------
class _FakeGraph:
    """Enough of a SparqlClient for `find_entities`."""

    endpoint = "http://fake/sparql"

    def __init__(self) -> None:
        self.queries = 0

    def select(self, query, use_prefixes=True):
        self.queries += 1
        return [{"e": "ws:postgresql", "label": "PostgreSQL"}], 0.1


def _load_plugin(name="samskara_trust"):
    spec = importlib.util.spec_from_file_location(name, REPO / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_end_makes_no_llm_call_without_a_host_model(tmp_path, monkeypatch, other_provider_keys):
    """The end-to-end version of finding 1: no socket, ever."""
    import urllib.request

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("samskara opened a network connection without a host model")

    monkeypatch.setattr(urllib.request, "urlopen", explode)

    module = _load_plugin("samskara_trust_session_end")
    provider = module.SamskaraMemoryProvider()
    provider.initialize("trust-session", hermes_home=str(tmp_path))
    provider._client = _FakeGraph()

    provider.on_session_end(
        [
            {"role": "user", "content": "We froze Onboarding Hub on PostgreSQL 15."},
            {"role": "assistant", "content": "Noted: no upgrade while the incident is open."},
        ]
    )

    # It still degrades usefully: the lexical fallback recorded what was mentioned.
    journal_dir = Path(provider._journal_dir())
    written = "".join(f.read_text(encoding="utf-8") for f in sorted(journal_dir.glob("session-*.ttl")))
    assert "/postgresql>" in written  # the entity mention, expanded to its URI
    assert provider._stats["errors"] == 0


def test_session_end_uses_the_host_model_when_present(tmp_path, monkeypatch):
    module = _load_plugin("samskara_trust_host_model")
    host = _HostLLM()
    module._CTX = _Ctx(host)
    try:
        provider = module.SamskaraMemoryProvider()
        provider.initialize("trust-host", hermes_home=str(tmp_path))
        provider._client = _FakeGraph()
        provider.on_session_end(
            [
                {"role": "user", "content": "Onboarding Hub runs PostgreSQL 15."},
                {"role": "assistant", "content": "Understood."},
            ]
        )
        assert host.calls == 1
    finally:
        module._CTX = None


# ---------------------------------------------------------------------------
# 4. credentials: secret scope, schema declaration, never in samskara.json
# ---------------------------------------------------------------------------
def _install_secret_scope(monkeypatch, mapping):
    """Stand in for `agent.secret_scope` — the host's profile-scoped reader."""
    fake_agent = types.ModuleType("agent")
    fake_scope = types.ModuleType("agent.secret_scope")
    fake_scope.get_secret = lambda name, default=None: mapping.get(name, default)
    fake_agent.secret_scope = fake_scope
    monkeypatch.setitem(sys.modules, "agent", fake_agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", fake_scope)
    return fake_scope


def test_password_is_read_through_the_secret_scope(tmp_path, monkeypatch):
    _install_secret_scope(monkeypatch, {"SAMSKARA_PASSWORD": "from-the-scope"})
    module = _load_plugin("samskara_trust_secret_scope")
    provider = module.SamskaraMemoryProvider()
    provider.initialize("trust-secret", hermes_home=str(tmp_path))
    assert provider._cfg["password"] == "from-the-scope"


def test_password_is_not_read_from_the_environment(tmp_path, monkeypatch):
    """Bypassing the scope would leak across profiles in a multiplexing gateway."""
    monkeypatch.setenv("SAMSKARA_PASSWORD", "from-the-environment")
    module = _load_plugin("samskara_trust_no_env_password")
    provider = module.SamskaraMemoryProvider()
    provider.initialize("trust-env", hermes_home=str(tmp_path))
    assert provider._cfg["password"] == ""


def test_schema_declares_the_password_as_a_secret(tmp_path, monkeypatch):
    module = _load_plugin("samskara_trust_schema")
    provider = module.SamskaraMemoryProvider()
    schema = provider.get_config_schema()

    # The host reads `field["key"]`; `name` would be a KeyError in `hermes memory setup`.
    assert all("key" in field for field in schema)
    assert not any("name" in field for field in schema)

    password = next(field for field in schema if field["key"] == "password")
    assert password["secret"] is True
    assert password["env_var"] == "SAMSKARA_PASSWORD"

    # Every declared field is a key the provider actually understands.
    assert {field["key"] for field in schema} <= set(module.DEFAULTS)


def test_save_config_never_writes_the_password(tmp_path, monkeypatch):
    _install_secret_scope(monkeypatch, {"SAMSKARA_PASSWORD": "s3cret"})
    module = _load_plugin("samskara_trust_save_config")
    provider = module.SamskaraMemoryProvider()
    provider.initialize("trust-save", hermes_home=str(tmp_path))

    provider.save_config({"password": "leaked?", "timeout": 5}, str(tmp_path))

    saved = json.loads((Path(tmp_path) / module.CONFIG_FILE).read_text(encoding="utf-8"))
    assert "password" not in saved
    assert saved["timeout"] == 5
