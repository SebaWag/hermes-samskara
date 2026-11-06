"""Offline unit tests — no graph, no network, no Hermes runtime required.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core import Entity, Fact, Journal, recall, render, render_triple  # noqa: E402
from samskara_core.client import SparqlError  # noqa: E402
from samskara_core.recall import find_entities, humanize, keywords, local_name  # noqa: E402


class FakeClient:
    """Stands in for SparqlClient: canned rows, controllable failure."""

    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.last_ms = 0.42
        self.endpoint = "http://fake/sparql"
        self.queries = 0

    def select(self, query, use_prefixes=True):
        self.queries += 1
        if self.fail:
            raise SparqlError("boom")
        # label lookups return matches; neighbourhood queries return facts
        if "rdfs:label ?label" in query:
            return self.rows, self.last_ms
        return [
            {"p": "https://wsolutionsai.com/ns/shiva/usaTecnologia",
             "o": "https://wsolutionsai.com/ns/shiva/postgresql", "oLabel": "PostgreSQL"},
            {"p": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
             "o": "https://wsolutionsai.com/ns/shiva/Proyecto", "oLabel": ""},
        ], self.last_ms

    def healthy(self):
        return not self.fail


# --------------------------------------------------------------------------
# text handling
# --------------------------------------------------------------------------
def test_keywords_drops_stopwords_and_short_tokens():
    got = keywords("¿Puedo subir PostgreSQL a la 17 en Onboarding Hub?")
    assert "postgresql" in got
    assert "onboarding" in got
    assert "puedo" not in got  # stopword
    assert "a" not in got       # too short
    assert len(got) <= 6


def test_keywords_prefers_longer_tokens():
    got = keywords("millenniumdb graph db")
    assert got[0] == "millenniumdb"


def test_local_name_and_humanize():
    assert local_name("https://wsolutionsai.com/ns/shiva/usaTecnologia") == "usaTecnologia"
    assert local_name("http://www.w3.org/2000/01/rdf-schema#label") == "label"
    assert humanize("usaTecnologia") == "usa tecnologia"


# --------------------------------------------------------------------------
# turtle serialisation (the write path must never emit invalid triples)
# --------------------------------------------------------------------------
NS = "https://wsolutionsai.com/ns/shiva/"


def test_triple_with_uri_object_is_unquoted():
    # the journal stores absolute identifiers so it can be merged into N-Triples
    assert render_triple(("ws:a", "ws:b", "ws:c")) == f"<{NS}a> <{NS}b> <{NS}c> ."


def test_triple_with_text_becomes_literal():
    assert render_triple(("ws:a", "rdfs:label", "Ledger App")) == (
        f'<{NS}a> <http://www.w3.org/2000/01/rdf-schema#label> "Ledger App" .'
    )


def test_triple_with_quoted_literal_is_not_double_quoted():
    assert render_triple(("ws:a", "ws:estado", '"vigente"')) == f'<{NS}a> <{NS}estado> "vigente" .'


def test_triple_escapes_embedded_quotes():
    out = render_triple(("ws:a", "ws:note", 'said "hi"'))
    assert out.count('"') == 4 and '\\"' in out


def test_triple_rejects_missing_subject():
    with pytest.raises(ValueError):
        render_triple(("", "ws:b", "c"))


# --------------------------------------------------------------------------
# journal: append-only, survives, counts what is pending
# --------------------------------------------------------------------------
def test_journal_appends_and_counts(tmp_path):
    j = Journal(tmp_path, session_id="s1")
    assert j.append([("ws:a", "ws:b", "ws:c"), ("ws:a", "rdfs:label", "A")]) == 2
    assert j.pending() == 2
    assert j.path.read_text().count("\n") == 2


def test_journal_accumulates_across_appends(tmp_path):
    j = Journal(tmp_path, session_id="s1")
    j.append([("ws:a", "ws:b", "ws:c")])
    j.append([("ws:d", "ws:e", "ws:f")])
    assert j.pending() == 2


def test_journal_sanitises_session_id(tmp_path):
    j = Journal(tmp_path, session_id="../../etc/passwd")
    assert ".." not in j.path.name and "/" not in j.path.name


def test_journal_isolates_sessions(tmp_path):
    Journal(tmp_path, session_id="s1").append([("ws:a", "ws:b", "ws:c")])
    Journal(tmp_path, session_id="s2").append([("ws:d", "ws:e", "ws:f")])
    assert len(Journal(tmp_path).files()) == 2


# --------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------
def test_find_entities_ranks_and_dedupes():
    rows = [
        {"e": "ws:onboarding_hub", "label": "Onboarding Hub"},
        {"e": "ws:onboarding_old", "label": "Onboarding Legacy"},
    ]
    client = FakeClient(rows=rows)
    found = find_entities(client, "onboarding hub")
    assert found and isinstance(found[0], Entity)
    assert found[0].uri == "ws:onboarding_hub"  # exact label wins
    assert len({e.uri for e in found}) == len(found)


def test_render_has_provenance_header():
    entities = [Entity("ws:postgresql", "PostgreSQL")]
    facts = {"ws:postgresql": [Fact("out", "versionActual", "15"),
                               Fact("in", "usaTecnologia", "Onboarding Hub")]}
    block = render(entities, facts, elapsed_ms=1.27)
    assert block.startswith("[memoria relacional")
    assert "1 entidad(es)" in block and "2 hecho(s)" in block
    assert "1.27 ms" in block
    assert "→" in block and "←" in block


def test_render_truncates_to_budget():
    entities = [Entity(f"ws:e{i}", f"Entity {i}") for i in range(30)]
    facts = {e.uri: [Fact("out", "rel", "x" * 40)] * 5 for e in entities}
    assert len(render(entities, facts, 1.0, max_chars=200)) <= 220


def test_recall_returns_block_for_known_entity():
    client = FakeClient(rows=[{"e": "ws:postgresql", "label": "PostgreSQL"}])
    block, elapsed = recall(client, "¿puedo subir postgresql?")
    assert "PostgreSQL" in block
    assert client.queries >= 2  # label lookup + neighbourhood


def test_recall_returns_empty_when_nothing_matches():
    client = FakeClient(rows=[])
    block, _ = recall(client, "tema que no existe")
    assert block == ""


def test_recall_swallows_graph_errors():
    client = FakeClient(fail=True)
    block, _ = recall(client, "postgresql")
    assert block == ""  # memory must never break a turn


# --------------------------------------------------------------------------
# the provider itself (loaded exactly like Hermes loads it: by path)
# --------------------------------------------------------------------------
@pytest.fixture()
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMSKARA_ENDPOINT", "http://fake/sparql")
    spec = importlib.util.spec_from_file_location("samskara_plugin", REPO / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inst = module.SamskaraMemoryProvider()
    inst.initialize("test-session", hermes_home=str(tmp_path))
    inst._client = FakeClient(rows=[{"e": "ws:postgresql", "label": "PostgreSQL"}])
    return inst


def test_provider_identity_and_availability(provider):
    assert provider.name == "samskara"
    assert provider.is_available() is True


def test_system_prompt_block_mentions_tools(provider):
    block = provider.system_prompt_block()
    assert "samskara_search" in block and "samskara_query" in block


def test_tool_schemas_are_well_formed(provider):
    schemas = provider.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"samskara_search", "samskara_query", "samskara_write", "samskara_status"} <= names
    for schema in schemas:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]


def test_prefetch_injects_block(provider):
    assert "PostgreSQL" in provider.prefetch("¿postgresql?")


def test_prefetch_survives_dead_graph(provider):
    provider._client = FakeClient(fail=True)
    assert provider.prefetch("postgresql") == ""


def test_prefetch_disabled_by_config(provider):
    provider._cfg["auto_recall"] = False
    assert provider.prefetch("postgresql") == ""


def test_query_tool_refuses_writes(provider):
    out = provider.handle_tool_call("samskara_query", {"sparql": "DELETE WHERE { ?s ?p ?o }"})
    assert "only SELECT/ASK" in out


def test_write_tool_journals_triple(provider, tmp_path):
    out = provider.handle_tool_call(
        "samskara_write",
        {"subject": "ws:x", "predicate": "ws:usaTecnologia", "object": "ws:postgresql"},
    )
    assert "queued" in out
    journals = list((tmp_path / "samskara" / "journal").glob("session-*.ttl"))
    assert journals and "ns/shiva/postgresql" in journals[0].read_text()


def test_pre_compress_preserves_relations(provider, tmp_path):
    block = provider.on_pre_compress([{"role": "user", "content": "hablamos de PostgreSQL"}])
    assert "preserved" in block
    journals = list((tmp_path / "samskara" / "journal").glob("session-*.ttl"))
    assert journals and "ns/shiva/menciona" in journals[0].read_text()


def test_status_tool_is_json(provider):
    payload = json.loads(provider.handle_tool_call("samskara_status", {}))
    assert payload["provider"] == "samskara"


def test_unknown_tool_raises(provider):
    with pytest.raises(NotImplementedError):
        provider.handle_tool_call("samskara_nope", {})


def test_config_schema_roundtrip(provider, tmp_path):
    assert provider.get_config_schema()
    provider.save_config({"max_entities": 7}, str(tmp_path))
    saved = json.loads((tmp_path / "samskara.json").read_text())
    assert saved["max_entities"] == 7
