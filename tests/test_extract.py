"""Offline tests for extraction, schema learning and session reading."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.extract import Extractor, build_prompt, validate  # noqa: E402
from samskara_core.schema import SchemaRegistry, normalise, slug  # noqa: E402
from samskara_core.sessions import (  # noqa: E402
    SessionContext,
    SessionReader,
    already_done,
    mark_done,
)

NS = "https://wsolutionsai.com/ns/shiva/"

PAYLOAD = {
    "entities": [
        {"label": "Onboarding Hub", "type": "Proyecto"},
        {"label": "PostgreSQL", "type": "Tecnologia"},
    ],
    "relations": [
        {"subject": "Onboarding Hub", "predicate": "usa", "object": "PostgreSQL"},
        {"subject": "Onboarding Hub", "predicate": "estado", "object": "produccion",
         "object_is_literal": True},
        {"subject": "Onboarding Hub", "predicate": "inventado_raro", "object": "PostgreSQL"},
    ],
    "new_predicates": ["inventado_raro"],
}


@pytest.fixture()
def registry(tmp_path):
    return SchemaRegistry(tmp_path / "schema.json")


# --------------------------------------------------------------------------
# schema: the vocabulary learns, but slowly
# --------------------------------------------------------------------------
def test_slug_is_stable_and_safe():
    assert slug("Onboarding Hub") == "onboarding_hub"
    assert slug("  ¿Qué?  ") == "que"     # accents decomposed and dropped
    assert slug("versión 15") == "version_15"
    assert slug("Ñandú") == "nandu"
    assert slug("") == "unknown"
    assert slug("x" * 200) and len(slug("x" * 200)) == 64


def test_normalise_ignores_case_and_punctuation():
    assert normalise("usaTecnologia") == normalise("usa tecnologia") == normalise("USA-TECNOLOGIA")


def test_alias_resolves_to_canonical(registry):
    assert registry.canonical("usa") == "usaTecnologia"
    assert registry.canonical("trabaja con") == "usaTecnologia"
    assert registry.canonical("nonsense") is None


def test_new_predicate_needs_to_recur(registry):
    assert registry.propose("migraA") is False       # first sighting: only counted
    assert registry.canonical("migraA") is None
    assert registry.propose("migraA") is True        # second: becomes canonical
    assert registry.canonical("migraA") == "migraa"


def test_registry_roundtrip(tmp_path):
    path = tmp_path / "schema.json"
    reg = SchemaRegistry(path)
    reg.propose("edge", threshold=1)
    reg.save()
    reloaded = SchemaRegistry(path)
    assert reloaded.canonical("edge") == "edge"


def test_unknown_type_falls_back(registry):
    assert registry.type_of("Proyecto") == "Proyecto"
    assert registry.type_of("algo inventado") == "Concepto"


# --------------------------------------------------------------------------
# validate: what we are willing to store
# --------------------------------------------------------------------------
def test_validate_mints_entities_with_label_and_type(registry):
    result = validate(PAYLOAD, registry, NS, "sess-1")
    uris = {t[0] for t in result.triples if t[1] == "rdfs:label"}
    assert f"{NS}onboarding_hub" in uris
    assert f"{NS}postgresql" in uris
    assert any(t[1] == "rdf:type" for t in result.triples)


def test_validate_maps_alias_to_canonical_predicate(registry):
    result = validate(PAYLOAD, registry, NS, "sess-1")
    predicates = {t[1] for t in result.triples}
    assert "ws:usaTecnologia" in predicates


def test_validate_rejects_unknown_predicate(registry):
    result = validate(PAYLOAD, registry, NS, "sess-1")
    assert not any("inventado_raro" in t[1] for t in result.triples)
    assert any("unknown predicate" in r for r in result.rejected)


def test_validate_keeps_literals_as_literals(registry):
    result = validate(PAYLOAD, registry, NS, "sess-1")
    estados = [t for t in result.triples if t[1] == "ws:estado"]
    assert estados and estados[0][2] == "produccion"  # not a URI


def test_validate_subjects_are_minted_uris(registry):
    """Regression: a relation's subject must be a URI, never a raw label."""
    result = validate(PAYLOAD, registry, NS, "s")
    object_subjects = [t[0] for t in result.triples if t[1].startswith("ws:")]
    assert object_subjects
    for uri in object_subjects:
        assert uri.startswith(NS), uri
        assert " " not in uri, uri


def test_validate_adds_provenance(registry):
    result = validate(PAYLOAD, registry, NS, "sess-1")
    assert any(t[1] == "ws:derivadoDe" and "sess-1" in str(t[2]) for t in result.triples)


def test_validate_dedupes(registry):
    twice = dict(PAYLOAD)
    twice["relations"] = PAYLOAD["relations"] + PAYLOAD["relations"]
    result = validate(twice, registry, NS, "s")
    keys = [(t[0], t[1], t[2]) for t in result.triples]
    assert len(keys) == len(set(keys))


def test_validate_ignores_self_reference(registry):
    payload = {
        "entities": [{"label": "X", "type": "Proyecto"}],
        "relations": [{"subject": "X", "predicate": "usa", "object": "X"}],
    }
    result = validate(payload, registry, NS, "s")
    assert not any(t[1] == "ws:usaTecnologia" for t in result.triples)


def test_validate_tolerates_garbage(registry):
    assert validate({}, registry, NS, "s").count >= 0
    assert validate({"entities": "nope", "relations": None}, registry, NS, "s").count >= 0


# --------------------------------------------------------------------------
# extractor: injected model, soft failure
# --------------------------------------------------------------------------
def test_extractor_uses_injected_llm(registry):
    seen = {}

    def fake_llm(instructions, prompt, schema):
        seen["instructions"] = instructions
        seen["has_schema"] = bool(schema.get("properties"))
        return PAYLOAD

    result = Extractor(registry, fake_llm, NS).extract(
        SessionContext(session_id="s1", messages=[("user", "uso PostgreSQL en Onboarding Hub")])
    )
    assert result.count > 0
    assert seen["has_schema"] and "knowledge graph" in seen["instructions"].lower()


def test_extractor_survives_model_failure(registry):
    def boom(*_):
        raise RuntimeError("provider down")

    result = Extractor(registry, boom, NS).extract(SessionContext("s1", messages=[("user", "x")]))
    assert result.count == 0 and result.rejected


def test_build_prompt_includes_vocabulary_and_metadata(registry):
    prompt = build_prompt(
        SessionContext(session_id="s9", title="Deploy", cwd="/srv", messages=[("user", "hola")]),
        registry,
    )
    assert "Known predicates" in prompt and "usaTecnologia" in prompt
    assert "s9" in prompt and "/srv" in prompt


# --------------------------------------------------------------------------
# session reader: reads a real sqlite file, never writes it
# --------------------------------------------------------------------------
def _make_db(path: Path, sessions: int = 2) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT, title TEXT, started_at REAL, ended_at REAL, cwd TEXT, model TEXT, archived INTEGER)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, api_content TEXT, timestamp REAL, active INTEGER)")
    for i in range(sessions):
        sid = f"s{i}"
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,0)", (sid, f"title {i}", 1000 + i, 1001 + i, "/w", "m"))
        conn.execute("INSERT INTO messages VALUES (NULL,?,?,?,NULL,?,1)", (sid, "user", f"hola {i}", 1.0))
        conn.execute("INSERT INTO messages VALUES (NULL,?,?,?,NULL,?,1)", (sid, "assistant", f"respuesta {i}", 2.0))
        conn.execute("INSERT INTO messages VALUES (NULL,?,?,?,NULL,?,1)", (sid, "tool", "ignored", 3.0))
    conn.commit()
    conn.close()
    return path


def test_session_reader_streams_and_filters(tmp_path):
    db = _make_db(tmp_path / "state.db")
    reader = SessionReader(db)
    sessions = list(reader.sessions())
    assert len(sessions) == 2
    assert sessions[0].session_id == "s0"  # oldest first
    assert all(role in {"user", "assistant"} for role, _ in sessions[0].messages)  # tool rows dropped


def test_session_reader_missing_db_is_empty(tmp_path):
    assert list(SessionReader(tmp_path / "nope.db").sessions()) == []


def test_state_tracks_processed_sessions(tmp_path):
    state = {"sessions": {}, "seen": []}
    assert not already_done(state, "s1")
    mark_done(state, "s1", 5)
    assert already_done(state, "s1")
    SessionReader.save_state(tmp_path / "st.json", state)
    assert already_done(SessionReader.load_state(tmp_path / "st.json"), "s1")
