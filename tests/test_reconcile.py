"""Offline tests for reconciliation: the layer that keeps one graph, not five."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.extract import Extractor, apply_bridge, validate  # noqa: E402
from samskara_core.reconcile import LabelIndex, normalize_label  # noqa: E402
from samskara_core.schema import SchemaRegistry  # noqa: E402
from samskara_core.sessions import SessionContext  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"


@pytest.fixture()
def registry(tmp_path):
    return SchemaRegistry(tmp_path / "schema.json")


@pytest.fixture()
def index():
    return LabelIndex().load_pairs([
        (f"{NS}dec_congelar_pg", "Congelar PostgreSQL en 15 hasta migrar las réplicas lógicas"),
        (f"{NS}postgresql", "PostgreSQL"),
        (f"{NS}onboarding_hub", "Onboarding Hub"),
        (f"{NS}inc_2026_03", "Incidente de upgrade a PostgreSQL 16 (inc_2026_03)"),
    ])


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def test_normalize_strips_accents_case_and_punctuation():
    assert normalize_label("Congelar  PostgreSQL en la VERSIÓN 15") == \
           "congelar postgresql en la version 15"


def test_normalize_drops_trailing_parenthetical_ids():
    assert normalize_label("Incidente de upgrade (inc_2026_03)") == "incidente de upgrade"


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------
def test_exact_match_reuses_uri(index):
    hit = index.resolve("ONBOARDING HUB")
    assert hit and hit.kind == "exact"
    assert hit.uri == f"{NS}onboarding_hub"


def test_parenthetical_and_accents_still_match(index):
    hit = index.resolve("incidente de upgrade a Postgresql 16")
    assert hit is not None and hit.uri == f"{NS}inc_2026_03"


def test_near_identical_label_resolves_fuzzy(index):
    hit = index.resolve("Postgresql")  # hyphens/case noise
    assert hit and hit.uri == f"{NS}postgresql"


def test_version_is_not_its_parent(index):
    """The dangerous false positive: 'PostgreSQL 16' must not become 'PostgreSQL'."""
    assert index.resolve("PostgreSQL 16") is None


def test_unknown_label_is_unresolved(index):
    assert index.resolve("Completely different thing") is None
    assert index.unresolved(["Onboarding Hub", "Completely different thing"]) == [
        "Completely different thing"
    ]


def test_sample_is_bounded(index):
    assert len(index.sample(2)) == 2


# --------------------------------------------------------------------------
# validate with an index: no minting when the meaning already exists
# --------------------------------------------------------------------------
def test_validate_reuses_existing_uri_without_minting(registry, index):
    payload = {
        "entities": [{"label": "Onboarding Hub", "type": "Proyecto"}],
        "relations": [{"subject": "Onboarding Hub", "predicate": "usa", "object": "PostgreSQL"}],
    }
    result = validate(payload, registry, NS, "s", index=index)
    labels = [t for t in result.triples if t[1] == "rdfs:label"]
    assert labels == []  # existing entity keeps its label, we do not duplicate
    types = [t for t in result.triples if t[1] == "rdf:type"]
    assert types == []
    rel = [t for t in result.triples if t[1] == "ws:usaTecnologia"]
    assert rel and rel[0][0] == f"{NS}onboarding_hub" and rel[0][2] == f"{NS}postgresql"
    assert result.unresolved == []


def test_validate_mints_only_what_is_new(registry, index):
    payload = {
        "entities": [
            {"label": "Onboarding Hub", "type": "Proyecto"},
            {"label": "Redis", "type": "Tecnologia"},
        ],
        "relations": [],
    }
    result = validate(payload, registry, NS, "s", index=index)
    assert result.unresolved == ["Redis"]
    assert any(t[0] == f"{NS}redis" for t in result.triples)


# --------------------------------------------------------------------------
# semantic bridge
# --------------------------------------------------------------------------
def test_bridge_rewrites_triples_to_existing_uri(registry, index):
    payload = {
        "entities": [
            {"label": "Congelar PostgreSQL en la versión 15", "type": "Decision"},
            {"label": "PostgreSQL", "type": "Tecnologia"},
        ],
        "relations": [
            {"subject": "Congelar PostgreSQL en la versión 15", "predicate": "estado",
             "object": "vigente", "object_is_literal": True},
        ],
    }
    result = validate(payload, registry, NS, "s", index=index)
    assert result.unresolved  # the decision is a new spelling

    mapping = {"Congelar PostgreSQL en la versión 15": "Congelar PostgreSQL en 15 hasta migrar las réplicas lógicas"}
    result = apply_bridge(result, mapping, index, NS)

    subjects = {t[0] for t in result.triples}
    assert f"{NS}dec_congelar_pg" in subjects
    assert f"{NS}congelar_postgresql_en_la_version_15" not in subjects
    assert result.bridged and result.bridged[0][0] == "Congelar PostgreSQL en la versión 15"
    # the alternative spelling is kept as an alias, not as a second node
    assert any(t[1] == "skos:altLabel" for t in result.triples)


def test_extractor_uses_bridge_when_index_present(registry, index):
    calls = []

    def fake_llm(instructions, prompt, schema):
        calls.append(instructions[:40])
        if "reconciling" in instructions.lower():
            return {"mapping": {"Postgres15": "PostgreSQL"}}
        return {
            "entities": [{"label": "Postgres15", "type": "Tecnologia"}],
            "relations": [],
        }

    result = Extractor(registry, fake_llm, NS, index=index).extract(
        SessionContext(session_id="s1", messages=[("user", "usamos Postgres15")])
    )
    assert len(calls) == 2  # extraction + bridge
    assert any(t[0] == f"{NS}postgresql" for t in result.triples)


def test_bridge_failure_is_not_fatal(registry, index):
    def flaky(instructions, prompt, schema):
        if "reconciling" in instructions.lower():
            raise RuntimeError("bridge down")
        return {"entities": [{"label": "Brand New Thing", "type": "Concepto"}], "relations": []}

    result = Extractor(registry, flaky, NS, index=index).extract(
        SessionContext("s", messages=[("user", "x")])
    )
    assert result.count > 0


def test_bridge_disabled_mints_normally(registry, index):
    def fake_llm(instructions, prompt, schema):
        return {"entities": [{"label": "Postgres15", "type": "Tecnologia"}], "relations": []}

    result = Extractor(registry, fake_llm, NS, index=index, bridge=False).extract(
        SessionContext("s", messages=[("user", "x")])
    )
    assert any(t[0] == f"{NS}postgres15" for t in result.triples)
