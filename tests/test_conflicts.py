"""Offline tests for layer 3: a memory that admits doubt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.facts import FactStore  # noqa: E402
from samskara_core.recall import Fact, recall, render_conflicts  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"
PRED = f"{NS}estado"
SUBJ = f"{NS}dec_congelar_pg"


class FakeClient:
    """Returns one entity and one attribute fact; enough to exercise the path."""

    def __init__(self):
        self.last_ms = 0.5

    def select(self, query, use_prefixes=True):
        if "rdfs:label ?label" in query:
            return [{"e": SUBJ, "label": "Congelar PostgreSQL en 15"}], 0.5
        return [{"p": PRED, "o": '"vigente"', "oLabel": ""}], 0.5

    def healthy(self):
        return True


@pytest.fixture()
def store(tmp_path):
    return FactStore(tmp_path / "facts.db")


def test_conflicts_for_returns_both_sides(store):
    store.record([(SUBJ, PRED, '"vigente"')], session_id="mar")
    store.record([(SUBJ, PRED, '"descongelado"')], session_id="hoy")
    sides = store.conflicts_for(SUBJ, PRED)
    assert len(sides) == 2
    assert {f.object for f in sides} == {'"vigente"', '"descongelado"'}
    assert all(f.status == "conflict" for f in sides)


def test_conflicts_for_empty_when_consistent(store):
    store.record([(SUBJ, PRED, '"vigente"')])
    assert store.conflicts_for(SUBJ, PRED) == []


def test_render_conflicts_is_explicit():
    class Row:
        subject = SUBJ
        predicate = PRED
        object = '"vigente"'
        weight = 0.9
        sessions = 3

    class Row2(Row):
        object = '"descongelado"'
        weight = 0.4
        sessions = 1

    text = render_conflicts([Row(), Row2()])
    assert "conflicto" in text
    assert "vigente" in text and "descongelado" in text
    assert "peso 0.90" in text and "peso 0.40" in text
    assert "no elijo por ti" in text


def test_render_conflicts_empty_is_empty_string():
    assert render_conflicts([]) == ""


def test_recall_appends_conflict_notice(store):
    store.record([(SUBJ, PRED, '"vigente"')], session_id="mar")
    store.record([(SUBJ, PRED, '"descongelado"')], session_id="hoy")
    block, _ = recall(FakeClient(), "¿qué decisión sobre congelar?", store=store)
    assert "conflicto" in block
    assert "vigente" in block and "descongelado" in block


def test_recall_without_conflict_has_no_notice(store):
    store.record([(SUBJ, PRED, '"vigente"')])
    block, _ = recall(FakeClient(), "¿qué decisión sobre congelar?", store=store)
    assert "conflicto" not in block


def test_recall_reinforces_what_it_shows(store):
    store.record([(SUBJ, PRED, '"vigente"')], confidence=0.6)
    before = store.weight((SUBJ, PRED, '"vigente"'))
    recall(FakeClient(), "¿qué decisión sobre congelar?", store=store)
    assert store.weight((SUBJ, PRED, '"vigente"')) > before


def test_explicit_write_outweighs_extraction(store):
    store.record([(SUBJ, PRED, '"vigente"')], session_id="a", confidence=0.65)  # extraction
    store.record([(f"{NS}other", PRED, '"x"')], session_id="b", confidence=0.9)  # user said it
    assert store.weight((f"{NS}other", PRED, '"x"')) > store.weight((SUBJ, PRED, '"vigente"'))
