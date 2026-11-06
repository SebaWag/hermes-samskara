"""Offline tests for weights: salience, reinforcement, decay and conflicts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.facts import FactStore, is_literal, triple_hash  # noqa: E402

T1 = ("ws:a", "ws:usaTecnologia", "ws:postgresql")
T2 = ("ws:a", "ws:estado", '"vigente"')
T3 = ("ws:b", "ws:nota", '"sin uso"')


@pytest.fixture()
def store(tmp_path):
    return FactStore(tmp_path / "facts.db")


def test_hash_is_stable_and_short():
    assert triple_hash(T1) == triple_hash(T1)
    assert len(triple_hash(T1)) == 16
    assert triple_hash(T1) != triple_hash(T2)


def test_literals_are_detected():
    assert is_literal('"vigente"')
    assert is_literal("texto suelto")
    assert not is_literal("https://example.org/x")
    assert not is_literal("ws:algo")


def test_record_returns_new_count(store):
    assert store.record([T1, T2]) == 2
    assert store.record([T1, T2]) == 0       # already known
    assert store.record([T3]) == 1


def test_reinforcement_raises_weight(store):
    store.record([T1], confidence=0.7)
    before = store.weight(T1)
    for _ in range(4):
        store.reinforce([T1])
    assert store.weight(T1) > before


def test_repetition_across_sessions_raises_weight(store):
    store.record([T1], session_id="s1", confidence=0.6)
    first = store.weight(T1)
    store.record([T1], session_id="s2", confidence=0.6)
    assert store.weight(T1) > first


def test_confidence_dominates(store):
    store.record([T1], confidence=0.9)
    store.record([T3], confidence=0.3)
    assert store.weight(T1) > store.weight(T3)


def test_weight_is_bounded(store):
    store.record([T1], confidence=1.0)
    for _ in range(50):
        store.reinforce([T1])
    assert 0.0 <= store.weight(T1) <= 1.0


def test_idle_facts_decay(store):
    now = time.time()
    fresh = FactStore.score(0.6, 1, 0, now, now, now=now)
    old_created = now - 200 * 86400
    old_used = now - 120 * 86400  # idle for 4 months
    stale = FactStore.score(0.6, 1, 0, old_created, old_used, now=now)
    assert stale < fresh


def test_unknown_triple_weighs_zero(store):
    assert store.weight(("ws:x", "ws:y", "ws:z")) == 0.0


def test_conflicting_literals_are_flagged(store):
    store.record([("ws:a", "ws:estado", '"vigente"')])
    store.record([("ws:a", "ws:estado", '"descongelado"')])
    conflicts = store.conflicts()
    assert len(conflicts) == 2
    assert {f.object for f in conflicts} == {'"vigente"', '"descongelado"'}


def test_non_literal_does_not_conflict(store):
    store.record([("ws:a", "ws:usaTecnologia", "ws:postgresql")])
    store.record([("ws:a", "ws:usaTecnologia", "ws:redis")])
    assert store.conflicts() == []


def test_same_value_twice_is_not_a_conflict(store):
    store.record([("ws:a", "ws:estado", '"vigente"')])
    store.record([("ws:a", "ws:estado", '"vigente"')])
    assert store.conflicts() == []


def test_rank_orders_by_weight(store):
    store.record([T1], confidence=0.9)
    store.record([T3], confidence=0.2)
    ranked = store.rank([T3, T1])
    assert ranked[0][0] == T1 and ranked[0][1] > ranked[1][1]


def test_stats_are_coherent(store):
    store.record([T1, T2])
    store.reinforce([T1])
    stats = store.stats()
    assert stats["facts"] == 2
    assert stats["conflicts"] == 0
    assert stats["ever_used"] == 1
    assert 0 <= stats["avg_weight"] <= 1


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "facts.db"
    first = FactStore(path)
    first.record([T1], confidence=0.8)
    weight = first.weight(T1)
    second = FactStore(path)
    assert second.weight(T1) == pytest.approx(weight)


def test_decay_all_recomputes(store):
    store.record([T1])
    assert store.decay_all() >= 0  # never raises; may change nothing on fresh facts
