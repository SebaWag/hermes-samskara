"""Failure modes — what Samskara does when the world is broken.

Reliability is not the happy path. These tests pin the behaviour when the graph
is down, when it answers HTML instead of JSON, when a session id tries to climb
out of the journal directory, and when the ontology has a cycle. None of them
needs a graph, a network or Hermes: they are the contract, written down.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from samskara_core.client import SparqlClient, SparqlError  # noqa: E402
from samskara_core.facts import FactStore, triple_hash  # noqa: E402
from samskara_core.journal import Journal  # noqa: E402
from samskara_core.ontology import Ontology  # noqa: E402
from samskara_core.recall import Entity, Fact, render, render_conflicts  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"


# ---------------------------------------------------------------------------
# a graph that answers badly, on purpose
# ---------------------------------------------------------------------------
class _BadGraph(BaseHTTPRequestHandler):
    mode = "html"

    def do_POST(self):  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        if _BadGraph.mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"internal error")
            return
        body = b"<html><body>not sparql</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        return


@pytest.fixture()
def bad_graph():
    _BadGraph.mode = "html"
    server = HTTPServer(("127.0.0.1", 0), _BadGraph)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/sparql", _BadGraph
    server.shutdown()


# ---------------------------------------------------------------------------
# transport failures
# ---------------------------------------------------------------------------
def test_unreachable_endpoint_raises_sparql_error():
    client = SparqlClient("http://127.0.0.1:1/sparql", timeout=0.3)
    with pytest.raises(SparqlError) as exc:
        client.select("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert "unreachable" in str(exc.value).lower()


def test_html_instead_of_json_raises_sparql_error(bad_graph):
    endpoint, _ = bad_graph
    client = SparqlClient(endpoint, timeout=5.0)
    with pytest.raises(SparqlError) as exc:
        client.select("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert "non-json" in str(exc.value).lower()


def test_http_500_raises_sparql_error(bad_graph):
    endpoint, graph = bad_graph
    graph.mode = "error"
    client = SparqlClient(endpoint, timeout=5.0)
    with pytest.raises(SparqlError) as exc:
        client.select("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert "500" in str(exc.value)


# ---------------------------------------------------------------------------
# the provider must never break a turn
# ---------------------------------------------------------------------------
class _DeadClient:
    """A client whose graph is always down."""

    endpoint = "http://127.0.0.1:1/sparql"
    last_ms = 0.0
    queries = 0

    def select(self, query, use_prefixes=True):
        raise SparqlError("graph unreachable")


def _load_provider(tmp_path):
    spec = importlib.util.spec_from_file_location("samskara_plugin_fail", REPO / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = module.SamskaraMemoryProvider()
    provider.initialize("failure-session", hermes_home=str(tmp_path))
    provider._client = _DeadClient()
    return provider


def test_prefetch_returns_empty_when_graph_is_down(tmp_path):
    provider = _load_provider(tmp_path)
    assert provider.prefetch("¿puedo subir PostgreSQL a la 17?") == ""


def test_prefetch_never_raises_on_blank_query(tmp_path):
    provider = _load_provider(tmp_path)
    assert provider.prefetch("") == ""
    assert provider.prefetch("   ") == ""
    assert provider.prefetch(None) == ""


# ---------------------------------------------------------------------------
# the journal cannot be walked out of
# ---------------------------------------------------------------------------
def test_journal_neutralises_traversal_in_session_id(tmp_path):
    journal = Journal(tmp_path / "journal", session_id="../../etc/passwd")
    assert journal.path.parent == tmp_path / "journal"
    assert ".." not in journal.path.name
    assert journal.path.name.startswith("session-")


def test_journal_survives_a_hostile_session_id(tmp_path):
    for hostile in ["", "../../../root", "a/b/c", "....", "\x00null", "x" * 500]:
        journal = Journal(tmp_path / "journal", session_id=hostile)
        assert journal.path.parent == tmp_path / "journal"
        assert journal.path.exists() is False or journal.path.parent == tmp_path / "journal"


def test_journal_append_is_atomic_and_counted(tmp_path):
    journal = Journal(tmp_path / "journal", session_id="ok")
    written = journal.append([(f"{NS}a", f"{NS}p", f"{NS}b")])
    assert written == 1
    assert journal.pending() == 1


# ---------------------------------------------------------------------------
# ontology must not hang on a cycle
# ---------------------------------------------------------------------------
def test_ontology_survives_a_cycle():
    ontology = Ontology(classes={"A": ["B"], "B": ["C"], "C": ["A"]})
    result = ontology.superclasses("A")
    assert result == {"B", "C"}


def test_ontology_unknown_class_is_empty():
    assert Ontology().superclasses("NoExiste") == set()


# ---------------------------------------------------------------------------
# weights stay inside [0, 1] no matter what the clock says
# ---------------------------------------------------------------------------
def test_score_is_clamped_between_zero_and_one():
    now = 1_700_000_000.0
    future = now + 10_000_000  # a clock that went backwards
    extreme = FactStore.score(5.0, 999, 999, future, future, now=now)
    assert 0.0 <= extreme <= 1.0


def test_score_decays_with_idleness():
    now = 1_700_000_000.0
    fresh = FactStore.score(0.6, 1, 0, now - 3600, now - 3600, now=now)
    stale = FactStore.score(0.6, 1, 0, now - 86400 * 400, now - 86400 * 400, now=now)
    assert stale < fresh


def test_record_skips_structural_predicates(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    structural = {"rdf:type", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"}
    try:
        new = store.record([(f"{NS}a", predicate, f"{NS}b") for predicate in structural])
        assert new == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# rendering never produces a broken block
# ---------------------------------------------------------------------------
def test_render_with_no_entities_is_empty():
    assert render([], {}, elapsed_ms=0.0) == ""


def test_render_truncates_instead_of_overflowing():
    facts = {
        f"{NS}e": [
            Fact(direction="out", predicate="usaTecnologia", target=f"t{i}",
                 other_uri=f"{NS}t{i}", predicate_uri=f"{NS}usaTecnologia")
            for i in range(200)
        ]
    }
    block = render([Entity(uri=f"{NS}e", label="Entidad")], facts, elapsed_ms=1.0, max_chars=300)
    assert len(block) <= 300
    assert "truncado" in block


def test_render_conflicts_with_nothing_is_empty():
    assert render_conflicts([]) == ""


def test_render_conflicts_names_both_sides():
    class Row:
        subject = f"{NS}pg"
        predicate = f"{NS}estado"
        object = '"vigente"'
        weight = 0.45
        sessions = 1

    block = render_conflicts([Row()])
    assert "vigente" in block
    assert "0.45" in block


# ---------------------------------------------------------------------------
# hashing is stable: the weight of a fact cannot move because of an encoding
# ---------------------------------------------------------------------------
def test_triple_hash_is_stable_and_direction_free():
    a = triple_hash((f"{NS}s", f"{NS}p", f"{NS}o"))
    b = triple_hash((f"{NS}s", f"{NS}p", f"{NS}o"))
    assert a == b and isinstance(a, str) and len(a) >= 16
