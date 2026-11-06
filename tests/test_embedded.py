"""The embedded backend: a graph on disk that needs nothing running.

Two kinds of test here:

  * the parser — because a quietly wrong parser would feed the model plausible
    nonsense, which is the one failure this project exists to prevent;
  * the contract — the embedded client must answer exactly like the HTTP one
    for the three shapes Samskara issues, and refuse everything else.

The seed assertion (`69 triples`) is deliberate: that is the count MillenniumDB
reports for `seed/demo.ttl`. If the two backends ever disagree about the corpus,
the demo would be lying.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.client import SparqlError  # noqa: E402
from samskara_core.embedded import (  # noqa: E402
    EmbeddedClient,
    EmbeddedGraph,
    parse_turtle,
)
from samskara_core.recall import recall  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------
def test_seed_parses_to_the_same_count_millenniumdb_reports():
    client = EmbeddedClient.from_seed("seed/demo.ttl", base=str(REPO))
    assert len(client.graph) == 69


def test_semicolon_and_comma_abbreviations():
    triples = parse_turtle(
        '@prefix ws: <https://x/> .\nws:a ws:p ws:b , ws:c ; ws:q "v" .'
    )
    assert (f"https://x/a", f"https://x/p", f"https://x/b") in triples
    assert ("https://x/a", "https://x/p", "https://x/c") in triples
    assert ("https://x/a", "https://x/q", "v") in triples


def test_a_is_rdf_type():
    triples = parse_turtle('@prefix ws: <https://x/> .\nws:a a ws:Thing .')
    assert triples == [("https://x/a", RDF_TYPE, "https://x/Thing")]


def test_prefix_table_accepts_a_custom_one():
    triples = parse_turtle("ex:subject ex:predicate ex:object .", {"ex": "https://ex/"})
    assert triples == [("https://ex/subject", "https://ex/predicate", "https://ex/object")]


def test_literals_keep_their_value_and_drop_the_tag():
    triples = parse_turtle('@prefix ws: <https://x/> .\nws:a ws:p "hola"@es ; ws:q "15"^^<http://www.w3.org/2001/XMLSchema#integer> .')
    assert ("https://x/a", "https://x/p", "hola") in triples
    assert ("https://x/a", "https://x/q", "15") in triples


def test_comments_and_a_dot_inside_a_string_do_not_split_statements():
    text = (
        "# a comment with a . inside\n"
        '@prefix ws: <https://x/> .\n'
        'ws:a ws:p "v1.2.3" .  # trailing comment\n'
        "ws:b ws:p <https://y/> .\n"
    )
    triples = parse_turtle(text)
    assert ("https://x/a", "https://x/p", "v1.2.3") in triples
    assert ("https://x/b", "https://x/p", "https://y/") in triples
    assert len(triples) == 2


def test_escaped_quote_inside_a_literal():
    triples = parse_turtle('@prefix ws: <https://x/> .\nws:a ws:p "say \\"hi\\"" .')
    assert triples == [("https://x/a", "https://x/p", 'say "hi"')]


# ---------------------------------------------------------------------------
# the three shapes Samskara issues
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    return EmbeddedClient.from_seed("seed/demo.ttl", base=str(REPO))


def test_search_shape_matches_labels_case_insensitively(client):
    rows, _ = client.select(
        'SELECT ?e ?label WHERE { ?e rdfs:label ?label . '
        'FILTER(CONTAINS(LCASE(STR(?label)), "postg")) } LIMIT 40'
    )
    assert any(row["e"] == f"{NS}postgresql" for row in rows)
    assert all("postg" in row["label"].lower() for row in rows)


def test_search_shape_honours_limit(client):
    rows, _ = client.select(
        'SELECT ?e ?label WHERE { ?e rdfs:label ?label . '
        'FILTER(CONTAINS(LCASE(STR(?label)), "a")) } LIMIT 2'
    )
    assert len(rows) <= 2


def test_outgoing_shape_returns_predicate_object_and_label(client):
    rows, _ = client.select(
        f"SELECT ?p ?o ?oLabel WHERE {{ <{NS}onboarding_hub> ?p ?o . "
        "OPTIONAL { ?o rdfs:label ?oLabel } } LIMIT 24"
    )
    labels = {row["p"]: row for row in rows}
    assert f"{NS}usaTecnologia" in labels
    assert any(row["oLabel"] == "PostgreSQL" for row in rows)


def test_incoming_shape_finds_who_points_at_an_entity(client):
    rows, _ = client.select(
        f"SELECT ?s ?p ?sLabel WHERE {{ ?s ?p <{NS}postgresql> . "
        "OPTIONAL { ?s rdfs:label ?sLabel } } LIMIT 24"
    )
    subjects = {row["sLabel"] for row in rows}
    assert "Onboarding Hub" in subjects


def test_health_probe_returns_a_row(client):
    rows, _ = client.select("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert len(rows) == 1 and "s" in rows[0]


def test_client_counts_its_queries(client):
    before = client.queries
    client.select('SELECT ?e ?label WHERE { ?e rdfs:label ?label . FILTER(CONTAINS(LCASE(STR(?label)), "a")) } LIMIT 1')
    assert client.queries == before + 1


# ---------------------------------------------------------------------------
# the honest refusal
# ---------------------------------------------------------------------------
def test_arbitrary_sparql_is_refused_not_guessed(client):
    with pytest.raises(SparqlError) as exc:
        client.select("SELECT ?x WHERE { ?x ex:anything ?y . ?y ex:else 1 }")
    assert "embedded backend" in str(exc.value)
    assert "SAMSKARA_ENDPOINT" in str(exc.value)


def test_missing_seed_is_an_error_not_an_empty_graph():
    with pytest.raises(SparqlError):
        EmbeddedClient.from_seed("seed/does-not-exist.ttl", base=str(REPO))


def test_empty_corpus_is_refused():
    with pytest.raises(SparqlError):
        EmbeddedClient.from_text("")

    graph = EmbeddedGraph()
    assert len(graph) == 0


# ---------------------------------------------------------------------------
# end to end: the same pipeline, no server
# ---------------------------------------------------------------------------
def test_recall_works_with_no_network_at_all(client):
    block, elapsed = recall(client, "¿puedo subir PostgreSQL a la 17 en Onboarding Hub?")
    assert block.startswith("[memoria relacional")
    assert "PostgreSQL" in block
    assert "consulta(s)" in block, "the header must report how much work it did"
    assert elapsed >= 0


def test_recall_returns_nothing_for_an_unknown_topic(client):
    block, _ = recall(client, "¿cuál es la capital de Mongolia?")
    assert block == ""
