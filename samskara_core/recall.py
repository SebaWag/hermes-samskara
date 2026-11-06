"""Relational recall: find entities, walk the graph, render with provenance.

The point of this module is the thing a vector store cannot do: follow edges.
"Which projects use PostgreSQL *and* are covered by a decision that is still
in force, and what incident motivated it?" is not a similarity search — it is
a path. Here the path is a SPARQL pattern, answered in microseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .client import SparqlClient, SparqlError

STOPWORDS = {
    "the", "and", "for", "with", "que", "los", "las", "del", "una", "por",
    "con", "para", "como", "pero", "sus", "mas", "más", "este", "esta",
    "esto", "sobre", "entre", "cuando", "donde", "puedo", "podemos", "hay",
    "tengo", "tiene", "hacer", "hago", "cual", "cuál", "qué", "que", "es",
    "son", "fue", "era", "ser", "está", "esta", "estan", "están", "mi", "mis",
    "tu", "tus", "su", "sus", "de", "en", "a", "el", "la", "lo", "un", "y",
    "o", "al", "se", "si", "sí", "no", "me", "te", "nos", "le", "les",
}

# Predicates that add noise rather than knowledge in a recall block.
SKIP_PREDICATES = {
    "rdf:type",
    "rdfs:label",
}


@dataclass
class Entity:
    uri: str
    label: str
    score: float = 0.0


@dataclass
class Fact:
    direction: str          # "out" (entity -> target) or "in" (source -> entity)
    predicate: str          # local name, e.g. "usaTecnologia"
    target: str             # human label of the other end
    other_uri: str = ""


def keywords(text: str, max_keywords: int = 6) -> List[str]:
    """Extract searchable tokens. Accent-insensitive enough for our ontology."""
    raw = re.findall(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_.+-]{3,}", text or "")
    out: List[str] = []
    for token in raw:
        low = token.lower().strip("._-")
        if len(low) < 3 or low in STOPWORDS:
            continue
        if low not in out:
            out.append(low)
    # Longer tokens carry more signal; keep the most specific ones.
    out.sort(key=len, reverse=True)
    return out[:max_keywords]


def local_name(uri: str) -> str:
    """`http://...#usaTecnologia` -> `usaTecnologia`."""
    for sep in ("#", "/"):
        if sep in uri:
            uri = uri.rsplit(sep, 1)[-1]
    return uri or "?"


def humanize(name: str) -> str:
    """`usaTecnologia` -> `usa tecnologia` (readable in a prompt, cheap)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return spaced.replace("_", " ").lower().strip()


def _escape(term: str) -> str:
    return term.replace("\\", "").replace('"', "")


def find_entities(
    client: SparqlClient,
    text: str,
    limit: int = 5,
    max_keywords: int = 6,
) -> List[Entity]:
    """Resolve mentions in free text to graph entities, best match first.

    One query per keyword (`FILTER(CONTAINS(LCASE(?label), kw))`), then rank by
    how many keywords each entity matched — so "PostgreSQL 17" lands on
    PostgreSQL and not on a project that merely mentions it.
    """
    terms = keywords(text, max_keywords=max_keywords)
    if not terms:
        return []
    seen: Dict[str, Entity] = {}
    for term in terms:
        query = (
            "SELECT ?e ?label WHERE {\n"
            "  ?e rdfs:label ?label .\n"
            f'  FILTER(CONTAINS(LCASE(STR(?label)), "{_escape(term)}"))\n'
            "} LIMIT 40"
        )
        try:
            rows, _ = client.select(query)
        except SparqlError:
            continue
        for row in rows:
            uri = row.get("e", "")
            label = row.get("label", "")
            if not uri or not label:
                continue
            entity = seen.get(uri)
            if entity is None:
                entity = Entity(uri=uri, label=label)
                seen[uri] = entity
            entity.score += 1.0
            if label.lower() == term:
                entity.score += 2.0
            elif label.lower().startswith(term):
                entity.score += 0.5
    ranked = sorted(seen.values(), key=lambda e: (-e.score, len(e.label)))
    return ranked[:limit]


def facts(client: SparqlClient, uri: str, max_facts: int = 24) -> List[Fact]:
    """Both directions of one entity: what it points at, and what points at it."""
    out: List[Fact] = []
    outgoing = (
        "SELECT ?p ?o ?oLabel WHERE {\n"
        f"  <{uri}> ?p ?o .\n"
        "  OPTIONAL { ?o rdfs:label ?oLabel }\n"
        f"}} LIMIT {int(max_facts)}"
    )
    incoming = (
        "SELECT ?s ?p ?sLabel WHERE {\n"
        f"  ?s ?p <{uri}> .\n"
        "  OPTIONAL { ?s rdfs:label ?sLabel }\n"
        f"}} LIMIT {int(max_facts)}"
    )
    for direction, query in (("out", outgoing), ("in", incoming)):
        try:
            rows, _ = client.select(query)
        except SparqlError:
            continue
        for row in rows:
            pred_uri = row.get("p", "")
            pred = local_name(pred_uri)
            if pred_uri in SKIP_PREDICATES or f"rdfs:{pred}" in SKIP_PREDICATES:
                continue
            if pred == "type":
                continue
            if direction == "out":
                other_uri = row.get("o", "")
                label = row.get("oLabel") or local_name(other_uri)
            else:
                other_uri = row.get("s", "")
                label = row.get("sLabel") or local_name(other_uri)
            out.append(Fact(direction=direction, predicate=pred, target=label, other_uri=other_uri))
    return out[:max_facts]


def render(
    entities: Sequence[Entity],
    facts_by_uri: Dict[str, List[Fact]],
    elapsed_ms: float,
    max_chars: int = 1200,
) -> str:
    """Render a recall block: compact, human-readable, with a provenance header."""
    if not entities:
        return ""
    fact_count = sum(len(v) for v in facts_by_uri.values())
    header = (
        f"[memoria relacional · samskara · {len(entities)} entidad(es) · "
        f"{fact_count} hecho(s) · {elapsed_ms:.2f} ms]"
    )
    lines: List[str] = [header]
    for entity in entities:
        lines.append(f"• {entity.label}  ({local_name(entity.uri)})")
        for fact in facts_by_uri.get(entity.uri, []):
            arrow = "→" if fact.direction == "out" else "←"
            lines.append(f"   {arrow} {humanize(fact.predicate)}: {fact.target}")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 20].rstrip() + "\n  … (truncado)"
    return block


def recall(
    client: SparqlClient,
    text: str,
    max_entities: int = 3,
    max_facts: int = 24,
    max_chars: int = 1200,
) -> Tuple[str, float]:
    """Full recall pipeline: text -> entities -> neighbourhood -> rendered block."""
    entities = find_entities(client, text, limit=max_entities)
    if not entities:
        return "", client.last_ms
    facts_by_uri = {e.uri: facts(client, e.uri, max_facts=max_facts) for e in entities}
    return render(entities, facts_by_uri, client.last_ms, max_chars=max_chars), client.last_ms
