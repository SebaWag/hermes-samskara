"""An in-process graph, so Samskara runs with nothing installed.

**Scope, stated up front, because it is the whole point.** This is not a SPARQL
engine. It answers the three query shapes Samskara itself issues — label search,
outgoing neighbourhood, incoming neighbourhood — plus the liveness probe used by
`samskara_status`. Any other query raises `SparqlError` instead of guessing,
because a wrong answer from memory is worse than an honest refusal.

It is read-only, and that is not a limitation of this module: the plugin is
read-only against the graph by design. MillenniumDB's open-source build rejects
`INSERT`, so writes go to the journal and are folded in at reindex time. The
embedded store follows the same contract.

Why it exists: the first minute. `python tools/demo.py` shows relational memory
working on a machine with no graph server, no Docker and no network. When a real
corpus outgrows this, point `SAMSKARA_ENDPOINT` at MillenniumDB or any SPARQL
store — nothing else changes.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .client import SparqlError

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

DEFAULT_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "ws": "https://wsolutionsai.com/ns/shiva/",
}

Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# a Turtle reader for the subset a hand-written corpus uses
# ---------------------------------------------------------------------------
def _strip_comments(text: str) -> str:
    out: List[str] = []
    in_str = in_uri = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_uri:
            out.append(ch)
            if ch == ">":
                in_uri = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "<":
            in_uri = True
        elif ch == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_statements(text: str) -> List[str]:
    """Split on `.` only when it is outside a string and outside a `<uri>`."""
    statements: List[str] = []
    buffer: List[str] = []
    in_str = in_uri = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            buffer.append(ch)
            if ch == "\\" and i + 1 < len(text):
                buffer.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_uri:
            buffer.append(ch)
            if ch == ">":
                in_uri = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "<":
            in_uri = True
        elif ch == ".":
            statements.append("".join(buffer))
            buffer = []
            i += 1
            continue
        buffer.append(ch)
        i += 1
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return [s.strip() for s in statements if s.strip()]


def _tokenize(statement: str) -> List[str]:
    tokens: List[str] = []
    i, n = 0, len(statement)
    while i < n:
        ch = statement[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n:
                if statement[j] == "\\":
                    j += 2
                    continue
                if statement[j] == '"':
                    break
                j += 1
            j = min(j + 1, n)
            # keep a language tag or datatype attached to the literal
            while j < n and not statement[j].isspace() and statement[j] not in ";,":
                j += 1
            tokens.append(statement[i:j])
            i = j
            continue
        if ch == "<":
            j = statement.find(">", i)
            j = n - 1 if j == -1 else j
            tokens.append(statement[i : j + 1])
            i = j + 1
            continue
        if ch in ";,":
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < n and not statement[j].isspace() and statement[j] not in ";,":
            j += 1
        tokens.append(statement[i:j])
        i = j
    return tokens


def _expand(token: str, prefixes: Dict[str, str]) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if token == "a":
        return RDF_TYPE
    if ":" in token:
        prefix, _, local = token.partition(":")
        if prefix in prefixes:
            return prefixes[prefix] + local
    return token


def _literal_value(token: str) -> str:
    """`"15"`, `"hola"@es`, `"15"^^<...int>` -> the value, unescaped."""
    if not token.startswith('"'):
        return token
    body = token[1:]
    out: List[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return "".join(out)


def parse_turtle(text: str, prefixes: Optional[Dict[str, str]] = None) -> List[Triple]:
    """Parse the Turtle subset a hand-written corpus uses.

    Supported: `@prefix`/`PREFIX`, `;` and `,` abbreviations, `a` for rdf:type,
    `<uri>` terms, prefixed names, quoted literals with an optional language tag
    or datatype, and `#` comments. Not supported (and refused by the caller):
    blank nodes, collections, and multi-line literals with `\"\"\"`.
    """
    table: Dict[str, str] = dict(DEFAULT_PREFIXES)
    if prefixes:
        table.update(prefixes)

    triples: List[Triple] = []
    for statement in _split_statements(_strip_comments(text)):
        tokens = _tokenize(statement)
        if not tokens:
            continue
        lowered = tokens[0].lower()
        if lowered in ("@prefix", "prefix"):
            if len(tokens) >= 3:
                name = tokens[1].rstrip(":")
                table[name] = _expand(tokens[2], table)
            continue

        subject = _expand(tokens[0], table)
        predicate: Optional[str] = None
        expect_predicate = True
        for token in tokens[1:]:
            if token == ";":
                expect_predicate = True
                predicate = None
                continue
            if token == ",":
                continue
            if expect_predicate:
                predicate = _expand(token, table)
                expect_predicate = False
                continue
            if predicate is None:
                continue
            obj = _literal_value(token) if token.startswith('"') else _expand(token, table)
            triples.append((subject, predicate, obj))
    return triples


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
class EmbeddedGraph:
    """Triples plus the three indexes Samskara's queries need."""

    def __init__(self) -> None:
        self.triples: set = set()
        self._out: Dict[str, List[Tuple[str, str]]] = {}
        self._in: Dict[str, List[Tuple[str, str]]] = {}
        self._labels: Dict[str, str] = {}

    def add(self, subject: str, predicate: str, obj: str) -> None:
        key = (subject, predicate, obj)
        if key in self.triples:
            return
        self.triples.add(key)
        self._out.setdefault(subject, []).append((predicate, obj))
        self._in.setdefault(obj, []).append((subject, predicate))
        if predicate == RDFS_LABEL and not obj.startswith(("http://", "https://")):
            self._labels.setdefault(subject, obj)

    def extend(self, triples: Iterable[Triple]) -> int:
        before = len(self.triples)
        for subject, predicate, obj in triples:
            self.add(subject, predicate, obj)
        return len(self.triples) - before

    def label(self, uri: str) -> str:
        return self._labels.get(uri, "")

    def outgoing(self, uri: str) -> List[Tuple[str, str]]:
        return self._out.get(uri, [])

    def incoming(self, uri: str) -> List[Tuple[str, str]]:
        return self._in.get(uri, [])

    def search(self, term: str, limit: int = 40) -> List[Tuple[str, str]]:
        needle = (term or "").lower()
        if not needle:
            return []
        hits = [(uri, label) for uri, label in self._labels.items() if needle in label.lower()]
        hits.sort(key=lambda pair: (len(pair[1]), pair[1].lower()))
        return hits[:limit]

    def __len__(self) -> int:
        return len(self.triples)


# ---------------------------------------------------------------------------
# the client — same surface as SparqlClient, so nothing downstream changes
# ---------------------------------------------------------------------------
_SEARCH = re.compile(r'CONTAINS\(\s*LCASE\(\s*STR\(\s*\?label\s*\)\s*\)\s*,\s*"([^"]*)"\s*\)', re.I)
_OUTGOING = re.compile(r"<\s*([^<>\s]+)\s*>\s+\?p\s+\?o\b")
_INCOMING = re.compile(r"\?s\s+\?p\s+<\s*([^<>\s]+)\s*>")
_LIMIT = re.compile(r"LIMIT\s+(\d+)", re.I)


class EmbeddedClient:
    """Read-only, in-process stand-in for `SparqlClient`.

    Same attributes (`endpoint`, `timeout`, `last_ms`, `queries`) and the same
    `select()` contract, so `recall()`, the tools and the tests do not know which
    backend they are talking to.
    """

    scope = (
        "the embedded backend answers the three query shapes Samskara issues "
        "(label search, outgoing and incoming neighbourhood). For arbitrary "
        "SPARQL, set SAMSKARA_ENDPOINT to MillenniumDB or another SPARQL store."
    )

    def __init__(self, graph: EmbeddedGraph, source: str = "embedded://seed/demo.ttl") -> None:
        self.graph = graph
        self.endpoint = source
        self.timeout = 1.0
        self.last_ms = 0.0
        self.queries = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_seed(cls, seed: str = "seed/demo.ttl", base: Optional[str] = None) -> "EmbeddedClient":
        path = Path(seed)
        if not path.is_absolute():
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            path = root / seed
        if not path.is_file():
            raise SparqlError(f"embedded seed not found: {path}")
        graph = EmbeddedGraph()
        graph.extend(parse_turtle(path.read_text(encoding="utf-8")))
        if not len(graph):
            raise SparqlError(f"embedded seed parsed to zero triples: {path}")
        return cls(graph, source=f"embedded://{path.name}")

    @classmethod
    def from_text(cls, text: str, source: str = "embedded://text") -> "EmbeddedClient":
        graph = EmbeddedGraph()
        graph.extend(parse_turtle(text))
        if not len(graph):
            # An empty graph is not "no memory yet": it is a corpus that failed
            # to parse. Refusing here beats returning silence to every question.
            raise SparqlError(f"embedded corpus parsed to zero triples: {source}")
        return cls(graph, source=source)

    # -- client surface ----------------------------------------------------
    def healthy(self) -> bool:
        return len(self.graph) > 0

    def select(self, query: str, use_prefixes: bool = True) -> Tuple[List[Dict[str, str]], float]:
        started = time.perf_counter()
        self.queries += 1
        rows = self._answer(query)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return rows, self.last_ms

    # -- query shapes ------------------------------------------------------
    def _answer(self, query: str) -> List[Dict[str, str]]:
        limit_match = _LIMIT.search(query)
        limit = int(limit_match.group(1)) if limit_match else 40

        search = _SEARCH.search(query)
        if search:
            return [{"e": uri, "label": label} for uri, label in self.graph.search(search.group(1), limit)]

        outgoing = _OUTGOING.search(query)
        if outgoing:
            uri = outgoing.group(1)
            return [
                {"p": predicate, "o": obj, "oLabel": self.graph.label(obj)}
                for predicate, obj in self.graph.outgoing(uri)[:limit]
            ]

        incoming = _INCOMING.search(query)
        if incoming:
            uri = incoming.group(1)
            return [
                {"s": subject, "p": predicate, "sLabel": self.graph.label(subject)}
                for subject, predicate in self.graph.incoming(uri)[:limit]
            ]

        if "?s ?p ?o" in " ".join(query.split()) or "?s ?p ?o" in query:
            if len(self.graph):
                subject, predicate, obj = next(iter(self.graph.triples))
                return [{"s": subject, "p": predicate, "o": obj}]

        raise SparqlError(f"unsupported query for the embedded backend — {self.scope}")
