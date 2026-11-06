"""Append-only write path for a graph that cannot be mutated in place.

MillenniumDB (open-source build) is import-only: it rejects SPARQL INSERT.
That constraint is a feature for a memory provider, because it forces the
happy path to be <1 ms: we append a line, we never block the agent's turn.
The journal is folded back into the graph by `samskara reindex` (see README).

Design rule: writes must never make a turn slower or fail a turn.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Triple = Tuple[str, str, str]

_URI_PREFIXES = ("http://", "https://", "urn:", "ws:", "rdfs:", "rdf:", "xsd:")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def render_triple(triple: Triple) -> str:
    """Serialise one triple to Turtle, quoting the object when it is a literal."""
    subject, predicate, obj = triple
    subject = subject.strip()
    predicate = predicate.strip()
    obj = (obj or "").strip()
    if not subject or not predicate:
        raise ValueError("triple needs a subject and a predicate")
    if obj.startswith('"') and obj.endswith('"') and len(obj) > 1:
        obj_token = obj  # already a literal: pass through untouched
    elif obj.lower().startswith(_URI_PREFIXES) or (":" in obj and _SAFE_NAME.match(obj)):
        obj_token = obj
    else:
        obj_token = '"' + _escape_literal(obj) + '"'
    return f"{subject} {predicate} {obj_token} ."


class Journal:
    """A crash-safe, append-only Turtle journal, one file per session."""

    def __init__(self, directory: Path, session_id: str = "") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")[:64]
        self.path = self.directory / f"session-{safe_session}.ttl"
        self._lock = threading.Lock()

    def append(self, triples: Iterable[Triple]) -> int:
        """Append triples atomically. Returns how many were written."""
        lines = [render_triple(t) for t in triples]
        if not lines:
            return 0
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return len(lines)

    def pending(self) -> int:
        """How many triples are waiting to be folded into the graph."""
        total = 0
        for path in self.files():
            try:
                with open(path, encoding="utf-8") as handle:
                    total += sum(1 for line in handle if line.strip())
            except OSError:
                continue
        return total

    def files(self) -> List[Path]:
        return sorted(self.directory.glob("session-*.ttl"))

    def snapshot(self) -> List[dict]:
        return [{"journal": str(p), "bytes": p.stat().st_size} for p in self.files()]


def reindex_commands(db_dir: str, journal_dir: str, merged_file: str = "/tmp/samskara-merged.ttl") -> List[str]:
    """The exact shell steps that fold the journal into the graph.

    Import-only databases are rebuilt, not patched: dump, concatenate, import.
    For a personal knowledge graph this is milliseconds of work, and it keeps
    the read path untouched — no locks, no downtime for the agent.
    """
    return [
        f"# 1. dump the current graph to Turtle",
        f"docker run --rm -v {db_dir}:/db imfd/millenniumdb:latest dump /db/db /db/dump.ttl",
        f"# 2. fold in every journal line",
        f"cat {db_dir}/dump.ttl {journal_dir}/session-*.ttl > {merged_file}",
        f"# 3. rebuild the database from the union",
        f"docker run --rm -v {db_dir}:/db -v {os.path.dirname(merged_file)}:/in "
        f"imfd/millenniumdb:latest import /in/{os.path.basename(merged_file)} /db/db --format ttl",
        f"# 4. archive the folded journals so they are not applied twice",
        f"mkdir -p {journal_dir}/applied && mv {journal_dir}/session-*.ttl {journal_dir}/applied/",
    ]


def stats(directory: Path) -> dict:
    journal = Journal(directory)
    return {"files": len(journal.files()), "pending_triples": journal.pending()}
