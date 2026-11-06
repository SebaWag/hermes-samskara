"""Weight: not every memory deserves the same trust.

A flat set of facts is a bag, not a memory. This module gives each triple a
*weight* in [0, 1] computed from four things that actually matter:

    peso = 0.5·confianza + repetición + 0.2·recencia + refuerzo por uso

- **confianza**: how sure the source was (explicit user statement > extraction).
- **repetición**: how many distinct sessions asserted it.
- **recencia**: exponential decay with age (half-life ≈ 60 days).
- **refuerzo**: every recall that used the fact and was answered from it.

And the part that makes it honest: a fact used in no recall for 30 days starts to
**decay**. Memory that never forgets is just a log file.

Conflicts are first-class. If a new literal contradicts a heavier existing one we
do not overwrite (that loses history) and we do not ignore it (that lies): both
are flagged `conflict`, and the recall says so out loud.

Storage is SQLite next to the journal, keyed by the triple's hash. RDF has no
place for metadata without reification, and reification would slow every read
query down to prove a point.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Triple = Tuple[str, str, str]

# Schema, not memory: types and labels describe structure and would drown the
# signal in any ranking. They are never weighted or reported.
STRUCTURAL_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "rdf:type",
    "a",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "rdfs:label",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "skos:altLabel",
    "https://wsolutionsai.com/ns/shiva/derivadoDe",
    "ws:derivadoDe",
}

HALF_LIFE_DAYS = 60.0
IDLE_DAYS_BEFORE_DECAY = 30.0
WEEKLY_DECAY = 0.95
MAX_REINFORCEMENT = 0.20
REINFORCE_STEP = 0.05


def triple_hash(triple: Triple) -> str:
    raw = "|".join(str(part) for part in triple)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


_PREFIXED_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.\-]+$")


def is_literal(value: str) -> bool:
    """A literal is a value; anything shaped like an identifier is not.

    `ws:postgresql` is a prefixed name and must not be mistaken for a literal —
    getting this wrong flags every attribute change as a contradiction.
    """
    text = (value or "").strip()
    if not text:
        return False
    if text.startswith('"'):
        return True
    if text.startswith(("http://", "https://", "urn:", "<")):
        return False
    if _PREFIXED_NAME.match(text):
        return False
    return True


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    weight: float = 0.0
    confidence: float = 0.6
    sessions: int = 1
    uses: int = 0
    status: str = "active"
    created: float = 0.0
    last_used: float = 0.0

    @property
    def triple(self) -> Triple:
        return (self.subject, self.predicate, self.object)

    @property
    def key(self) -> str:
        return triple_hash(self.triple)


class FactStore:
    """The weights, the usage counts and the conflicts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure()

    # -- storage -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.6,
                    sessions INTEGER NOT NULL DEFAULT 1,
                    uses INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created REAL NOT NULL,
                    last_used REAL NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(subject, predicate)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status)")

    # -- weight maths ------------------------------------------------------
    @staticmethod
    def score(
        confidence: float,
        sessions: int,
        uses: int,
        created: float,
        last_used: float,
        now: Optional[float] = None,
    ) -> float:
        now = now if now is not None else time.time()
        age_days = max(0.0, (now - created) / 86400.0)
        idle_days = max(0.0, (now - last_used) / 86400.0)
        recency = math.exp(-age_days / HALF_LIFE_DAYS)
        repetition = min(0.3, 0.1 * max(0, sessions - 1))
        reinforcement = min(MAX_REINFORCEMENT, REINFORCE_STEP * max(0, uses))
        raw = 0.5 * confidence + repetition + 0.2 * recency + reinforcement
        if idle_days > IDLE_DAYS_BEFORE_DECAY:
            raw *= WEEKLY_DECAY ** ((idle_days - IDLE_DAYS_BEFORE_DECAY) / 7.0)
        return round(min(1.0, max(0.0, raw)), 3)

    # -- writes ------------------------------------------------------------
    def record(
        self,
        triples: Iterable[Triple],
        session_id: str = "",
        confidence: float = 0.6,
    ) -> int:
        """Store or reinforce triples. Returns how many were new."""
        now = time.time()
        new = 0
        with self._connect() as conn:
            for triple in triples:
                subject, predicate, obj = (str(p) for p in triple)
                if predicate in STRUCTURAL_PREDICATES:
                    continue
                key = triple_hash((subject, predicate, obj))
                row = conn.execute("SELECT * FROM facts WHERE key = ?", (key,)).fetchone()
                if row is None:
                    conn.execute(
                        """INSERT INTO facts
                           (key, subject, predicate, object, weight, confidence, sessions, uses, status, created, last_used)
                           VALUES (?, ?, ?, ?, ?, ?, 1, 0, 'active', ?, ?)""",
                        (key, subject, predicate, obj,
                         self.score(confidence, 1, 0, now, now), confidence, now, now),
                    )
                    new += 1
                    self._flag_conflicts(conn, subject, predicate, obj, key, now)
                else:
                    sessions = int(row["sessions"]) + (1 if session_id else 0)
                    uses = int(row["uses"])
                    conf = max(float(row["confidence"]), confidence)
                    conn.execute(
                        "UPDATE facts SET sessions = ?, confidence = ?, weight = ? WHERE key = ?",
                        (sessions, conf, self.score(conf, sessions, uses, row["created"], row["last_used"], now), key),
                    )
        return new

    def reinforce(self, triples: Iterable[Triple]) -> int:
        """A recall used these facts; nudge them up (Hebbian, capped)."""
        now = time.time()
        touched = 0
        with self._connect() as conn:
            for triple in triples:
                key = triple_hash(tuple(str(p) for p in triple))
                row = conn.execute("SELECT * FROM facts WHERE key = ?", (key,)).fetchone()
                if row is None:
                    continue
                uses = int(row["uses"]) + 1
                conn.execute(
                    "UPDATE facts SET uses = ?, last_used = ?, weight = ? WHERE key = ?",
                    (uses, now,
                     self.score(float(row["confidence"]), int(row["sessions"]), uses,
                                row["created"], now, now),
                     key),
                )
                touched += 1
        return touched

    def decay_all(self, now: Optional[float] = None) -> int:
        """Recompute every weight (recency + idle decay). Returns rows touched."""
        now = now if now is not None else time.time()
        touched = 0
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM facts").fetchall():
                weight = self.score(float(row["confidence"]), int(row["sessions"]),
                                    int(row["uses"]), row["created"], row["last_used"], now)
                if abs(weight - float(row["weight"])) > 1e-9:
                    touched += 1
                conn.execute("UPDATE facts SET weight = ? WHERE key = ?", (weight, row["key"]))
        return touched

    def _flag_conflicts(self, conn, subject: str, predicate: str, obj: str, key: str, now: float) -> None:
        """Two different literal values for the same (subject, predicate) conflict."""
        if not is_literal(obj):
            return
        rows = conn.execute(
            "SELECT key, object FROM facts WHERE subject = ? AND predicate = ? AND key != ? AND status = 'active'",
            (subject, predicate, key),
        ).fetchall()
        for other in rows:
            if is_literal(other["object"]) and other["object"] != obj:
                conn.execute("UPDATE facts SET status = 'conflict' WHERE key IN (?, ?)",
                             (key, other["key"]))

    # -- reads -------------------------------------------------------------
    def weight(self, triple: Triple) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT weight FROM facts WHERE key = ?",
                               (triple_hash(tuple(str(p) for p in triple)),)).fetchone()
        return float(row["weight"]) if row else 0.0

    def weights(self, triples: Iterable[Triple]) -> Dict[str, float]:
        keys = [triple_hash(tuple(str(p) for p in t)) for t in triples]
        if not keys:
            return {}
        with self._connect() as conn:
            marks = ",".join("?" * len(keys))
            rows = conn.execute(f"SELECT key, weight, status FROM facts WHERE key IN ({marks})", keys).fetchall()
        return {row["key"]: float(row["weight"]) for row in rows}

    def rank(self, triples: Sequence[Triple]) -> List[Tuple[Triple, float]]:
        table = self.weights(triples)
        scored = [(t, table.get(triple_hash(tuple(str(p) for p in t)), 0.0)) for t in triples]
        return sorted(scored, key=lambda pair: -pair[1])

    def conflicts(self, limit: int = 50) -> List[Fact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE status = 'conflict' ORDER BY subject, predicate LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_fact(row) for row in rows]

    def conflicts_for(self, subject: str, predicate: str) -> List[Fact]:
        """Every conflicting value for one attribute — the two sides of a doubt."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM facts
                   WHERE subject = ? AND predicate = ? AND status = 'conflict'
                   ORDER BY weight DESC""",
                (subject, predicate),
            ).fetchall()
        return [self._to_fact(row) for row in rows]

    def top(self, limit: int = 10) -> List[Fact]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM facts ORDER BY weight DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_fact(row) for row in rows]

    def weakest(self, limit: int = 5) -> List[Fact]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM facts ORDER BY weight ASC LIMIT ?", (limit,)).fetchall()
        return [self._to_fact(row) for row in rows]

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
            conflicts = conn.execute("SELECT COUNT(*) AS n FROM facts WHERE status = 'conflict'").fetchone()["n"]
            avg = conn.execute("SELECT AVG(weight) AS w FROM facts").fetchone()["w"] or 0.0
            strongest = conn.execute("SELECT MAX(weight) AS w FROM facts").fetchone()["w"] or 0.0
            used = conn.execute("SELECT COUNT(*) AS n FROM facts WHERE uses > 0").fetchone()["n"]
        return {
            "facts": total,
            "conflicts": conflicts,
            "avg_weight": round(float(avg), 3),
            "max_weight": round(float(strongest), 3),
            "ever_used": used,
        }

    @staticmethod
    def _to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            subject=row["subject"], predicate=row["predicate"], object=row["object"],
            weight=float(row["weight"]), confidence=float(row["confidence"]),
            sessions=int(row["sessions"]), uses=int(row["uses"]), status=row["status"],
            created=float(row["created"]), last_used=float(row["last_used"]),
        )

    def close(self) -> None:
        return None
