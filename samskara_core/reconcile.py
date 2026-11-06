"""Entity reconciliation: keep one graph, not five.

Extraction mints URIs from labels, and labels are unstable — the same decision
arrives as "Congelar PostgreSQL en 15 hasta migrar las réplicas lógicas" in one
session and "dec_congelar_pg" in the next. Left alone that fragments the graph
and, worse, silently breaks multi-hop recall: the path stops at the duplicate.

Two layers, in order of cheapness:

1. **Deterministic** (this module): normalise the label, look for an exact match,
   then a near-identical one. Catches casing, accents, punctuation, parenthetical
   ids — the majority of real duplicates.
2. **Semantic** (`extract.bridge`): the model is shown the labels it could not
   match and asked to point them at an existing entity or declare them new.
   Only runs for the leftovers, so it costs one extra call per session at most.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

_PARENTHETICAL = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*$")
_NON_WORD = re.compile(r"[^a-z0-9]+")

FUZZY_THRESHOLD = 0.90


def normalize_label(label: str) -> str:
    """Lowercase, accent-free, punctuation-free, without trailing ids."""
    text = unicodedata.normalize("NFKD", str(label or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PARENTHETICAL.sub("", text.lower())
    return _NON_WORD.sub(" ", text).strip()


@dataclass
class Resolution:
    uri: str
    label: str
    kind: str          # "exact" | "fuzzy"
    score: float = 1.0


class LabelIndex:
    """A snapshot of the graph's labels, with exact and fuzzy lookup."""

    def __init__(self) -> None:
        self._by_norm: Dict[str, Resolution] = {}
        self._norms: List[str] = []

    def __len__(self) -> int:
        return len(self._by_norm)

    # -- loading -----------------------------------------------------------
    def add(self, uri: str, label: str) -> None:
        key = normalize_label(label)
        if not key or key in self._by_norm:
            return
        self._by_norm[key] = Resolution(uri=uri, label=label, kind="exact")
        self._norms.append(key)

    def load_pairs(self, pairs: Iterable[Tuple[str, str]]) -> "LabelIndex":
        for uri, label in pairs:
            self.add(uri, label)
        return self

    def load_from_client(self, client, limit: int = 5000) -> "LabelIndex":
        """Snapshot labels from a SPARQL endpoint. Failure means an empty index."""
        try:
            rows, _ = client.select(
                f"SELECT ?e ?l WHERE {{ ?e rdfs:label ?l }} LIMIT {int(limit)}"
            )
        except Exception:  # noqa: BLE001 - a missing index degrades to minting
            return self
        for row in rows:
            uri, label = row.get("e", ""), row.get("l", "")
            if uri and label:
                self.add(uri, label)
        return self

    # -- lookup ------------------------------------------------------------
    def resolve(self, label: str) -> Optional[Resolution]:
        key = normalize_label(label)
        if not key:
            return None
        exact = self._by_norm.get(key)
        if exact:
            return exact
        best: Optional[Resolution] = None
        best_score = 0.0
        for candidate in self._norms:
            score = difflib.SequenceMatcher(None, key, candidate).ratio()
            if score > best_score:
                best_score, best_match = score, candidate
        if best_score >= FUZZY_THRESHOLD:
            hit = self._by_norm[best_match]
            best = Resolution(uri=hit.uri, label=hit.label, kind="fuzzy", score=best_score)
        return best

    def unresolved(self, labels: Iterable[str]) -> List[str]:
        return [label for label in labels if self.resolve(label) is None]

    def sample(self, limit: int = 80) -> List[str]:
        """A compact label list to show the model during extraction."""
        return [res.label for res in list(self._by_norm.values())[:limit]]
