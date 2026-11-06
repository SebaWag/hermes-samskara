"""A vocabulary that grows with the user's data.

A fixed ontology is a lie: nobody's projects, clients and decisions look like
the ones we imagined. So Samskara starts with a small seed vocabulary and
*learns predicates from usage*, keeping aliases so that "uses", "usaTecnologia"
and "depends on" converge on one edge instead of three.

Everything here is plain JSON on disk, so a user can read, edit or version it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

SEED_PREDICATES: Dict[str, List[str]] = {
    "usaTecnologia": ["usa", "uses", "utiliza", "trabaja con", "depende de", "depends on", "stack"],
    "desplegadoEn": ["corre en", "hosteado en", "runs on", "deployed to", "servidor", "vps"],
    "participaEn": ["participa", "trabaja en", "owns", "es responsable de", "maintains"],
    "sobreProyecto": ["sobre", "about", "referente a", "para el proyecto"],
    "motivadaPor": ["porque", "causada por", "motivada por", "debido a", "triggered by"],
    "afectoTecnologia": ["afectó a", "rompió", "broke", "impactó"],
    "estado": ["status", "situación", "state"],
    "fecha": ["date", "cuándo", "when"],
    "nota": ["nota", "note", "observación", "comentario"],
    "decideSobre": ["decide sobre", "decision about"],
    "prefiere": ["prefiere", "prefers", "le gusta", "favorite"],
    "evita": ["evita", "avoids", "no quiere", "dislikes"],
}

SEED_TYPES: List[str] = [
    "Proyecto", "Tecnologia", "Persona", "Organizacion", "Evento",
    "Decision", "Infraestructura", "Concepto", "Producto",
]

_SLUG_CLEAN = re.compile(r"[^a-z0-9]+")


def slug(label: str) -> str:
    """`Onboarding Hub` -> `onboarding_hub`, `versión` -> `version`.

    Accents are decomposed and dropped: an identifier that reads correctly beats
    one that preserves orthography, and a URI with `versi_n` in it is a bug you
    only notice a year later.
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", (label or "").strip().lower())
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _SLUG_CLEAN.sub("_", ascii_only).strip("_")
    return cleaned[:64] or "unknown"


def normalise(predicate: str) -> str:
    """Compare predicates by their letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", (predicate or "").lower())


class SchemaRegistry:
    """Canonical predicates/types + aliases, persisted next to the journal."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.predicates: Dict[str, List[str]] = {k: list(v) for k, v in SEED_PREDICATES.items()}
        self.types: List[str] = list(SEED_TYPES)
        self.proposed: Dict[str, int] = {}  # predicate -> times seen
        self._index: Dict[str, str] = {}
        self._rebuild()
        if self.path and self.path.is_file():
            self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.predicates.update(data.get("predicates", {}))
        for t in data.get("types", []):
            if t not in self.types:
                self.types.append(t)
        self.proposed.update(data.get("proposed", {}))
        self._rebuild()

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"predicates": self.predicates, "types": self.types, "proposed": self.proposed},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- lookup ------------------------------------------------------------
    def _rebuild(self) -> None:
        self._index = {}
        for canonical, aliases in self.predicates.items():
            self._index[normalise(canonical)] = canonical
            for alias in aliases:
                self._index[normalise(alias)] = canonical

    def canonical(self, predicate: str) -> Optional[str]:
        return self._index.get(normalise(predicate))

    def propose(self, predicate: str, threshold: int = 2) -> bool:
        """A new predicate becomes canonical after it shows up `threshold` times.

        One stray phrasing should not create an edge type; a recurring one is a
        real concept in this user's world.
        """
        key = slug(predicate)
        if not key or key in self.predicates:
            return False
        self.proposed[key] = self.proposed.get(key, 0) + 1
        if self.proposed[key] >= threshold:
            self.predicates[key] = []
            self.proposed.pop(key, None)
            self._rebuild()
            return True
        return False

    def type_of(self, name: str) -> str:
        for known in self.types:
            if normalise(known) == normalise(name):
                return known
        return "Concepto"

    # -- prompt material ---------------------------------------------------
    def prompt_block(self, max_items: int = 40) -> str:
        preds = ", ".join(list(self.predicates)[:max_items])
        types = ", ".join(self.types[:max_items])
        return (
            f"Known predicates: {preds}\n"
            f"Known entity types: {types}\n"
            "If the conversation needs a relation that does not exist here, propose "
            "a short camelCase predicate name — it will be registered if it recurs."
        )
