"""An ontology the graph can actually reason with.

Labels alone are a bag of strings: "the incident broke the replica" and "the
deployment happened" are the same shape and tell you nothing. A hierarchy gives
memory its first kind of weight — *subsumption*:

    Incidente   ⊑ Evento      → "what events happened?" now includes incidents
    Tecnologia  ⊑ Artefacto   → "what artefacts does this project depend on?"
                                reaches its stack and its infrastructure at once

Two things make this usable rather than decorative:

1. **The hierarchy is declared here, in one readable file**, not scattered
   through prompts.
2. **Inference is materialised at reindex time.** MillenniumDB is import-only,
   so we compute the closure when we rebuild — which means read queries stay
   plain SPARQL and stay fast (0.1 ms, no reasoning at query time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

# Class hierarchy: child -> parents. Multiple parents are allowed (it is a DAG).
CLASS_HIERARCHY: Dict[str, List[str]] = {
    # agents
    "Persona": ["Agente"],
    "Organizacion": ["Agente"],
    "Agente": ["Entidad"],
    # artefacts
    "Proyecto": ["Artefacto"],
    "Producto": ["Artefacto"],
    "Tecnologia": ["Artefacto"],
    "Libreria": ["Tecnologia"],
    "BaseDeDatos": ["Tecnologia"],
    "Infraestructura": ["Artefacto"],
    "Servidor": ["Infraestructura"],
    "Artefacto": ["Entidad"],
    # knowledge
    "Concepto": ["Conocimiento"],
    "Decision": ["Conocimiento"],
    "Politica": ["Decision"],
    "Restriccion": ["Conocimiento"],
    "Conocimiento": ["Entidad"],
    # occurrences
    "Evento": ["Ocurrencia"],
    "Incidente": ["Evento"],
    "Reunion": ["Evento"],
    "Despliegue": ["Evento"],
    "Ocurrencia": ["Entidad"],
    # top
    "Entidad": [],
}

# Predicate hierarchy: child -> parents. A parent is the general form of the edge.
PREDICATE_HIERARCHY: Dict[str, List[str]] = {
    "usaTecnologia": ["dependeDe"],
    "desplegadoEn": ["dependeDe"],
    "dependeDe": ["relacionadoCon"],
    "participaEn": ["relacionadoCon"],
    "sobreProyecto": ["relacionadoCon"],
    "decideSobre": ["relacionadoCon"],
    "estado": ["atributoDe"],
    "fecha": ["atributoDe"],
    "nota": ["atributoDe"],
    "afectoTecnologia": ["relacionadoCon"],
    "motivadaPor": ["relacionadoCon"],
    "derivadoDe": ["procedenciaDe"],
    "relacionadoCon": [],
}

_TERM = re.compile(r"<([^>]+)>")


@dataclass
class Ontology:
    """The declared hierarchy plus the closures we compute from it."""

    classes: Dict[str, List[str]] = field(default_factory=lambda: dict(CLASS_HIERARCHY))
    predicates: Dict[str, List[str]] = field(default_factory=lambda: dict(PREDICATE_HIERARCHY))
    _class_cache: Dict[str, Set[str]] = field(default_factory=dict, repr=False)
    _pred_cache: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    # -- closures ----------------------------------------------------------
    def superclasses(self, name: str) -> Set[str]:
        """Every ancestor of a class, transitively (cycle-safe)."""
        if name in self._class_cache:
            return self._class_cache[name]
        out: Set[str] = set()
        stack = list(self.classes.get(name, []))
        while stack:
            current = stack.pop()
            if current in out or current == name:
                continue
            out.add(current)
            stack.extend(self.classes.get(current, []))
        self._class_cache[name] = out
        return out

    def subclasses(self, name: str) -> Set[str]:
        return {child for child in self.classes if name in self.superclasses(child)}

    def superpredicates(self, name: str) -> Set[str]:
        if name in self._pred_cache:
            return self._pred_cache[name]
        out: Set[str] = set()
        stack = list(self.predicates.get(name, []))
        while stack:
            current = stack.pop()
            if current in out or current == name:
                continue
            out.add(current)
            stack.extend(self.predicates.get(current, []))
        self._pred_cache[name] = out
        return out

    def is_a(self, child: str, ancestor: str) -> bool:
        return ancestor in self.superclasses(child)

    # -- materialisation ---------------------------------------------------
    def materialise_types(
        self,
        triples: Iterable[Tuple[str, str, str]],
        namespace: str = "https://wsolutionsai.com/ns/shiva/",
    ) -> List[Tuple[str, str, str]]:
        """Given (s, p, o) triples, add the entailed rdf:type triples.

        Only `rdf:type` edges produce new triples, so the closure stays small:
        a handful per entity instead of a combinatorial explosion.
        """
        extra: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for subject, predicate, obj in triples:
            if predicate not in (RDF_TYPE, "rdf:type", "a"):
                continue
            cls_name = _local(obj)
            if not cls_name or cls_name not in self.classes:
                continue
            for ancestor in sorted(self.superclasses(cls_name)):
                triple = (subject, RDF_TYPE, namespace + ancestor)
                if triple in seen:
                    continue
                seen.add(triple)
                extra.append(triple)
        return extra

    # -- query helpers -----------------------------------------------------
    def predicate_path(self, name: str) -> str:
        """SPARQL property path that also matches every specialisation.

        `dependeDe` becomes `(dependeDe|usaTecnologia|desplegadoEn)` so one query
        answers the general question without knowing the leaf vocabulary.
        """
        leaves = [name] + sorted(self.subclasses_of_predicate(name))
        return "(" + "|".join(f"ws:{leaf}" for leaf in leaves) + ")"

    def subclasses_of_predicate(self, name: str) -> Set[str]:
        return {child for child in self.predicates if name in self.superpredicates(child)}

    def describe(self) -> str:
        roots = [c for c in self.classes if not self.classes[c] or self.classes[c] == ["Entidad"]]
        return (
            f"{len(self.classes)} classes, {len(self.predicates)} predicates; "
            f"top-level: {', '.join(sorted(roots))}"
        )


def _local(token: str) -> str:
    match = _TERM.search(token or "")
    value = match.group(1) if match else (token or "")
    for sep in ("#", "/", ":"):
        if sep in value:
            value = value.rsplit(sep, 1)[-1]
    return value


def _is_uri(token: str) -> bool:
    return (token or "").startswith(("<http", "http"))


def as_uri(token: str) -> str:
    return token if token.startswith("<") else f"<{token}>"


def as_object(token: str) -> str:
    return token if token.startswith(("<", '"')) else f"<{token}>"


def parse_nt_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Parse the subset of N-Triples we emit: <s> <p> <o> .  (o may be a literal)."""
    line = line.strip()
    if not line or line.startswith("#") or not line.endswith("."):
        return None
    body = line[:-1].strip()
    parts: List[str] = []
    for chunk in body.split(" ", 2):
        parts.append(chunk.strip())
    if len(parts) != 3:
        return None
    subject, predicate, obj = parts
    # subjects and predicates come wrapped in <> in N-Triples; normalise them so
    # comparisons (`predicate == RDF_TYPE`) actually work. Objects keep their
    # form, because a literal is not a URI.
    return subject.strip("<>"), predicate.strip("<>"), obj


def materialise_nt(
    text: str,
    ontology: Optional[Ontology] = None,
    namespace: str = "https://wsolutionsai.com/ns/shiva/",
) -> Tuple[str, int]:
    """Add entailed type triples to an N-Triples document.

    Returns the extended document and how many triples were added — a number
    worth printing, because "inference that changes nothing" is a smell.
    """
    ontology = ontology or Ontology()
    triples: List[Tuple[str, str, str]] = []
    for line in text.splitlines():
        parsed = parse_nt_line(line)
        if parsed:
            triples.append(parsed)
    extra = ontology.materialise_types(triples, namespace)
    if not extra:
        return text, 0
    rendered = []
    for subject, predicate, obj in extra:
        rendered.append(f"{as_uri(subject)} {as_uri(predicate)} {as_object(obj)} .")
    return text.rstrip("\n") + "\n" + "\n".join(rendered) + "\n", len(rendered)
