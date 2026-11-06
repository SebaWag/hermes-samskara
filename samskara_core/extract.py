"""Turn a conversation into typed relations — without fragmenting the graph.

The hard part of agent memory is not storage, it is deciding what is a *fact*
and whether it is a *new* one. Four rules keep a graph from turning into a
rumour mill or a hall of mirrors:

1. **Bounded output.** The model answers against a JSON schema, not free prose.
2. **Whitelisted predicates.** Anything outside the (learnable) vocabulary is
   rejected or merely *proposed* — never silently invented.
3. **Reconciliation before minting.** A label is matched against the entities
   the graph already has (see `reconcile`), and leftovers go through a semantic
   bridge, so the same decision does not become two nodes.
4. **Provenance on every edge.** Each triple records the session it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .reconcile import LabelIndex
from .schema import SchemaRegistry, slug
from .sessions import SessionContext

DEFAULT_NAMESPACE = "https://wsolutionsai.com/ns/shiva/"
MAX_RELATIONS = 40
MAX_LITERAL = 300
MAX_LABEL = 80

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "type": {"type": "string"}},
                "required": ["label", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_is_literal": {"type": "boolean"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        "new_predicates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities", "relations"],
}

BRIDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mapping": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
    },
    "required": ["mapping"],
}

INSTRUCTIONS = """You extract a knowledge graph from an agent conversation.

Rules:
- Only extract facts the user stated or confirmed. Never infer, never speculate.
- Prefer durable facts (projects, stack, people, decisions, preferences,
  incidents, constraints) over transient chatter.
- Entities are concrete things (a project, a technology, a person, an
  organisation, an infrastructure, a decision, an event). Not sentences.
- If an entity appears in the "existing entities" list under the same meaning,
  reuse that label EXACTLY as written. Never invent a variant of an existing
  entity, and never put an internal id (like dec_x_2026) in a label.
- Predicates describe a relation between two entities, or an attribute of one.
  Use the known predicates whenever one fits. If none fits, propose one short
  camelCase name in `new_predicates` (do not invent synonyms of known ones).
- Set `object_is_literal` to true when the object is a value (a date, a version,
  a status, a note) rather than another entity.
- If the conversation contains no durable fact, return empty arrays.
"""

BRIDGE_INSTRUCTIONS = """You are reconciling entity names against a knowledge graph.

You receive a list of entity names extracted from a conversation and a list of
entities that already exist in the graph. Decide, for each extracted name,
whether it refers to one of the existing entities.

- If it does, map the extracted name to the existing name, copied EXACTLY.
- If it does not, map it to an empty string.
- Never map something to an existing entity just because the words overlap: a
  version ("PostgreSQL 16") is not its parent ("PostgreSQL"), and a decision is
  not the incident that motivated it.

Return {"mapping": {"<extracted name>": "<existing name or empty>"}}.
"""

_PUNCT = re.compile(r"\s+")


@dataclass
class ExtractionResult:
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    entities: List[Tuple[str, str]] = field(default_factory=list)  # (uri, label)
    rejected: List[str] = field(default_factory=list)
    proposed: List[str] = field(default_factory=list)
    resolved: List[Tuple[str, str]] = field(default_factory=list)   # (label, existing uri)
    unresolved: List[str] = field(default_factory=list)
    bridged: List[Tuple[str, str]] = field(default_factory=list)    # (label, existing label)

    @property
    def count(self) -> int:
        return len(self.triples)


def _clean(text: Any) -> str:
    return _PUNCT.sub(" ", str(text or "")).strip()


def build_prompt(ctx: SessionContext, registry: SchemaRegistry, index: Optional[LabelIndex] = None) -> str:
    meta = [f"session: {ctx.session_id}"]
    if ctx.title:
        meta.append(f"title: {ctx.title}")
    if ctx.cwd:
        meta.append(f"working directory: {ctx.cwd}")
    existing = ""
    if index is not None and len(index):
        existing = "\n\nexisting entities (reuse these labels when they mean the same thing):\n- " + "\n- ".join(
            index.sample(80)
        )
    return (
        "\n".join(meta)
        + "\n\n"
        + registry.prompt_block()
        + existing
        + "\n\n--- conversation ---\n"
        + ctx.text[:12000]
    )


def _resolve(index: Optional[LabelIndex], label: str) -> Optional[str]:
    if index is None:
        return None
    hit = index.resolve(label)
    return hit.uri if hit else None


def validate(
    payload: Dict[str, Any],
    registry: SchemaRegistry,
    namespace: str = DEFAULT_NAMESPACE,
    session_id: str = "",
    index: Optional[LabelIndex] = None,
) -> ExtractionResult:
    """Turn the model's JSON into triples we are willing to store."""
    result = ExtractionResult()
    if not isinstance(payload, dict):
        result.rejected.append("payload is not an object")
        return result
    for key in ("entities", "relations", "new_predicates"):
        if payload.get(key) is not None and not isinstance(payload[key], list):
            result.rejected.append(f"{key} is not a list")
            payload[key] = []

    known_uris: Set[str] = set()

    # 1. entities: reuse an existing URI when the meaning already exists
    for item in payload.get("entities") or []:
        if not isinstance(item, dict):
            result.rejected.append("entity is not an object")
            continue
        label = _clean(item.get("label"))
        if not label or len(label) > MAX_LABEL:
            continue
        resolved = _resolve(index, label)
        if resolved:
            hit = index.resolve(label)
            result.resolved.append((label, resolved))
            if hit.kind == "fuzzy":
                result.triples.append((resolved, "skos:altLabel", label))
            if resolved in known_uris:
                continue
            known_uris.add(resolved)
            result.entities.append((resolved, hit.label))
            continue
        uri = namespace + slug(label)
        if uri in known_uris:
            continue
        known_uris.add(uri)
        result.unresolved.append(label)
        result.entities.append((uri, label))
        entity_type = registry.type_of(_clean(item.get("type")) or "Concepto")
        result.triples.append((uri, "rdf:type", f"ws:{entity_type}"))
        result.triples.append((uri, "rdfs:label", label))

    # 2. new predicates are registered only if they recur
    for name in payload.get("new_predicates") or []:
        if not isinstance(name, (str, int, float)):
            continue
        candidate = _clean(name)
        if not candidate:
            continue
        if registry.propose(candidate):
            result.proposed.append(slug(candidate))
        else:
            result.rejected.append(f"predicate proposed once: {candidate}")

    # 3. relations
    seen_triples: Set[Tuple[str, str, str]] = set()
    for item in payload.get("relations") or []:
        if not isinstance(item, dict):
            result.rejected.append("relation is not an object")
            continue
        if len(result.triples) >= MAX_RELATIONS * 3:
            break
        subject = _clean(item.get("subject"))
        predicate_raw = _clean(item.get("predicate"))
        obj = _clean(item.get("object"))
        if not subject or not predicate_raw or not obj:
            continue
        if len(subject) > MAX_LABEL or len(obj) > MAX_LITERAL:
            result.rejected.append(f"too long: {subject[:30]}…")
            continue
        canonical = registry.canonical(predicate_raw)
        if not canonical:
            registry.propose(predicate_raw)
            result.rejected.append(f"unknown predicate: {predicate_raw}")
            continue
        subject_uri = _resolve(index, subject) or (namespace + slug(subject))
        if _resolve(index, subject) is None and subject not in result.unresolved:
            result.unresolved.append(subject)
        if subject == obj:
            continue
        is_literal = bool(item.get("object_is_literal")) or canonical in {"estado", "fecha", "nota"}
        if is_literal:
            object_token = obj
        else:
            object_token = _resolve(index, obj) or (namespace + slug(obj))
            if _resolve(index, obj) is None and obj not in result.unresolved:
                result.unresolved.append(obj)
        triple = (subject_uri, f"ws:{canonical}", object_token)
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        result.triples.append(triple)

    # 4. provenance: every edge records the session it came from
    if session_id:
        result.triples.append(
            (f"{namespace}session_{slug(session_id)}", "ws:derivadoDe", f'"{session_id}"')
        )
    return result


def apply_bridge(
    result: ExtractionResult,
    mapping: Dict[str, str],
    index: LabelIndex,
    namespace: str = DEFAULT_NAMESPACE,
) -> ExtractionResult:
    """Point bridged entities at the URIs that already exist.

    Rewrites triples in place, drops the now-orphan minted labels and merges the
    alternative spelling as `skos:altLabel`, so the graph learns the synonym
    instead of growing a second node for it.
    """
    if not mapping:
        return result
    rewritten: List[Tuple[str, str, str]] = []
    for label, target in mapping.items():
        target = _clean(target)
        if not target:
            continue
        hit = index.resolve(target)
        if not hit:
            continue
        old_uri = namespace + slug(label)
        result.bridged.append((label, hit.label))
        if old_uri not in [t[0] for t in rewritten]:
            rewritten.append((hit.uri, "skos:altLabel", label))
        for subject, predicate, obj in result.triples:
            subject = hit.uri if subject == old_uri else subject
            obj = hit.uri if obj == old_uri else obj
            if predicate == "rdfs:label" and subject == hit.uri:
                continue  # the existing entity already has a label
            rewritten.append((subject, predicate, obj))
    # keep everything that was not part of the rewrite
    bridged_uris = {namespace + slug(label) for label in mapping}
    for triple in result.triples:
        if triple[0] in bridged_uris:
            continue
        if triple not in rewritten:
            rewritten.append(triple)
    seen: Set[Tuple[str, str, str]] = set()
    deduped: List[Tuple[str, str, str]] = []
    for triple in rewritten:
        if triple in seen:
            continue
        seen.add(triple)
        deduped.append(triple)
    result.triples = deduped
    return result


class Extractor:
    """Extraction pipeline with an injected model and an optional graph index."""

    def __init__(
        self,
        registry: SchemaRegistry,
        llm: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
        namespace: str = DEFAULT_NAMESPACE,
        index: Optional[LabelIndex] = None,
        bridge: bool = True,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.namespace = namespace
        self.index = index
        self.bridge_enabled = bridge

    def extract(self, ctx: SessionContext) -> ExtractionResult:
        prompt = build_prompt(ctx, self.registry, self.index)
        try:
            payload = self.llm(INSTRUCTIONS, prompt, EXTRACTION_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - extraction must never break a session
            return ExtractionResult(rejected=[f"llm error: {exc}"])
        result = validate(payload, self.registry, self.namespace, ctx.session_id, self.index)

        if self.bridge_enabled and self.index is not None and result.unresolved:
            mapping = self._bridge(result.unresolved)
            if mapping:
                result = apply_bridge(result, mapping, self.index, self.namespace)
        return result

    def _bridge(self, unresolved: List[str]) -> Dict[str, str]:
        candidates = self.index.sample(80) if self.index is not None else []
        if not candidates:
            return {}
        prompt = (
            "extracted names:\n- " + "\n- ".join(unresolved[:30])
            + "\n\nexisting entities:\n- " + "\n- ".join(candidates)
        )
        try:
            payload = self.llm(BRIDGE_INSTRUCTIONS, prompt, BRIDGE_SCHEMA)
        except Exception:  # noqa: BLE001 - the bridge is an optimisation, not a requirement
            return {}
        mapping = payload.get("mapping") if isinstance(payload, dict) else None
        return mapping if isinstance(mapping, dict) else {}
