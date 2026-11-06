## What it does

Samskara is a **memory provider** backed by a knowledge graph (MillenniumDB or any SPARQL endpoint). Where vector memory answers *"what looks similar?"*, Samskara answers *"what depends on this, and why?"* — multi-hop paths, with the provenance of every fact attached.

## Hermes surfaces used

- `register_memory_provider` (MemoryProvider ABC): `prefetch`, `sync_turn`, `get_tool_schemas`, `handle_tool_call`, `on_pre_compress`, `on_session_end`, `on_delegation`, `on_session_switch`, `on_memory_write`, `get_config_schema`, `save_config`, `backup_paths`
- 5 tools: `samskara_search`, `samskara_query` (read-only SPARQL), `samskara_write`, `samskara_conflicts`, `samskara_status`
- `ctx.llm` (Plugin LLM Access) for session-end extraction and backfill — **no provider/model override, no capability request**; the host's model and credentials are used as-is.
- `$HERMES_HOME/state.db` opened **read-only** for the onboarding backfill.

## Three things that are not a prompt or a skill

1. **Subsumption inference**, materialised at reindex time (`Incidente ⊑ Evento`, `BaseDeDatos ⊑ Tecnologia`). *"What events happened?"* returns incidents even though nothing asserted `a Evento`; reads stay plain SPARQL — the engine reports 0.05 ms of execution, 0.87 ms wall clock over loopback — because the closure is computed when the graph is rebuilt, not per query.
2. **Per-fact weight**: `0.5·confidence + repetition + 0.2·recency + reinforcement by use`, with decay after 30 days of disuse. It lives in a SQLite sidecar, so the graph stays plain RDF and fast. Measured on the demo graph: **0.45 → 0.55 after one recall**, while unused facts stayed at 0.45. Trust is a hierarchy, not a constant: explicit write 0.90 > extraction 0.65 > import 0.50.
3. **Conflicts are surfaced, never resolved silently.** Two contradictory values for one attribute are both kept and flagged; the recall block and the `samskara_conflicts` tool report both, with weights, and the agent declines to choose:

   > *"No está claro: mi memoria guarda dos estados contradictorios ('descongelado', peso 0.65, y 'vigente', peso 0.45). No elijo por ti."*

## Onboarding

Install, and the graph is built from the sessions the agent already had: `tools/backfill.py` reads `state.db` read-only, extracts durable relations per session, and reconciles them against existing entities in two layers (deterministic normalisation, then a semantic pass) so the same decision does not become two nodes. After that, each session close adds only what is new.

## Disclosure (rule 13)

- With `extract_on_session_end` enabled (default), the recent turns of a session are sent to **the user's own model** via `ctx.llm` to extract facts. Nothing else leaves the machine.
- The only other network call is the graph endpoint the user configures; the default is loopback.
- No telemetry, no self-updater, no bundled user data. **Stdlib only — zero Python dependencies**, so nothing in the install can rot.
- Extraction is **plausible, not certain**: it can produce an edge that is wrong. That is why extracted facts weigh 0.65 against 0.90 for an explicit write, and why contradictions are flagged instead of overwritten. A review queue for pending facts is the known next step.

## Checks

`hermes plugins validate <clone> --install-deps` → **Validation passed**:

```
✓ manifest · ✓ loadable · ✓ capability probe (register() in isolation)
✓ declared tools match registrations · ✓ security scan safe
✓ no core override · ✓ requires_hermes '>=0.19.0' parses
```

118 offline tests (`python -m pytest tests/ -q`), no graph and no network required — including the failure modes: graph down, HTML instead of JSON, path traversal in a session id, an ontology cycle.
