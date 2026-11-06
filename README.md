# Samskara — relational memory for Hermes Agent

> **Samskara** (Sanskrit: *saṃskāra*) — the latent impression an experience leaves
> behind, which shapes what happens next.

Most agent memory stores **text** and retrieves it by similarity. Samskara stores
**relations** and retrieves them by **path**. That is the difference between:

> *"Here are five paragraphs that look like your question"*

and

> *"You cannot upgrade PostgreSQL on Onboarding Hub — a decision from March is
> still in force, and here is the incident that caused it, in 1.3 ms."*

Every recall block Samskara injects into a turn carries its own **provenance**:
how many entities, how many facts, and how many milliseconds the graph took to
answer. Memory you cannot audit is memory you cannot trust.

---

## Why a graph instead of another vector store

| | Vector memory | Samskara |
|---|---|---|
| Unit of storage | chunk of text + embedding | entity + relation |
| Retrieval | nearest neighbours | traversal (1..n hops) |
| Answers "what is similar?" | ✅ | ✅ (via label match) |
| Answers "what depends on this, and why?" | ❌ | ✅ |
| Can it explain itself? | similarity score | the actual path + source facts |
| Works on a 3-hop question | degrades | one query |

Vector search cannot answer *"which projects use PostgreSQL **and** are covered
by a decision that is still in force"* — that is not a distance, it is a path.

---

## Install

```bash
hermes plugins install SebaWag/hermes-samskara
hermes plugins enable samskara
hermes config set memory.provider samskara
```

Then point Samskara at your graph. It speaks plain SPARQL-over-HTTP, so any
endpoint that answers `POST /sparql` works (MillenniumDB, Oxigraph, Jena,
Blazegraph, Neptune, …):

```bash
# in $HERMES_HOME/.env
SAMSKARA_ENDPOINT=http://127.0.0.1:1234/sparql
```

Or persistently in `$HERMES_HOME/samskara.json`:

```json
{
  "endpoint": "http://127.0.0.1:1234/sparql",
  "auto_recall": true,
  "max_entities": 3,
  "max_facts": 24,
  "timeout": 8.0
}
```

Verify:

```bash
hermes memory status     # → Plugin: installed ✓ · Status: available ✓ · samskara ← active
```

### Running a graph next to the agent

MillenniumDB is a 49 MB container that holds this demo corpus in **18 MiB of RAM**:

```bash
docker run -d --name samskara-mdb --restart unless-stopped \
  -p 127.0.0.1:1234:1234 \
  -v /srv/samskara:/data \
  imfd/millenniumdb:latest bash -c "mdb server /data/db -p 1234 --browser false"
```

> **Co-locate the graph with the agent.** We benchmarked the same corpus over an
> SSH tunnel between two datacentres: **0.8 ms local vs ~1000 ms remote** per
> query. A recall makes ~10 queries, so the tunnel turns a 10 ms recall into an
> 11 second turn. Memory belongs next to the agent — for latency *and* for
> sovereignty.

---

## What it actually does in a turn

```
user asks ──► prefetch()      find entities → expand both directions → inject block
              (Samskara)      [memoria relacional · 3 entidades · 18 hechos · 1.27 ms]
                     │
model answers ───────┘
                     │
              sync_turn()     buffer the turn (no I/O, no latency)
                     │
context full ─► on_pre_compress()  extract entities BEFORE compression, journal
                                    the relations → the graph keeps what the
                                    transcript is about to forget
```

### The part that matters: `on_pre_compress`

Compression is lossy for prose and, with Samskara, **lossless for relations**.
Before Hermes summarises the context away, Samskara resolves the entities the
session talked about and writes the edges to the journal — so a fact mentioned
in turn 3 is still retrievable in session 40.

---

## Tools exposed to the model

| Tool | Purpose |
|---|---|
| `samskara_search` | Resolve names in free text to real entities, with facts in both directions |
| `samskara_query` | Read-only SPARQL — follow relations across hops (SELECT/ASK only) |
| `samskara_write` | Persist one fact; appended to the journal, folded in on reindex |
| `samskara_status` | Health + telemetry (endpoint, entity count, latency, error count) |

---

## Write path: why a journal

The open-source MillenniumDB build is **import-only** — it rejects SPARQL
`INSERT`. Rather than hide that, Samskara turns it into a guarantee:

1. `samskara_write` and `on_pre_compress` append to
   `$HERMES_HOME/samskara/journal/session-*.ttl` (single write, `fsync`, done).
2. A reindex folds the journal into the graph: dump → concatenate → import →
   archive the applied journals.

A write therefore costs microseconds and **can never slow down or fail a turn**.
Print the exact commands for your layout:

```python
from samskara_core.journal import reindex_commands
print("\n".join(reindex_commands("/srv/samskara", "/root/.hermes/samskara/journal")))
```

---

## Ontology

Any RDF works: entity resolution only needs `rdfs:label` on the things you want
to find. The bundled demo corpus uses this prefix:

```
ws:  https://wsolutionsai.com/ns/shiva/
```

Change it in `samskara_core/client.py` (`PREFIXES`) or pass full IRIs.

---

## Demo corpus

`seed/demo.ttl` (72 lines) models a real dev team: projects, technologies, an
infrastructure, an incident, and the decision that is still in force. It exists
to demonstrate the one thing vector memory cannot do:

```sparql
SELECT ?proyecto ?decisionLabel WHERE {
  ?proyecto ws:usaTecnologia ws:postgresql ;
            rdfs:label ?proyectoLabel .
  ?decision ws:sobreProyecto ?proyecto ;
            ws:estado "vigente" ;
            rdfs:label ?decisionLabel ;
            ws:motivadaPor ?incidente .
}
```

→ *2 projects, 1 decision still in force, 1 incident, answered in 0.14 ms.*

---

## Testing

```bash
python -m pytest tests/ -v                       # offline: parsing, rendering, journal
python tests/integration_graph.py <endpoint>     # needs a live graph
```

---

## The ontology: memory with weight

Labels alone are a bag of strings. Samskara declares a small hierarchy
(`samskara_core/ontology.py`, one readable file) and **materialises the closure
at reindex time** — so reads stay plain SPARQL at ~0.1 ms, with no reasoning in
the hot path:

```
Incidente ⊑ Evento ⊑ Ocurrencia ⊑ Entidad
BaseDeDatos ⊑ Tecnologia ⊑ Artefacto ⊑ Entidad
usaTecnologia ⊑ dependeDe ⊑ relacionadoCon
```

What that buys you, concretely:

| You ask | Without a hierarchy | With Samskara |
|---|---|---|
| "what events happened?" | misses incidents (they are typed `Incidente`) | ✅ includes them |
| "what technologies does X use?" | misses `BaseDeDatos` nodes | ✅ includes them |
| "what does X depend on?" | one query per predicate | ✅ `(dependeDe\|usaTecnologia\|desplegadoEn)` in one |

The closure is printed on every reindex (`+ 35 inferred triple(s)`), because
inference that changes nothing is a smell.

---

## Scope & safety — what this plugin does *not* do

Installing Samskara gives you a copy of this software and nothing else. It is
worth being explicit about that, because "memory provider" means it handles
everything you have told your agent:

- **No telemetry.** No analytics endpoint, no phone-home, no counters.
- **No outbound network calls** other than the graph endpoint *you* configure
  (default `http://127.0.0.1:1234/sparql`). Grep for yourself: `client.py` is
  the only module that opens a socket.
- **No bundled data beyond the demo corpus.** `seed/demo.ttl` is a synthetic
  example (fictional projects, generic infrastructure). Your graph is not in
  this repository and is never uploaded.
- **Writes stay local.** The journal is written to *your* `$HERMES_HOME`.
- **No self-updater.** Updates only flow through a catalog SHA bump, so an
  installed copy cannot silently move to a commit nobody reviewed.
- **Read-only tool surface.** `samskara_query` rejects anything that is not
  `SELECT`/`ASK`: a model cannot rewrite your memory graph with a tool call.

If you run the graph yourself, bind it to loopback:

```bash
-p 127.0.0.1:1234:1234   # not 0.0.0.0 — memory is not a public service
```

---

## Limitations (honest list)

- **Single write path.** Concurrent writers to the same journal should serialise
  reindex runs; the journal itself is append-only and safe.
- **Entity resolution is lexical** (`FILTER(CONTAINS(LCASE(...)))`). It handles
  the names in your graph well and ambiguity poorly; it does not embed.
- **No automatic reindex.** Writes are journalled, not live. Run reindex on a
  schedule (cron) or on session end.
- **Read-only tool surface.** `samskara_query` refuses non-SELECT/ASK statements
  on purpose: a model should not be able to rewrite your memory graph.

---

## License

MIT © Wagner Solutions
