#!/usr/bin/env python3
"""Reproducible micro-benchmarks for Samskara's offline hot paths.

Everything measured here runs on a bare Python install: no graph, no network,
no Hermes runtime. That is the point -- the numbers in the README should be
re-runnable by anyone who clones the repo.

    python tools/bench.py            # markdown table
    python tools/bench.py --json     # machine-readable
    python tools/bench.py --endpoint http://127.0.0.1:1234/sparql
                                     # adds the full recall pipeline (needs a graph)

The graph-dependent rows are marked `n/a` when no endpoint is given. They are
never estimated: if it was not measured, it is not printed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.facts import FactStore, triple_hash  # noqa: E402
from samskara_core.journal import Journal  # noqa: E402
from samskara_core.ontology import Ontology  # noqa: E402
from samskara_core.recall import Entity, Fact, keywords, render  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"


def _time_once(fn: Callable[[], Any], inner: int) -> float:
    start = time.perf_counter()
    for _ in range(inner):
        fn()
    return (time.perf_counter() - start) / inner


def bench(fn: Callable[[], Any], *, inner: int = 1, reps: int = 5) -> Dict[str, float]:
    """Best-of-N, because the fastest run is the one least polluted by the OS."""
    samples = [_time_once(fn, inner) for _ in range(reps)]
    best = min(samples)
    return {
        "best_ms": best * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "ops_per_s": (1.0 / best) if best > 0 else float("inf"),
    }


def synth_facts(count: int, entities: int = 5) -> List[Fact]:
    predicates = ["usaTecnologia", "desplegadoEn", "sobreProyecto", "motivadaPor"]
    return [
        Fact(
            direction="out" if i % 2 == 0 else "in",
            predicate=predicates[i % len(predicates)],
            target=f"objeto_{i}",
            other_uri=f"{NS}objeto_{i}",
            predicate_uri=f"{NS}{predicates[i % len(predicates)]}",
        )
        for i in range(count)
    ]


def run(endpoint: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rows": [],
    }
    rows = results["rows"]

    tmp = tempfile.mkdtemp(prefix="samskara-bench-")
    store = FactStore(Path(tmp) / "facts.db")
    journal = Journal(Path(tmp) / "journal", session_id="bench")
    ontology = Ontology()

    triples = [(f"{NS}e{i}", f"{NS}usaTecnologia", f"{NS}t{i}") for i in range(2000)]

    def add(name: str, fn: Callable[[], Any], unit: str, inner: int = 1, reps: int = 5,
            per_call: int = 1, note: str = "") -> None:
        """`bench` already returns time per single call of `fn`; `per_call` says
        how many units of work one call performs (e.g. 2000 triples)."""
        r = bench(fn, inner=inner, reps=reps)
        call_ms = r["best_ms"]
        call_s = call_ms / 1000.0
        rows.append({
            "operation": name,
            "best_ms": call_ms,
            "median_ms": r["median_ms"],
            "unit": unit,
            "units_per_call": per_call,
            "per_s": (per_call / call_s) if call_s > 0 else float("inf"),
            "note": note,
        })

    # --- pure Python hot paths -------------------------------------------
    add("triple_hash()", lambda: triple_hash((f"{NS}a", f"{NS}p", f"{NS}b")), "call", inner=200_000)
    add("FactStore.score()", lambda: FactStore.score(0.9, 3, 2, time.time() - 86400, time.time() - 3600), "call", inner=200_000)
    add("keywords() on a sentence", lambda: keywords("¿puedo subir PostgreSQL a la 17 en Onboarding Hub mañana?"), "call", inner=50_000)
    add("ontology.superclasses() cached", lambda: ontology.superclasses("Incidente"), "call", inner=200_000)

    def cold_closure() -> None:
        fresh = Ontology()
        fresh.superclasses("Incidente")

    add("ontology.superclasses() cold", cold_closure, "call", inner=2_000)

    # --- storage ----------------------------------------------------------
    def journal_batch() -> None:
        journal.append(triples[:50])

    add("Journal.append() — batch of 50, fsync", journal_batch, "triple", inner=20, per_call=50)

    def record_batch() -> None:
        store.record(triples, session_id="bench")

    add("FactStore.record() — 2000 triples", record_batch, "triple", inner=5, per_call=2000)

    add("FactStore.weights() — 200 triples", lambda: store.weights(triples[:200]), "triple", inner=200, per_call=200)

    # --- rendering --------------------------------------------------------
    facts_by_uri = {f"{NS}e{i}": synth_facts(24) for i in range(5)}
    entities = [Entity(uri=f"{NS}e{i}", label=f"Entidad {i}") for i in range(5)]

    add("recall.render() — 5 entities x 24 facts",
        lambda: render(entities, facts_by_uri, elapsed_ms=1.3), "fact", inner=2_000, per_call=120)

    # --- full pipeline (graph required) -----------------------------------
    if endpoint:
        from samskara_core.client import SparqlClient
        from samskara_core.recall import recall as recall_pipeline

        client = SparqlClient(endpoint, timeout=8.0)
        query = "¿puedo subir PostgreSQL a la 17 en Onboarding Hub?"

        def full_recall() -> None:
            recall_pipeline(client, query, max_entities=3, max_facts=24, store=store)

        add("recall() end-to-end (query -> block), graph-backed",
            full_recall, "call", inner=20, reps=5, note=f"endpoint {endpoint}")
    else:
        rows.append({
            "operation": "recall() end-to-end (query -> block), graph-backed",
            "best_ms": None, "median_ms": None, "unit": "call", "units_per_call": 1, "per_s": None,
            "note": "n/a without --endpoint (not estimated)",
        })

    store.close()
    return results


def print_markdown(results: Dict[str, Any]) -> None:
    print(f"Python {results['python']} · {results['machine']} · {platform.system()}\n")
    print("| Operation | Best (per call) | Median | Throughput | Note |")
    print("|---|---:|---:|---:|---|")
    for row in results["rows"]:
        if row["best_ms"] is None:
            best = median = thr = "n/a"
        else:
            best = f"{row['best_ms']:.4f} ms"
            median = f"{row['median_ms']:.4f} ms"
            thr = f"{row['per_s']:,.0f} {row['unit']}/s"
        print(f"| {row['operation']} | {best} | {median} | {thr} | {row['note']} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--endpoint", default="", help="SPARQL endpoint for the full recall row")
    args = parser.parse_args()

    results = run(args.endpoint)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_markdown(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
