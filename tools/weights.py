#!/usr/bin/env python3
"""Inspect and maintain the weight of memory.

    python tools/weights.py bootstrap --endpoint http://127.0.0.1:1236/sparql
    python tools/weights.py decay
    python tools/weights.py report
    python tools/weights.py conflicts

`bootstrap` reads every triple in the graph into the weight store. Seeded or
imported facts start at the base confidence (0.5): the agent did not witness
them, so they should not outrank something the user said yesterday.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.client import SparqlClient  # noqa: E402
from samskara_core.facts import FactStore, is_literal  # noqa: E402


def store_path(hermes_home: Path) -> Path:
    return hermes_home / "samskara" / "facts.db"


def bootstrap(store: FactStore, endpoint: str, limit: int = 50000, confidence: float = 0.5) -> int:
    client = SparqlClient(endpoint)
    if not client.healthy():
        sys.exit(f"graph unreachable at {endpoint}")
    rows, ms = client.select(f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {int(limit)}")
    triples = [
        (row.get("s", ""), row.get("p", ""), row.get("o", ""))
        for row in rows
        if row.get("s") and row.get("p") and row.get("o")
    ]
    new = store.record(triples, confidence=confidence)
    print(f"read {len(triples)} triple(s) in {ms:.1f} ms · {new} new in the weight store")
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage Samskara's memory weights.")
    ap.add_argument("command", choices=["bootstrap", "decay", "report", "conflicts"], nargs="?", default="report")
    ap.add_argument("--hermes-home", default="", help="Default: $HERMES_HOME or ~/.hermes")
    ap.add_argument("--endpoint", default="", help="Graph endpoint (default: $SAMSKARA_ENDPOINT)")
    ap.add_argument("--confidence", type=float, default=0.5, help="Base confidence for imported facts")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    home = Path(args.hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    store = FactStore(store_path(home))

    if args.command == "bootstrap":
        endpoint = args.endpoint or os.environ.get("SAMSKARA_ENDPOINT", "")
        if not endpoint:
            sys.exit("no endpoint: pass --endpoint or set SAMSKARA_ENDPOINT")
        bootstrap(store, endpoint, confidence=args.confidence)

    elif args.command == "decay":
        touched = store.decay_all()
        print(f"recomputed weights · {touched} fact(s) changed")

    elif args.command == "conflicts":
        rows = store.conflicts(limit=100)
        if not rows:
            print("no conflicts — every subject/predicate pair has a single value")
            return 0
        print(f"{len(rows)} fact(s) in conflict:")
        for fact in rows:
            print(f"  {fact.subject.split('/')[-1]}  {fact.predicate.split('/')[-1]} = {fact.object[:70]}")
            print(f"      peso {fact.weight:.2f} · {fact.sessions} sesión(es) · confianza {fact.confidence:.2f}")

    else:
        stats = store.stats()
        print("=== memoria ===")
        print(f"  hechos     : {stats['facts']}")
        print(f"  conflictos : {stats['conflicts']}")
        print(f"  peso medio : {stats['avg_weight']}  (máx {stats['max_weight']})")
        print(f"  usados     : {stats['ever_used']}")
        print("\n=== los que más pesan ===")
        for fact in store.top(args.limit):
            print(f"  {fact.weight:.2f}  {fact.subject.split('/')[-1]} —{fact.predicate.split('/')[-1]}→ {fact.object[:52]}")
        print("\n=== los que se están olvidando ===")
        for fact in store.weakest(min(5, args.limit)):
            print(f"  {fact.weight:.2f}  {fact.subject.split('/')[-1]} —{fact.predicate.split('/')[-1]}→ {fact.object[:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
