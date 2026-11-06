#!/usr/bin/env python3
"""Samskara in one command, with nothing installed.

    python tools/demo.py

No graph server, no Docker, no network, no API key. The corpus is parsed in
process, the weights live in a temporary SQLite sidecar, and everything you see
below is the real pipeline: the same `recall()` Hermes calls before a turn.

What it shows, in order:
  1. a question that resolves to real entities and pulls their neighbourhood;
  2. the same question again — the facts it used got heavier (Hebbian, capped);
  3. a contradicting fact taught afterwards — the answer stops being clean.

    python tools/demo.py --keep    # leave the temp dir on disk to poke at it
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.embedded import EmbeddedClient  # noqa: E402
from samskara_core.facts import FactStore  # noqa: E402
from samskara_core.recall import recall  # noqa: E402

NS = "https://wsolutionsai.com/ns/shiva/"
RULE = "─" * 74


def _block(block: str, elapsed: float) -> str:
    if not block:
        return "  (nothing recalled — no entity in the question matched the corpus)"
    return block + f"\n  ↳ {elapsed:.2f} ms, in-process"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="seed/demo.ttl", help="Turtle corpus to load")
    parser.add_argument("--keep", action="store_true", help="keep the temporary store")
    parser.add_argument("--no-prime", action="store_true", help="skip seeding weights")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="samskara-demo-"))
    client = EmbeddedClient.from_seed(args.seed, base=str(REPO))
    store = FactStore(workdir / "facts.db")

    entities = len(client.graph._labels)
    print(RULE)
    print("Samskara — relational memory that admits doubt")
    print(RULE)
    print(f"corpus      {args.seed} · {len(client.graph)} triples · {entities} labelled entities")
    print(f"backend     {client.endpoint} (in-process, read-only, zero dependencies)")
    print(f"weights     {workdir / 'facts.db'}")
    print()

    if not args.no_prime:
        # Everything in the seed is an import, so it enters at the lowest trust
        # tier: confidence 0.5 -> weight 0.45. Explicit writes weigh 0.90,
        # extraction 0.65. Trust is a hierarchy, not a constant.
        store.record(list(client.graph.triples), session_id="import", confidence=0.5)
        print(f"primed      {store.stats().get('facts', 0)} facts at import confidence (weight 0.45)\n")

    w_import = store.weight((f"{NS}onboarding_hub", f"{NS}usaTecnologia", f"{NS}postgresql"))

    question = "¿puedo subir PostgreSQL a la 17 en Onboarding Hub?"
    watched = (f"{NS}onboarding_hub", f"{NS}usaTecnologia", f"{NS}postgresql")

    print(RULE)
    print(f"1) {question}")
    print(RULE)
    block, elapsed = recall(client, question, store=store)
    print(_block(block, elapsed))
    print()
    w_first = store.weight(watched)

    print(RULE)
    print("2) the same question again — memory that is used gets heavier")
    print(RULE)
    block, elapsed = recall(client, question, store=store)
    w_second = store.weight(watched)
    print(_block(block, elapsed))
    recall(client, question, store=store)
    w_third = store.weight(watched)
    print(f"\n  Onboarding Hub → usa → PostgreSQL, recall after recall:")
    print(f"    imported {w_import:.2f}  →  used once {w_first:.2f}  →  used twice {w_second:.2f}"
          f"  →  used three times {w_third:.2f}")
    print("  It climbs while someone keeps asking, and stops at +0.20 over its base.")
    print("  A fact nobody recalls stays at 0.45 and starts decaying after 30 idle")
    print("  days. Memory with a pulse.\n")

    print(RULE)
    print("3) now teach the opposite and ask again")
    print(RULE)
    decision = f"{NS}dec_congelar_pg"
    state = f"{NS}estado"
    print(f'  write: dec_congelar_pg · estado = "descongelado"  (conflict, not overwrite)')
    store.record([(decision, state, "descongelado")], session_id="demo", confidence=0.65)
    block, elapsed = recall(client, "¿sigue vigente el congelamiento de PostgreSQL?", store=store)
    print()
    print(_block(block, elapsed))
    print()

    print(RULE)
    print("That is the whole plugin: no server, no key, no network.")
    print("For a real corpus, point it at a graph and nothing above changes:")
    print()
    print("    export SAMSKARA_ENDPOINT=http://127.0.0.1:1234/sparql")
    print("    hermes plugins install SebaWag/hermes-samskara && hermes memory status")
    print(RULE)

    store.close()
    if args.keep:
        print(f"\nkept: {workdir}")
    else:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
