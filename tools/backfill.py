#!/usr/bin/env python3
"""Build the graph from the sessions the agent already had.

Samskara's onboarding is not "wait three weeks until you have history": on the
first run this reads `$HERMES_HOME/state.db`, extracts durable relations from
every past session, journals them and lets `reindex` fold them into the graph.
After that, each session close adds the new facts only.

It is idempotent: processed sessions are recorded in a state file, so re-running
never duplicates work or triples.

Disclosure: this reads your past sessions from `state.db` (read-only) and sends
their text to the model endpoint you configure here — `SAMSKARA_LLM_BASE_URL`
plus `SAMSKARA_LLM_API_KEY` (or `--base-url`). There is no default host: if you
do not name an endpoint, the script refuses to run rather than guessing one.

Usage:
    python tools/backfill.py --hermes-home ~/.hermes [--limit 20] [--dry-run]

    # then, exactly as the journal always works:
    python tools/reindex.py --data-dir /srv/samskara --container samskara-mdb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core import Journal  # noqa: E402
from samskara_core.extract import Extractor  # noqa: E402
from samskara_core.llm import cli_setup_help, make_cli_llm  # noqa: E402
from samskara_core.client import SparqlClient  # noqa: E402
from samskara_core.reconcile import LabelIndex  # noqa: E402
from samskara_core.schema import SchemaRegistry  # noqa: E402
from samskara_core.sessions import SessionReader, already_done, mark_done  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill the Samskara graph from past Hermes sessions.")
    ap.add_argument("--hermes-home", default="", help="Default: $HERMES_HOME or ~/.hermes")
    ap.add_argument("--db", default="", help="state.db path (default: <hermes-home>/state.db)")
    ap.add_argument("--journal-dir", default="", help="Default: <hermes-home>/samskara/journal")
    ap.add_argument("--schema", default="", help="Default: <hermes-home>/samskara/schema.json")
    ap.add_argument("--limit", type=int, default=0, help="Only the N oldest unprocessed sessions")
    ap.add_argument("--min-messages", type=int, default=2)
    ap.add_argument("--model", default="", help="Override the model for extraction")
    ap.add_argument("--base-url", default="", help="Override the OpenAI-compatible base URL")
    ap.add_argument("--include-archived", action="store_true")
    ap.add_argument("--endpoint", default="", help="Graph endpoint to reconcile against (default: $SAMSKARA_ENDPOINT)")
    ap.add_argument("--no-bridge", action="store_true", help="Skip the semantic reconciliation pass")
    ap.add_argument("--dry-run", action="store_true", help="Extract but write nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import os

    home = Path(args.hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    db = Path(args.db) if args.db else home / "state.db"
    journal_dir = Path(args.journal_dir) if args.journal_dir else home / "samskara" / "journal"
    schema_path = Path(args.schema) if args.schema else home / "samskara" / "schema.json"
    state_path = home / "samskara" / "backfill-state.json"

    if not db.is_file():
        sys.exit(f"no session database at {db} — has Hermes run at least once?")
    if not journal_dir.is_dir():
        journal_dir.mkdir(parents=True, exist_ok=True)

    registry = SchemaRegistry(schema_path)
    # CLI path: an explicit endpoint and an explicit key, named SAMSKARA_*-only.
    # A key you exported for another provider is never presented to this one.
    llm = make_cli_llm(
        base_url=args.base_url or None,
        model=args.model or None,
    )
    if not llm.available:
        sys.exit(
            "no model configured: "
            + llm.unavailable_reason
            + "\n  "
            + "\n  ".join(cli_setup_help())
        )

    index = LabelIndex()
    endpoint = args.endpoint or os.environ.get("SAMSKARA_ENDPOINT", "")
    if endpoint:
        index = index.load_from_client(SparqlClient(endpoint))
    extractor = Extractor(registry, llm, index=index, bridge=not args.no_bridge)
    journal = Journal(journal_dir, session_id="backfill")
    state = SessionReader.load_state(state_path)
    reader = SessionReader(db, min_messages=args.min_messages)

    total_triples = processed = skipped = 0
    started = time.time()
    print(f"session db : {db}")
    print(f"model      : {getattr(llm, 'model', 'host model')}")
    print(f"journal    : {journal_dir}")
    print(f"already    : {len(state.get('sessions') or {})} session(s) processed before")
    print(f"known nodes: {len(index)} label(s) to reconcile against")
    print(f"bridge     : {'off' if args.no_bridge else 'on'}\n")

    for ctx in reader.sessions(limit=0, include_archived=args.include_archived):
        if already_done(state, ctx.session_id):
            skipped += 1
            continue
        if args.limit and processed >= args.limit:
            break
        result = extractor.extract(ctx)
        processed += 1
        title = (ctx.title or ctx.session_id)[:52]
        if result.count:
            if not args.dry_run:
                journal.append(result.triples)
                mark_done(state, ctx.session_id, result.count)
            total_triples += result.count
            print(f"  + {result.count:3d} triple(s)  {title}")
            if args.verbose:
                for triple in result.triples[:8]:
                    print(f"        {triple[0].split('/')[-1]} —{triple[1].split('/')[-1]}→ {str(triple[2])[:60]}")
                if len(result.triples) > 8:
                    print(f"        … {len(result.triples) - 8} more")
        else:
            if not args.dry_run:
                mark_done(state, ctx.session_id, 0)
            print(f"  · 0 triple(s)  {title}   ({', '.join(result.rejected[:2]) or 'no durable facts'})")

    if not args.dry_run:
        registry.save()
        SessionReader.save_state(state_path, state)

    elapsed = time.time() - started
    print(
        f"\ndone in {elapsed:.1f}s — {processed} session(s) processed, {skipped} skipped, "
        f"{total_triples} triple(s) journalled"
    )
    if total_triples and not args.dry_run:
        print("next: fold the journal into the graph")
        print(f"  python tools/reindex.py --data-dir <data-dir> --journal-dir {journal_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
