#!/usr/bin/env python3
"""Fold the Samskara journal into the graph.

MillenniumDB (open-source build) is import-only, so memory writes are journalled
and applied here: dump the graph, concatenate the pending journals, rebuild, and
archive what was applied.

Downtime is the rebuild of a personal knowledge graph — seconds, not minutes —
and the journal is fsync'd, so nothing is lost if this is interrupted halfway:
worst case you re-run it, because applied journals are only archived at the end.

Usage:
    python tools/reindex.py --data-dir /srv/samskara \
                            --journal-dir ~/.hermes/samskara/journal \
                            [--container samskara-mdb] [--dry-run]

Layout assumed (the same one the README tells users to create):
    <data-dir>/db            <- MillenniumDB folder, mounted at /data in the container
    <data-dir>/journal       <- default journal dir if --journal-dir is omitted
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samskara_core.ontology import Ontology, materialise_nt  # noqa: E402

DEFAULT_IMAGE = "imfd/millenniumdb:latest"


def run(cmd: list[str], dry: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    if dry:
        print("  $ " + " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def docker_mdb(data_dir: Path, command: str, image: str, dry: bool, extra_volumes: dict | None = None) -> None:
    """Run `mdb <command>` inside the image with <data-dir> mounted at /data."""
    volumes = ["-v", f"{data_dir}:/data"]
    for host, container in (extra_volumes or {}).items():
        volumes += ["-v", f"{host}:{container}"]
    cmd = ["docker", "run", "--rm", "--entrypoint", "bash", *volumes, image, "-c", command]
    result = run(cmd, dry=dry)
    if not dry and result.returncode != 0:
        sys.exit(f"mdb command failed: {result.stderr.strip()[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fold pending Samskara journal triples into the graph.")
    ap.add_argument("--data-dir", required=True, help="Host directory mounted at /data (contains db/)")
    ap.add_argument("--journal-dir", default="", help="Journal directory (default: <data-dir>/journal)")
    ap.add_argument("--container", default="", help="Running server container to stop/start (optional)")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--keep-merged", action="store_true", help="Keep the merged .nt file after import")
    ap.add_argument("--no-infer", action="store_true", help="Skip ontology materialisation (types closure)")
    ap.add_argument("--force", action="store_true", help="Rebuild even with no pending journals (e.g. after editing the ontology)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    journal_dir = Path(args.journal_dir).resolve() if args.journal_dir else data_dir / "journal"
    db_dir = data_dir / "db"
    stamp = time.strftime("%Y%m%d-%H%M%S")

    journals = sorted(journal_dir.glob("session-*.ttl")) if journal_dir.is_dir() else []
    pending = sum(
        sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()) for p in journals
    )
    print(f"data-dir   : {data_dir}")
    print(f"journal-dir: {journal_dir}")
    print(f"journals   : {len(journals)} file(s), {pending} pending triple(s)")

    if not pending and not args.force:
        print("nothing to do — the graph is already up to date (use --force to re-run inference)")
        return 0
    if not db_dir.is_dir():
        sys.exit(f"no database at {db_dir} — import a corpus first (see README)")

    # 1. dump the current graph (read the live db, no writes)
    dump_prefix = f"/data/reindex-{stamp}"
    print(f"\n[1/5] dumping graph")
    docker_mdb(data_dir, f"mdb dump /data/db {dump_prefix} nt", args.image, args.dry_run)
    dump_file = data_dir / f"reindex-{stamp}.nt"

    # 2. merge dump + journals
    print(f"[2/5] merging {len(journals)} journal(s) into the dump")
    merged = data_dir / f"reindex-{stamp}-merged.nt"
    if args.dry_run:
        print(f"  $ cat {dump_file} {journal_dir}/session-*.ttl > {merged}")
        merged_lines = total_lines = 0
    else:
        with open(merged, "w", encoding="utf-8") as out:
            out.write(dump_file.read_text(encoding="utf-8"))
            if not dump_file.read_text(encoding="utf-8").endswith("\n"):
                out.write("\n")
            for journal in journals:
                out.write(journal.read_text(encoding="utf-8"))
        if not args.no_infer:
            ontology = Ontology()
            before, added = materialise_nt(merged.read_text(encoding="utf-8"), ontology)
            if added:
                merged.write_text(before, encoding="utf-8")
                print(f"      + {added} inferred triple(s) from {ontology.describe()}")
        merged_lines = len(merged.read_text(encoding="utf-8").splitlines())
        total_lines = len(dump_file.read_text(encoding="utf-8").splitlines())

    # 3. stop the server so the db folder can be replaced
    if args.container:
        print(f"[3/5] stopping {args.container}")
        run(["docker", "stop", args.container], dry=args.dry_run, check=False)
        if not args.dry_run:
            time.sleep(2)

    # 4. rebuild: retire the old db folder, import the union
    print("[4/5] rebuilding database from the union")
    retired = data_dir / f"db.pre-reindex-{stamp}"
    if not args.dry_run:
        db_dir.rename(retired)
    try:
        docker_mdb(
            data_dir,
            f"mdb import /data/{merged.name} /data/db --format nt",
            args.image,
            args.dry_run,
        )
    except SystemExit:
        # Never leave the agent without memory: put the previous database back.
        print("import failed — restoring the previous database")
        if not args.dry_run:
            if db_dir.exists():
                shutil.rmtree(db_dir, ignore_errors=True)
            retired.rename(db_dir)
            if args.container:
                run(["docker", "start", args.container], check=False)
        raise

    # 5. start the server, archive applied journals
    if args.container:
        print(f"[5/5] starting {args.container}")
        run(["docker", "start", args.container], dry=args.dry_run, check=False)
    if not args.dry_run:
        applied = journal_dir / "applied"
        applied.mkdir(exist_ok=True)
        for journal in journals:
            journal.rename(applied / journal.name)
        if not args.keep_merged:
            merged.unlink(missing_ok=True)
            dump_file.unlink(missing_ok=True)

    print(
        f"\ndone: {total_lines} -> {merged_lines} triple(s) "
        f"(+{merged_lines - total_lines} from the journal)"
    )
    print(f"previous database kept at {retired if not args.dry_run else '(dry-run)'} — delete it once verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
