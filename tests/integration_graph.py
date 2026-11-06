"""Integration check: talk to a real MillenniumDB through the sampler graph.

    python3 integration_graph.py [endpoint]

Not a unit test — this one needs a live graph. Kept in the repo because the
hackathon demo is exactly this: a question that cannot be answered by
similarity search.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from samskara_core import SparqlClient, recall  # noqa: E402

QUESTIONS = [
    "¿Puedo subir PostgreSQL a la 17 en Onboarding Hub?",
    "¿Qué pasó con el upgrade de postgres?",
    "¿Qué proyectos dependen de la misma base de datos?",
]


def main() -> int:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234/sparql"
    client = SparqlClient(endpoint, timeout=8.0)
    print(f"endpoint : {client.endpoint}")
    print(f"healthy  : {client.healthy()}")
    if not client.healthy():
        return 1
    rows, ms = client.select("SELECT (COUNT(*) AS ?n) WHERE { ?e rdfs:label ?l }")
    print(f"entities : {rows[0].get('n') if rows else '?'} (count in {ms:.2f} ms)")
    for question in QUESTIONS:
        block, elapsed = recall(client, question)
        print("\n" + "─" * 72)
        print(f"Q: {question}")
        print(block if block else "  (no entities matched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
