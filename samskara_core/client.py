"""Minimal SPARQL-over-HTTP client for MillenniumDB. Standard library only.

Why no third-party deps: a memory provider runs on every turn of every
session. Fewer imports = fewer ways for someone else's upgrade to break
the agent's memory.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

# Ontology prefixes exported for query builders and for the user's own queries.
PREFIXES = {
    "ws": "https://wsolutionsai.com/ns/shiva/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

PREFIX_HEADER = "\n".join(f"PREFIX {p}: <{u}>" for p, u in PREFIXES.items())


class SparqlError(RuntimeError):
    """Raised when the graph endpoint rejects a query or is unreachable."""


class SparqlClient:
    """A small, synchronous SPARQL client.

    Deliberately synchronous: MillenniumDB answers the demo's 3-hop query in
    well under a millisecond, so a thread pool would add latency, not remove it.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 8.0,
        username: str = "",
        password: str = "",
    ) -> None:
        self.endpoint = (endpoint or "").strip().rstrip("/")
        if self.endpoint and not self.endpoint.endswith("/sparql"):
            self.endpoint += "/sparql"
        self.timeout = float(timeout or 8.0)
        self._auth = ""
        if username:
            raw = (username + ":" + password).encode("utf-8")
            self._auth = "Basic " + base64.b64encode(raw).decode("ascii")
        self.last_ms: float = 0.0
        self.queries: int = 0

    # -- transport ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        if self.endpoint.endswith("/sparql"):
            return self.endpoint[: -len("/sparql")]
        return self.endpoint

    def _request(self, url: str, data: Optional[bytes] = None) -> bytes:
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if data:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            req.add_header("Accept", "application/sparql-results+json")
        if self._auth:
            req.add_header("Authorization", self._auth)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise SparqlError(f"HTTP {exc.code} from graph: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise SparqlError(f"graph unreachable at {self.base_url}: {exc}") from exc

    # -- API ---------------------------------------------------------------
    def healthy(self) -> bool:
        """Cheap liveness check. Used by the status command and the demo."""
        if not self.endpoint:
            return False
        try:
            self._request(f"{self.base_url}/health")
            return True
        except SparqlError:
            pass
        # Not every MillenniumDB build (and almost no other SPARQL store) exposes
        # /health: fall back to the cheapest query that proves the store answers.
        try:
            self.select("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
            return True
        except SparqlError:
            return False

    def select(self, query: str, use_prefixes: bool = True) -> Tuple[List[Dict[str, str]], float]:
        """Run a SELECT/ASK and return (rows, elapsed_ms).

        Rows are flattened: {"proyecto": "http://...", "label": "Onboarding Hub"}.
        """
        full = f"{PREFIX_HEADER}\n{query}" if use_prefixes else query
        payload = urllib.parse.urlencode({"query": full}).encode()
        started = time.perf_counter()
        raw = self._request(self.endpoint, payload)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        self.queries += 1
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise SparqlError(f"non-JSON response from graph: {raw[:200]!r}") from exc
        rows: List[Dict[str, str]] = []
        for binding in doc.get("results", {}).get("bindings", []):
            rows.append({var: cell.get("value", "") for var, cell in binding.items()})
        return rows, self.last_ms
