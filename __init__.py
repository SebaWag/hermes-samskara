"""Samskara — relational memory for Hermes Agent.

Most memory providers store *text* and search it by similarity. Samskara stores
*relations* and searches them by path: project → incident → decision. That is
the difference between "here are 5 similar paragraphs" and "you cannot upgrade
PostgreSQL on Onboarding Hub, because a decision from March is still in force
and here is the incident that caused it".

Every recall block Samskara injects carries its own provenance: how many facts,
how many hops, and how many milliseconds it took the graph to answer.

Layout (Hermes memory-provider plugin):
    $HERMES_HOME/plugins/samskara/
        __init__.py      <- this file: MemoryProvider + register(ctx)
        samskara_core/   <- transport, recall, journal (stdlib only)
        plugin.yaml
        README.md

Author: Wagner Solutions (Sebastian Wagner). License: MIT.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Hermes runtime
    from agent.memory_provider import MemoryProvider
except Exception:  # pragma: no cover - lets the module be imported for tests
    class MemoryProvider:  # type: ignore[no-redef]
        """Stand-in ABC so unit tests can import this plugin outside Hermes."""

# The plugin is loaded by path, not as an installed package: make the sibling
# core package importable regardless of how Hermes discovered us.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from samskara_core import (  # noqa: E402  (import after sys.path bootstrap)
    Journal,
    SparqlClient,
    SparqlError,
    find_entities,
    local_name,
    recall as run_recall,
    reindex_commands,
)
from samskara_core.extract import DEFAULT_NAMESPACE, Extractor  # noqa: E402
from samskara_core.llm import make_llm  # noqa: E402
from samskara_core.reconcile import LabelIndex  # noqa: E402
from samskara_core.schema import SchemaRegistry  # noqa: E402
from samskara_core.sessions import SessionContext  # noqa: E402

# Set by register(ctx): lets the provider use the host's model (ctx.llm)
# without ever touching a credential.
_CTX: Any = None

log = logging.getLogger("samskara")

DEFAULTS: Dict[str, Any] = {
    "endpoint": "http://127.0.0.1:1234/sparql",  # MillenniumDB default port
    "timeout": 8.0,
    "max_entities": 3,
    "max_facts": 24,
    "max_chars": 1200,
    "auto_recall": True,
    "write_mode": "journal",
    "extract_on_session_end": True,
    "username": "",
    "password": "",
}

CONFIG_FILE = "samskara.json"


class SamskaraMemoryProvider(MemoryProvider):
    """MemoryProvider backed by a MillenniumDB knowledge graph."""

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = dict(DEFAULTS)
        self._client: Optional[SparqlClient] = None
        self._journal: Optional[Journal] = None
        self._registry: Optional[SchemaRegistry] = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._turns: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "recalls": 0,
            "hits": 0,
            "last_ms": 0.0,
            "journal_triples": 0,
            "errors": 0,
        }

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "samskara"

    def is_available(self) -> bool:
        """Config check only — never a network call (Hermes calls this on init)."""
        endpoint = str(self._cfg.get("endpoint") or "")
        return endpoint.startswith(("http://", "https://"))

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._hermes_home = str(kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or "")
        self._cfg.update(self._load_config(self._hermes_home))
        self._client = SparqlClient(
            self._cfg["endpoint"],
            timeout=self._cfg["timeout"],
            username=self._cfg["username"],
            password=self._cfg["password"],
        )
        self._journal = Journal(Path(self._journal_dir()), session_id=self._session_id)
        # The vocabulary is per-user and grows with their data.
        schema_path = Path(self._journal_dir()).parent / "schema.json"
        self._registry = SchemaRegistry(schema_path)
        log.info(
            "samskara ready: endpoint=%s session=%s journal=%s",
            self._client.endpoint,
            self._session_id or "-",
            self._journal.path,
        )

    def shutdown(self) -> None:
        """Flush nothing, lose nothing: the journal is already fsync'd per append."""
        with self._lock:
            self._turns.clear()

    def system_prompt_block(self) -> str:
        return (
            "You have relational memory (samskara) backed by a knowledge graph.\n"
            "Use samskara_search to resolve a name to real entities, and samskara_query\n"
            "to follow relations (project -> incident -> decision) with SPARQL.\n"
            "Use samskara_write to persist a fact you want to remember across sessions.\n"
            "Prefer one precise multi-hop query over several similarity searches.\n"
            "When you state a fact that came from memory, say where it came from."
        )

    # -- recall ------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Context injected before the model answers. Must never raise."""
        if not self._cfg.get("auto_recall", True) or not self._client or not (query or "").strip():
            return ""
        try:
            block, _ = run_recall(
                self._client,
                query,
                max_entities=int(self._cfg["max_entities"]),
                max_facts=int(self._cfg["max_facts"]),
                max_chars=int(self._cfg["max_chars"]),
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break a turn
            self._stats["errors"] += 1
            log.debug("prefetch failed: %s", exc)
            return ""
        with self._lock:
            self._stats["recalls"] += 1
            self._stats["last_ms"] = self._client.last_ms
            if block:
                self._stats["hits"] += 1
        return block

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No background thread needed: a recall is sub-millisecond here."""
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Buffer the turn. Extraction happens at compress/session end, not here,
        so that writing memory never slows the conversation down."""
        with self._lock:
            self._turns.append({"user": user_content or "", "assistant": assistant_content or ""})
            if len(self._turns) > 12:
                self._turns = self._turns[-12:]

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        return None

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """The part that makes this relational memory and not a summary.

        Before Hermes compresses the context away, we walk the turn text, resolve
        the entities it mentions against the graph and journal them against the
        session node. Compression becomes lossy for prose and lossless for
        relations — the graph keeps what the transcript is about to forget.
        """
        text = self._messages_to_text(messages) or self._buffered_text()
        if not text.strip() or not self._client:
            return ""
        try:
            entities = find_entities(self._client, text, limit=8)
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            log.debug("pre-compress extraction failed: %s", exc)
            return ""
        if not entities:
            return ""
        session_node = self._session_node()
        triples = [(session_node, "ws:menciona", entity.uri) for entity in entities]
        written = self._append(triples)
        labels = ", ".join(entity.label for entity in entities[:5])
        return (
            f"[samskara] {written} relationship(s) preserved in the graph before "
            f"compression: {labels}"
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract durable relations from the session that just ended.

        This is what makes the graph grow by itself: the model reads the
        conversation, proposes typed triples against a learnable vocabulary, and
        we journal them. Validation happens in `extract.validate` — unknown
        predicates are proposed, not accepted, and every edge keeps provenance.
        """
        if not self._cfg.get("extract_on_session_end", True):
            return
        pairs = [
            (m.get("role", ""), m.get("content", ""))
            for m in (messages or [])
            if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str) and m["content"].strip()
        ]
        if not pairs:
            return

        llm = make_llm(_CTX)
        triples: List[tuple] = []
        if getattr(llm, "available", False):
            try:
                ctx = SessionContext(session_id=self._session_id or "session", messages=pairs)
                index = LabelIndex().load_from_client(self._client) if self._client else LabelIndex()
                result = Extractor(self._registry, llm, DEFAULT_NAMESPACE, index=index).extract(ctx)
                triples = result.triples
                if result.rejected:
                    log.debug("samskara: %d item(s) rejected during extraction", len(result.rejected))
                if result.proposed:
                    log.info("samskara: new predicates registered: %s", ", ".join(result.proposed))
                self._registry.save()
            except Exception as exc:  # noqa: BLE001 - never break session close
                self._stats["errors"] += 1
                log.debug("extraction failed: %s", exc)

        # Fallback (and complement): at minimum, record what the session was about.
        if self._client:
            try:
                entities = find_entities(self._client, "\n".join(c for _, c in pairs), limit=10)
                session_node = self._session_node()
                triples.extend((session_node, "ws:menciona", e.uri) for e in entities)
            except Exception:  # noqa: BLE001
                pass

        if triples:
            written = self._append(triples)
            log.info("samskara: %d triple(s) journalled at session end", written)
            self._stats["extracted"] = self._stats.get("extracted", 0) + written

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """Keep the delegation edge: who asked whom, for what."""
        parent = self._session_node()
        child = f"ws:session_{self._slug(child_session_id or 'child')}"
        self._append(
            [
                (parent, "ws:delegoA", child),
                (child, "ws:tarea", (task or "")[:280]),
            ]
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id or self._session_id
        if self._journal is not None:
            self._journal = Journal(Path(self._journal_dir()), session_id=self._session_id)
        # The vocabulary is per-user and grows with their data.
        schema_path = Path(self._journal_dir()).parent / "schema.json"
        self._registry = SchemaRegistry(schema_path)
        with self._lock:
            self._turns.clear()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror durable MEMORY.md/USER.md writes as graph events."""
        self._append(
            [
                (self._session_node(), "ws:memoriaEscrita", f"{action or 'write'} {target or ''}".strip()),
            ]
        )

    def backup_paths(self) -> List[str]:
        return [self._journal_dir()]

    # -- tools -------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "samskara_search",
                    "description": (
                        "Resolve names mentioned by the user to real entities in the "
                        "relational memory graph. Returns each entity with its facts "
                        "in both directions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Names or topic to look up."},
                            "limit": {"type": "integer", "description": "Max entities (default 3)."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "samskara_query",
                    "description": (
                        "Run a read-only SPARQL query against the memory graph. Use this "
                        "to follow relations across hops (project -> incident -> decision)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sparql": {"type": "string", "description": "A SELECT or ASK query."},
                        },
                        "required": ["sparql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "samskara_write",
                    "description": (
                        "Persist one fact as a triple. Written to the append-only journal "
                        "and folded into the graph on the next reindex."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "e.g. ws:onboarding_hub"},
                            "predicate": {"type": "string", "description": "e.g. ws:usaTecnologia"},
                            "object": {
                                "type": "string",
                                "description": "A prefixed name/URI, or a plain string literal.",
                            },
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "samskara_status",
                    "description": "Health and telemetry of the relational memory backend.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        if tool_name == "samskara_search":
            return self._tool_search(str(args.get("query", "")), int(args.get("limit", 3) or 3))
        if tool_name == "samskara_query":
            return self._tool_query(str(args.get("sparql", "")))
        if tool_name == "samskara_write":
            return self._tool_write(
                str(args.get("subject", "")),
                str(args.get("predicate", "")),
                str(args.get("object", "")),
            )
        if tool_name == "samskara_status":
            return self._tool_status()
        raise NotImplementedError(f"samskara does not handle tool {tool_name}")

    # -- config ------------------------------------------------------------
    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "endpoint",
                "label": "Graph endpoint",
                "type": "text",
                "default": DEFAULTS["endpoint"],
                "help": "SPARQL endpoint. A base URL gets /sparql appended automatically.",
            },
            {
                "name": "auto_recall",
                "label": "Recall on every turn",
                "type": "bool",
                "default": True,
                "help": "Inject relational context before the model answers.",
            },
            {
                "name": "max_entities",
                "label": "Max entities per recall",
                "type": "number",
                "default": 3,
                "help": "How many matched entities to expand. More = more context tokens.",
            },
            {
                "name": "timeout",
                "label": "Query timeout (s)",
                "type": "number",
                "default": 8.0,
                "help": "Hard ceiling for one SPARQL call.",
            },
            {
                "name": "username",
                "label": "Basic auth user",
                "type": "text",
                "default": "",
                "help": "For endpoints behind a proxy that requires auth.",
            },
            {
                "name": "password",
                "label": "Basic auth password",
                "type": "secret",
                "default": "",
                "help": "Stored in $HERMES_HOME/samskara.json, never logged.",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home or self._hermes_home or ".") / CONFIG_FILE
        merged = dict(self._cfg)
        merged.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self._cfg = merged

    # -- internals ---------------------------------------------------------
    def _load_config(self, hermes_home: str) -> Dict[str, Any]:
        found: Dict[str, Any] = {}
        if hermes_home:
            path = Path(hermes_home) / CONFIG_FILE
            if path.is_file():
                try:
                    found.update(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError) as exc:
                    log.debug("ignoring unreadable %s: %s", path, exc)
        env_map = {
            "SAMSKARA_ENDPOINT": "endpoint",
            "SAMSKARA_TIMEOUT": "timeout",
            "SAMSKARA_USERNAME": "username",
            "SAMSKARA_PASSWORD": "password",
            "SAMSKARA_AUTO_RECALL": "auto_recall",
        }
        for env_key, cfg_key in env_map.items():
            value = os.environ.get(env_key)
            if value:
                found[cfg_key] = value
        if "auto_recall" in found and isinstance(found["auto_recall"], str):
            found["auto_recall"] = found["auto_recall"].lower() not in {"0", "false", "no"}
        return found

    def _journal_dir(self) -> str:
        base = self._hermes_home or os.path.expanduser("~/.hermes")
        return str(Path(base) / "samskara" / "journal")

    def _session_node(self) -> str:
        return f"ws:session_{self._slug(self._session_id or 'anon')}"

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value or "")[:64] or "anon"

    def _append(self, triples: List[tuple]) -> int:
        if self._journal is None or not triples:
            return 0
        try:
            written = self._journal.append(triples)
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            log.debug("journal append failed: %s", exc)
            return 0
        self._stats["journal_triples"] += written
        return written

    def _buffered_text(self) -> str:
        with self._lock:
            turns = list(self._turns)
        return "\n".join(f"{t['user']}\n{t['assistant']}" for t in turns)

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for message in messages or []:
            role = message.get("role", "")
            content = message.get("content", "")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            parts.append(content)
        return "\n".join(parts)[-6000:]

    # -- tool implementations ---------------------------------------------
    def _tool_search(self, query: str, limit: int) -> str:
        if not self._client:
            return "samskara: not initialized"
        try:
            block, elapsed = run_recall(self._client, query, max_entities=max(1, limit), max_facts=24)
        except SparqlError as exc:
            self._stats["errors"] += 1
            return f"samskara: graph error: {exc}"
        if not block:
            return f"samskara: no entity matched '{query}'."
        return block

    def _tool_query(self, sparql: str) -> str:
        if not self._client:
            return "samskara: not initialized"
        head = (sparql or "").strip().lower()
        if not (head.startswith("select") or head.startswith("prefix") or head.startswith("ask")):
            return "samskara: only SELECT/ASK queries are allowed (the graph is read-only here)."
        try:
            rows, elapsed = self._client.select(sparql)
        except SparqlError as exc:
            self._stats["errors"] += 1
            return f"samskara: query error: {exc}"
        if not rows:
            return f"samskara: 0 rows in {elapsed:.2f} ms."
        keys = list(rows[0].keys())
        lines = [f"samskara: {len(rows)} row(s) in {elapsed:.2f} ms"]
        lines.append(" | ".join(keys))
        for row in rows[:25]:
            lines.append(" | ".join(str(row.get(k, "")) for k in keys))
        if len(rows) > 25:
            lines.append(f"… {len(rows) - 25} more row(s)")
        return "\n".join(lines)

    def _tool_write(self, subject: str, predicate: str, obj: str) -> str:
        if not subject or not predicate or not obj:
            return "samskara: write needs subject, predicate and object."
        written = self._append([(subject, predicate, obj)])
        if not written:
            return "samskara: write rejected (journal unavailable)."
        pending = 0
        if self._journal is not None:
            try:
                pending = self._journal.pending()
            except OSError:
                pending = 0
        return (
            f"samskara: fact queued ({subject} {local_name(predicate)} {obj}). "
            f"{pending} triple(s) pending reindex."
        )

    def _tool_status(self) -> str:
        healthy = self._client.healthy() if self._client else False
        payload = {
            "provider": "samskara",
            "endpoint": self._client.endpoint if self._client else "",
            "healthy": healthy,
            "session": self._session_id or None,
            "journal_dir": self._journal_dir(),
            "stats": dict(self._stats),
        }
        if self._client and healthy:
            try:
                rows, elapsed = self._client.select(
                    "SELECT (COUNT(*) AS ?n) WHERE { ?e rdfs:label ?l }"
                )
                payload["entities"] = rows[0].get("n") if rows else "0"
                payload["count_ms"] = round(elapsed, 3)
            except SparqlError:
                payload["entities"] = "unavailable"
        return json.dumps(payload, indent=2, ensure_ascii=False)


def register(ctx) -> None:
    """Hermes entry point for a memory-provider plugin."""
    global _CTX
    _CTX = ctx
    ctx.register_memory_provider(SamskaraMemoryProvider())
