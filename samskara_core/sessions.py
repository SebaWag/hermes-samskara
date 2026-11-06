"""Read the agent's own history so the graph is not born empty.

Hermes keeps sessions in `$HERMES_HOME/state.db` (SQLite): `sessions` plus
`messages`. We open it read-only — a memory provider must never be able to
corrupt the transcript it is learning from.

This is what makes the onboarding different: on first run, the graph is built
from everything the user already told the agent, instead of starting at zero.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


@dataclass
class SessionContext:
    session_id: str
    title: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    cwd: str = ""
    model: str = ""
    messages: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(f"{role}: {content}" for role, content in self.messages)


class SessionReader:
    """Streams sessions from state.db, oldest first."""

    def __init__(self, db_path: Path | str, min_messages: int = 2, max_chars: int = 12000) -> None:
        self.db_path = Path(db_path)
        self.min_messages = min_messages
        self.max_chars = max_chars

    def available(self) -> bool:
        return self.db_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        # mode=ro: never write to the agent's own database
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def sessions(self, limit: int = 0, include_archived: bool = False) -> Iterator[SessionContext]:
        if not self.available():
            return
        conn = self._connect()
        try:
            where = "" if include_archived else "WHERE COALESCE(archived, 0) = 0"
            rows = conn.execute(
                f"""SELECT id, COALESCE(title, ''), COALESCE(started_at, 0), COALESCE(ended_at, 0),
                           COALESCE(cwd, ''), COALESCE(model, '')
                    FROM sessions {where} ORDER BY started_at ASC
                    {"LIMIT " + str(int(limit)) if limit else ""}"""
            ).fetchall()
            for session_id, title, started, ended, cwd, model in rows:
                ctx = SessionContext(
                    session_id=session_id,
                    title=title or "",
                    started_at=float(started or 0),
                    ended_at=float(ended or 0),
                    cwd=cwd or "",
                    model=model or "",
                )
                ctx.messages = self._messages(conn, session_id)
                if len(ctx.messages) >= self.min_messages:
                    yield ctx
        finally:
            conn.close()

    def _messages(self, conn: sqlite3.Connection, session_id: str) -> List[Tuple[str, str]]:
        try:
            rows = conn.execute(
                """SELECT role, COALESCE(NULLIF(content, ''), api_content, '')
                   FROM messages
                   WHERE session_id = ? AND COALESCE(active, 1) = 1
                     AND role IN ('user', 'assistant')
                   ORDER BY COALESCE(timestamp, 0) ASC, id ASC""",
                (session_id,),
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
        out: List[Tuple[str, str]] = []
        budget = self.max_chars
        for role, content in rows:
            text = (content or "").strip()
            if not text:
                continue
            if len(text) > budget:
                text = text[:budget]
            out.append((role, text))
            budget -= len(text)
            if budget <= 0:
                break
        return out

    # -- state (idempotency) ----------------------------------------------
    @staticmethod
    def load_state(path: Path) -> dict:
        if not Path(path).is_file():
            return {"sessions": {}, "seen": []}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"sessions": {}, "seen": []}
        data.setdefault("sessions", {})
        data.setdefault("seen", [])
        return data

    @staticmethod
    def save_state(path: Path, state: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def already_done(state: dict, session_id: str) -> bool:
    return session_id in (state.get("sessions") or {})


def mark_done(state: dict, session_id: str, triples: int) -> None:
    import time

    state.setdefault("sessions", {})[session_id] = {"triples": triples, "at": time.time()}
