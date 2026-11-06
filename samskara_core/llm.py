"""Two ways to reach a model, one interface.

- `HermesLLM` uses `ctx.llm.complete_structured` inside the agent: the host owns
  routing, auth and fallback, and the plugin never sees a credential.
- `OpenAICompatLLM` is for the CLI (backfill from a terminal, CI): any
  OpenAI-compatible endpoint, key from the environment.

Both return a plain dict that `extract.validate` can consume, and both fail
soft: a broken model must degrade memory, never break the agent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, Optional

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def _parse_json(text: str) -> Dict[str, Any]:
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    if not cleaned:
        return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # tolerate prose around the object: take the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


class HermesLLM:
    """Adapter over the host's `ctx.llm` (Plugin LLM Access)."""

    def __init__(self, ctx: Any) -> None:
        self._llm = getattr(ctx, "llm", None)

    @property
    def available(self) -> bool:
        return self._llm is not None and hasattr(self._llm, "complete_structured")

    def __call__(self, instructions: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("ctx.llm.complete_structured is not available in this host")
        out = self._llm.complete_structured(
            instructions=instructions,
            input=[{"type": "text", "text": prompt}],
            json_schema=schema,
        )
        if isinstance(out, dict):
            return out
        return _parse_json(str(out))


class OpenAICompatLLM:
    """Minimal client for any OpenAI-compatible chat endpoint (CLI/backfill)."""

    def __init__(
        self,
        key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("SAMSKARA_LLM_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
        # Extraction is a bounded, schema-constrained task: the fast model is the
        # right default (the flagship spends reasoning tokens for no gain here).
        self.model = model or os.environ.get("SAMSKARA_LLM_MODEL") or "deepseek-flash"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def __call__(self, instructions: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("no API key (set DEEPSEEK_API_KEY or OPENAI_API_KEY)")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": prompt
                    + "\n\nReturn ONLY JSON matching this schema:\n"
                    + json.dumps(schema, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        content = payload["choices"][0]["message"].get("content", "")
        return _parse_json(content)


def make_llm(ctx: Any = None) -> Any:
    """Prefer the host's model (no key handling); fall back to the environment."""
    if ctx is not None:
        host = HermesLLM(ctx)
        if host.available:
            return host
    return OpenAICompatLLM()
