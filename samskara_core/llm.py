"""Two ways to reach a model — two separate trust domains.

- `HermesLLM` is the **only** model path inside the agent. It borrows
  `ctx.llm.complete_structured`, so the host owns routing, auth and fallback and
  the plugin never sees a credential. If the host does not expose `ctx.llm`,
  `make_llm` returns an *unavailable* adapter: session-end extraction is skipped
  and the lexical entity-mention fallback runs instead.

  There is deliberately **no environment fallback here**. A missing `ctx.llm`
  must never silently become a network call to some default host carrying
  whatever key happened to be in the environment: that would send a transcript
  somewhere the README (and the user) does not expect.

- `OpenAICompatLLM` is for the CLI scripts in `tools/` only. It requires an
  explicit `SAMSKARA_LLM_BASE_URL` **and** an explicit `SAMSKARA_LLM_API_KEY`:
  there is no default host and no reuse of `OPENAI_API_KEY` /
  `DEEPSEEK_API_KEY`, so a key written for one provider is never presented to
  another one.

Both return a plain dict that `extract.validate` can consume, and both fail
soft: a broken model must degrade memory, never break the agent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)

# The CLI client names its configuration explicitly. Two variables, both
# required, no implicit provider: whoever runs a tools/ script states where the
# turns are going.
CLI_BASE_URL_VAR = "SAMSKARA_LLM_BASE_URL"
CLI_API_KEY_VAR = "SAMSKARA_LLM_API_KEY"
CLI_MODEL_VAR = "SAMSKARA_LLM_MODEL"
CLI_DEFAULT_MODEL = "deepseek-flash"


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
    """Adapter over the host's `ctx.llm` (Plugin LLM Access).

    Unavailable when the host does not pass `ctx.llm` — the in-process
    memory-provider path does not today. Callers must check `available` and
    degrade; they must not look for a credential of their own.
    """

    def __init__(self, ctx: Any = None) -> None:
        self._llm = getattr(ctx, "llm", None) if ctx is not None else None

    @property
    def available(self) -> bool:
        return self._llm is not None and hasattr(self._llm, "complete_structured")

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        return "host did not expose ctx.llm (no model call is made from this plugin)"

    def __call__(self, instructions: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(
                "ctx.llm.complete_structured is not available in this host; "
                "samskara does not fall back to an API key of its own"
            )
        out = self._llm.complete_structured(
            instructions=instructions,
            input=[{"type": "text", "text": prompt}],
            json_schema=schema,
        )
        if isinstance(out, dict):
            return out
        return _parse_json(str(out))


class OpenAICompatLLM:
    """Minimal client for any OpenAI-compatible chat endpoint.

    CLI-only: `tools/backfill.py` and `tools/bench.py`. Both the base URL and the
    key must be declared explicitly (`SAMSKARA_LLM_BASE_URL`,
    `SAMSKARA_LLM_API_KEY`); nothing is inherited from another provider's key.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = key if key is not None else os.environ.get(CLI_API_KEY_VAR, "")
        self.base_url = (
            base_url if base_url is not None else os.environ.get(CLI_BASE_URL_VAR, "")
        ).rstrip("/")
        # Extraction is a bounded, schema-constrained task: the fast model is the
        # right default (the flagship spends reasoning tokens for no gain here).
        self.model = model or os.environ.get(CLI_MODEL_VAR) or CLI_DEFAULT_MODEL
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        missing = [
            var
            for var, value in ((CLI_BASE_URL_VAR, self.base_url), (CLI_API_KEY_VAR, self.api_key))
            if not value
        ]
        return "missing " + " and ".join(missing)

    def __call__(self, instructions: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(f"no model configured: {self.unavailable_reason}")
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
    """The plugin's model path: the host's model, or nothing.

    Never falls back to the environment. `on_session_end` reads `available` and
    keeps the entity-mention fallback when the host has no `ctx.llm`, so a
    transcript is never posted anywhere the user did not configure.
    """
    return HermesLLM(ctx)


def make_cli_llm(
    key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> OpenAICompatLLM:
    """For `tools/` scripts run from a terminal: explicit endpoint, explicit key."""
    return OpenAICompatLLM(key=key, base_url=base_url, model=model)


def cli_setup_help() -> List[str]:
    """What a CLI user must export before `tools/backfill.py` can extract."""
    return [
        f"export {CLI_BASE_URL_VAR}=https://your-endpoint/v1",
        f"export {CLI_API_KEY_VAR}=...",
        f"# optional: {CLI_MODEL_VAR} (default: {CLI_DEFAULT_MODEL})",
    ]
