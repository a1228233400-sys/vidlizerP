from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from vidlizer.frames import encode_frame


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    endpoint: str
    api_key: str | None


class ProviderError(RuntimeError):
    pass


class ProviderClient:
    """Bounded provider client: one request at a time per shot."""

    def __init__(self, config: ProviderConfig, timeout: int = 600, max_cost: float = 0.0, context_tokens: int = 8192) -> None:
        self.config = config
        self.timeout = timeout
        self.max_cost = max_cost
        self.context_tokens = max(4096, context_tokens)
        self.requests = 0
        self.cost_usd = 0.0

    @classmethod
    def from_env(
        cls,
        provider: str | None = None,
        model: str | None = None,
        timeout: int = 600,
        max_cost: float = 0.0,
        context_tokens: int = 8192,
    ) -> "ProviderClient":
        p = (provider or os.getenv("PROVIDER") or "ollama").lower()
        m = model or os.getenv("MOVIEMIND_MODEL") or os.getenv("MODEL")
        if p == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            return cls(
                ProviderConfig(p, m or "qwen2.5vl:7b", f"{host}/api/chat", None),
                timeout,
                max_cost,
                context_tokens,
            )
        if p == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ProviderError("OPENROUTER_API_KEY is required.")
            return cls(
                ProviderConfig(
                    p,
                    m or "google/gemini-2.5-flash",
                    "https://openrouter.ai/api/v1/chat/completions",
                    key,
                ),
                timeout,
                max_cost,
                context_tokens,
            )
        if p in {"openai", "openai-compat"}:
            base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
            key = os.getenv("OPENAI_API_KEY", "lm-studio")
            if not m:
                raise ProviderError("Set --model or MOVIEMIND_MODEL for an OpenAI-compatible server.")
            return cls(
                ProviderConfig("openai", m, f"{base}/chat/completions", key),
                timeout,
                max_cost,
                context_tokens,
            )
        raise ProviderError(f"Unsupported provider: {p}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.provider != "ollama":
            headers["Authorization"] = f"Bearer {self.config.api_key or ''}"
        if self.config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/arizawan/vidlizer"
            headers["X-Title"] = "MovieMind"
        return headers

    def _record_usage(self, usage: dict[str, Any]) -> None:
        self.requests += 1
        if self.config.provider == "ollama":
            return
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        from vidlizer.models import get_pricing
        inp, out = get_pricing(self.config.model, None)
        self.cost_usd += (pt * inp + ct * out) / 1_000_000
        if self.max_cost > 0 and self.cost_usd > self.max_cost:
            raise ProviderError(
                f"MovieMind cost cap exceeded: {self.cost_usd:.4f} USD > {self.max_cost:.2f} USD."
            )

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Conservatively close JSON that was truncated at end-of-output."""
        stack: list[str] = []
        in_string = False
        escaped = False
        out: list[str] = []

        for char in text:
            out.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == chr(92):
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()

        if in_string:
            out.append('"')
        while stack:
            opener = stack.pop()
            out.append("]" if opener == "[" else "}")
        return "".join(out)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text[len(fence):]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
            if fence in text:
                text = text.rsplit(fence, 1)[0]

        # Some local VLMs add a short preamble despite JSON mode.
        first_object = text.find("{")
        if first_object >= 0:
            text = text[first_object:]

        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as first_exc:
            # A trailing comma is safe to normalize.
            normalized = re.sub(r",(\s*[}\]])", r"\1", text)
            try:
                value, _ = decoder.raw_decode(normalized)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                # Only attempt structural repair when the decoder is effectively
                # reaching the end of a truncated response.
                if (
                    "Unterminated string" in str(first_exc)
                    or first_exc.pos >= max(0, len(text) - 32)
                ):
                    repaired = ProviderClient._repair_truncated_json(normalized.rstrip().rstrip(","))
                    repaired = re.sub(r",(s*[}]])", r"\1", repaired)
                    repaired = re.sub(r",s*([}]])", r"\1", repaired)
                    try:
                        value, _ = decoder.raw_decode(repaired)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass

        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Invalid JSON from model: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("Expected a JSON object.")
        return value

    def _request(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool,
        max_tokens: int,
        retries: int = 3,
    ) -> str:
        if self.config.provider == "ollama":
            native_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts: list[str] = []
                    images: list[str] = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part["text"])
                        elif part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            images.append(url.split(",", 1)[1] if "," in url else url)
                    native = {"role": msg["role"], "content": "\n".join(text_parts)}
                    if images:
                        native["images"] = images
                    native_messages.append(native)
                else:
                    native_messages.append(msg)
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": native_messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": self.context_tokens,
                    "num_predict": max_tokens,
                },
            }
            if json_mode:
                payload["format"] = "json"
        else:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                if not response.ok:
                    raise ProviderError(
                        f"{self.config.provider} HTTP {response.status_code}: {response.text[:500]}"
                    )
                body = response.json()
                if self.config.provider == "ollama":
                    content = (body.get("message") or {}).get("content") or ""
                    usage = {
                        "prompt_tokens": body.get("prompt_eval_count", 0) or 0,
                        "completion_tokens": body.get("eval_count", 0) or 0,
                    }
                else:
                    choices = body.get("choices") or []
                    content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
                    usage = body.get("usage") or {}
                self._record_usage(usage)
                return content
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(str(exc)) from exc
        raise ProviderError(str(last_error) if last_error else "provider request failed")

    def analyze(
        self,
        prompt: str,
        images: list[Path],
        timestamps: list[float],
        max_output_tokens: int = 2048,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for i, image in enumerate(images):
            content.append(
                {"type": "text", "text": f"[OBSERVATION t={timestamps[i]:.3f}s]"}
            )
            content.append(encode_frame(image))
        raw = self._request(
            [{"role": "user", "content": content}],
            True,
            max_output_tokens,
        )
        return self._parse_json(raw)

    def complete_json(self, prompt: str, max_output_tokens: int = 2048) -> dict[str, Any]:
        raw = self._request(
            [{"role": "user", "content": prompt}],
            True,
            max_output_tokens,
        )
        return self._parse_json(raw)

    def preflight(self) -> None:
        if self.config.provider == "ollama":
            host = self.config.endpoint.rsplit("/api/chat", 1)[0]
            try:
                response = requests.get(f"{host}/api/tags", timeout=5)
                response.raise_for_status()
                names = [x.get("name", "") for x in response.json().get("models", [])]
            except requests.RequestException as exc:
                raise ProviderError(
                    f"Ollama is not reachable at {host}. Start it first."
                ) from exc
            if self.config.model in names:
                return

            # The canonical model name is qwen2.5vl:7b, while many local
            # Ollama installs use a quantized tag such as qwen2.5vl:7b-q4_K_M.
            # Resolve the canonical default to an installed concrete tag.
            if self.config.model == "qwen2.5vl:7b":
                family = [
                    name for name in names
                    if name.split(":")[0] == "qwen2.5vl"
                ]
                if family:
                    preferred = next(
                        (name for name in family if "q4" in name.lower()),
                        family[0],
                    )
                    self.config = ProviderConfig(
                        self.config.provider,
                        preferred,
                        self.config.endpoint,
                        self.config.api_key,
                    )
                    return

            available = ", ".join(names[:12]) or "(none)"
            raise ProviderError(
                f"Ollama model '{self.config.model}' is not installed. "
                f"Available models: {available}"
            )
        elif not self.config.api_key:
            raise ProviderError("Provider API key is missing.")

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "requests": self.requests,
            "estimated_cost_usd": round(self.cost_usd, 6),
        }
