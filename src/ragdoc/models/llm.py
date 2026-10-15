"""Provider-agnostic chat LLM client with structured (JSON) output.

Supported providers: OpenAI, Anthropic, and any OpenAI-compatible server (Ollama, vLLM,
LM Studio, ...). Every call reports token usage to the active trace span.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ragdoc.config import LLMConfig, Secrets
from ragdoc.observability.tracing import add_usage

Role = Literal["fast", "strong", "vision"]
T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ImageInput(BaseModel):
    """An image sent to a vision-capable model (attached to the first user message)."""

    data: bytes
    mime: str = "image/png"

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode()


def guess_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    """Minimal interface the rest of the system depends on (easy to fake in tests)."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        role: Role = "strong",
        task: str = "generate",
        json_mode: bool = False,
        cache_system: bool = False,
        images: Sequence[ImageInput] | None = None,
    ) -> LLMResponse: ...


class StructuredOutputError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array found in ``text`` (tolerates code fences and prose)."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        for i, ch in enumerate(candidate):
            if ch in "{[":
                try:
                    obj, _ = decoder.raw_decode(candidate[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
    raise StructuredOutputError(f"No JSON object found in model output: {text[:200]!r}")


def _schema_hint(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), indent=None)


def structured(
    llm: LLM,
    *,
    system: str,
    user: str,
    schema: type[T],
    role: Role = "fast",
    task: str = "structured",
    max_attempts: int = 2,
    images: Sequence[ImageInput] | None = None,
) -> T:
    """Ask the model for JSON matching ``schema``; retries once with the validation error."""
    instructions = (
        f"{system}\n\nRespond with a single JSON object only, no prose, matching this JSON schema:\n"
        f"{_schema_hint(schema)}"
    )
    messages: list[Message] = [Message(role="user", content=user)]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        response = llm.generate(
            messages,
            system=instructions,
            role=role,
            task=task,
            json_mode=True,
            images=images,
        )
        try:
            return schema.model_validate(extract_json(response.text))
        except (StructuredOutputError, ValidationError) as exc:
            last_error = exc
            messages = [
                *messages,
                Message(role="assistant", content=response.text),
                Message(
                    role="user",
                    content=f"That was not valid. Error: {exc}. Return only the corrected JSON object.",
                ),
            ]
    raise StructuredOutputError(f"{task}: could not parse structured output: {last_error}")


# --------------------------------------------------------------------------- providers


def _model_for(config: LLMConfig, role: Role) -> str:
    if role == "fast":
        return config.fast_model
    if role == "vision":
        return config.vision_model or config.strong_model
    return config.strong_model


class OpenAIChat:
    """OpenAI or any OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig, secrets: Secrets) -> None:
        from openai import OpenAI

        self.config = config
        compatible = config.provider == "openai_compatible"
        api_key = secrets.get("llm_api_key") if compatible else secrets.get("openai_api_key")
        self._compatible = compatible
        self._client = OpenAI(
            api_key=api_key or ("not-needed" if compatible else None),
            base_url=config.base_url,
            timeout=config.request_timeout_s,
            max_retries=config.max_retries,
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        role: Role = "strong",
        task: str = "generate",
        json_mode: bool = False,
        cache_system: bool = False,  # OpenAI caches long prompt prefixes automatically
        images: Sequence[ImageInput] | None = None,
    ) -> LLMResponse:
        model = _model_for(self.config, role)
        payload: list[dict[str, Any]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(m.model_dump() for m in messages)
        if images:
            last = next(m for m in payload if m["role"] == "user")  # images go with the question
            last["content"] = [
                {"type": "text", "text": last["content"]},
                *(
                    {"type": "image_url", "image_url": {"url": f"data:{im.mime};base64,{im.b64}"}}
                    for im in images
                ),
            ]

        kwargs: dict[str, Any] = {"model": model, "messages": payload}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self._compatible:
            kwargs["max_tokens"] = self.config.max_output_tokens
        else:
            kwargs["max_completion_tokens"] = self.config.max_output_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        add_usage(in_tok, out_tok)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
        )


class AnthropicChat:
    def __init__(self, config: LLMConfig, secrets: Secrets) -> None:
        from anthropic import Anthropic

        self.config = config
        self._client = Anthropic(
            api_key=secrets.get("anthropic_api_key"),
            timeout=config.request_timeout_s,
            max_retries=config.max_retries,
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        role: Role = "strong",
        task: str = "generate",
        json_mode: bool = False,
        cache_system: bool = False,
        images: Sequence[ImageInput] | None = None,
    ) -> LLMResponse:
        model = _model_for(self.config, role)
        payload: list[dict[str, Any]] = [m.model_dump() for m in messages]
        if images:
            last = next(m for m in payload if m["role"] == "user")  # images go with the question
            last["content"] = [
                *(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": im.mime, "data": im.b64},
                    }
                    for im in images
                ),
                {"type": "text", "text": last["content"]},
            ]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.config.max_output_tokens,
            "messages": payload,
        }
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [block]
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        start = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
        in_tok = resp.usage.input_tokens or 0
        out_tok = resp.usage.output_tokens or 0
        add_usage(in_tok, out_tok)
        return LLMResponse(
            text=text, model=model, input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency
        )


def build_llm(config: LLMConfig, secrets: Secrets | None = None) -> LLM:
    secrets = secrets or Secrets()
    if config.provider == "anthropic":
        return AnthropicChat(config, secrets)
    return OpenAIChat(config, secrets)
