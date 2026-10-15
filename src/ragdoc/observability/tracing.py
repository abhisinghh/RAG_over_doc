"""Per-request tracing.

Every pipeline step runs inside ``recorder.span("step")``. Spans capture latency, a small
detail dict and LLM/embedding token usage (reported by the model clients via ``add_usage``).
The trace is returned with each answer, and, when an OTLP endpoint is configured
(Langfuse, Arize Phoenix, Jaeger...), mirrored to OpenTelemetry.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ragdoc.types import TraceEvent, Usage


@dataclass
class _Span:
    step: str
    detail: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    recorder: TraceRecorder | None = None


_current_span: ContextVar[_Span | None] = ContextVar("ragdoc_span", default=None)
_current_recorder: ContextVar[TraceRecorder | None] = ContextVar("ragdoc_recorder", default=None)
_otel_tracer: Any = None


def add_usage(input_tokens: int, output_tokens: int) -> None:
    """Attribute token usage to the innermost open span (and its recorder's total)."""
    span = _current_span.get()
    if span is not None:
        span.input_tokens += input_tokens
        span.output_tokens += output_tokens
    recorder = span.recorder if span is not None else _current_recorder.get()
    if recorder is not None:
        recorder.usage.add(input_tokens, output_tokens)


class TraceRecorder:
    def __init__(self, name: str = "request") -> None:
        self.name = name
        self.events: list[TraceEvent] = []
        self.usage = Usage()
        self._token = None

    def __enter__(self) -> TraceRecorder:
        self._token = _current_recorder.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _current_recorder.reset(self._token)

    @contextmanager
    def span(self, step: str, **detail: Any) -> Iterator[dict[str, Any]]:
        span = _Span(step=step, detail=dict(detail), recorder=self)
        token = _current_span.set(span)
        start = time.perf_counter()
        otel_cm = _otel_tracer.start_as_current_span(f"ragdoc.{step}") if _otel_tracer else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        try:
            yield span.detail
        except Exception as exc:
            span.detail["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency = (time.perf_counter() - start) * 1000
            _current_span.reset(token)
            self.events.append(
                TraceEvent(
                    step=step,
                    latency_ms=round(latency, 1),
                    detail=span.detail,
                    input_tokens=span.input_tokens,
                    output_tokens=span.output_tokens,
                )
            )
            if otel_span is not None:
                otel_span.set_attribute("ragdoc.input_tokens", span.input_tokens)
                otel_span.set_attribute("ragdoc.output_tokens", span.output_tokens)
                otel_span.set_attribute(
                    "ragdoc.detail", json.dumps(span.detail, default=str)[:4000]
                )
                otel_cm.__exit__(None, None, None)


def init_tracing(otlp_endpoint: str | None, service_name: str = "ragdoc") -> bool:
    """Enable OpenTelemetry export if an endpoint is configured and the SDK is installed."""
    global _otel_tracer
    if not otlp_endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    _otel_tracer = trace.get_tracer("ragdoc")
    return True
