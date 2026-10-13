"""Speech-to-text with segment timestamps (OpenAI Whisper API or local faster-whisper)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from ragdoc.config import MediaConfig, Secrets


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, offset_s: float = 0.0) -> list[Segment]: ...


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


class OpenAITranscriber:
    """whisper-1 returns segment-level timestamps (verbose_json). Files must be < 25 MB, so
    the media pipeline sends ~10-minute mono 64 kbps pieces (~5 MB each)."""

    def __init__(self, config: MediaConfig, secrets: Secrets) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=secrets.get("openai_api_key"))
        self.config = config

    def transcribe(self, audio_path: Path, offset_s: float = 0.0) -> list[Segment]:
        kwargs: dict[str, Any] = {
            "model": self.config.transcription_model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if self.config.language:
            kwargs["language"] = self.config.language
        with audio_path.open("rb") as fh:
            resp = self._client.audio.transcriptions.create(file=fh, **kwargs)
        segments = _get(resp, "segments") or []
        if not segments:  # models without timestamps: one segment for the whole piece
            text = (_get(resp, "text") or "").strip()
            duration = float(_get(resp, "duration") or 0.0)
            return [Segment(start=offset_s, end=offset_s + duration, text=text)] if text else []
        return [
            Segment(
                start=offset_s + float(_get(s, "start") or 0.0),
                end=offset_s + float(_get(s, "end") or 0.0),
                text=(_get(s, "text") or "").strip(),
            )
            for s in segments
            if (_get(s, "text") or "").strip()
        ]


class LocalWhisperTranscriber:
    """faster-whisper on CPU/GPU (pip install faster-whisper). No data leaves the machine."""

    def __init__(self, config: MediaConfig) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(config.local_whisper_model, compute_type="int8")
        self.config = config

    def transcribe(self, audio_path: Path, offset_s: float = 0.0) -> list[Segment]:
        segments, _info = self._model.transcribe(
            str(audio_path), language=self.config.language, vad_filter=True
        )
        return [
            Segment(start=offset_s + s.start, end=offset_s + s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]


def build_transcriber(config: MediaConfig, secrets: Secrets | None = None) -> Transcriber:
    if config.transcriber == "local":
        return LocalWhisperTranscriber(config)
    return OpenAITranscriber(config, secrets or Secrets())
