"""Audio and video ingestion.

Audio:  ffmpeg → 16 kHz mono pieces → timestamped transcript segments → time-window chunks.
Video:  the same for the soundtrack, plus keyframes (fixed interval or scene changes) that the
        VLM describes with the nearby transcript as context. Answers cite ``mm:ss`` ranges and
        frames, so a user can jump to the exact moment.

Requires the ``ffmpeg`` and ``ffprobe`` binaries on PATH.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ragdoc.config import MediaConfig
from ragdoc.ingestion.parsers import _doc_id, file_sha256
from ragdoc.models.transcribe import Segment, Transcriber
from ragdoc.types import Block, ParsedDocument

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
_PTS_RE = re.compile(r"pts_time:([\d.]+)")


class MediaToolMissing(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise MediaToolMissing(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg to ingest audio and video."
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_duration(path: Path) -> float | None:
    _require_ffmpeg()
    out = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def has_audio_stream(path: Path) -> bool:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    ).stdout.strip()
    return bool(out)


def split_audio(src: Path, out_dir: Path, segment_s: int = 600) -> list[tuple[Path, float]]:
    """Mono 16 kHz 64 kbps MP3 pieces of ``segment_s`` seconds: (path, start offset)."""
    _require_ffmpeg()
    pattern = out_dir / "piece_%04d.mp3"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            "-f",
            "segment",
            "-segment_time",
            str(segment_s),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    pieces = sorted(out_dir.glob("piece_*.mp3"))
    return [(p, i * float(segment_s)) for i, p in enumerate(pieces)]


def extract_keyframes(
    src: Path,
    out_dir: Path,
    interval_s: int = 30,
    scene_threshold: float | None = None,
    max_frames: int = 120,
) -> list[tuple[Path, float]]:
    """JPEG keyframes with their timestamps (seconds)."""
    _require_ffmpeg()
    if scene_threshold:
        proc = _run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(src),
                "-vf",
                f"select='gt(scene,{scene_threshold})',showinfo,scale='min(1280,iw)':-2",
                "-fps_mode",
                "vfr",
                "-q:v",
                "3",
                str(out_dir / "frame_%05d.jpg"),
            ]
        )
        times = [float(t) for t in _PTS_RE.findall(proc.stderr)]
        frames = sorted(out_dir.glob("frame_*.jpg"))
        pairs = list(zip(frames, times, strict=False))
    else:
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(src),
                "-vf",
                f"fps=1/{interval_s},scale='min(1280,iw)':-2",
                "-q:v",
                "3",
                str(out_dir / "frame_%05d.jpg"),
            ]
        )
        frames = sorted(out_dir.glob("frame_*.jpg"))
        pairs = [(f, i * float(interval_s)) for i, f in enumerate(frames)]
    if len(pairs) > max_frames:  # subsample uniformly
        step = len(pairs) / max_frames
        pairs = [pairs[int(i * step)] for i in range(max_frames)]
    return pairs


def transcribe_media(path: Path, transcriber: Transcriber, workdir: Path) -> list[Segment]:
    segments: list[Segment] = []
    for piece, offset in split_audio(path, workdir):
        segments.extend(transcriber.transcribe(piece, offset_s=offset))
    return segments


def _segment_blocks(segments: list[Segment], modality: str) -> list[Block]:
    return [
        Block(
            text=s.text,
            kind="segment",
            modality=modality,  # type: ignore[arg-type]
            start_s=s.start,
            end_s=s.end,
            meta={"speaker": s.speaker} if s.speaker else {},
        )
        for s in segments
    ]


def parse_audio(path: Path, transcriber: Transcriber) -> ParsedDocument:
    sha = file_sha256(path)
    with tempfile.TemporaryDirectory(prefix="ragdoc-audio-") as tmp:
        segments = transcribe_media(path, transcriber, Path(tmp))
    if not segments:
        raise ValueError(f"No speech found in {path.name}")
    return ParsedDocument(
        doc_id=_doc_id(path, sha),
        source=path.name,
        title=path.stem,
        blocks=_segment_blocks(segments, "audio"),
        sha256=sha,
        kind="audio",
        duration_s=probe_duration(path),
    )


def parse_video(path: Path, transcriber: Transcriber | None, cfg: MediaConfig) -> ParsedDocument:
    sha = file_sha256(path)
    blocks: list[Block] = []
    segments: list[Segment] = []
    with tempfile.TemporaryDirectory(prefix="ragdoc-video-") as tmp:
        work = Path(tmp)
        if transcriber is not None and has_audio_stream(path):
            (work / "audio").mkdir()
            segments = transcribe_media(path, transcriber, work / "audio")
            blocks += _segment_blocks(segments, "video")
        (work / "frames").mkdir()
        for frame, t in extract_keyframes(
            path,
            work / "frames",
            cfg.keyframe_interval_s,
            cfg.keyframe_scene_threshold,
            cfg.max_keyframes,
        ):
            nearby = " ".join(s.text for s in segments if s.end >= t - 30 and s.start <= t + 30)
            blocks.append(
                Block(
                    text="",
                    kind="figure",
                    modality="image",
                    start_s=t,
                    end_s=t,
                    meta={"_image": frame.read_bytes(), "nearby": nearby[:1500]},
                )
            )
    if not blocks:
        raise ValueError(f"Nothing could be extracted from {path.name}")
    return ParsedDocument(
        doc_id=_doc_id(path, sha),
        source=path.name,
        title=path.stem,
        blocks=blocks,
        sha256=sha,
        kind="video",
        duration_s=probe_duration(path),
    )
