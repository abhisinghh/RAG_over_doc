"""Repository scanning and code → ParsedDocument conversion.

Each source file becomes one document whose blocks are its symbols (functions, methods,
class outlines) plus module-level code, each with exact line ranges. Large symbols are split
into overlapping windows. A deterministic header (path, language, symbol, service, imports)
replaces the LLM context step, so ingesting thousands of files costs embeddings only.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path

from ragdoc.code.parser import CodeFile, Symbol, detect_language, parse_code
from ragdoc.config import CodeConfig
from ragdoc.types import Block, ParsedDocument

DOC_SUFFIXES = {".md", ".markdown", ".rst", ".txt", ".pdf", ".docx"}
CONFIG_LANGUAGES = {"yaml", "toml", "json", "xml", "dockerfile", "make", "groovy", "terraform"}


def is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return True


def iter_repo_files(root: Path, cfg: CodeConfig) -> Iterator[Path]:
    """Source, config and documentation files, skipping vendored/build directories."""
    excluded = set(cfg.exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in DOC_SUFFIXES or detect_language(path):
                try:
                    if path.stat().st_size > cfg.max_file_bytes or path.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                if path.suffix.lower() not in {".pdf", ".docx"} and is_binary(path):
                    continue
                if name.endswith((".min.js", ".lock", "-lock.json", ".map")) or name in (
                    "package-lock.json",
                    "yarn.lock",
                    "pnpm-lock.yaml",
                ):
                    continue
                yield path


def code_doc_id(repo: str, rel: str) -> str:
    """Stable per path (not per content), so an edited file replaces its previous version."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{repo}/{rel}").strip("-").lower()[-60:]
    return f"{slug}-{hashlib.sha256(f'{repo}/{rel}'.encode()).hexdigest()[:8]}"


def _windows(sym: Symbol, max_lines: int, overlap: int) -> list[tuple[int, int, int]]:
    """(start, end, part) windows over a long symbol."""
    span = sym.line_end - sym.line_start + 1
    if span <= max_lines:
        return [(sym.line_start, sym.line_end, 0)]
    out = []
    start, part = sym.line_start, 1
    step = max(1, max_lines - overlap)
    while start <= sym.line_end:
        end = min(sym.line_end, start + max_lines - 1)
        out.append((start, end, part))
        if end == sym.line_end:
            break
        start += step
        part += 1
    return out


def _uncovered_ranges(n_lines: int, symbols: list[Symbol]) -> list[tuple[int, int]]:
    covered = [False] * (n_lines + 2)
    for s in symbols:
        if s.parent is None:  # top-level symbols cover their nested members too
            for i in range(s.line_start, min(s.line_end, n_lines) + 1):
                covered[i] = True
    ranges, start = [], None
    for i in range(1, n_lines + 1):
        if not covered[i] and start is None:
            start = i
        elif covered[i] and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, n_lines))
    return ranges


def code_file_to_document(
    path: Path,
    root: Path,
    cfg: CodeConfig,
    repo_name: str | None = None,
    service: str | None = None,
) -> tuple[ParsedDocument, CodeFile]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    sha = hashlib.sha256(raw).hexdigest()
    rel = path.relative_to(root).as_posix()
    repo = repo_name or root.name
    language = detect_language(path) or "text"
    parsed = parse_code(text, rel, language)
    lines = text.splitlines()

    def snippet(a: int, b: int) -> str:
        return "\n".join(lines[a - 1 : b])

    blocks: list[Block] = []
    base_meta = {"language": language, "path": rel}
    if service:
        base_meta["service"] = service
    header_path = [repo, *([service] if service else []), rel]

    # Module-level code (imports, constants, top-level statements, config files).
    for a, b in _uncovered_ranges(parsed.n_lines, parsed.symbols):
        body = snippet(a, b)
        if not body.strip():
            continue
        for wa, wb, part in _windows(
            Symbol(name="module", qualname="module", kind="module", line_start=a, line_end=b),
            cfg.symbol_max_lines,
            cfg.window_overlap_lines,
        ):
            blocks.append(
                Block(
                    text=snippet(wa, wb),
                    kind="symbol",
                    modality="code",
                    heading_path=header_path,
                    line_start=wa,
                    line_end=wb,
                    meta={
                        **base_meta,
                        "symbol": "<module>" + (f" part {part}" if part else ""),
                        "kind": "module",
                    },
                )
            )

    for sym in parsed.symbols:
        if sym.outline:  # class/struct: index an outline; members are indexed separately
            blocks.append(
                Block(
                    text=sym.outline,
                    kind="symbol",
                    modality="code",
                    heading_path=header_path,
                    line_start=sym.line_start,
                    line_end=sym.line_end,
                    meta={**base_meta, "symbol": sym.qualname, "kind": sym.kind, "outline": True},
                )
            )
            continue
        has_members = any(s.parent == sym.qualname for s in parsed.symbols)
        if has_members:
            continue
        for a, b, part in _windows(sym, cfg.symbol_max_lines, cfg.window_overlap_lines):
            blocks.append(
                Block(
                    text=snippet(a, b),
                    kind="symbol",
                    modality="code",
                    heading_path=header_path,
                    line_start=a,
                    line_end=b,
                    meta={
                        **base_meta,
                        "symbol": sym.qualname + (f" (part {part})" if part else ""),
                        "kind": sym.kind,
                        "calls": sym.calls[:40],
                        "docstring": sym.docstring[:300],
                    },
                )
            )

    blocks.sort(key=lambda b: b.line_start or 0)
    doc = ParsedDocument(
        doc_id=code_doc_id(repo, rel),
        source=rel,
        title=rel,
        blocks=blocks,
        sha256=sha,
        kind="code",
        meta={
            "repo": repo,
            "language": language,
            "service": service,
            "imports": parsed.imports[:100],
            "config": language in CONFIG_LANGUAGES,
        },
    )
    return doc, parsed


def code_context(doc: ParsedDocument, block_meta: dict[str, object]) -> str:
    """Deterministic contextual header for a code chunk (no LLM call)."""
    parts = [f"{doc.meta.get('language', 'code')} {block_meta.get('kind', 'code')}"]
    if block_meta.get("symbol"):
        parts.append(str(block_meta["symbol"]))
    parts.append(f"in {doc.source}")
    if doc.meta.get("service"):
        parts.append(f"(service {doc.meta['service']})")
    header = " ".join(parts) + "."
    doc_line = str(block_meta.get("docstring") or "").split("\n")[0]
    if doc_line:
        header += f" {doc_line}"
    calls = block_meta.get("calls")
    if calls:
        header += f" Calls: {', '.join(list(calls)[:12])}."  # type: ignore[arg-type]
    return header
