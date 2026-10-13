"""Figures and charts: extract images from documents and turn them into searchable text.

Pipeline per figure: extract → dedupe/filter (logos, icons) → normalise to PNG → store as an
asset → VLM description + OCR (+ chart → data table) → indexed as an ``image``/``chart`` chunk
that cites its page and links to the image. Optional CLIP vectors add text→image search.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from ragdoc.config import MultimodalConfig
from ragdoc.ingestion.assets import AssetStore, to_png
from ragdoc.ingestion.contextualize import run_parallel
from ragdoc.models.llm import LLM
from ragdoc.models.vision import FigureDescription, describe_image
from ragdoc.types import Block, ParsedDocument

log = logging.getLogger(__name__)

IMAGE_KEY = "_image"  # transient raw bytes on a Block, removed once stored as an asset


def extract_pdf_images_pypdf(path: Path, max_images: int = 200) -> list[Block]:
    """Embedded raster images per page (pypdf). Vector drawings are not captured; use Docling
    (which renders picture regions) for documents whose charts are vector graphics."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    seen: set[str] = set()
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception as exc:  # malformed image streams are common in the wild
            log.debug("Skipping images on page %d: %s", page_no, exc)
            continue
        for img in images:
            data = img.data
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:  # repeated logos / backgrounds
                continue
            seen.add(digest)
            blocks.append(
                Block(
                    text="",
                    page=page_no,
                    kind="figure",
                    modality="image",
                    meta={IMAGE_KEY: data, "name": img.name},
                )
            )
            if len(blocks) >= max_images:
                return blocks
    return blocks


def store_figures(doc: ParsedDocument, assets: AssetStore, cfg: MultimodalConfig) -> ParsedDocument:
    """Normalise, filter and persist figure images; drop tiny/decorative ones."""
    kept: list[Block] = []
    figures = 0
    for block in doc.blocks:
        raw = block.meta.get(IMAGE_KEY)
        if raw is None:
            kept.append(block)
            continue
        meta = {k: v for k, v in block.meta.items() if k != IMAGE_KEY}
        if figures >= cfg.max_images_per_doc:
            continue
        try:
            png, width, height = to_png(raw)
        except Exception as exc:
            log.debug("Unreadable image on page %s: %s", block.page, exc)
            continue
        if min(width, height) < cfg.min_image_px:
            continue
        rel = assets.save(doc.doc_id, png, ".png")
        figures += 1
        kept.append(
            block.model_copy(
                update={"asset": rel, "meta": {**meta, "width": width, "height": height}}
            )
        )
    return doc.model_copy(update={"blocks": kept})


def _nearby_text(doc: ParsedDocument, index: int, window: int = 3) -> str:
    lo, hi = max(0, index - window), min(len(doc.blocks), index + window + 1)
    return "\n".join(
        b.text for b in doc.blocks[lo:hi] if b.modality == "text" and b.kind != "figure"
    )


def describe_figures(
    llm: LLM,
    doc: ParsedDocument,
    assets: AssetStore,
    cfg: MultimodalConfig,
    workers: int = 8,
    on_table: Callable[[Block, FigureDescription], None] | None = None,
) -> ParsedDocument:
    """Replace each figure block's text with a VLM description (and chart data)."""
    targets = [i for i, b in enumerate(doc.blocks) if b.asset and b.modality in ("image", "chart")]
    if not targets:
        return doc

    def one(index: int) -> tuple[int, Block | None]:
        block = doc.blocks[index]
        caption = block.meta.get("caption", "")
        where = (
            f"page {block.page}"
            if block.page
            else (f"{block.start_s:.0f}s" if block.start_s is not None else "")
        )
        if not cfg.describe_images:
            text = caption or f"Figure ({where})"
            return index, block.model_copy(update={"text": text})
        try:
            desc = describe_image(
                llm,
                assets.read(block.asset or ""),
                document_title=doc.title,
                location=where,
                caption=caption,
                nearby_text=block.meta.get("nearby", "") or _nearby_text(doc, index),
                want_data=cfg.chart_to_table,
            )
        except Exception as exc:
            log.warning("Figure description failed (%s, %s): %s", doc.source, where, exc)
            return index, block.model_copy(update={"text": caption or f"Figure ({where})"})
        if not desc.informative and not caption:
            return index, None  # decorative: drop
        label = "Video frame" if doc.kind == "video" else ("Chart" if desc.is_chart else "Figure")
        text = desc.as_text(label)
        if caption and caption not in text:
            text = f"Caption: {caption}\n{text}"
        updated = block.model_copy(
            update={
                "text": text,
                "modality": "chart" if desc.is_chart else block.modality,
                "meta": {**block.meta, "figure_kind": desc.kind, "title": desc.title},
            }
        )
        if on_table and desc.data_table:
            on_table(updated, desc)
        return index, updated

    results = dict(run_parallel(one, targets, workers))
    blocks = []
    for i, block in enumerate(doc.blocks):
        if i in results:
            if results[i] is not None:
                blocks.append(results[i])
        else:
            blocks.append(block)
    return doc.model_copy(update={"blocks": blocks})
