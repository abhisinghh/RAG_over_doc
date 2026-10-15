"""Versioned prompt templates stored as Markdown files next to this module.

Templates use ``$name`` placeholders (``string.Template``) so documents containing braces
are safe to substitute.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

_DIR = Path(__file__).parent


@cache
def _load(name: str) -> Template:
    path = _DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return Template(path.read_text(encoding="utf-8").strip())


def render(name: str, **values: object) -> str:
    return _load(name).safe_substitute({k: str(v) for k, v in values.items()})
