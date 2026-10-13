"""Prompt-injection screening for retrieved text.

Retrieved passages are untrusted. Every prompt already tells the model to treat them as
data; this module adds a heuristic screen that flags passages containing instruction-like
text so they can be down-weighted, logged and wrapped defensively.
"""

from __future__ import annotations

import re

_SUSPICIOUS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\bsystem prompt\b",
    r"reveal (your|the) (instructions|prompt|api key)",
    r"\bdo anything now\b",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"\bBEGIN (SYSTEM|ADMIN) (PROMPT|INSTRUCTIONS)\b",
]
_SUSPICIOUS_RE = re.compile("|".join(_SUSPICIOUS), re.IGNORECASE)


def injection_score(text: str) -> int:
    """Number of instruction-like patterns found (0 = clean)."""
    return len(_SUSPICIOUS_RE.findall(text))


def neutralise(text: str) -> str:
    """Escape tag-like sequences so a passage cannot close the <source> wrapper."""
    return text.replace("</source>", "&lt;/source&gt;").replace("<source", "&lt;source")
