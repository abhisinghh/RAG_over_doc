"""Lightweight regex PII redaction applied at ingestion (opt-in).

This is a baseline, not a compliance tool: for regulated data use a dedicated detector
(e.g. Microsoft Presidio, which also covers national ID numbers) and keep this module as
the integration point. Patterns are conservative so numeric tables (years, amounts) are
not mangled; card numbers must pass a Luhn check.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b")
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ -]?\d{2,5}[ -]?\d{3,5}[ -]?\d{3,5}"  # +91 98765 43210
    r"|\(\d{3}\)\s?\d{3}-\d{4}"  # (555) 123-4567
    r"|\d{3}-\d{3}-\d{4}"  # 555-123-4567
    r"|\b[6-9]\d{9}\b)"  # Indian mobile
)


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def sub(label: str, pattern: re.Pattern[str], value: str, check=None) -> str:
        def repl(match: re.Match[str]) -> str:
            if check and not check(match.group(0)):
                return match.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(repl, value)

    text = sub("EMAIL", _EMAIL, text)
    text = sub("CARD", _CARD, text, _luhn_ok)
    text = sub("PHONE", _PHONE, text)
    return text, counts
