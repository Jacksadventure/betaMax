#!/usr/bin/env python3
"""Regex oracle that distinguishes full, partial, and invalid matches.

Exit codes:
 0   -> full match (string already satisfies the pattern)
 255 -> partial match (string could become valid by appending characters)
 1   -> invalid (must change existing characters to become valid)
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict

try:
    import regex as re_mod  # type: ignore

    HAS_REGEX = True
except Exception:  # pragma: no cover - best-effort fallback
    import re as re_mod  # type: ignore

    HAS_REGEX = False

# Patterns are identical to match.py to ensure consistent acceptance criteria.
PATTERNS: Dict[str, str] = {
    "Date": r"^\d{4}-\d{2}-\d{2}$",
    "Time": r"^\d{2}:\d{2}:\d{2}$",
    "URL": r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",
    "ISBN": r"^(?:\d[- ]?){9}[\dX]$",
    "IPv4": r"^(\d{1,3}\.){3}\d{1,3}$",
    "IPv6": r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$",
}

PARTIAL_EXIT = 255


def _partial_date(text: str) -> bool:
    if len(text) >= 10:
        return False
    for idx, ch in enumerate(text):
        if idx in (4, 7):
            if ch != "-":
                return False
        else:
            if not ch.isdigit():
                return False
    return True


def _partial_time(text: str) -> bool:
    if len(text) >= 8:
        return False
    for idx, ch in enumerate(text):
        if idx in (2, 5):
            if ch != ":":
                return False
        else:
            if not ch.isdigit():
                return False
    return True


def _partial_isbn(text: str) -> bool:
    digits = 0
    last_sep = False
    for ch in text:
        if ch.isdigit():
            digits += 1
            if digits > 10:
                return False
            last_sep = False
        elif ch in "- ":
            if digits == 0 or last_sep:
                return False
            last_sep = True
        elif ch.upper() == "X":
            if digits != 9:
                return False
            digits += 1
            last_sep = False
        else:
            return False
    return digits < 10


def _partial_ipv4(text: str) -> bool:
    if any(ch not in "0123456789." for ch in text):
        return False
    parts = text.split(".")
    if len(parts) > 4:
        return False
    for idx, part in enumerate(parts):
        if not part:
            if idx == 0:
                return False
            # allow trailing dot to indicate unfinished octet
            if idx < len(parts) - 1:
                return False
            continue
        if not part.isdigit() or len(part) > 3:
            return False
        if int(part) > 255:
            return False
    return len(parts) < 4 or text.endswith(".") or len(parts[-1]) < 3


def _partial_ipv6(text: str) -> bool:
    allowed = "0123456789abcdefABCDEF:"
    if any(ch not in allowed for ch in text):
        return False
    if text.count(":::") > 0:
        return False
    parts = text.split(":")
    double_colon = "::" in text
    for part in parts:
        if part and len(part) > 4:
            return False
    non_empty = sum(1 for part in parts if part)
    if double_colon:
        return non_empty <= 8
    return non_empty < 8 or text.endswith(":")


def _partial_url(text: str) -> bool:
    if not text:
        return True
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=%"
    if any(ch not in allowed for ch in text):
        return False
    lower = text.lower()
    http_prefixes = ("http://", "https://")
    if any(prefix.startswith(lower) for prefix in http_prefixes):
        return True
    if lower.startswith(http_prefixes):
        rest = lower.split("://", 1)[1]
        if not rest:
            return True
        domain_stop = len(rest)
        for sep in ("/", "?", "#"):
            pos = rest.find(sep)
            if pos != -1 and pos < domain_stop:
                domain_stop = pos
        domain = rest[:domain_stop]
        if not domain:
            return True
        if ".." in domain:
            return False
        for ch in domain:
            if ch not in "abcdefghijklmnopqrstuvwxyz0123456789-.":
                return False
        if "." not in domain:
            return True
        return not rest[domain_stop:]  # still building path/query
    return False


FALLBACK_PARTIAL: Dict[str, Callable[[str], bool]] = {
    "Date": _partial_date,
    "Time": _partial_time,
    "ISBN": _partial_isbn,
    "IPv4": _partial_ipv4,
    "IPv6": _partial_ipv6,
    "URL": _partial_url,
}


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match with partial awareness for repair oracles.")
    parser.add_argument("category", choices=sorted(PATTERNS.keys()), help="Category name (e.g., Date).")
    parser.add_argument("file_path", help="Path to input file.")
    args = parser.parse_args()

    data = _read_file(args.file_path)
    pattern = PATTERNS[args.category]
    compiled = re_mod.compile(pattern)

    if compiled.fullmatch(data):
        sys.exit(0)

    if HAS_REGEX:
        match = compiled.fullmatch(data, partial=True)  # type: ignore[call-arg]
        if match and getattr(match, "partial", False):
            sys.exit(PARTIAL_EXIT)
    else:
        handler = FALLBACK_PARTIAL.get(args.category)
        if handler and handler(data):
            sys.exit(PARTIAL_EXIT)

    sys.exit(1)


if __name__ == "__main__":
    main()
