#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _tiny_json_variants(n: int) -> list[str]:
    """
    Generate up to n tiny valid JSON objects.
    Target size: <= 10 bytes (no trailing newline), includes braces.
    """
    keys = list("abcdefghij")
    vals = list("0123456789")
    out: list[str] = []
    for k in keys:
        for v in vals:
            out.append(f'{{"{k}":{v}}}')
            if len(out) >= n:
                return out
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a tiny JSON corpus (<=10 bytes per file).")
    ap.add_argument("--out-dir", default="original_files/json_tiny_data", help="Output directory")
    ap.add_argument("--count", type=int, default=100, help="Number of JSON files to generate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _tiny_json_variants(max(0, int(args.count)))
    for i, text in enumerate(variants, start=1):
        path = out_dir / f"{i:04d}_tiny.json"
        path.write_text(text, encoding="ascii")
    print(f"[ok] wrote {len(variants)} file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

