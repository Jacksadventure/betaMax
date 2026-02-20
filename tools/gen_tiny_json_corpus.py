#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _gen_families() -> list[list[str]]:
    letters = "abcdefghij"
    digits = "0123456789"
    families: list[list[str]] = []

    # Two-field numeric objects (len=13 for single-digit values): {"a":0,"b":0}
    fam_obj_pair: list[str] = []
    for x in digits:
        for y in digits:
            fam_obj_pair.append(f'{{"a":{x},"b":{y}}}')
    families.append(fam_obj_pair)

    # Same but with a space after comma (len=14): {"a":0, "b":0}
    fam_obj_pair_sp: list[str] = []
    for x in digits:
        for y in digits:
            fam_obj_pair_sp.append(f'{{"a":{x}, "b":{y}}}')
    families.append(fam_obj_pair_sp)

    # Two-digit numbers (len=14/15): {"a":10,"b":0} / {"a":10,"b":10}
    fam_two_digit: list[str] = []
    for x in range(10, 20):
        for y in range(10):
            fam_two_digit.append(f'{{"a":{x},"b":{y}}}')
    for x in range(10, 20):
        fam_two_digit.append(f'{{"a":{x},"b":10}}')
    families.append(fam_two_digit)

    # Short string values (len=10): {"s":"ab"}
    fam_str: list[str] = []
    for a in letters:
        for b in letters:
            fam_str.append(f'{{"s":"{a}{b}"}}')
    families.append(fam_str)

    # Nested objects (len=13): {"a":{"b":0}}
    fam_nested: list[str] = []
    for d in digits:
        fam_nested.append(f'{{"a":{{"b":{d}}}}}')
    families.append(fam_nested)

    # Arrays inside object (len=11): {"a":[0,1]}
    fam_obj_arr: list[str] = []
    for x in digits:
        for y in digits:
            fam_obj_arr.append(f'{{"a":[{x},{y}]}}')
    families.append(fam_obj_arr)

    # Top-level arrays (len=11): [0,1,2,3,4]
    fam_arr5: list[str] = []
    for start in range(10):
        seq = ",".join(str((start + i) % 10) for i in range(5))
        fam_arr5.append(f"[{seq}]")
    families.append(fam_arr5)

    # Booleans/null (len=10/11)
    families.append(['{"a":null}', '{"a":true}', '{"a":false}'])

    return families


def _json_variants(n: int, *, min_bytes: int, max_bytes: int) -> list[str]:
    """
    Generate up to n small valid JSON values (objects/arrays/etc).
    Target size: min_bytes..max_bytes (no trailing newline).
    """
    min_b = max(0, int(min_bytes))
    max_b = max(min_b, int(max_bytes))

    families = _gen_families()
    idx = [0 for _ in families]

    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> bool:
        if s in seen:
            return False
        if not (min_b <= len(s) <= max_b):
            return False
        seen.add(s)
        out.append(s)
        return True

    made_progress = True
    while len(out) < n and made_progress:
        made_progress = False
        for fi, fam in enumerate(families):
            while idx[fi] < len(fam):
                s = fam[idx[fi]]
                idx[fi] += 1
                if add(s):
                    made_progress = True
                    break
            if len(out) >= n:
                break

    if len(out) < n:
        raise RuntimeError(
            f"could not generate enough JSON variants within size range: "
            f"requested={n}, got={len(out)}, min={min_b}, max={max_b}"
        )
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a small JSON corpus (size-bounded, ASCII).")
    ap.add_argument("--out-dir", default="original_files/json_small_data", help="Output directory")
    ap.add_argument("--count", type=int, default=100, help="Number of JSON files to generate")
    ap.add_argument("--min-bytes", type=int, default=10, help="Minimum bytes per file (no trailing newline)")
    ap.add_argument("--max-bytes", type=int, default=15, help="Maximum bytes per file (no trailing newline)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _json_variants(
        max(0, int(args.count)),
        min_bytes=int(args.min_bytes),
        max_bytes=int(args.max_bytes),
    )
    for i, text in enumerate(variants, start=1):
        path = out_dir / f"{i:04d}_small.json"
        path.write_text(text, encoding="ascii")
    print(f"[ok] wrote {len(variants)} file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

