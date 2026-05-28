#!/usr/bin/env python3
"""
mutate_and_store.py  (two-byte contiguous mutation)
---------------------------------------------------

For each file, try contiguous two-byte spans directly.  A span is recorded as
soon as applying mutations across that span makes the validator fail.

Only one contiguous (p,p+1) pair is stored per original file.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Tuple

PRINTABLE_ASCII = ["!", "^", "$", "%", "&"]

# ─────────────────────────── helpers ────────────────────────────────────────
def run_validator(exe: str, path: Path) -> bool:
    try:
        res = subprocess.run(
            [exe, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False

def mutate_at_positions(data: bytearray, *positions: int) -> bytearray:
    """
    For each pos in positions (from the original data):
      - randomly choose one of: replace, delete, or insert
      - apply it to the working buffer, adjusting an offset so that
        subsequent positions still refer to the original byte indices.
    """
    out = bytearray(data)
    offset = 0

    for pos in sorted(positions):
        idx = pos + offset
        op = random.choice(["replace", "delete", "insert"])

        if op == "replace" and idx < len(out):
            orig = out[idx]
            new_ch = random.choice(PRINTABLE_ASCII)
            while ord(new_ch) == orig:
                new_ch = random.choice(PRINTABLE_ASCII)
            out[idx] = ord(new_ch)

        elif op == "delete" and idx < len(out):
            del out[idx]
            offset -= 1

        else:
            # insert (or fallback if replace/delete invalid)
            new_ch = random.choice(PRINTABLE_ASCII)
            out.insert(idx, ord(new_ch))
            offset += 1

    return out

def ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS mutations (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               file_path TEXT,
               mutation_pos TEXT,            -- "p1,p2"
               original_text TEXT,
               mutated_text TEXT
           )"""
    )
    conn.commit()

def store_pair(
    conn: sqlite3.Connection,
    fpath: str,
    pos_pair: Tuple[int, int],
    orig: str,
    mut: str
):
    conn.execute(
        "INSERT INTO mutations (file_path, mutation_pos, original_text, mutated_text) "
        "VALUES (?, ?, ?, ?)",
        (fpath, f"{pos_pair[0]},{pos_pair[1]}", orig, mut),
    )
    conn.commit()

def find_contiguous_fault(
    data: bytes,
    validator: str,
    path: Path,
    span_len: int,
    max_attempts: int,
    prefix_len: int | None = None,
) -> tuple[Tuple[int, ...], bytearray] | None:
    """Return a contiguous span whose combined mutation fails validation."""
    if len(data) < span_len:
        return None

    tmp = path.with_suffix(f".tmp_mut_span{span_len}")
    tries = 0
    while tries < max_attempts:
        tries += 1
        if prefix_len is not None and random.random() < 0.8:
            upper = min(len(data), prefix_len) - span_len + 1
            if upper <= 0:
                continue
            start = random.randrange(upper)
        else:
            start = random.randrange(len(data) - span_len + 1)

        positions = tuple(range(start, start + span_len))
        if any(data[pos] == ord('"') for pos in positions):
            continue

        mutated = mutate_at_positions(bytearray(data), *positions)
        tmp.write_bytes(mutated)
        if not run_validator(validator, tmp):
            tmp.unlink(missing_ok=True)
            return positions, mutated

    tmp.unlink(missing_ok=True)
    return None

# ────────────────────────── core logic ──────────────────────────────────────
def process_file(
    path: Path,
    validator: str,
    conn: sqlite3.Connection,
    max_attempts: int
):
    data = path.read_bytes()
    if not data:
        return

    # Only proceed if original is valid ASCII
    try:
        data.decode('ascii')
    except UnicodeDecodeError:
        print(f"[skip] non-ASCII file skipped: {path}")
        return
    if not run_validator(validator, path):
        print(f"[skip] original already invalid: {path}")
        return

    cur = conn.execute("SELECT 1 FROM mutations WHERE file_path=? LIMIT 1",
                       (str(path),))
    if cur.fetchone():
        return

    print(f"[info] processing {path}")

    focus_prefix = 7 if "url" in path.parts else None

    result = find_contiguous_fault(data, validator, path, 2, max_attempts, focus_prefix)
    if result is None:
        print("  [✗] no failing contiguous two-byte span found")
        return

    positions, mutated_double = result
    store_pair(
        conn,
        str(path),
        (positions[0], positions[1]),
        data.decode(errors="ignore"),
        mutated_double.decode(errors="ignore"),
    )
    print(f"  [✓] found contiguous pair ({positions[0]},{positions[1]})")

def traverse(
    folder: Path,
    validator: str,
    db: Path,
    max_attempts: int,
    seed: int | None
):
    if seed is not None:
        random.seed(seed)

    with sqlite3.connect(db) as conn:
        ensure_table(conn)
        for file in folder.rglob("*"):
            if file.is_file():
                try:
                    process_file(file, validator, conn, max_attempts)
                except Exception as e:
                    print(f"[err] {file}: {e}", file=sys.stderr)

# ───────────────────────────────── main ─────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Two-byte contiguous mutate & store"
    )
    ap.add_argument("--folder", required=True)
    ap.add_argument("--validator", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument(
        "--max-attempts", type=int, default=100,
        help="max contiguous-span attempts per file"
    )
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    traverse(
        Path(args.folder).resolve(),
        args.validator,
        Path(args.database).resolve(),
        args.max_attempts,
        args.seed
    )
    print("[done] all files processed")

if __name__ == "__main__":
    main()
