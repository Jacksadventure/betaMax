#!/usr/bin/env python3
"""
mutate_and_store.py   –   tail-truncation variant (one success per file)
-------
CREATE TABLE IF NOT EXISTS mutations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    mutation_pos INTEGER,   -- 截断偏移 cut_pos
    original_text TEXT,
    mutated_text TEXT
);
"""

import argparse
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ─────────────────── validator ─────────────────── #
def run_validator(exe: str, path: Path) -> bool:
    """Return True if validator exit-code == 0."""
    try:
        res = subprocess.run([exe, str(path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             timeout=10)
        return res.returncode == 0
    except Exception:
        return False

# ────────────────── mutation logic ─────────────── #
def truncate_tail(data: bytes) -> tuple[bytes, int]:
    """
    Choose a cut position in [len/2, len-1], biased toward the tail,
    and return (truncated_bytes, cut_pos).
    """
    if len(data) < 4:
        raise ValueError("file too small to truncate")
    lo = len(data) // 2
    hi = len(data) - 1
    # triangular distribution, mode at hi -> 更倾向靠后截断
    cut = int(random.triangular(lo, hi, hi))
    # Avoid cutting exactly at a quote character (can produce trivial parse artifacts).
    # NOTE: bytes indexing returns ints.
    for _ in range(20):
        if data[cut] != ord('"'):
            break
        cut = int(random.triangular(lo, hi, hi))
    return data[:cut], cut


def find_failing_truncation(data: bytes, validator: str, tmp: Path, max_attempts: int) -> tuple[bytes, int] | None:
    """
    Try to find a tail truncation that makes the validator fail.
    Returns (mutated_bytes, cut_pos) or None if not found.
    """
    try:
        lo = len(data) // 2
        hi = len(data) - 1
    except Exception:
        return None
    if len(data) < 4 or hi <= lo:
        return None

    # Random attempts first (triangular bias toward tail).
    for _attempt in range(max(1, int(max_attempts))):
        try:
            mutated, cut_pos = truncate_tail(data)
        except ValueError:
            return None
        tmp.write_bytes(mutated)
        if not run_validator(validator, tmp):
            return mutated, cut_pos

    # Deterministic fallback: walk the tail backwards.
    for cut_pos in range(hi, lo - 1, -1):
        if cut_pos <= 0 or cut_pos >= len(data):
            continue
        if data[cut_pos] == ord('"'):
            continue
        mutated = data[:cut_pos]
        tmp.write_bytes(mutated)
        if not run_validator(validator, tmp):
            return mutated, cut_pos
    return None

# ───────────── SQLite helpers ───────────────────── #
def ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS mutations(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               file_path TEXT,
               mutation_pos INTEGER,
               original_text TEXT,
               mutated_text TEXT
           );"""
    ); conn.commit()

def store_pair(conn: sqlite3.Connection, path: str,
               cut: int, orig: str, mut: str):
    conn.execute(
        "INSERT INTO mutations(file_path, mutation_pos, original_text, mutated_text) "
        "VALUES (?,?,?,?)", (path, cut, orig, mut))
    conn.commit()

# ───────────── per-file processing ─────────────── #
def process_file(path: Path, validator: str, conn: sqlite3.Connection,
                 max_attempts: int):
    data = path.read_bytes()
    if not data:
        return
    if not run_validator(validator, path):
        print(f"[skip] original invalid: {path}")
        return

    print(f"[info] processing {path}")
    tmp = path.with_suffix(".tmp_trunc")
    try:
        res = find_failing_truncation(data, validator, tmp, max_attempts)
        if not res:
            print(f"  [✗] no failing truncation within {max_attempts} attempts")
            return
        mutated, cut_pos = res
        store_pair(conn, str(path), cut_pos, data.decode(errors="ignore"), mutated.decode(errors="ignore"))
        print(f"  [✓] success cut @ {cut_pos}")
    finally:
        tmp.unlink(missing_ok=True)


def _guess_input_table(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    for candidate in ("mutations", "mutations_triple"):
        if candidate in tables:
            return candidate
    raise sqlite3.OperationalError("No table named 'mutations' or 'mutations_triple' found")


def process_text_row(
    *,
    source_label: str,
    row_id: int,
    original_text: str,
    validator: str,
    conn: sqlite3.Connection,
    max_attempts: int,
):
    data = (original_text or "").encode("utf-8", errors="ignore")
    if not data:
        return
    tmp = Path(f".tmp_trunc_{os.getpid()}_{random.randint(0, 999999)}.txt")
    try:
        tmp.write_bytes(data)
        if not run_validator(validator, tmp):
            print(f"[skip] original invalid: {source_label} id={row_id}")
            return
        res = find_failing_truncation(data, validator, tmp, max_attempts)
        if not res:
            print(f"[warn] no failing truncation: {source_label} id={row_id}")
            return
        mutated, cut_pos = res
        store_pair(
            conn,
            f"{source_label}#id={row_id}",
            cut_pos,
            original_text or "",
            mutated.decode(errors="ignore"),
        )
    finally:
        tmp.unlink(missing_ok=True)

# ─────────── directory traversal ───────────────── #
def traverse_folder(folder: Path, validator: str, db: Path, max_attempts: int, seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
    with sqlite3.connect(db) as conn:
        ensure_table(conn)
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        for p in files:
            # Keep truncation datasets deterministic and focused on JSON inputs.
            if p.suffix.lower() != ".json":
                continue
            try:
                process_file(p, validator, conn, max_attempts)
            except Exception as e:
                print(f"[err] {p}: {e}", file=sys.stderr)


def traverse_input_db(
    input_db: Path,
    validator: str,
    db: Path,
    max_attempts: int,
    seed: Optional[int],
    offset: int,
    limit: Optional[int],
):
    if seed is not None:
        random.seed(seed)
    source_label = str(input_db)
    with sqlite3.connect(db) as out_conn:
        ensure_table(out_conn)
        with sqlite3.connect(input_db) as in_conn:
            table = _guess_input_table(in_conn)
            cur = in_conn.cursor()
            sql = f"SELECT id, original_text FROM {table} ORDER BY id"
            params: list[object] = []
            if limit is not None:
                sql += " LIMIT ?"
                params.append(int(limit))
            if offset:
                sql += " OFFSET ?"
                params.append(int(offset))
            cur.execute(sql, params)
            for row_id, original_text in cur.fetchall():
                try:
                    process_text_row(
                        source_label=source_label,
                        row_id=int(row_id),
                        original_text=original_text or "",
                        validator=validator,
                        conn=out_conn,
                        max_attempts=max_attempts,
                    )
                except Exception as e:
                    print(f"[err] {source_label} id={row_id}: {e}", file=sys.stderr)

# ───────────────────── main ────────────────────── #
def main():
    ap = argparse.ArgumentParser("Tail-truncation mutator")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder", help="Traverse folder recursively and truncate each file")
    src.add_argument("--input-db", help="Read original_text from an input mutation DB (mutations/mutations_triple)")
    ap.add_argument("--validator", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--input-offset", type=int, default=0, help="Offset for --input-db (ORDER BY id)")
    ap.add_argument("--input-limit", type=int, default=None, help="Limit for --input-db (ORDER BY id)")
    args = ap.parse_args()

    out_db = Path(args.database).resolve()
    if args.folder:
        traverse_folder(Path(args.folder).resolve(), args.validator, out_db, args.max_attempts, args.seed)
    else:
        traverse_input_db(
            Path(args.input_db).resolve(),
            args.validator,
            out_db,
            args.max_attempts,
            args.seed,
            offset=int(args.input_offset or 0),
            limit=args.input_limit,
        )
    print("[done] all files processed")

if __name__ == "__main__":
    main()
