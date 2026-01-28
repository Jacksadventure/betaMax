#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CopyPair:
    src: Path
    dst: Path


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def _detect_key_columns(cols: set[str]) -> list[str]:
    candidates: list[list[str]] = [
        ["format", "file_id", "corrupted_index", "algorithm"],
        ["format", "fid", "cidx", "algorithm"],
        ["format", "broken_text", "algorithm"],
    ]
    for key in candidates:
        if all(c in cols for c in key):
            return key
    raise RuntimeError(f"Cannot detect a stable key from columns={sorted(cols)}")


def copy_erepair_rows(pair: CopyPair, *, dry_run: bool = False) -> tuple[int, int]:
    if not pair.src.is_file():
        raise FileNotFoundError(pair.src)
    if not pair.dst.is_file():
        raise FileNotFoundError(pair.dst)

    dst_conn = sqlite3.connect(str(pair.dst))
    try:
        dst_cols = _get_columns(dst_conn, "results")
        if not dst_cols:
            raise RuntimeError(f"{pair.dst}: missing table results")

        src_conn = sqlite3.connect(str(pair.src))
        try:
            src_cols = _get_columns(src_conn, "results")
            if not src_cols:
                raise RuntimeError(f"{pair.src}: missing table results")

            if set(dst_cols) != set(src_cols):
                raise RuntimeError(
                    f"Schema mismatch: dst={len(dst_cols)} cols, src={len(src_cols)} cols"
                )
        finally:
            src_conn.close()

        cols_no_id = [c for c in dst_cols if c != "id"]
        if not cols_no_id:
            raise RuntimeError("No insertable columns found (unexpected schema)")

        key_cols = _detect_key_columns(set(dst_cols))
        where_match = " AND ".join([f"t.{c} = s.{c}" for c in key_cols])

        dst_conn.execute("ATTACH DATABASE ? AS src", (str(pair.src),))
        try:
            before = dst_conn.execute(
                "SELECT count(*) FROM results WHERE algorithm='erepair'"
            ).fetchone()[0]

            insert_sql = (
                f"INSERT INTO results ({', '.join(cols_no_id)}) "
                f"SELECT {', '.join(['s.' + c for c in cols_no_id])} "
                "FROM src.results s "
                "WHERE s.algorithm='erepair' "
                f"AND NOT EXISTS (SELECT 1 FROM results t WHERE {where_match})"
            )

            if dry_run:
                # Compute how many would be inserted using the same predicate.
                would = dst_conn.execute(
                    "SELECT count(*) FROM src.results s "
                    "WHERE s.algorithm='erepair' "
                    f"AND NOT EXISTS (SELECT 1 FROM results t WHERE {where_match})"
                ).fetchone()[0]
            else:
                cur = dst_conn.execute(insert_sql)
                would = int(cur.rowcount if cur.rowcount is not None else 0)
                dst_conn.commit()

            after = dst_conn.execute(
                "SELECT count(*) FROM results WHERE algorithm='erepair'"
            ).fetchone()[0]
            return int(would), int(after - before)
        finally:
            dst_conn.execute("DETACH DATABASE src")
    finally:
        dst_conn.close()


def _default_pairs() -> list[CopyPair]:
    modes = ["single", "double", "triple"]
    pairs: list[CopyPair] = []
    for mode in modes:
        pairs.append(CopyPair(src=Path(f"{mode}_erepair.db"), dst=Path(f"{mode}.db")))
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Copy algorithm='erepair' rows from *_erepair.db into the corresponding *.db."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many rows would be inserted (no DB writes).",
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        help="Optional explicit pairs as src:dst (e.g., single_erepair.db:single.db).",
    )
    args = ap.parse_args(argv)

    pairs: Iterable[CopyPair]
    if args.pairs:
        parsed: list[CopyPair] = []
        for item in args.pairs:
            if ":" not in item:
                raise SystemExit(f"Invalid pair '{item}' (expected src:dst)")
            src_s, dst_s = item.split(":", 1)
            parsed.append(CopyPair(src=Path(src_s), dst=Path(dst_s)))
        pairs = parsed
    else:
        pairs = _default_pairs()

    for pair in pairs:
        planned, delta = copy_erepair_rows(pair, dry_run=bool(args.dry_run))
        action = "would insert" if args.dry_run else "inserted"
        print(f"{pair.src} -> {pair.dst}: {action} {planned} row(s) (erepair count change: {delta})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

