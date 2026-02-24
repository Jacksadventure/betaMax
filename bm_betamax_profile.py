#!/usr/bin/env python3
"""
bm_betamax_profile.py
--------------------
Benchmark/profiling runner for BETAMAX only.

For each test entry, records:
- total runtime (seconds)
- Earley-correction runtime (seconds)
- loop count (L* attempts, parsed from stdout "[ATTEMPT N]")
- EC/total ratio

Timeout is enforced at the subprocess level (default: 300s).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from bm_lstar_mutations import get_mutation_table_name
from bm_betamax_backend import build_betamax_cmd, get_engine

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------

VALID_FORMATS = ["date", "iso8601", "time", "isbn", "ipv4", "ipv6", "url"]
REGEX_DIR_TO_CATEGORY = {
    "date": "Date",
    "iso8601": "ISO8601",
    "time": "Time",
    "url": "URL",
    "isbn": "ISBN",
    "ipv4": "IPv4",
    "ipv6": "IPv6",
}

BETAMAX_ENGINE = get_engine()


def create_database(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            format TEXT,
            file_id INTEGER,
            corrupted_index INTEGER,
            algorithm TEXT,
            original_text TEXT,
            broken_text TEXT,
            repaired_text TEXT,
            fixed INTEGER,
            iterations INTEGER,
            repair_time REAL,
            ec_time REAL,
            ec_ratio REAL,
            learn_time REAL,
            learn_ratio REAL,
            oracle_time REAL,
            oracle_ratio REAL,
            timed_out INTEGER,
            return_code INTEGER
        )
        """
    )
    # Lightweight schema migration for older DBs.
    cur.execute("PRAGMA table_info(results)")
    existing = {row[1] for row in cur.fetchall()}
    if "learn_time" not in existing:
        cur.execute("ALTER TABLE results ADD COLUMN learn_time REAL DEFAULT 0.0")
    if "learn_ratio" not in existing:
        cur.execute("ALTER TABLE results ADD COLUMN learn_ratio REAL DEFAULT 0.0")
    if "oracle_time" not in existing:
        cur.execute("ALTER TABLE results ADD COLUMN oracle_time REAL DEFAULT 0.0")
    if "oracle_ratio" not in existing:
        cur.execute("ALTER TABLE results ADD COLUMN oracle_ratio REAL DEFAULT 0.0")
    conn.commit()
    conn.close()


def load_test_samples_from_db(mutation_db_path: str) -> list[tuple[int, int, str, str]]:
    if not os.path.exists(mutation_db_path):
        raise FileNotFoundError(mutation_db_path)
    conn = sqlite3.connect(mutation_db_path)
    table_name = get_mutation_table_name(mutation_db_path, conn)
    cur = conn.cursor()
    cur.execute(f"SELECT id, original_text, mutated_text FROM {table_name} ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [(int(r[0]), 0, r[1] or "", r[2] or "") for r in rows]


def insert_test_samples_to_db(db_path: str, format_key: str, test_samples: list[tuple[int, int, str, str]]) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for (file_id, cindex, orig_text, broken_text) in test_samples:
        alg = "betamax"
        cur.execute(
            "SELECT 1 FROM results WHERE format=? AND file_id=? AND corrupted_index=? AND algorithm=? LIMIT 1",
            (format_key, file_id, cindex, alg),
        )
        if cur.fetchone():
            continue
        cur.execute(
            """
            INSERT INTO results (
              format, file_id, corrupted_index, algorithm,
              original_text, broken_text,
              repaired_text, fixed, iterations,
              repair_time, ec_time, ec_ratio, learn_time, learn_ratio, oracle_time, oracle_ratio, timed_out, return_code
            )
            VALUES (?, ?, ?, ?, ?, ?, '', 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, NULL)
            """,
            (format_key, file_id, cindex, alg, orig_text, broken_text),
        )
    conn.commit()
    conn.close()


def validate_with_external_tool(file_path: str, base_format: str) -> bool:
    category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
    wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
    if os.path.exists(wrapper):
        cmd = [wrapper, file_path]
    else:
        cmd = ["python3", "match.py", category, file_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def extract_lstar_attempts(stdout: str) -> int:
    max_attempt = -1
    try:
        for m in re.finditer(r"\[ATTEMPT\s+(\d+)\]", stdout):
            n = int(m.group(1))
            if n > max_attempt:
                max_attempt = n
        if max_attempt >= 0:
            return max_attempt + 1
        m2 = re.findall(r"attempt\s+(\d+)\s*/\s*(\d+)", stdout, flags=re.IGNORECASE)
        if m2:
            max_attempt = max(int(x) for x, _ in m2)
            return max_attempt + 1
    except Exception:
        pass
    return 0


_RE_METRICS_EC_TOTAL = re.compile(r"^\[METRICS\]\s+ec_seconds_total=([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_RE_PROFILE_EC = re.compile(r"^\[PROFILE\]\s+ec_earley(?:\\(relearn\\))?:\\s+([0-9]*\\.?[0-9]+)s\\s*$", re.MULTILINE)
_RE_METRICS_LEARN_TOTAL = re.compile(r"^\[METRICS\]\s+learn_seconds_total=([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_RE_PROFILE_LEARN_TOTAL = re.compile(r"^\[PROFILE\]\s+learn_grammar\\(total\\):\\s+([0-9]*\\.?[0-9]+)s\\s*;", re.MULTILINE)
_RE_PROFILE_LEARN_RELEARN = re.compile(r"^\[PROFILE\]\s+learn_grammar\\(relearn\\):\\s+([0-9]*\\.?[0-9]+)s\\s*;", re.MULTILINE)
_RE_METRICS_ORACLE_TOTAL = re.compile(r"^\[METRICS\]\s+oracle_seconds_total=([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_RE_PROFILE_ORACLE = re.compile(r"^\[PROFILE\]\s+oracle_validate(?:\\([^\\)]*\\))?:\\s+([0-9]*\\.?[0-9]+)s\\s*$", re.MULTILINE)


def extract_ec_seconds(stdout: str) -> float:
    m = _RE_METRICS_EC_TOTAL.search(stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    total = 0.0
    for m2 in _RE_PROFILE_EC.finditer(stdout):
        try:
            total += float(m2.group(1))
        except ValueError:
            continue
    return total


def extract_learn_seconds(stdout: str) -> float:
    m = _RE_METRICS_LEARN_TOTAL.search(stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    total = 0.0
    for m2 in _RE_PROFILE_LEARN_TOTAL.finditer(stdout):
        try:
            total += float(m2.group(1))
        except ValueError:
            continue
    for m3 in _RE_PROFILE_LEARN_RELEARN.finditer(stdout):
        try:
            total += float(m3.group(1))
        except ValueError:
            continue
    return total


def extract_oracle_seconds(stdout: str) -> float:
    m = _RE_METRICS_ORACLE_TOTAL.search(stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    total = 0.0
    for m2 in _RE_PROFILE_ORACLE.finditer(stdout):
        try:
            total += float(m2.group(1))
        except ValueError:
            continue
    return total


def _ensure_text(data: object) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _cache_path(fmt: str, learner: str, cache_root: str) -> str:
    learner_name = (learner or "rpni").replace("-", "_")
    print("cache_root: ", os.path.join(cache_root, f"lstar_{fmt}_{learner_name}.json"))
    return os.path.join(cache_root, f"lstar_{fmt}_{learner_name}.json")


def run_one_betamax(
    *,
    format_key: str,
    entry_id: int,
    original_text: str,
    broken_text: str,
    timeout_s: int,
    train_k: int,
    learner: str,
    cache_root: str,
    quiet: bool,
) -> dict[str, Any]:
    base_format = format_key.split("_")[-1]
    category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
    ext = base_format

    input_file = f"temp_{entry_id}_{random.randint(0, 9999)}_input.{ext}"
    output_file = os.path.join("repair_results", f"repair_{entry_id}_output.{ext}")
    pos_file = f"temp_pos_{entry_id}_{random.randint(0, 9999)}.txt"
    neg_file = f"temp_neg_{entry_id}_{random.randint(0, 9999)}.txt"

    cache_path = _cache_path(base_format, learner, cache_root)

    mdb_path = os.path.join("mutated_files", f"{format_key}.db")
    if not os.path.exists(mdb_path):
        raise FileNotFoundError(mdb_path)

    # Build positives/negatives from mutation DB (leave-one-out for positives).
    with sqlite3.connect(mdb_path) as conn:
        table_name = get_mutation_table_name(mdb_path, conn)
        cur = conn.cursor()
        cur.execute(f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {train_k * 3}")
        pos_rows = [r[0] or "" for r in cur.fetchall()]
        cur.execute(f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {train_k * 3}")
        neg_rows = [r[0] or "" for r in cur.fetchall()]

    pos_lines: list[str] = []
    for s in pos_rows:
        if s == original_text:
            continue
        s = s.rstrip("\n")
        if s:
            pos_lines.append(s)
        if len(pos_lines) >= train_k:
            break
    if not pos_lines:
        pos_lines = [original_text.rstrip("\n")] if original_text else [""]

    neg_lines: list[str] = []
    for s in neg_rows:
        s = s.rstrip("\n")
        if s:
            neg_lines.append(s)
        if len(neg_lines) >= train_k:
            break
    if broken_text:
        neg_lines.append(broken_text.rstrip("\n"))

    Path("repair_results").mkdir(parents=True, exist_ok=True)
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(broken_text)
    with open(pos_file, "w", encoding="utf-8") as pf:
        for line in pos_lines:
            pf.write(line + "\n")
    with open(neg_file, "w", encoding="utf-8") as nf:
        for line in neg_lines:
            nf.write(line + "\n")

    oracle_override = os.environ.get("LSTAR_ORACLE_VALIDATOR")
    oracle_bin = os.path.join("validators", f"validate_{base_format}")
    oracle_wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
    if oracle_override:
        oracle_cmd = oracle_override
    elif os.path.exists(oracle_bin):
        oracle_cmd = oracle_bin
    elif os.path.exists(oracle_wrapper):
        oracle_cmd = oracle_wrapper
    else:
        oracle_cmd = None

    cmd = build_betamax_cmd(
        engine=BETAMAX_ENGINE,
        positives=pos_file,
        negatives=neg_file,
        cache_path=(cache_path if BETAMAX_ENGINE == "python" else None),
        category=category,
        broken_file=input_file,
        output_file=output_file,
        attempts=500,
        mutations=0,
        learner=learner,
        oracle_cmd=oracle_cmd,
    )

    env = dict(os.environ)
    if BETAMAX_ENGINE == "python":
        env["BETAMAX_EMIT_METRICS"] = "1"
        env.setdefault("LSTAR_PARSE_TIMEOUT", os.environ.get("LSTAR_PARSE_TIMEOUT", "100.0"))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    t0 = time.time()
    timed_out = 0
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired as e:
        timed_out = 1
        try:
            proc.kill()
        except Exception:
            pass
        try:
            out2, err2 = proc.communicate(timeout=10)
        except Exception:
            out2, err2 = "", ""
        stdout = _ensure_text(e.output) + _ensure_text(out2)
        stderr = _ensure_text(e.stderr) + _ensure_text(err2)
        elapsed = float(timeout_s)

    rc = proc.returncode
    if BETAMAX_ENGINE == "python":
        attempts = extract_lstar_attempts(stdout)
        ec_time = extract_ec_seconds(stdout)
        ec_ratio = (ec_time / elapsed) if elapsed > 0 else 0.0
        learn_time = extract_learn_seconds(stdout)
        learn_ratio = (learn_time / elapsed) if elapsed > 0 else 0.0
        oracle_time = extract_oracle_seconds(stdout)
        oracle_ratio = (oracle_time / elapsed) if elapsed > 0 else 0.0
    else:
        attempts = 0
        ec_time = 0.0
        ec_ratio = 0.0
        learn_time = 0.0
        learn_ratio = 0.0
        oracle_time = 0.0
        oracle_ratio = 0.0

    repaired_text = ""
    fixed = 0
    if rc == 0 and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as rf:
                repaired_text = rf.read()
            fixed = 1 if validate_with_external_tool(output_file, base_format) else 0
        except Exception:
            repaired_text = ""
            fixed = 0

    if not quiet and (stdout.strip() or stderr.strip()):
        print(f"--- STDOUT (ID={entry_id}) ---\n{stdout}\n")
        print(f"--- STDERR (ID={entry_id}) ---\n{stderr}\n")

    for p in (input_file, output_file, pos_file, neg_file):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return {
        "fixed": fixed,
        "iterations": attempts,
        "repair_time": float(elapsed),
        "ec_time": float(ec_time),
        "ec_ratio": float(ec_ratio),
        "learn_time": float(learn_time),
        "learn_ratio": float(learn_ratio),
        "oracle_time": float(oracle_time),
        "oracle_ratio": float(oracle_ratio),
        "timed_out": int(timed_out),
        "return_code": rc if rc is not None else None,
        "repaired_text": repaired_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="BETAMAX-only benchmark with EC profiling")
    ap.add_argument("--db", default="betamax_profile.db", help="Output SQLite DB path")
    ap.add_argument("--formats", nargs="+", choices=VALID_FORMATS, help="Formats to include (default: all)")
    ap.add_argument("--mutations", nargs="+", default=["single", "double", "triple"], help="Mutation types (single/double/triple)")
    ap.add_argument("--train-k", type=int, default=int(os.environ.get("BM_TRAIN_K", "50")))
    ap.add_argument("--test-k", type=int, default=int(os.environ.get("BM_TEST_K", "50")))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-workers", type=int, help="Parallel workers (default: cpu count)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-entry stdout/stderr dumps")
    ap.add_argument("--limit", type=int, help="Limit entries processed (after filtering)")
    ap.add_argument("--resume-only", action="store_true", help="Skip sample insertion; only run unfinished rows")
    ap.add_argument("--learner", default=os.environ.get("BM_BETAMAX_LEARNER", os.environ.get("LSTAR_LEARNER", "rpni")))
    ap.add_argument("--cache-root", default=os.environ.get("LSTAR_CACHE_ROOT", "cache"))
    ap.add_argument(
        "--betamax-engine",
        choices=["python", "cpp"],
        default=None,
        help="Select betaMax engine backend (default: env BM_BETAMAX_ENGINE or 'python')",
    )
    args = ap.parse_args()

    global BETAMAX_ENGINE
    BETAMAX_ENGINE = get_engine(args.betamax_engine)

    create_database(args.db)

    selected_formats = args.formats if args.formats else VALID_FORMATS
    selected_mutations = args.mutations if args.mutations else ["single", "double", "triple"]

    if not args.resume_only:
        for mut in selected_mutations:
            for fmt in selected_formats:
                format_key = f"{mut}_{fmt}"
                mdb_path = os.path.join("mutated_files", f"{format_key}.db")
                if not os.path.exists(mdb_path):
                    print(f"[INFO] Skipping, not found: {mdb_path}")
                    continue
                samples = load_test_samples_from_db(mdb_path)
                if not samples:
                    continue
                budget = args.train_k + args.test_k
                limited = samples[:budget] if budget > 0 else samples
                train_limit = min(args.train_k, len(limited))
                test_samples = limited[train_limit : train_limit + min(args.test_k, max(0, len(limited) - train_limit))]
                if not test_samples:
                    continue
                print(f"[INFO] Inserting {len(test_samples)} test samples for {format_key}")
                insert_test_samples_to_db(args.db, format_key, test_samples)

    conn = sqlite3.connect(args.db, timeout=30)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, format, file_id, corrupted_index, algorithm, original_text, broken_text,
               fixed, iterations, repair_time, ec_time, ec_ratio, learn_time, learn_ratio,
               oracle_time, oracle_ratio, timed_out, return_code
        FROM results
        """
    )
    rows = cur.fetchall()
    conn.close()

    def wanted(row: tuple[Any, ...]) -> bool:
        fmt_key = row[1]
        alg = row[4]
        if alg != "betamax":
            return False
        base = fmt_key.split("_")[-1]
        mut = fmt_key.split("_")[0]
        if base not in selected_formats:
            return False
        if mut not in selected_mutations:
            return False
        return True

    filtered = [r for r in rows if wanted(r)]
    # Unfinished: keep re-running if timed_out or return_code not 0 or fixed not 1? user asked record per entry, so allow rerun.
    # Here we resume only missing runtime info (repair_time==0 and ec_time==0 and return_code is NULL).
    if args.resume_only:
        filtered = [r for r in filtered if (r[9] == 0.0 and r[10] == 0.0 and r[17] is None)]

    if args.limit is not None:
        filtered = filtered[: args.limit]

    if not filtered:
        print("[INFO] No entries to process.")
        return

    max_workers = args.max_workers or (os.cpu_count() or 4)
    print(f"[INFO] Processing {len(filtered)} entries with max_workers={max_workers}")

    def worker(row: tuple[Any, ...]) -> None:
        (id_, format_key, file_id, cidx, _alg, original_text, broken_text, *_rest) = row
        try:
            result = run_one_betamax(
                format_key=str(format_key),
                entry_id=int(id_),
                original_text=original_text or "",
                broken_text=broken_text or "",
                timeout_s=int(args.timeout),
                train_k=int(args.train_k),
                learner=str(args.learner),
                cache_root=str(args.cache_root),
                quiet=bool(args.quiet),
            )
        except Exception as e:
            if not args.quiet:
                print(f"[ERROR] Failed to run betamax for id={id_}, format={format_key}: {e}")
            result = {
                "repaired_text": "",
                "fixed": 0,
                "iterations": 0,
                "repair_time": 0.0,
                "ec_time": 0.0,
                "ec_ratio": 0.0,
                "learn_time": 0.0,
                "learn_ratio": 0.0,
                "oracle_time": 0.0,
                "oracle_ratio": 0.0,
                "timed_out": 0,
                "return_code": -1,
            }
        thread_conn = sqlite3.connect(args.db, timeout=30)
        thread_cur = thread_conn.cursor()
        thread_cur.execute(
            """
            UPDATE results
            SET repaired_text=?, fixed=?, iterations=?, repair_time=?, ec_time=?, ec_ratio=?,
                learn_time=?, learn_ratio=?, oracle_time=?, oracle_ratio=?, timed_out=?, return_code=?
            WHERE id=?
            """,
            (
                result["repaired_text"],
                result["fixed"],
                result["iterations"],
                result["repair_time"],
                result["ec_time"],
                result["ec_ratio"],
                result["learn_time"],
                result["learn_ratio"],
                result["oracle_time"],
                result["oracle_ratio"],
                result["timed_out"],
                result["return_code"],
                int(id_),
            ),
        )
        thread_conn.commit()
        thread_conn.close()

    # Run in parallel
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(worker, filtered))

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
