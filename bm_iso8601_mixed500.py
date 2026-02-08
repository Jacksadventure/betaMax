#!/usr/bin/env python3
"""
Benchmark: ISO8601 mixed500 (shuffled), with custom split:
  - Train positives: first 400 lines from data/iso8601_500_mixed.txt
  - Test set: 100 mutated samples derived from lines 400..499

This benchmark is intentionally standalone (custom split + unlimited precompute timeout).
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
from typing import Optional

from bm_betamax_backend import build_betamax_cmd, cpp_bin_path, get_engine
from bm_lstar_mutations import get_mutation_table_name


DATASET_PATH = Path("data/iso8601_500_mixed.txt")
VALIDATOR = Path("validators/validate_iso8601_mixed")
DEFAULT_MUTATION_TYPES = ["single", "double", "triple", "truncated"]


RESULTS_TABLE_SQL = """
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
    correct_runs INTEGER,
    incorrect_runs INTEGER,
    incomplete_runs INTEGER,
    distance_original_broken INTEGER,
    distance_broken_repaired INTEGER,
    distance_original_repaired INTEGER
)
"""


def _ensure_results_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(results)")
    existing = {row[1] for row in cursor.fetchall()}
    additions = [
        ("timed_out", "INTEGER DEFAULT 0"),
        ("return_code", "INTEGER"),
        ("ec_time", "REAL DEFAULT 0.0"),
        ("ec_ratio", "REAL DEFAULT 0.0"),
        ("learn_time", "REAL DEFAULT 0.0"),
        ("learn_ratio", "REAL DEFAULT 0.0"),
        ("oracle_time", "REAL DEFAULT 0.0"),
        ("oracle_ratio", "REAL DEFAULT 0.0"),
    ]
    for name, decl in additions:
        if name in existing:
            continue
        try:
            cursor.execute(f"ALTER TABLE results ADD COLUMN {name} {decl}")
        except Exception:
            pass


def create_database(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(RESULTS_TABLE_SQL)
    _ensure_results_schema(cur)
    conn.commit()
    conn.close()


def _normalize_for_distance(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.rstrip("\r\n")


def levenshtein_distance(a: str, b: str) -> int:
    a = _normalize_for_distance(a)
    b = _normalize_for_distance(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


_RE_METRIC = re.compile(r"^\[METRICS\]\s+([a-zA-Z0-9_]+)=([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_RE_ATTEMPT = re.compile(r"\[ATTEMPT\s+(\d+)\]")


def _extract_betamax_metrics(stdout: str) -> tuple[float, float, float]:
    kv: dict[str, float] = {}
    for m in _RE_METRIC.finditer(stdout or ""):
        try:
            kv[m.group(1)] = float(m.group(2))
        except Exception:
            continue
    return (
        float(kv.get("ec_seconds_total", 0.0)),
        float(kv.get("learn_seconds_total", 0.0)),
        float(kv.get("oracle_seconds_total", 0.0)),
    )


def _extract_attempts(stdout: str) -> int:
    max_attempt = -1
    for m in _RE_ATTEMPT.finditer(stdout or ""):
        try:
            max_attempt = max(max_attempt, int(m.group(1)))
        except Exception:
            pass
    return (max_attempt + 1) if max_attempt >= 0 else 0


def _parse_line_id(file_path: str) -> int:
    name = Path(file_path).name
    m = re.match(r"^line_(\d{4})\.txt$", name)
    if not m:
        return -1
    return int(m.group(1))


def load_mutation_db_samples(mutation_db_path: str) -> list[tuple[int, str, str]]:
    """
    Return [(file_id, original_text, broken_text)].
    DBs are expected to have one row per file_path (we generate them that way).
    """
    def _clean(s: str) -> str:
        # Mutation DBs are generated from per-line files; ensure samples are single-line even
        # if the mutator inserted after a trailing newline from the source file.
        return (s or "").replace("\r", "").replace("\n", "")

    conn = sqlite3.connect(mutation_db_path)
    table = get_mutation_table_name(mutation_db_path, conn)
    cur = conn.cursor()
    cur.execute(f"SELECT file_path, original_text, mutated_text FROM {table} ORDER BY file_path, id")
    rows = cur.fetchall()
    conn.close()

    out: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for fp, orig, mut in rows:
        fp_s = str(fp or "")
        if fp_s in seen:
            continue
        seen.add(fp_s)
        out.append((_parse_line_id(fp_s), _clean(orig or ""), _clean(mut or "")))
    out.sort(key=lambda t: t[0])
    return out


def insert_tasks(*, db_path: str, format_key: str, samples: list[tuple[int, str, str]]) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _ensure_results_schema(cur)
    conn.commit()
    for file_id, orig_text, broken_text in samples:
        cur.execute(
            "SELECT 1 FROM results WHERE format=? AND file_id=? AND corrupted_index=? AND algorithm=? LIMIT 1",
            (format_key, file_id, 0, "betamax"),
        )
        if cur.fetchone():
            continue
        cur.execute(
            """
            INSERT INTO results (
              format, file_id, corrupted_index, algorithm,
              original_text, broken_text,
              repaired_text, fixed, iterations, repair_time,
              correct_runs, incorrect_runs, incomplete_runs,
              distance_original_broken, distance_broken_repaired, distance_original_repaired
            )
            VALUES (?, ?, ?, 'betamax', ?, ?, '', 0, 0, 0.0, 0, 0, 0, 0, 0, 0)
            """,
            (format_key, file_id, 0, orig_text, broken_text),
        )
    conn.commit()
    conn.close()


def _validate_file(file_path: str) -> bool:
    try:
        res = subprocess.run([str(VALIDATOR), file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        return res.returncode == 0
    except Exception:
        return False


def _precompute_cache(*, positives_file: str, negatives_file: str, cache_path: str, learner: str) -> None:
    os.makedirs(Path(cache_path).parent, exist_ok=True)
    cmd = [
        "python3",
        "betamax/app/betamax.py",
        "--positives",
        positives_file,
        "--negatives",
        negatives_file,
        "--category",
        "ISO8601",
        "--grammar-cache",
        cache_path,
        "--init-cache",
        "--learner",
        learner,
        "--oracle-validator",
        str(VALIDATOR),
    ]

    pre_mut = int(os.environ.get("LSTAR_PRECOMPUTE_MUTATIONS", "60"))
    if pre_mut > 0:
        cmd += ["--mutations", str(pre_mut)]

    env = dict(os.environ)
    env.setdefault("BETAMAX_EMIT_METRICS", "1")
    env.setdefault("LSTAR_PARSE_TIMEOUT", os.environ.get("LSTAR_PARSE_TIMEOUT", "100.0"))

    t0 = time.time()
    print(f"[INFO] Precomputing cache (no timeout): {cache_path}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=False)
    dt = time.time() - t0
    try:
        sz = os.path.getsize(cache_path) if os.path.exists(cache_path) else "NA"
    except Exception:
        sz = "NA"
    print(f"[INFO] Precompute finished in {dt:.2f}s, cache_size={sz}")


def _precompute_cache_cpp(*, positives_file: str, negatives_file: str, cache_path: str, learner: str) -> None:
    exe = cpp_bin_path()
    cmd = [
        exe,
        "--positives",
        positives_file,
        "--negatives",
        negatives_file,
        "--category",
        "ISO8601",
        "--learner",
        learner,
        "--repo-root",
        ".",
        "--oracle-validator",
        str(VALIDATOR),
        "--dfa-cache",
        cache_path,
        "--init-cache",
    ]
    pre_mut = int(os.environ.get("LSTAR_PRECOMPUTE_MUTATIONS", "60"))
    if pre_mut > 0:
        cmd += ["--mutations", str(pre_mut)]
    print(f"[INFO] Precomputing C++ DFA cache (no timeout): {cache_path}")
    t0 = time.time()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    dt = time.time() - t0
    try:
        sz = os.path.getsize(cache_path) if os.path.exists(cache_path) else "NA"
    except Exception:
        sz = "NA"
    print(f"[INFO] C++ precompute finished in {dt:.2f}s, cache_size={sz}")

def run_repairs(
    *,
    db_path: str,
    engine: str,
    learner: str,
    cache_path: Optional[str],
    positives_file: str,
    negatives_file: str,
    attempts: int,
    train_mutations: int,
    repair_timeout_s: int,
    limit: Optional[int],
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _ensure_results_schema(cur)
    conn.commit()

    cur.execute(
        """
        SELECT id, format, file_id, original_text, broken_text
        FROM results
        WHERE algorithm='betamax' AND fixed=0
        ORDER BY format, file_id, id
        """
    )
    rows = cur.fetchall()
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    print(f"[INFO] Pending rows: {len(rows)}")

    for (id_, format_key, file_id, original_text, broken_text) in rows:
        input_file = f"temp_iso8601_mixed_input_{id_}_{random.randint(0, 9999)}.txt"
        output_file = f"temp_iso8601_mixed_output_{id_}_{random.randint(0, 9999)}.txt"

        Path(input_file).write_text((broken_text or "").rstrip("\n") + "\n", encoding="utf-8")

        cmd = build_betamax_cmd(
            engine=engine,
            positives=positives_file,
            negatives=negatives_file,
            cache_path=cache_path,
            category="ISO8601",
            broken_file=input_file,
            output_file=output_file,
            attempts=attempts,
            mutations=train_mutations,
            learner=learner,
            oracle_cmd=str(VALIDATOR),
        )

        repaired_text = ""
        fixed = 0
        iterations = 0
        timed_out = 0
        return_code = -1
        repair_time = 0.0
        ec_time = learn_time = oracle_time = 0.0
        ec_ratio = learn_ratio = oracle_ratio = 0.0

        dist_ob = levenshtein_distance(original_text or "", broken_text or "")
        dist_br = 0
        dist_or = 0

        try:
            t0 = time.time()
            env = dict(os.environ)
            env.setdefault("BETAMAX_EMIT_METRICS", "1")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            stdout, _stderr = proc.communicate(timeout=repair_timeout_s)
            repair_time = time.time() - t0
            return_code = proc.returncode

            if engine == "python":
                iterations = _extract_attempts(stdout)
                ec_time, learn_time, oracle_time = _extract_betamax_metrics(stdout)
                if repair_time > 0:
                    ec_ratio = ec_time / repair_time
                    learn_ratio = learn_time / repair_time
                    oracle_ratio = oracle_time / repair_time

            if return_code == 0 and Path(output_file).exists():
                repaired_text = Path(output_file).read_text(encoding="utf-8", errors="replace")
                if _validate_file(output_file):
                    fixed = 1
                dist_br = levenshtein_distance(broken_text or "", repaired_text)
                dist_or = levenshtein_distance(original_text or "", repaired_text)
        except subprocess.TimeoutExpired:
            timed_out = 1
            repair_time = float(repair_timeout_s)
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

        cur.execute(
            """
            UPDATE results
            SET repaired_text=?,
                fixed=?,
                iterations=?,
                repair_time=?,
                distance_original_broken=?,
                distance_broken_repaired=?,
                distance_original_repaired=?,
                timed_out=?,
                return_code=?,
                ec_time=?,
                ec_ratio=?,
                learn_time=?,
                learn_ratio=?,
                oracle_time=?,
                oracle_ratio=?
            WHERE id=?
            """,
            (
                repaired_text,
                fixed,
                iterations,
                repair_time,
                dist_ob,
                dist_br,
                dist_or,
                timed_out,
                return_code,
                ec_time,
                ec_ratio,
                learn_time,
                learn_ratio,
                oracle_time,
                oracle_ratio,
                id_,
            ),
        )
        conn.commit()

        print(f"[INFO] {format_key} file_id={file_id} id={id_} fixed={fixed} rc={return_code} t={repair_time:.2f}s")

    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark: iso8601_500_mixed (train=400, test=100 mutated)")
    ap.add_argument("--db", default="iso8601_mixed500.db")
    ap.add_argument("--betamax-engine", default=None, help="cpp (default) or python")
    ap.add_argument("--learner", default=os.environ.get("BM_BETAMAX_LEARNER", "rpni"))
    ap.add_argument(
        "--precompute-learner",
        default=os.environ.get("BM_BETAMAX_CACHE_LEARNER", os.environ.get("LSTAR_CACHE_LEARNER")),
        help="learner to use for python precompute cache (default: same as --learner)",
    )
    ap.add_argument("--attempts", type=int, default=int(os.environ.get("BM_BETAMAX_ATTEMPTS", "500")))
    ap.add_argument("--train-mutations", type=int, default=int(os.environ.get("BM_TRAIN_MUTATIONS", "0")))
    ap.add_argument("--repair-timeout", type=int, default=int(os.environ.get("BM_REPAIR_TIMEOUT", "600")))
    ap.add_argument("--limit", type=int, default=None, help="process only N pending rows")
    ap.add_argument("--resume-only", action="store_true", help="skip task insertion and precompute")
    ap.add_argument("--skip-precompute", action="store_true", help="skip precompute stage")
    ap.add_argument("--mutation-types", nargs="*", default=DEFAULT_MUTATION_TYPES)
    args = ap.parse_args()

    engine = get_engine(args.betamax_engine, default="cpp")
    precompute_learner = (args.precompute_learner or "").strip() or args.learner

    if not DATASET_PATH.exists():
        raise SystemExit(f"[ERROR] Dataset not found: {DATASET_PATH}")
    if not VALIDATOR.exists():
        raise SystemExit(f"[ERROR] Validator not found: {VALIDATOR}")

    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) != 500:
        raise SystemExit(f"[ERROR] Expected 500 lines in {DATASET_PATH}, got {len(lines)}")

    train_lines = lines[:400]
    pos_file = f"temp_pos_iso8601_mixed500_{random.randint(0, 9999)}.txt"
    neg_file = f"temp_neg_iso8601_mixed500_{random.randint(0, 9999)}.txt"
    Path(pos_file).write_text("".join([(s.rstrip("\n") + "\n") for s in train_lines]), encoding="utf-8")
    Path(neg_file).write_text("", encoding="utf-8")

    cache_path: Optional[str] = None
    if engine == "python":
        cache_path = os.path.join("cache", f"iso8601_mixed500_{precompute_learner}.json")
    else:
        cache_path = os.path.join("cache", f"iso8601_mixed500_{precompute_learner}.dfa")

    try:
        create_database(args.db)

        if (not args.resume_only) and (not args.skip_precompute):
            # Custom requirement: NO timeout in precompute for this dataset.
            if precompute_learner != args.learner:
                print(
                    f"[WARN] Precompute learner ({precompute_learner}) != runtime learner ({args.learner}). "
                    "Runtime will still use the precomputed cache file path."
                )
            if engine == "python":
                _precompute_cache(
                    positives_file=pos_file,
                    negatives_file=neg_file,
                    cache_path=str(cache_path),
                    learner=precompute_learner,
                )
            else:
                _precompute_cache_cpp(
                    positives_file=pos_file,
                    negatives_file=neg_file,
                    cache_path=str(cache_path),
                    learner=precompute_learner,
                )

        if not args.resume_only:
            for mt in args.mutation_types:
                mt_norm = (mt or "").strip().lower()
                if not mt_norm:
                    continue
                mdb = Path("mutated_files") / f"{mt_norm}_iso8601_mixed500_test100.db"
                if not mdb.exists():
                    raise SystemExit(
                        f"[ERROR] Mutation DB missing: {mdb}\n"
                        f"Generate it with mutation_{mt_norm}.py first."
                    )
                samples = load_mutation_db_samples(str(mdb))
                samples = [s for s in samples if 400 <= s[0] <= 499]
                insert_tasks(db_path=args.db, format_key=f"{mt_norm}_iso8601_mixed500", samples=samples)

        run_repairs(
            db_path=args.db,
            engine=engine,
            learner=args.learner,
            cache_path=cache_path,
            positives_file=pos_file,
            negatives_file=neg_file,
            attempts=args.attempts,
            train_mutations=args.train_mutations,
            repair_timeout_s=args.repair_timeout,
            limit=args.limit,
        )
    finally:
        Path(pos_file).unlink(missing_ok=True)
        Path(neg_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
