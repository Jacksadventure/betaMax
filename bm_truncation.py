#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bm_betamax_backend import build_betamax_cmd, cpp_bin_path, get_engine, should_precompute_cache
from bm_ddmax_backend import (
    ddmax_repair_by_deletion,
    oracle_cmd_from_env_or_default,
    select_regex_oracle_arg,
    select_regex_oracle_cmd,
)
from bm_lstar_mutations import LStarMutationPool, get_mutation_table_name

# ------------------------------------------------------------------------------
# Configuration (defaults; most are overridable via CLI/env)
# ------------------------------------------------------------------------------
DATABASE_PATH = "truncation.db"
REPAIR_OUTPUT_DIR = "repair_results"
os.makedirs(REPAIR_OUTPUT_DIR, exist_ok=True)

ALL_ALGORITHMS = ["erepair", "earley", "betamax", "ddmax"]
REPAIR_ALGORITHMS = ["betamax"]

PROJECT_PATHS = {
    "dot": "project/bin/subjects/dot/build/dot_parser",
    "ini": "project/bin/subjects/ini/ini",
    "json": "project/bin/subjects/cjson/cjson",
    "lisp": "project/bin/subjects/sexp-parser/sexp",
    "obj": "project/bin/subjects/obj/build/obj_parser",
    "c": "project/bin/subjects/tiny/tiny",
    # Regex-based categories use match.py as oracle command string
    "date": "python3 match.py Date",
    "iso8601": "python3 match.py ISO8601",
    "time": "python3 match.py Time",
    "url": "python3 match.py URL",
    "isbn": "python3 match.py ISBN",
    "ipv4": "python3 match.py IPv4",
    "ipv6": "python3 match.py IPv6",
}

REGEX_DIR_TO_CATEGORY = {
    "date": "Date",
    "iso8601": "ISO8601",
    "time": "Time",
    "url": "URL",
    "isbn": "ISBN",
    "ipv4": "IPv4",
    "ipv6": "IPv6",
}
REGEX_FORMATS = set(REGEX_DIR_TO_CATEGORY.keys())

# For truncation experiments, default to JSON (can be overridden via --formats).
DEFAULT_FORMATS = ["json"]
FORMAT_CHOICES = list(dict.fromkeys(DEFAULT_FORMATS + list(REGEX_FORMATS) + ["dot", "ini", "lisp", "obj", "c"]))
VALID_FORMATS = FORMAT_CHOICES

# Mutation DB prefixes used by this runner. We keep the canonical prefix as "truncated"
# to match mutation_truncated.py and existing artifacts like truncated_*.db.
MUTATION_TYPES = ["truncated"]

# Train/Test split counts (default 50/50). Override via env BM_TRAIN_K, BM_TEST_K.
TRAIN_K = int(os.environ.get("BM_TRAIN_K", "50"))
TEST_K = int(os.environ.get("BM_TEST_K", "50"))

# Shared seed pool manager. Even though mutation augmentation is regex-only, seed pools
# are useful for all formats (e.g., JSON).
LSTAR_MUTATION_POOL = LStarMutationPool(TRAIN_K, REGEX_FORMATS, REGEX_DIR_TO_CATEGORY)

CACHE_ROOT = os.environ.get("LSTAR_CACHE_ROOT", "cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

BM_NEGATIVES_FROM_DB = os.environ.get("BM_NEGATIVES_FROM_DB", "0").lower() in ("1", "true", "yes")

# betaMax engine selection
BETAMAX_ENGINE = get_engine(default="cpp")

# Timeouts (seconds)
VALIDATION_TIMEOUT = int(os.environ.get("BM_VALIDATION_TIMEOUT", "600"))
REPAIR_TIMEOUT = int(os.environ.get("BM_REPAIR_TIMEOUT", "600"))

# Verbosity and run-control (set by CLI)
QUIET = False
LIMIT_N: Optional[int] = None
PAUSE_ON_EXIT = False


def _cache_path(fmt: str, learner: Optional[str] = None) -> str:
    learner_name = (learner or _runtime_betamax_learner()).replace("-", "_")
    ext = "json" if BETAMAX_ENGINE == "python" else "dfa"
    return os.path.join(CACHE_ROOT, f"lstar_{fmt}_{learner_name}.{ext}")


def _runtime_betamax_learner() -> str:
    return "rpni"


def _cache_betamax_learner() -> str:
    return "rpni_xover"


def _cpp_dfa_cache_ready(cache_path: str) -> bool:
    try:
        if not os.path.isfile(cache_path):
            return False
        with open(cache_path, "rb") as f:
            head = f.readline().decode("ascii", errors="ignore").strip()
            if head != "BMXDFA1":
                return False
            f.seek(0, os.SEEK_END)
            size = f.tell()
            n = min(size, 65536)
            f.seek(size - n)
            tail = f.read(n).decode("latin-1", errors="ignore")
        return ("LEARNER rpni_xover" in tail) and ("MERGE_HISTORY" in tail)
    except Exception:
        return False


def _ensure_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


# ------------------------------------------------------------------------------
# Mutation DB generation (truncated JSON)
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class TruncatedJsonConfig:
    input_db: str
    input_offset: int
    input_limit: Optional[int]
    folder: str
    validator: str
    max_attempts: int
    seed: Optional[int]


def _default_truncated_json_config() -> TruncatedJsonConfig:
    default_limit = None
    try:
        if (TRAIN_K + TEST_K) > 0:
            default_limit = TRAIN_K + TEST_K
    except Exception:
        default_limit = None
    return TruncatedJsonConfig(
        # Default source: the folder of JSON originals for truncation experiments.
        # (You can override to use an input DB via BM_TRUNCATED_JSON_INPUT_DB / --truncated-json-input-db.)
        input_db=os.environ.get("BM_TRUNCATED_JSON_INPUT_DB", ""),
        input_offset=int(os.environ.get("BM_TRUNCATED_JSON_INPUT_OFFSET", "0")),
        input_limit=int(os.environ["BM_TRUNCATED_JSON_INPUT_LIMIT"]) if os.environ.get("BM_TRUNCATED_JSON_INPUT_LIMIT") else default_limit,
        folder=os.environ.get("BM_TRUNCATED_JSON_FOLDER", "original_files/json_small_data"),
        validator=os.environ.get("BM_TRUNCATED_JSON_VALIDATOR", PROJECT_PATHS["json"]),
        max_attempts=int(os.environ.get("BM_TRUNCATED_MAX_ATTEMPTS", "100")),
        seed=int(os.environ["BM_TRUNCATED_SEED"]) if os.environ.get("BM_TRUNCATED_SEED") else None,
    )


def _db_has_mutations(db_path: str) -> bool:
    if not os.path.exists(db_path):
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            table = get_mutation_table_name(db_path, conn)
            cur = conn.cursor()
            cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
            return cur.fetchone() is not None
    except Exception:
        return False


def ensure_truncated_json_db(
    *,
    db_path: str,
    cfg: TruncatedJsonConfig,
    regen: bool = False,
) -> None:
    """
    Ensure mutated_files/truncated_json.db exists and has at least one row.
    Uses mutation_truncated.py to generate it.

    Default behavior (for parity with bm_single's JSON train/test split):
      - Read originals from `mutated_files/single_json.db` (or cfg.input_db)
      - Generate truncations for the first (TRAIN_K + TEST_K) rows in id order
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if regen and os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
    if not regen and _db_has_mutations(db_path):
        return

    validator_path = Path(cfg.validator)
    if not validator_path.exists():
        raise SystemExit(
            f"[ERROR] JSON validator not found: {cfg.validator}\n"
            "Build subjects first (see build_all.sh) or override via BM_TRUNCATED_JSON_VALIDATOR / --truncated-json-validator."
        )

    cmd = [
        "python3",
        "mutation_truncated.py",
        "--validator",
        str(validator_path),
        "--database",
        db_path,
        "--max-attempts",
        str(int(cfg.max_attempts)),
    ]
    input_db_path = Path(cfg.input_db) if (cfg.input_db or "").strip() else None
    if input_db_path and input_db_path.exists():
        cmd += ["--input-db", str(input_db_path)]
        if cfg.input_offset:
            cmd += ["--input-offset", str(int(cfg.input_offset))]
        if cfg.input_limit is not None:
            cmd += ["--input-limit", str(int(cfg.input_limit))]
    else:
        folder_path = Path(cfg.folder)
        if not folder_path.exists():
            raise SystemExit(
                f"[ERROR] JSON source folder not found: {cfg.folder}\n"
                "Override via BM_TRUNCATED_JSON_FOLDER / --truncated-json-folder."
            )
        cmd += ["--folder", str(folder_path)]
    if cfg.seed is not None:
        cmd += ["--seed", str(int(cfg.seed))]

    print(f"[INFO] Generating truncated JSON mutation DB: {db_path}")
    if not QUIET:
        if input_db_path and input_db_path.exists():
            print(f"[INFO]   input_db={input_db_path} (offset={cfg.input_offset}, limit={cfg.input_limit})")
        else:
            print(f"[INFO]   folder={folder_path}")
        print(f"[INFO]   validator={validator_path}")
        print(f"[INFO]   max_attempts={cfg.max_attempts}, seed={cfg.seed}")
    subprocess.run(cmd, check=True)

    if not _db_has_mutations(db_path):
        raise SystemExit(f"[ERROR] Truncated mutation DB generated but empty: {db_path}")


# ------------------------------------------------------------------------------
# Results DB helpers
# ------------------------------------------------------------------------------
def create_database(db_path: str) -> None:
    if os.path.exists(db_path):
        print(f"[WARNING] Database '{db_path}' already exists. It will be reused/overwritten.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
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
            correct_runs INTEGER,
            incorrect_runs INTEGER,
            incomplete_runs INTEGER,
            distance_original_broken INTEGER,
            distance_broken_repaired INTEGER,
            distance_original_repaired INTEGER
        )
        """
    )
    _ensure_results_schema(cursor)
    conn.commit()
    conn.close()
    if not QUIET:
        print(f"[INFO] Created/checked table 'results' in database '{db_path}'.")


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


def _results_columns(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("PRAGMA table_info(results)")
    return {row[1] for row in cursor.fetchall()}


def _update_results_row(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    *,
    id_: int,
    repaired_text: str,
    fixed: int,
    iterations: int,
    repair_time: float,
    correct_runs: int,
    incorrect_runs: int,
    incomplete_runs: int,
    distance_original_broken: int,
    distance_broken_repaired: int,
    distance_original_repaired: int,
    timed_out: int,
    return_code: int,
    ec_time: float,
    ec_ratio: float,
    learn_time: float,
    learn_ratio: float,
    oracle_time: float,
    oracle_ratio: float,
) -> None:
    cols = _results_columns(cursor)
    assignments = [
        "repaired_text = ?",
        "fixed = ?",
        "iterations = ?",
        "repair_time = ?",
        "correct_runs = ?",
        "incorrect_runs = ?",
        "incomplete_runs = ?",
        "distance_original_broken = ?",
        "distance_broken_repaired = ?",
        "distance_original_repaired = ?",
    ]
    params: list[object] = [
        repaired_text,
        fixed,
        iterations,
        repair_time,
        correct_runs,
        incorrect_runs,
        incomplete_runs,
        distance_original_broken,
        distance_broken_repaired,
        distance_original_repaired,
    ]

    optional = [
        ("timed_out", timed_out),
        ("return_code", return_code),
        ("ec_time", ec_time),
        ("ec_ratio", ec_ratio),
        ("learn_time", learn_time),
        ("learn_ratio", learn_ratio),
        ("oracle_time", oracle_time),
        ("oracle_ratio", oracle_ratio),
    ]
    for col, val in optional:
        if col not in cols:
            continue
        assignments.append(f"{col} = ?")
        params.append(val)

    params.append(id_)
    cursor.execute(
        f"UPDATE results SET {', '.join(assignments)} WHERE id = ?",
        params,
    )
    conn.commit()


# ------------------------------------------------------------------------------
# Mutation DB I/O
# ------------------------------------------------------------------------------
def load_test_samples_from_db(mutation_db_path: str):
    if not os.path.exists(mutation_db_path):
        print(f"[ERROR] Mutation database not found: {mutation_db_path}")
        return []

    conn = sqlite3.connect(mutation_db_path)
    table_name = get_mutation_table_name(mutation_db_path, conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, original_text, mutated_text FROM {table_name} ORDER BY id")
    samples = cursor.fetchall()
    conn.close()
    return [(row[0], 0, row[1], row[2]) for row in samples]


def insert_test_samples_to_db(db_path: str, format_key: str, test_samples: list) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for (file_id, cindex, orig_text, broken_text) in test_samples:
        for alg in REPAIR_ALGORITHMS:
            base_f = format_key.split("_")[-1]
            if base_f in REGEX_FORMATS and alg not in ("erepair", "earley", "betamax", "ddmax"):
                continue
            cursor.execute(
                "SELECT 1 FROM results WHERE format=? AND file_id=? AND corrupted_index=? AND algorithm=? LIMIT 1",
                (format_key, file_id, cindex, alg),
            )
            if cursor.fetchone():
                continue
            cursor.execute(
                """
                INSERT INTO results (format, file_id, corrupted_index, algorithm,
                                     original_text, broken_text,
                                     repaired_text, fixed, iterations, repair_time,
                                     correct_runs, incorrect_runs, incomplete_runs,
                                     distance_original_broken, distance_broken_repaired, distance_original_repaired)
                VALUES (?, ?, ?, ?, ?, ?, '', 0, 0, 0.0, 0, 0, 0, 0, 0, 0)
                """,
                (format_key, file_id, cindex, alg, orig_text, broken_text),
            )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Validation + distance
# ------------------------------------------------------------------------------
def validate_with_external_tool(file_path: str, format_key: str, algorithm: str) -> bool:
    base_format = format_key.split("_")[-1]
    try:
        if base_format in REGEX_FORMATS:
            category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
            if algorithm == "erepair":
                cmd = ["python3", "match_partial.py", category, file_path]
            elif algorithm == "betamax":
                validator_bin = os.path.join("validators", f"validate_{base_format}")
                wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
                if os.path.exists(validator_bin):
                    cmd = [validator_bin, file_path]
                elif os.path.exists(wrapper):
                    cmd = [wrapper, file_path]
                else:
                    cmd = ["python3", "match.py", category, file_path]
            else:
                validator_path = os.path.join("validators", f"validate_{base_format}")
                if os.path.exists(validator_path):
                    cmd = [validator_path, file_path]
                else:
                    wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
                    if os.path.exists(wrapper):
                        cmd = [wrapper, file_path]
                    else:
                        cmd = ["python3", "match.py", category, file_path]
        else:
            exe = PROJECT_PATHS.get(base_format)
            if not exe:
                print(f"[ERROR] No validator configured for format '{base_format}'")
                return False
            cmd = [exe, file_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=VALIDATION_TIMEOUT,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Validation timeout for '{file_path}', format '{format_key}' (algorithm={algorithm})")
        return False
    except Exception as e:
        print(f"[ERROR] Could not run validator for format '{format_key}' (algorithm={algorithm}): {e}")
        return False


def _normalize_for_distance(text: Optional[str]) -> str:
    return (text or "").rstrip("\r\n")


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


# ------------------------------------------------------------------------------
# Metrics extraction
# ------------------------------------------------------------------------------
_RE_METRIC = re.compile(r"^\[METRICS\]\s+([a-zA-Z0-9_]+)=([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_RE_PROFILE_EC = re.compile(r"^\[PROFILE\]\s+ec_earley(?:\(relearn\))?:\s+([0-9]*\.?[0-9]+)s\s*$", re.MULTILINE)
_RE_PROFILE_LEARN_TOTAL = re.compile(r"^\[PROFILE\]\s+learn_grammar\(total\):\s+([0-9]*\.?[0-9]+)s\s*;", re.MULTILINE)
_RE_PROFILE_LEARN_RELEARN = re.compile(r"^\[PROFILE\]\s+learn_grammar\(relearn\):\s+([0-9]*\.?[0-9]+)s\s*;", re.MULTILINE)
_RE_PROFILE_ORACLE = re.compile(r"^\[PROFILE\]\s+oracle_validate(?:\([^\)]*\))?:\s+([0-9]*\.?[0-9]+)s\s*$", re.MULTILINE)


def _extract_betamax_metrics(stdout: str) -> tuple[float, float, float]:
    kv: dict[str, float] = {}
    for m in _RE_METRIC.finditer(stdout or ""):
        try:
            kv[m.group(1)] = float(m.group(2))
        except Exception:
            continue
    if "ec_seconds_total" in kv or "learn_seconds_total" in kv or "oracle_seconds_total" in kv:
        return (
            float(kv.get("ec_seconds_total", 0.0)),
            float(kv.get("learn_seconds_total", 0.0)),
            float(kv.get("oracle_seconds_total", 0.0)),
        )
    ec_time = sum(float(m.group(1)) for m in _RE_PROFILE_EC.finditer(stdout or "") if m.group(1))
    learn_time = 0.0
    for m in _RE_PROFILE_LEARN_TOTAL.finditer(stdout or ""):
        try:
            learn_time += float(m.group(1))
        except Exception:
            pass
    for m in _RE_PROFILE_LEARN_RELEARN.finditer(stdout or ""):
        try:
            learn_time += float(m.group(1))
        except Exception:
            pass
    oracle_time = 0.0
    for m in _RE_PROFILE_ORACLE.finditer(stdout or ""):
        try:
            oracle_time += float(m.group(1))
        except Exception:
            pass
    return ec_time, learn_time, oracle_time


def extract_oracle_info(stdout: str) -> tuple[int, int, int, int]:
    match = re.search(
        r"\*\*\* Number of required oracle runs: (\d+) correct: (\d+) incorrect: (\d+)\s*$",
        stdout,
        re.MULTILINE,
    )
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), 0
    match = re.search(
        r"\*\*\* Number of required oracle runs: (\d+) correct: (\d+) incorrect: (\d+) incomplete: (\d+) \*\*\*",
        stdout,
    )
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return 0, 0, 0, 0


def extract_oracle_seconds_total(stdout: str) -> float:
    last_val = 0.0
    for m in _RE_METRIC.finditer(stdout or ""):
        if m.group(1) != "oracle_seconds_total":
            continue
        try:
            last_val = float(m.group(2))
        except Exception:
            continue
    return last_val


def extract_lstar_attempts(stdout: str) -> int:
    max_attempt = -1
    try:
        for m in re.finditer(r"\[ATTEMPT\s+(\d+)\]", stdout or ""):
            n = int(m.group(1))
            if n > max_attempt:
                max_attempt = n
        if max_attempt >= 0:
            return max_attempt + 1
        m2 = re.findall(r"attempt\s+(\d+)\s*/\s*(\d+)", stdout or "", flags=re.IGNORECASE)
        if m2:
            max_attempt = max(int(x) for x, _ in m2)
            return max_attempt + 1
    except Exception:
        pass
    return 0


def _materialize_jar_output(output_dir: str, input_file: str, output_file: str) -> bool:
    try:
        base = os.path.basename(input_file)
        direct = os.path.join(output_dir, base)
        if os.path.isfile(direct):
            shutil.copyfile(direct, output_file)
            return True

        candidates: list[str] = []
        for root, _dirs, files in os.walk(output_dir):
            for fn in files:
                p = os.path.join(root, fn)
                if os.path.isfile(p):
                    candidates.append(p)

        if not candidates:
            return False

        ext = os.path.splitext(base)[1].lower()
        same_ext = [p for p in candidates if os.path.splitext(p)[1].lower() == ext] or candidates
        same_ext.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        shutil.copyfile(same_ext[0], output_file)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Repair execution
# ------------------------------------------------------------------------------
def repair_and_update_entry(cursor: sqlite3.Cursor, conn: sqlite3.Connection, row) -> None:
    (
        id_,
        format_key,
        file_id,
        corrupted_index,
        algorithm,
        original_text,
        broken_text,
        _repaired,
        _fixed,
        _iter,
        _rtime,
        _correct,
        _incorrect,
        _incomplete,
        _distOB,
        _distBR,
        _distOR,
    ) = row

    if not QUIET:
        print(
            f"[INFO] Repairing ID={id_}, format={format_key}, algorithm={algorithm}, "
            f"file_id={file_id}, corrupted_index={corrupted_index}"
        )

    base_format = format_key.split("_")[-1]
    ext = base_format

    if algorithm != "erepair":
        input_file = f"temp_{id_}_{random.randint(0, 9999)}_input.{ext}"
        output_file = os.path.join(REPAIR_OUTPUT_DIR, f"repair_{id_}_output.{ext}")
    else:
        input_file = f"temp_{id_}_{random.randint(0, 9999)}_input.{format_key}"
        output_file = os.path.join(REPAIR_OUTPUT_DIR, f"repair_{id_}_output.{format_key}")

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(broken_text or "")

    distance_original_broken = levenshtein_distance(original_text or "", broken_text or "")
    distance_broken_repaired = -1
    distance_original_repaired = -1

    repaired_text = ""
    fixed = 0
    iterations, correct_runs, incorrect_runs, incomplete_runs = 0, 0, 0, 0
    repair_time = 0.0
    timed_out = 0
    return_code = None
    ec_time = 0.0
    ec_ratio = 0.0
    learn_time = 0.0
    learn_ratio = 0.0
    oracle_time = 0.0
    oracle_ratio = 0.0

    pos_file: Optional[str] = None
    neg_file: Optional[str] = None
    jar_out_dir: Optional[str] = None
    internal_ddmax = False
    ddmax_oracle_cmd: Optional[list[str]] = None

    cmd: Optional[list[str]] = None
    if algorithm == "erepair":
        if base_format in REGEX_FORMATS:
            category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
            oracle_executable = f"python3 match_partial.py {category}"
        else:
            oracle_executable = PROJECT_PATHS.get(base_format)
        if not oracle_executable:
            print(f"[ERROR] No oracle executable for format {base_format}")
            return
        cmd = ["./erepair", oracle_executable, input_file, output_file]
    elif algorithm == "earley":
        if base_format in REGEX_FORMATS:
            category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
            oracle_executable = f"re2-server:{category}"
        else:
            oracle_executable = PROJECT_PATHS.get(base_format)
        if not oracle_executable:
            print(f"[ERROR] No oracle executable for format {base_format}")
            return
        cmd = ["./earleyrepairer", oracle_executable, input_file, output_file]
    elif algorithm == "betamax":
        K = TRAIN_K
        category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
        runtime_learner = _runtime_betamax_learner()
        cache_learner = _cache_betamax_learner()
        cache_path = _cache_path(base_format, cache_learner)
        if not os.path.exists(cache_path):
            cache_path = _cache_path(base_format, runtime_learner)

        pos_file = f"temp_pos_{id_}_{random.randint(0, 9999)}.txt"
        neg_file = f"temp_neg_{id_}_{random.randint(0, 9999)}.txt"

        seed_pos: list[str] = []
        seed_neg: list[str] = []
        mut_pos: list[str] = []
        mut_neg: list[str] = []

        # For JSON truncation experiments, default learning pool is the same set
        # as the truncation DB itself (i.e., truncated_json). Override via env:
        #   BM_TRUNC_JSON_TRAIN_SOURCE=single  -> use single_json for learning
        seed_key = format_key
        if base_format == "json":
            src_mut = os.environ.get("BM_TRUNC_JSON_TRAIN_SOURCE", "truncated").strip() or "truncated"
            if src_mut == "truncated":
                seed_key = "truncated_json"
            else:
                seed_key = f"{src_mut}_json"

        if LSTAR_MUTATION_POOL:
            try:
                LSTAR_MUTATION_POOL.ensure_format(seed_key)
            except Exception as pool_err:
                if not QUIET:
                    print(f"[WARN] (ID={id_}) Failed to prime mutation pool for {seed_key}: {pool_err}")
            seed_pos = LSTAR_MUTATION_POOL.get_seed_positives(seed_key)
            seed_neg = LSTAR_MUTATION_POOL.get_seed_negatives(seed_key)
            mut_pos = LSTAR_MUTATION_POOL.get_mutation_positives(seed_key)
            mut_neg = LSTAR_MUTATION_POOL.get_mutation_negatives(seed_key)

        filtered_pos: list[str] = []
        if seed_pos:
            filtered_pos = [(s or "").rstrip("\n") for s in seed_pos if (s or "") != (original_text or "")]

        try:
            if not filtered_pos:
                mdb_key = seed_key
                mdb_path = os.path.join("mutated_files", f"{mdb_key}.db")
                with sqlite3.connect(mdb_path) as conn2:
                    table_name = get_mutation_table_name(mdb_path, conn2)
                    cur2 = conn2.cursor()
                    cur2.execute(
                        f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {K}"
                    )
                    rows = cur2.fetchall()
                filtered_pos = [(r[0] or "").rstrip("\n") for r in rows if (r[0] or "") != (original_text or "")]
            if mut_pos:
                filtered_pos.extend([(s or "").rstrip("\n") for s in mut_pos])
            with open(pos_file, "w", encoding="utf-8") as pf:
                for line in filtered_pos:
                    pf.write(line + "\n")
        except Exception as e:
            try:
                with open(pos_file, "w", encoding="utf-8") as pf:
                    pass
            except Exception:
                pass
            if not QUIET:
                print(f"[DEBUG] (ID={id_}) Failed to build positives file: {e}")

        try:
            neg_lines: list[str] = []
            if seed_neg:
                neg_lines = [(s or "").rstrip("\n") for s in seed_neg]
            if mut_neg:
                neg_lines.extend([(s or "").rstrip("\n") for s in mut_neg])
            if not neg_lines and BM_NEGATIVES_FROM_DB:
                mdb_key = seed_key
                mdb_path = os.path.join("mutated_files", f"{mdb_key}.db")
                with sqlite3.connect(mdb_path) as conn3:
                    table_name = get_mutation_table_name(mdb_path, conn3)
                    cur3 = conn3.cursor()
                    cur3.execute(
                        f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {K}"
                    )
                    rows = cur3.fetchall()
                neg_lines = [(r[0] or "").rstrip("\n") for r in rows]
            with open(neg_file, "w", encoding="utf-8") as nf:
                for line in neg_lines:
                    nf.write(line + "\n")
        except Exception as e:
            try:
                with open(neg_file, "w", encoding="utf-8") as nf:
                    pass
            except Exception:
                pass
            if not QUIET:
                print(f"[DEBUG] (ID={id_}) Failed to build negatives file: {e}")

        attempts = int(os.environ.get("BM_TRUNCATION_ATTEMPTS", "500"))

        oracle_override = os.environ.get("LSTAR_ORACLE_VALIDATOR")
        oracle_bin = os.path.join("validators", f"validate_{base_format}")
        oracle_wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
        if oracle_override:
            oracle_cmd = oracle_override
        else:
            if base_format in REGEX_FORMATS:
                if os.path.exists(oracle_bin):
                    oracle_cmd = oracle_bin
                elif os.path.exists(oracle_wrapper):
                    oracle_cmd = oracle_wrapper
                else:
                    oracle_cmd = select_regex_oracle_arg(base_format, category)
            else:
                oracle_cmd = PROJECT_PATHS.get(base_format)

        cmd = build_betamax_cmd(
            engine=BETAMAX_ENGINE,
            positives=pos_file,
            negatives=neg_file,
            cache_path=cache_path,
            category=category,
            broken_file=input_file,
            output_file=output_file,
            attempts=attempts,
            mutations=int(os.environ.get("BM_BETAMAX_MUTATIONS", "0")),
            learner=runtime_learner,
            oracle_cmd=oracle_cmd,
        )
    elif algorithm == "ddmax" and base_format in REGEX_FORMATS:
        internal_ddmax = True
        category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
        ddmax_oracle_cmd = oracle_cmd_from_env_or_default(select_regex_oracle_cmd(base_format, category))
        cmd = None
    else:
        raise RuntimeError(f"Unsupported algorithm for format={base_format}: {algorithm}")

    proc = None
    try:
        start_time = time.time()
        stdout = ""
        stderr = ""

        local_timeout = REPAIR_TIMEOUT
        if algorithm == "betamax":
            try:
                if os.environ.get("LSTAR_EC_TIMEOUT"):
                    local_timeout = int(os.environ["LSTAR_EC_TIMEOUT"])
            except Exception:
                pass

        if internal_ddmax:
            if not ddmax_oracle_cmd:
                raise RuntimeError("ddmax oracle command not configured")
            per_call = float(os.environ.get("DDMAX_PER_CALL_TIMEOUT", "2.0"))
            repaired_text, stats = ddmax_repair_by_deletion(
                broken_text=broken_text or "",
                oracle_cmd_prefix=ddmax_oracle_cmd,
                file_suffix=ext,
                timeout_s=float(local_timeout),
                per_call_timeout_s=per_call,
            )
            with open(output_file, "w", encoding="utf-8") as rf:
                rf.write(repaired_text)
            repair_time = time.time() - start_time
            return_code = 0
            timed_out = 1 if stats.timed_out else 0
            stdout = (
                f"*** Number of required oracle runs: {stats.total_runs} "
                f"correct: {stats.accepted_runs} incorrect: {stats.rejected_runs} incomplete: 0 ***\n"
            )
        else:
            if not cmd:
                raise RuntimeError("command not configured")
            if not QUIET:
                print(f"[DEBUG] (ID={id_}, ALG={algorithm}) Running command: {' '.join(str(x) for x in cmd)}")
            env = dict(os.environ)
            if algorithm == "betamax":
                env.setdefault("LSTAR_PARSE_TIMEOUT", os.environ.get("LSTAR_PARSE_TIMEOUT", "100.0"))
                env["BETAMAX_EMIT_METRICS"] = "1"
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            stdout, stderr = proc.communicate(timeout=local_timeout)
            repair_time = time.time() - start_time
            return_code = proc.returncode

        it_o, correct_runs, incorrect_runs, incomplete_runs = extract_oracle_info(stdout)
        if algorithm == "betamax":
            if BETAMAX_ENGINE == "python":
                iterations = extract_lstar_attempts(stdout)
                ec_time, learn_time, oracle_time = _extract_betamax_metrics(stdout)
                if repair_time > 0:
                    ec_ratio = ec_time / repair_time
                    learn_ratio = learn_time / repair_time
                    oracle_ratio = oracle_time / repair_time
            else:
                iterations = it_o
        else:
            iterations = it_o
            if algorithm == "erepair":
                oracle_time = extract_oracle_seconds_total(stdout)
                if repair_time > 0:
                    oracle_ratio = oracle_time / repair_time

        if not QUIET:
            print(f"--- STDOUT (ID={id_}) ---\n{stdout}\n")
            print(f"--- STDERR (ID={id_}) ---\n{stderr}\n")

        if (return_code == 0) and jar_out_dir:
            _materialize_jar_output(jar_out_dir, input_file, output_file)

        if (return_code == 0) and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as rf:
                repaired_text = rf.read()
            if validate_with_external_tool(output_file, format_key, algorithm):
                fixed = 1
            distance_broken_repaired = levenshtein_distance(broken_text or "", repaired_text or "")
            distance_original_repaired = levenshtein_distance(original_text or "", repaired_text or "")

    except subprocess.TimeoutExpired as e:
        print(f"[ERROR] Repair timed out for entry ID={id_}, alg={algorithm}, timeout={local_timeout}s")
        timed_out = 1
        repair_time = float(local_timeout)
        stdout = _ensure_text(getattr(e, "output", None))
        stderr = _ensure_text(getattr(e, "stderr", None))
        try:
            if proc:
                proc.kill()
        except Exception:
            pass
        try:
            if proc:
                out2, err2 = proc.communicate(timeout=5)
                stdout += _ensure_text(out2)
                stderr += _ensure_text(err2)
        except Exception:
            pass
        return_code = proc.returncode if (proc and proc.returncode is not None) else -9
        if algorithm == "betamax":
            iterations = extract_lstar_attempts(stdout)
            ec_time, learn_time, oracle_time = _extract_betamax_metrics(stdout)
            if repair_time > 0:
                ec_ratio = ec_time / repair_time
                learn_ratio = learn_time / repair_time
                oracle_ratio = oracle_time / repair_time
    except Exception as e:
        print(f"[ERROR] Repair failed for entry ID={id_}, alg={algorithm}: {e}")
        return_code = return_code if return_code is not None else -1
    finally:
        for p in (pos_file, neg_file):
            if not p:
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            if os.path.exists(input_file):
                os.remove(input_file)
        except Exception:
            pass
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception:
            pass
        if jar_out_dir:
            try:
                if os.path.isdir(jar_out_dir):
                    shutil.rmtree(jar_out_dir, ignore_errors=True)
            except Exception:
                pass

    _update_results_row(
        cursor,
        conn,
        id_=id_,
        repaired_text=repaired_text,
        fixed=fixed,
        iterations=iterations,
        repair_time=repair_time,
        correct_runs=correct_runs,
        incorrect_runs=incorrect_runs,
        incomplete_runs=incomplete_runs,
        distance_original_broken=distance_original_broken,
        distance_broken_repaired=distance_broken_repaired,
        distance_original_repaired=distance_original_repaired,
        timed_out=timed_out,
        return_code=int(return_code if return_code is not None else -1),
        ec_time=ec_time,
        ec_ratio=ec_ratio,
        learn_time=learn_time,
        learn_ratio=learn_ratio,
        oracle_time=oracle_time,
        oracle_ratio=oracle_ratio,
    )


def rerun_repairs_for_selected_formats(db_path: str, selected_formats: list[str], max_workers: Optional[int]) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _ensure_results_schema(cursor)
    conn.commit()

    cursor.execute(
        """
        SELECT id, format, file_id, corrupted_index, algorithm,
               original_text, broken_text, repaired_text, fixed,
               iterations, repair_time, correct_runs, incorrect_runs,
               incomplete_runs, distance_original_broken, distance_broken_repaired,
               distance_original_repaired
        FROM results
        """
    )
    entries = cursor.fetchall()

    def _allowed(row) -> bool:
        fmt_key = row[1]
        alg = row[4]
        base_f = fmt_key.split("_")[-1]
        if alg not in REPAIR_ALGORITHMS:
            return False
        if base_f in REGEX_FORMATS and alg not in ("earley", "erepair", "betamax", "ddmax"):
            return False
        return fmt_key in selected_formats

    filtered = [row for row in entries if _allowed(row)]
    if LIMIT_N:
        filtered = filtered[: int(LIMIT_N)]
    if not QUIET:
        print(f"[INFO] Found {len(filtered)} entries to (re)process.")

    if not max_workers:
        max_workers = os.cpu_count() or 4
    if not QUIET:
        print(f"[INFO] Starting ThreadPoolExecutor with max_workers={max_workers}")

    def _worker(row):
        thread_conn = sqlite3.connect(db_path, timeout=30)
        thread_cursor = thread_conn.cursor()
        try:
            repair_and_update_entry(thread_cursor, thread_conn, row)
        finally:
            thread_conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(_worker, filtered)

    conn.close()
    print("[INFO] Repair process completed!")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Truncation benchmark runner (supports betamax + truncated mutations)")
    parser.add_argument("--db", default=DATABASE_PATH, help="Path to results SQLite DB")
    parser.add_argument("--formats", nargs="+", choices=FORMAT_CHOICES, help="Formats to include (default: json)")
    parser.add_argument(
        "--mutations",
        nargs="+",
        default=MUTATION_TYPES,
        help="Mutation DB prefixes (default: truncated). Accepts alias: truncation -> truncated.",
    )
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS, help="Override algorithms to run")
    parser.add_argument("--resume-only", action="store_true", help="Skip sample insertion, only resume repairs")
    parser.add_argument("--resume", action="store_true", help="Alias of --resume-only (also skips precompute)")
    parser.add_argument("--max-workers", type=int, help="Max parallel workers (default: cpu count)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--limit", type=int, help="Limit number of entries to (re)process")
    parser.add_argument("--pause-on-exit", action="store_true", help="Pause for keypress before exiting")
    parser.add_argument(
        "--betamax-engine",
        choices=["python", "cpp"],
        default=None,
        help="Select betaMax engine backend for algorithm=betamax (default: env BM_BETAMAX_ENGINE or 'cpp')",
    )
    parser.add_argument(
        "--generate-mutations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate missing truncated JSON mutation DBs via mutation_truncated.py (default: enabled).",
    )
    parser.add_argument(
        "--regen-mutations",
        action="store_true",
        help="Force regeneration of truncated JSON mutation DB even if it exists.",
    )
    parser.add_argument("--truncated-json-folder", default=None, help="Source folder for truncated JSON generation")
    parser.add_argument("--truncated-json-validator", default=None, help="Validator executable for truncated JSON generation")
    parser.add_argument("--truncated-json-max-attempts", type=int, default=None, help="Max attempts per file for truncation")
    parser.add_argument("--truncated-json-seed", type=int, default=None, help="Optional RNG seed for truncation generation")
    parser.add_argument(
        "--truncated-json-input-db",
        default=None,
        help="Input mutation DB to source original_text from (default: mutated_files/single_json.db)",
    )
    parser.add_argument("--truncated-json-input-offset", type=int, default=None, help="Offset into --truncated-json-input-db (ORDER BY id)")
    parser.add_argument("--truncated-json-input-limit", type=int, default=None, help="Limit for --truncated-json-input-db (ORDER BY id)")
    args = parser.parse_args()

    global QUIET, LIMIT_N, PAUSE_ON_EXIT, BETAMAX_ENGINE
    QUIET = bool(args.quiet)
    LIMIT_N = args.limit
    PAUSE_ON_EXIT = bool(args.pause_on_exit)
    BETAMAX_ENGINE = get_engine(args.betamax_engine, default="cpp")

    if BETAMAX_ENGINE == "cpp" and ("betamax" in (args.algorithms or REPAIR_ALGORITHMS)):
        exe = cpp_bin_path()
        if not os.path.exists(exe):
            raise SystemExit(
                f"[ERROR] C++ betaMax binary not found: {exe}\n"
                "Build it first:\n"
                "  cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build betamax_cpp/build -j\n"
                "Or run with: --betamax-engine python"
            )

    if args.algorithms:
        REPAIR_ALGORITHMS[:] = args.algorithms

    # Normalize mutation type tokens (support old naming).
    normalized: list[str] = []
    for tok in (args.mutations or MUTATION_TYPES):
        if tok == "truncation":
            normalized.append("truncated")
        else:
            normalized.append(tok)
    args.mutations = normalized

    selected_formats = args.formats if args.formats else DEFAULT_FORMATS
    db_path = args.db

    # 0) Ensure truncated JSON mutation DB exists (only when inserting tasks).
    trunc_cfg = _default_truncated_json_config()
    if args.truncated_json_input_db is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=args.truncated_json_input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=trunc_cfg.input_limit,
            folder=trunc_cfg.folder,
            validator=trunc_cfg.validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_input_offset is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=int(args.truncated_json_input_offset),
            input_limit=trunc_cfg.input_limit,
            folder=trunc_cfg.folder,
            validator=trunc_cfg.validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_input_limit is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=int(args.truncated_json_input_limit),
            folder=trunc_cfg.folder,
            validator=trunc_cfg.validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_folder is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=trunc_cfg.input_limit,
            folder=args.truncated_json_folder,
            validator=trunc_cfg.validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_validator is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=trunc_cfg.input_limit,
            folder=trunc_cfg.folder,
            validator=args.truncated_json_validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_max_attempts is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=trunc_cfg.input_limit,
            folder=trunc_cfg.folder,
            validator=trunc_cfg.validator,
            max_attempts=int(args.truncated_json_max_attempts),
            seed=trunc_cfg.seed,
        )
    if args.truncated_json_seed is not None:
        trunc_cfg = TruncatedJsonConfig(
            input_db=trunc_cfg.input_db,
            input_offset=trunc_cfg.input_offset,
            input_limit=trunc_cfg.input_limit,
            folder=trunc_cfg.folder,
            validator=trunc_cfg.validator,
            max_attempts=trunc_cfg.max_attempts,
            seed=int(args.truncated_json_seed),
        )

    if args.generate_mutations and not (args.resume_only or args.resume):
        for m in args.mutations:
            if m != "truncated":
                continue
            if "json" not in selected_formats:
                continue
            ensure_truncated_json_db(
                db_path=os.path.join("mutated_files", "truncated_json.db"),
                cfg=trunc_cfg,
                regen=bool(args.regen_mutations),
            )

    # 1) Create or reuse results DB
    create_database(db_path)

    # 2) Optional precompute caches (mirrors bm_single/bm_triple behavior).
    if "betamax" in REPAIR_ALGORITHMS and should_precompute_cache(BETAMAX_ENGINE) and not (args.resume_only or args.resume):
        os.makedirs(CACHE_ROOT, exist_ok=True)
        cache_learner = _cache_betamax_learner()
        runtime_learner = _runtime_betamax_learner()
        for mutation_type in args.mutations:
            for fmt in selected_formats:
                format_key = f"{mutation_type}_{fmt}"
                cache_path = _cache_path(fmt, cache_learner)
                if BETAMAX_ENGINE == "cpp":
                    if _cpp_dfa_cache_ready(cache_path):
                        continue
                else:
                    if os.path.exists(cache_path):
                        continue

                preferred = os.environ.get("LSTAR_CACHE_SOURCE_MUTATION", "single")
                mutation_db_path = None
                for src in [preferred, "single", "double", "triple", mutation_type]:
                    cand = os.path.join("mutated_files", f"{src}_{fmt}.db")
                    if os.path.exists(cand):
                        mutation_db_path = cand
                        break
                if not mutation_db_path:
                    continue

                pos_file = f"temp_pos_cache_{format_key}_{random.randint(0,9999)}.txt"
                neg_file = f"temp_neg_cache_{format_key}_{random.randint(0,9999)}.txt"
                try:
                    pre_k = int(os.environ.get("LSTAR_PRECOMP_K", str(TRAIN_K)))
                    connc = sqlite3.connect(mutation_db_path)
                    curc = connc.cursor()
                    table_name = get_mutation_table_name(mutation_db_path, connc)
                    curc.execute(
                        f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {pre_k}"
                    )
                    rows = curc.fetchall()
                    with open(pos_file, "w", encoding="utf-8") as pf:
                        for r in rows:
                            pf.write((r[0] or "").rstrip("\n") + "\n")
                    with open(neg_file, "w", encoding="utf-8") as nf:
                        if BM_NEGATIVES_FROM_DB:
                            curc.execute(
                                f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {pre_k}"
                            )
                            rows2 = curc.fetchall()
                            for r in rows2:
                                nf.write((r[0] or "").rstrip("\n") + "\n")
                    connc.close()

                    category = REGEX_DIR_TO_CATEGORY.get(fmt, fmt)
                    oracle_override = os.environ.get("LSTAR_ORACLE_VALIDATOR")
                    oracle_bin = os.path.join("validators", f"validate_{fmt}")
                    oracle_wrapper = os.path.join("validators", "regex", f"validate_{fmt}")
                    if oracle_override:
                        oracle_cmd = oracle_override
                    else:
                        if fmt in REGEX_FORMATS:
                            if os.path.exists(oracle_bin):
                                oracle_cmd = oracle_bin
                            elif os.path.exists(oracle_wrapper):
                                oracle_cmd = oracle_wrapper
                            else:
                                oracle_cmd = select_regex_oracle_arg(fmt, category)
                        else:
                            oracle_cmd = PROJECT_PATHS.get(fmt)

                    pre_mut = int(os.environ.get("LSTAR_PRECOMPUTE_MUTATIONS", "60"))

                    if BETAMAX_ENGINE == "python":
                        cmd = [
                            "python3",
                            "betamax/app/betamax.py",
                            "--positives",
                            pos_file,
                            "--negatives",
                            neg_file,
                            "--category",
                            category,
                            "--grammar-cache",
                            cache_path,
                            "--init-cache",
                            "--learner",
                            cache_learner,
                        ]
                        if oracle_cmd:
                            cmd += ["--oracle-validator", oracle_cmd]
                        # Equivalence speed knobs via env for precompute as well
                        if os.environ.get("LSTAR_EQ_MAX_LENGTH"):
                            cmd += ["--eq-max-length", os.environ["LSTAR_EQ_MAX_LENGTH"]]
                        if os.environ.get("LSTAR_EQ_SAMPLES_PER_LENGTH"):
                            cmd += ["--eq-samples-per-length", os.environ["LSTAR_EQ_SAMPLES_PER_LENGTH"]]
                        if os.environ.get("LSTAR_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
                            cmd += ["--eq-disable-sampling"]
                        if os.environ.get("LSTAR_EQ_SKIP_NEGATIVES", "").lower() in ("1", "true", "yes"):
                            cmd += ["--eq-skip-negatives"]
                        if os.environ.get("LSTAR_EQ_MAX_ORACLE"):
                            cmd += ["--eq-max-oracle", os.environ["LSTAR_EQ_MAX_ORACLE"]]
                        if pre_mut > 0:
                            cmd += ["--mutations", str(pre_mut)]
                    else:
                        exe = cpp_bin_path()
                        cmd = [
                            exe,
                            "--positives",
                            pos_file,
                            "--negatives",
                            neg_file,
                            "--category",
                            category,
                            "--dfa-cache",
                            cache_path,
                            "--init-cache",
                            "--learner",
                            cache_learner,
                            "--repo-root",
                            ".",
                            "--incremental",
                        ]
                        if oracle_cmd:
                            cmd += ["--oracle-validator", oracle_cmd]
                        if os.environ.get("LSTAR_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
                            cmd += ["--eq-disable-sampling"]
                        if os.environ.get("LSTAR_EQ_MAX_LENGTH"):
                            cmd += ["--eq-max-length", os.environ["LSTAR_EQ_MAX_LENGTH"]]
                        if os.environ.get("LSTAR_EQ_SAMPLES_PER_LENGTH"):
                            cmd += ["--eq-samples-per-length", os.environ["LSTAR_EQ_SAMPLES_PER_LENGTH"]]
                        if os.environ.get("LSTAR_EQ_MAX_ORACLE"):
                            cmd += ["--eq-max-oracle", os.environ["LSTAR_EQ_MAX_ORACLE"]]
                        if os.environ.get("LSTAR_EQ_MAX_ROUNDS"):
                            cmd += ["--eq-max-rounds", os.environ["LSTAR_EQ_MAX_ROUNDS"]]
                        if pre_mut > 0:
                            cmd += ["--mutations", str(pre_mut)]

                    if not QUIET:
                        print(f"[INFO] Precompute cache for {format_key}: {cache_path}")
                    pre_tmo = int(os.environ.get("LSTAR_PRECOMPUTE_TIMEOUT", "600"))
                    env = dict(os.environ)
                    _t0 = time.time()
                    run_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                    if pre_tmo > 0:
                        run_kwargs["timeout"] = pre_tmo
                    _res = subprocess.run(cmd, **run_kwargs)
                    if not QUIET:
                        dt = time.time() - _t0
                        try:
                            sz = os.path.getsize(cache_path) if os.path.exists(cache_path) else "NA"
                        except Exception:
                            sz = "NA"
                        print(f"[INFO] Precompute cache for {format_key} finished in {dt:.2f}s, size={sz}")
                        if _res.returncode != 0:
                            print(f"[WARN] Precompute command returned code {_res.returncode} for {format_key}")
                            if _res.stderr:
                                print(_res.stderr)
                except Exception as e:
                    if not QUIET:
                        print(f"[WARN] Precompute failed for {format_key}: {e}")
                finally:
                    for p in (pos_file, neg_file):
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass

    # 3) Insert tasks (idempotent)
    if not (args.resume_only or args.resume):
        for mutation_type in args.mutations:
            for fmt in selected_formats:
                mutation_db_path = os.path.join("mutated_files", f"{mutation_type}_{fmt}.db")
                if not os.path.exists(mutation_db_path):
                    print(f"[INFO] Skipping, not found: {mutation_db_path}")
                    continue
                print(f"[INFO] Loading samples from {mutation_db_path}")
                samples = load_test_samples_from_db(mutation_db_path)
                total_samples = len(samples)
                if not total_samples:
                    print(f"[INFO] No samples found in '{mutation_db_path}'")
                    continue

                max_budget = TRAIN_K + TEST_K
                if max_budget <= 0 or total_samples <= max_budget:
                    budget = total_samples
                else:
                    budget = max_budget

                limited = samples[:budget]
                train_limit = min(TRAIN_K, len(limited))
                remaining = len(limited) - train_limit
                if remaining <= 0:
                    train_limit = 0
                    remaining = len(limited)
                test_limit = min(TEST_K, remaining) if TEST_K > 0 else remaining
                if test_limit <= 0:
                    test_limit = remaining

                test_samples = limited[train_limit : train_limit + test_limit]
                format_key = f"{mutation_type}_{fmt}"
                print(
                    f"[INFO] Inserting {len(test_samples)} test samples "
                    f"(train_used={train_limit}, test_used={len(test_samples)}) for {format_key}"
                )
                insert_test_samples_to_db(db_path, format_key, test_samples)

    formats_for_rerun = [f"{m}_{f}" for m in args.mutations for f in selected_formats]
    if LSTAR_MUTATION_POOL:
        LSTAR_MUTATION_POOL.ensure_many(formats_for_rerun)

    rerun_repairs_for_selected_formats(db_path, formats_for_rerun, args.max_workers)

    if PAUSE_ON_EXIT:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
