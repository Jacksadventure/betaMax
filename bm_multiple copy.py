#!/usr/bin/env python3
import os
import sqlite3
import subprocess
import re
import time
import random
import concurrent.futures
import argparse
from typing import Optional

from bm_lstar_mutations import LStarMutationPool, get_mutation_table_name

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATABASE_PATH = "double.db"  # Name of the new database to create
REPAIR_OUTPUT_DIR = "repair_results"  # Directory where repair outputs are stored
os.makedirs(REPAIR_OUTPUT_DIR, exist_ok=True)

# Possible repair algorithms you want to test
REPAIR_ALGORITHMS = ["betamax"]

PROJECT_PATHS = {
    "dot": "project/erepair-subjects/dot/build/dot_parser",
    "ini": "project/erepair-subjects/ini/ini",
    "json": "project/erepair-subjects/cjson/cjson",
    "lisp": "project/erepair-subjects/sexp-parser/sexp",
    "obj": "project/erepair-subjects/obj/build/obj_parser",
    "c": "project/erepair-subjects/tiny/tiny",
    # Regex-based categories use match.py as oracle command string
    "date": "python3 match.py Date",
    "time": "python3 match.py Time",
    "url":  "python3 match.py URL",
    "isbn": "python3 match.py ISBN",
    "ipv4": "python3 match.py IPv4",
    "ipv6": "python3 match.py IPv6"
}

# Mapping for regex-based categories
REGEX_DIR_TO_CATEGORY = {
    "date": "Date",
    "time": "Time",
    "url": "URL",
    "isbn": "ISBN",
    "ipv4": "IPv4",
    "ipv6": "IPv6"
}
REGEX_FORMATS = set(REGEX_DIR_TO_CATEGORY.keys())

# Valid formats/folders to process
VALID_FORMATS = ["date", "time", "url", "isbn", "ipv4", "ipv6"]


MUTATION_TYPES = ["double"]

# Train/Test split counts (default 50/50). Override via env BM_TRAIN_K, BM_TEST_K.
TRAIN_K = int(os.environ.get("BM_TRAIN_K", "50"))
TEST_K = int(os.environ.get("BM_TEST_K", "50"))

# Shared mutation pool manager
LSTAR_MUTATION_POOL = LStarMutationPool(TRAIN_K, REGEX_FORMATS, REGEX_DIR_TO_CATEGORY)

# Parser timeout (in seconds)
VALIDATION_TIMEOUT = 30

# Repair timeout (in seconds)
REPAIR_TIMEOUT = 240

# Verbosity and run-control
QUIET = False          # suppress per-entry stdout/stderr and noisy logs
LIMIT_N = None         # limit number of entries to (re)process
PAUSE_ON_EXIT = False  # wait for keypress before exiting (optional)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def create_database(db_path: str):
    """
    Creates a new SQLite database (or overwrites if it already exists).
    This function will create a 'results' table with columns that store
    original/corrupted text, repaired text, and various repair metrics.
    """
    if os.path.exists(db_path):
        print(f"[WARNING] Database '{db_path}' already exists. It will be reused/overwritten.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if not exists
    cursor.execute("""
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
    """)
    conn.commit()
    conn.close()
    print(f"[INFO] Created/checked table 'results' in database '{db_path}'.")


def load_test_samples_from_db(mutation_db_path: str):
    """
    Loads test samples from a mutation database.
    """
    if not os.path.exists(mutation_db_path):
        print(f"[ERROR] Mutation database not found: {mutation_db_path}")
        return []

    conn = sqlite3.connect(mutation_db_path)
    table_name = get_mutation_table_name(mutation_db_path, conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, original_text, mutated_text FROM {table_name} ORDER BY id")
    samples = cursor.fetchall()
    conn.close()

    # The format of test_samples should be (file_id, corrupted_index, original_text, corrupted_text)
    test_samples = [(row[0], 0, row[1], row[2]) for row in samples]
    return test_samples


def insert_test_samples_to_db(db_path: str, format_key: str, test_samples: list):
    """
    Insert the given list of (file_id, corrupted_index, original_text, corrupted_text)
    into the 'results' table in the database, for the specified format.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert each entry only if it doesn't already exist (resume capability)
    for (file_id, cindex, orig_text, broken_text) in test_samples:
        for alg in REPAIR_ALGORITHMS:
            base_f = format_key.split('_')[-1]
            if base_f in REGEX_FORMATS and alg not in ("erepair", "earley", "betamax"):
                continue
            # Skip if this combination already exists (enables resume)
            cursor.execute(
                "SELECT 1 FROM results WHERE format=? AND file_id=? AND corrupted_index=? AND algorithm=? LIMIT 1",
                (format_key, file_id, cindex, alg)
            )
            if cursor.fetchone():
                continue
            cursor.execute("""
                INSERT INTO results (format, file_id, corrupted_index, algorithm,
                                     original_text, broken_text,
                                     repaired_text, fixed, iterations, repair_time,
                                     correct_runs, incorrect_runs, incomplete_runs,
                                     distance_original_broken, distance_broken_repaired, distance_original_repaired)
                VALUES (?, ?, ?, ?, ?, ?, '', 0, 0, 0.0, 0, 0, 0, 0, 0, 0)
            """, (format_key, file_id, cindex, alg, orig_text, broken_text))
    conn.commit()
    conn.close()


def validate_with_external_tool(file_path: str, format_key: str, algorithm: str) -> bool:
    """
    Validate a repaired file using validators/regex validate_* if available,
    otherwise validators/validate_* (earley), or fallback to Python validators.
    - erepair: use match_partial.py Category
    - betamax: prefer validators/regex/validate_*
    - earley: prefer validators/validate_* (binary); fallback to validators/regex or match.py
    """
    base_format = format_key.split('_')[-1]
    try:
        if base_format in REGEX_FORMATS:
            category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
            if algorithm == "erepair":
                cmd = ["python3", "match_partial.py", category, file_path]
            elif algorithm == "betamax":
                wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
                if os.path.exists(wrapper):
                    cmd = [wrapper, file_path]
                else:
                    cmd = ["python3", "match.py", category, file_path]
            else:
                # earley or others: prefer binary validator, then regex wrapper, then match.py
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
            # Fallback for non-regex formats (not used in current VALID_FORMATS)
            proj = PROJECT_PATHS.get(base_format)
            if not proj:
                print(f"[ERROR] No validator for non-regex format: {base_format}")
                return False
            cmd = [proj, file_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=VALIDATION_TIMEOUT
        )
        return (result.returncode == 0)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Validation timeout for '{file_path}', format '{format_key}' (algorithm={algorithm})")
        return False
    except Exception as e:
        print(f"[ERROR] Could not run validator for format '{format_key}' (algorithm={algorithm}): {e}")
        return False


def _normalize_for_distance(text: Optional[str]) -> str:
    """
    Disregard trailing newline/CR characters for edit-distance metrics so they
    only reflect meaningful content changes.
    """
    if text is None:
        return ""
    return text.rstrip("\r\n")


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    a = _normalize_for_distance(a)
    b = _normalize_for_distance(b)
    if not a: return len(b)
    if not b: return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[-1][-1]


def extract_oracle_info(stdout: str):
    """
    Parse oracle summary lines. Supports both:
      - *** Number of required oracle runs: <n> correct: <ok> incorrect: <bad> incomplete: <inc> ***
      - *** Number of required oracle runs: <n> correct: <ok> incorrect: <bad>
    """
    # New format (no incomplete, no trailing ***)
    match = re.search(r"\*\*\* Number of required oracle runs: (\d+) correct: (\d+) incorrect: (\d+)\s*$", stdout, re.MULTILINE)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), 0
    # Old format (with incomplete and trailing ***)
    match = re.search(r"\*\*\* Number of required oracle runs: (\d+) correct: (\d+) incorrect: (\d+) incomplete: (\d+) \*\*\*", stdout)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return 0, 0, 0, 0


def extract_lstar_attempts(stdout: str) -> int:
    """
    Extract number of LSTAR EC attempts from stdout logs.
    Counts '[ATTEMPT N]' lines and returns N+1 (to include attempt 0).
    Falls back to parsing 'attempt x/y' info lines.
    """
    max_attempt = -1
    try:
        for m in re.finditer(r"\[ATTEMPT\s+(\d+)\]", stdout):
            n = int(m.group(1))
            if n > max_attempt:
                max_attempt = n
        if max_attempt >= 0:
            return max_attempt + 1
        # Fallback: parse 'attempt x/y' lines
        m2 = re.findall(r"attempt\s+(\d+)\s*/\s*(\d+)", stdout, flags=re.IGNORECASE)
        if m2:
            max_attempt = max(int(x) for x, _ in m2)
            return max_attempt + 1  # include initial attempt 0
    except Exception:
        pass
    return 0


def repair_and_update_entry(cursor, conn, row):
    """
    Given a single row from the 'results' table, run the repair tool, measure results,
    and update the row in the database.
    """
    (id_, format_key, file_id, corrupted_index, algorithm,
     original_text, broken_text, _repaired, _fixed, _iter, _rtime,
     _correct, _incorrect, _incomplete, _distOB, _distBR, _distOR) = row

    if not QUIET:
        print(f"[INFO] Repairing ID={id_}, format={format_key}, algorithm={algorithm}, file_id={file_id}, corrupted_index={corrupted_index}")

    # Prepare temporary input and output files
    base_format = format_key.split('_')[-1]
    ext = base_format
    if algorithm != "erepair":
        input_file = f"temp_{id_}_{random.randint(0, 9999)}_input.{ext}"
        output_file = os.path.join(REPAIR_OUTPUT_DIR, f"repair_{id_}_output.{ext}")
    else:
        input_file = f"temp_{id_}_{random.randint(0, 9999)}_input.{format_key}"
        output_file = os.path.join(REPAIR_OUTPUT_DIR, f"repair_{id_}_output.{format_key}")

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(broken_text)

    distance_original_broken = levenshtein_distance(original_text, broken_text)
    distance_broken_repaired = -1
    distance_original_repaired = -1

    # By default, we mark as not fixed
    repaired_text = ""
    fixed = 0
    iterations, correct_runs, incorrect_runs, incomplete_runs = 0, 0, 0, 0
    repair_time = 0.0

    # Choose the repair command
    pos_file = None
    if algorithm == "erepair":
        base_format = format_key.split('_')[-1]
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
        base_format = format_key.split('_')[-1]
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
        # Build top-K positives (K=20) from the corresponding mutation DB (e.g., mutated_files/single_date.db)
        K=TRAIN_K
        base_format = format_key.split('_')[-1]
        mutation_type = format_key.split('_')[0]  # e.g., "single"
        category = REGEX_DIR_TO_CATEGORY.get(base_format, base_format)
        pos_file = f"temp_pos_{id_}_{random.randint(0, 9999)}.txt"
        neg_file = f"temp_neg_{id_}_{random.randint(0, 9999)}.txt"
        seed_pos: list[str] = []
        seed_neg: list[str] = []
        mut_pos: list[str] = []
        mut_neg: list[str] = []
        if LSTAR_MUTATION_POOL:
            try:
                LSTAR_MUTATION_POOL.ensure_format(format_key)
            except Exception as pool_err:
                if not QUIET:
                    print(f"[WARN] (ID={id_}) Failed to prime mutation pool for {format_key}: {pool_err}")
            seed_pos = LSTAR_MUTATION_POOL.get_seed_positives(format_key)
            seed_neg = LSTAR_MUTATION_POOL.get_seed_negatives(format_key)
            mut_pos = LSTAR_MUTATION_POOL.get_mutation_positives(format_key)
            mut_neg = LSTAR_MUTATION_POOL.get_mutation_negatives(format_key)

        filtered_pos: list[str] = []
        if seed_pos:
            filtered_pos = [
                (s or "").rstrip("\n")
                for s in seed_pos
                if (s or "") != (original_text or "")
            ]
            if not QUIET:
                print(f"[DEBUG] (ID={id_}) Loaded {len(filtered_pos)} positives from seed pool (LOO applied).")

        try:
            if not filtered_pos:
                mdb_path = os.path.join("mutated_files", f"{format_key}.db")
                with sqlite3.connect(mdb_path) as conn2:
                    table_name = get_mutation_table_name(mdb_path, conn2)
                    cur2 = conn2.cursor()
                    cur2.execute(f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {K}")
                    rows = cur2.fetchall()
                filtered_pos = [
                    (r[0] or "").rstrip("\n")
                    for r in rows
                    if (r[0] or "") != (original_text or "")
                ]
                if not QUIET:
                    print(f"[DEBUG] (ID={id_}) Loaded positives directly from DB: count={len(filtered_pos)}")
            if mut_pos:
                extra = [(s or "").rstrip("\n") for s in mut_pos]
                filtered_pos.extend(extra)
                if not QUIET:
                    print(f"[DEBUG] (ID={id_}) Added {len(extra)} mutation positives to pool.")
            with open(pos_file, "w", encoding="utf-8") as pf:
                for line in filtered_pos:
                    pf.write(line + "\n")
            pos_count = len(filtered_pos)
        except Exception as e:
            pos_count = 0
            try:
                with open(pos_file, "w", encoding="utf-8") as pf:
                    pass
                if not QUIET:
                    print(f"[DEBUG] (ID={id_}) LOO positives fallback: wrote empty pos file due to error: {e}")
            except Exception:
                pass

        # Build initial negatives from first 20 mutated_text (initial hypothesis)
        try:
            neg_lines: list[str] = []
            if seed_neg:
                neg_lines = [(s or "").rstrip("\n") for s in seed_neg]
            if not neg_lines:
                mdb_path = os.path.join("mutated_files", f"{format_key}.db")
                with sqlite3.connect(mdb_path) as conn3:
                    table_name = get_mutation_table_name(mdb_path, conn3)
                    cur3 = conn3.cursor()
                    cur3.execute(f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {K}")
                    rows = cur3.fetchall()
                neg_lines = [(r[0] or "").rstrip("\n") for r in rows]
            if mut_neg:
                extra_neg = [(s or "").rstrip("\n") for s in mut_neg]
                neg_lines.extend(extra_neg)
                if not QUIET:
                    print(f"[DEBUG] (ID={id_}) Added {len(extra_neg)} mutation negatives to pool.")
            with open(neg_file, "w", encoding="utf-8") as nf:
                for line in neg_lines:
                    nf.write(line + "\n")
        except Exception as e:
            # Ensure the negatives file exists even if empty
            try:
                with open(neg_file, "w", encoding="utf-8") as nf:
                    pass
                if not QUIET:
                    print(f"[DEBUG] (ID={id_}) Negatives fallback: wrote empty neg file due to error: {e}")
            except Exception:
                pass

        # Use our Python repairer with validators/regex oracle (handled inside repairer)
        attempts = 500
        # Prefer validators/regex validator; allow override via LSTAR_ORACLE_VALIDATOR
        oracle_override = os.environ.get("LSTAR_ORACLE_VALIDATOR")
        oracle_wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
        oracle_cmd = oracle_override if oracle_override else (oracle_wrapper if os.path.exists(oracle_wrapper) else None)
        cmd = [
            "python3", "betamax/app/betamax.py",
            "--positives", pos_file,
            "--negatives", neg_file,
            # Use shared cache per format across all bm_* scripts
            "--grammar-cache", os.path.join("cache", f"lstar_{base_format}.json"),
            "--category", category,
            "--broken-file", input_file,
            "--output-file", output_file,
            "--max-attempts", str(attempts),
            "--mutations", "0"
        ]
        if oracle_cmd:
            cmd += ["--oracle-validator", oracle_cmd]
        # Equivalence speed knobs via env (optional; to reduce oracle query cost)
        eq_flags = []
        if os.environ.get("LSTAR_EQ_MAX_LENGTH"):
            eq_flags += ["--eq-max-length", os.environ["LSTAR_EQ_MAX_LENGTH"]]
        if os.environ.get("LSTAR_EQ_SAMPLES_PER_LENGTH"):
            eq_flags += ["--eq-samples-per-length", os.environ["LSTAR_EQ_SAMPLES_PER_LENGTH"]]
        if os.environ.get("LSTAR_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
            eq_flags += ["--eq-disable-sampling"]
        if os.environ.get("LSTAR_EQ_SKIP_NEGATIVES", "").lower() in ("1", "true", "yes"):
            eq_flags += ["--eq-skip-negatives"]
        if os.environ.get("LSTAR_EQ_MAX_ORACLE"):
            eq_flags += ["--eq-max-oracle", os.environ["LSTAR_EQ_MAX_ORACLE"]]
        learner = "rpni_xover"
        cmd += ["--learner", learner]
        cmd += eq_flags
    else:
        # Example usage of your erepair.jar approach
        cmd = [
            "java", "-jar", "./project/bin/erepair.jar",
            "-r", "-a", algorithm,
            "-i", input_file,
            "-o", output_file
        ]

    try:
        start_time = time.time()
        if not QUIET:
            print(f"[DEBUG] (ID={id_}, ALG={algorithm}) Running command: {' '.join(str(x) for x in cmd)}")
        # Prepare environment for subprocess (ensure penalty pruning is applied in ec_runtime)
        env = dict(os.environ)
        if algorithm == "betamax":
            # Raise EC parse timeout per attempt unless overridden (set to 100s)
            env.setdefault("LSTAR_PARSE_TIMEOUT", os.environ.get("LSTAR_PARSE_TIMEOUT", "100.0"))
            cache_p = os.path.join("cache", f"lstar_{base_format}.json")
            if not QUIET:
                try:
                    sz = os.path.getsize(cache_p) if os.path.exists(cache_p) else 'NA'
                    print(f"[DEBUG] (ID={id_}, ALG={algorithm}) Cache path: {cache_p}, exists={os.path.exists(cache_p)}, size={sz}")
                except Exception as _e:
                    print(f"[DEBUG] (ID={id_}, ALG={algorithm}) Cache stats unavailable: {cache_p}, err={_e}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        local_timeout = REPAIR_TIMEOUT
        if algorithm == "betamax":
            try:
                local_timeout = int(os.environ.get("LSTAR_EC_TIMEOUT", "600"))
            except Exception:
                pass
        if not QUIET:
            print(f"[DEBUG] (ID={id_}, ALG={algorithm}) Timeout set to: {local_timeout}s")
        stdout, stderr = proc.communicate(timeout=local_timeout)
        repair_time = time.time() - start_time

        # Extract oracle info (optional)
        it_o, correct_runs, incorrect_runs, incomplete_runs = extract_oracle_info(stdout)
        if algorithm == "betamax":
            iterations = extract_lstar_attempts(stdout)
        else:
            iterations = it_o

        if not QUIET:
            print(f"--- STDOUT (ID={id_}) ---\n{stdout}\n")
            print(f"--- STDERR (ID={id_}) ---\n{stderr}\n")

        if proc.returncode == 0 and os.path.exists(output_file):
            # Read the repaired output
            with open(output_file, "r", encoding="utf-8") as rf:
                repaired_text = rf.read()

            # Validate the repaired file
            if validate_with_external_tool(output_file, format_key, algorithm):
                fixed = 1

            # Compute Levenshtein distances
            distance_broken_repaired = levenshtein_distance(broken_text, repaired_text)
            distance_original_repaired = levenshtein_distance(original_text, repaired_text)

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Repair timed out for entry ID={id_}, alg={algorithm}, timeout={local_timeout}s")
    except Exception as e:
        print(f"[ERROR] Repair failed for entry ID={id_}: {e}")
    finally:
        # Clean up temp files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        if pos_file and os.path.exists(pos_file):
            os.remove(pos_file)
        if 'neg_file' in locals() and neg_file and os.path.exists(neg_file):
            # Persist negatives for future refinement (append into per-format accumulator)
            try:
                os.makedirs("negative", exist_ok=True)
                accum_path = os.path.join("negative", f"accum_{base_format}.txt")
                with open(neg_file, "r", encoding="utf-8") as nf, open(accum_path, "a", encoding="utf-8") as af:
                    for line in nf:
                        af.write(line)
            except Exception:
                pass
            os.remove(neg_file)

    # Update the database record
    cursor.execute("""
        UPDATE results
        SET repaired_text = ?, fixed = ?, iterations = ?, repair_time = ?,
            correct_runs = ?, incorrect_runs = ?, incomplete_runs = ?,
            distance_original_broken = ?, distance_broken_repaired = ?, distance_original_repaired = ?
        WHERE id = ?
    """, (
        repaired_text, fixed, iterations, repair_time,
        correct_runs, incorrect_runs, incomplete_runs,
        distance_original_broken, distance_broken_repaired, distance_original_repaired,
        id_
    ))
    conn.commit()


def rerun_repairs_for_selected_formats(db_path: str, selected_formats=None, max_workers=None):
    """
    Re-run (or run for the first time) repairs for the specified formats.
    If selected_formats is None, it will use all in VALID_FORMATS.
    """
    if not selected_formats:
        selected_formats = VALID_FORMATS

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch entries for the desired formats
    cursor.execute("""
        SELECT id, format, file_id, corrupted_index, algorithm,
               original_text, broken_text, repaired_text, fixed,
               iterations, repair_time, correct_runs, incorrect_runs,
               incomplete_runs, distance_original_broken, distance_broken_repaired,
               distance_original_repaired
        FROM results
    """)
    entries = cursor.fetchall()
    if not QUIET:
        print(f"[INFO] Loaded {len(entries)} unfinished entries from DB")

    # Filter only those in the selected formats and allowed algorithms for regex formats
    def _allowed(row):
        fmt_key = row[1]     # e.g., "single_date"
        alg = row[4]         # algorithm column
        base_f = fmt_key.split('_')[-1]
        # Honor CLI --algorithms selection; skip entries not requested this run
        if alg not in REPAIR_ALGORITHMS:
            return False
        if base_f in REGEX_FORMATS and alg not in ("earley", "erepair", "betamax"):
            return False
        return fmt_key in selected_formats

    filtered_entries = [row for row in entries if _allowed(row)]

    if LIMIT_N:
        filtered_entries = filtered_entries[:LIMIT_N]
    if not QUIET:
        print(f"[INFO] Found {len(filtered_entries)} entries to (re)process.")

    def _worker(row):
        # Each thread uses its own connection to avoid SQLite locking issues
        thread_conn = sqlite3.connect(db_path, timeout=30)
        thread_cursor = thread_conn.cursor()
        try:
            repair_and_update_entry(thread_cursor, thread_conn, row)
        finally:
            thread_conn.close()

    if not max_workers:
        max_workers = os.cpu_count() or 4
    if not QUIET:
        print(f"[INFO] Starting ThreadPoolExecutor with max_workers={max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_worker, filtered_entries)

    conn.close()
    print("[INFO] Repair process completed!")


# ------------------------------------------------------------------------------
# Main script flow
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark runner with resume support")
    parser.add_argument("--db", default=DATABASE_PATH, help="Path to results SQLite DB")
    parser.add_argument("--formats", nargs="+", choices=VALID_FORMATS, help="Formats to include (default: all)")
    parser.add_argument("--mutations", nargs="+", default=MUTATION_TYPES,
                        help="Either mutation type names (e.g. 'double') or a numeric cap "
                             "that limits how many samples per format to load.")
    parser.add_argument("--algorithms", nargs="+", choices=REPAIR_ALGORITHMS, help="Override algorithms to run")
    parser.add_argument("--resume-only", action="store_true", help="Skip sample insertion, only resume unfinished repairs")
    parser.add_argument("--resume", action="store_true", help="Alias of --resume-only (also skips precompute)")
    parser.add_argument("--max-workers", type=int, help="Max parallel workers (default: cpu count)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output (suppress per-entry stdout/stderr)")
    parser.add_argument("--limit", type=int, help="Limit number of entries to (re)process")
    parser.add_argument("--pause-on-exit", action="store_true", help="Pause for keypress before exiting")
    parser.add_argument("--lstar-mutation-count", type=int, default=0,
                        help="Number of oracle-classified mutations to pre-generate per format for L* pools (0 disables).")
    parser.add_argument("--lstar-mutation-deterministic", action="store_true",
                        help="Use deterministic mutation enumeration (default: random sampling).")
    parser.add_argument("--lstar-mutation-seed", type=int,
                        help="Optional seed applied to the random sampler (ignored for deterministic mode).")
    args = parser.parse_args()

    # Allow "--mutations 20" style usage by separating numeric caps from
    # actual mutation-type tokens.
    sample_cap_override: int | None = None
    normalized_mutation_types: list[str] = []
    for token in args.mutations:
        if token.isdigit():
            sample_cap_override = int(token)
        else:
            normalized_mutation_types.append(token)
    if not normalized_mutation_types:
        normalized_mutation_types = MUTATION_TYPES.copy()
    args.mutations = normalized_mutation_types
    args.sample_cap = sample_cap_override

    # Apply runtime flags
    global QUIET, LIMIT_N, PAUSE_ON_EXIT
    QUIET = bool(args.quiet)
    LIMIT_N = args.limit
    PAUSE_ON_EXIT = bool(args.pause_on_exit)

    db_path = args.db

    if args.algorithms:
        # override algorithms in-place
        REPAIR_ALGORITHMS[:] = args.algorithms

    # 1) Create or reuse the database
    create_database(db_path)

    # Precompute L* grammar caches for betamax (per format) using first 20 pos/neg
    if "betamax" in REPAIR_ALGORITHMS:
        os.makedirs("cache", exist_ok=True)
        for mutation_type in (args.mutations if args.mutations else MUTATION_TYPES):
            for fmt in (args.formats if args.formats else VALID_FORMATS):
                format_key = f"{mutation_type}_{fmt}"
                # Use shared cache per format across all bm_* (single/double/triple)
                cache_path = os.path.join("cache", f"lstar_{fmt}.json")
                if os.path.exists(cache_path):
                    continue
                # Pick a source DB for precompute: prefer env LSTAR_CACHE_SOURCE_MUTATION, else fallback to single/double/triple
                preferred = os.environ.get("LSTAR_CACHE_SOURCE_MUTATION", "single")
                mutation_db_path = None
                for src in [preferred, "single", "double", "triple"]:
                    cand = os.path.join("mutated_files", f"{src}_{fmt}.db")
                    if os.path.exists(cand):
                        mutation_db_path = cand
                        break
                if not mutation_db_path:
                    continue
                try:
                    pos_file = f"temp_pos_cache_{format_key}_{random.randint(0,9999)}.txt"
                    neg_file = f"temp_neg_cache_{format_key}_{random.randint(0,9999)}.txt"
                    pre_k = int(os.environ.get("LSTAR_PRECOMP_K", str(TRAIN_K)))
                    connc = sqlite3.connect(mutation_db_path)
                    curc = connc.cursor()
                    table_name = get_mutation_table_name(mutation_db_path, connc)
                    curc.execute(f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {pre_k}")
                    rows = curc.fetchall()
                    with open(pos_file, "w", encoding="utf-8") as pf:
                        for r in rows:
                            line = (r[0] or "").rstrip("\n")
                            pf.write(line + "\n")
                    curc.execute(f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {pre_k}")
                    rows = curc.fetchall()
                    with open(neg_file, "w", encoding="utf-8") as nf:
                        for r in rows:
                            line = (r[0] or "").rstrip("\n")
                            nf.write(line + "\n")
                    connc.close()
                    category = REGEX_DIR_TO_CATEGORY.get(fmt, fmt)
                    # Prefer validators/regex validator for precompute; allow override via LSTAR_ORACLE_VALIDATOR
                    oracle_override = os.environ.get("LSTAR_ORACLE_VALIDATOR")
                    oracle_wrapper = os.path.join("validators", "regex", f"validate_{fmt}")
                    oracle_cmd = oracle_override if oracle_override else (oracle_wrapper if os.path.exists(oracle_wrapper) else None)
                    cmd = [
                        "python3", "betamax/app/betamax.py",
                        "--positives", pos_file,
                        "--negatives", neg_file,
                        "--category", category,
                        "--grammar-cache", cache_path,
                        "--init-cache",
                        # Avoid betamax's default mutation augmentation during benchmarks.
                        "--mutations", "0",
                    ]
                    if oracle_cmd:
                        cmd += ["--oracle-validator", oracle_cmd]
                    # Equivalence speed knobs via env for precompute as well
                    eq_flags = []
                    if os.environ.get("LSTAR_EQ_MAX_LENGTH"):
                        eq_flags += ["--eq-max-length", os.environ["LSTAR_EQ_MAX_LENGTH"]]
                    if os.environ.get("LSTAR_EQ_SAMPLES_PER_LENGTH"):
                        eq_flags += ["--eq-samples-per-length", os.environ["LSTAR_EQ_SAMPLES_PER_LENGTH"]]
                    if os.environ.get("LSTAR_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
                        eq_flags += ["--eq-disable-sampling"]
                    if os.environ.get("LSTAR_EQ_SKIP_NEGATIVES", "").lower() in ("1", "true", "yes"):
                        eq_flags += ["--eq-skip-negatives"]
                    if os.environ.get("LSTAR_EQ_MAX_ORACLE"):
                        eq_flags += ["--eq-max-oracle", os.environ["LSTAR_EQ_MAX_ORACLE"]]
                    learner_pre = "rpni_xover"
                    cmd += ["--learner", learner_pre]
                    cmd += eq_flags
                    print(f"[DEBUG] Precompute cache for {format_key}: {' '.join(cmd)} (K={pre_k})")
                    pre_tmo = int(os.environ.get("LSTAR_PRECOMPUTE_TIMEOUT", "600"))
                    env = dict(os.environ)
                    _t0 = time.time()
                    _res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=pre_tmo, env=env)
                    try:
                        _sz = os.path.getsize(cache_path) if os.path.exists(cache_path) else "NA"
                        _dt = time.time() - _t0
                        print(f"[INFO] Precompute cache for {format_key} finished in {_dt:.2f}s, size={_sz}")
                    except Exception:
                        pass
                except subprocess.TimeoutExpired:
                    print(f"[WARN] Precompute timeout for {format_key} (K=20). Retrying with smaller K ...")
                    try:
                        small_k = int(os.environ.get("LSTAR_PRECOMP_K_FALLBACK", "10"))
                        # Rebuild pos/neg with smaller K
                        connc = sqlite3.connect(mutation_db_path)
                        curc = connc.cursor()
                        table_name = get_mutation_table_name(mutation_db_path, connc)
                        curc.execute(f"SELECT original_text FROM {table_name} ORDER BY LENGTH(original_text), id LIMIT {small_k}")
                        rows = curc.fetchall()
                        with open(pos_file, "w", encoding="utf-8") as pf:
                            for r in rows:
                                line = (r[0] or "").rstrip("\n")
                                pf.write(line + "\n")
                        curc.execute(f"SELECT mutated_text FROM {table_name} ORDER BY LENGTH(mutated_text), id LIMIT {small_k}")
                        rows = curc.fetchall()
                        with open(neg_file, "w", encoding="utf-8") as nf:
                            for r in rows:
                                line = (r[0] or "").rstrip("\n")
                                nf.write(line + "\n")
                        connc.close()
                        print(f"[DEBUG] Retry precompute cache for {format_key} with K={small_k}")
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=pre_tmo, env=env)
                    except subprocess.TimeoutExpired:
                        print(f"[WARN] Precompute second timeout for {format_key}, skipping.")
                    except Exception as e2:
                        print(f"[WARN] Precompute retry failed for {format_key}: {e2}")
                except Exception as e:
                    print(f"[WARN] Precompute failed for {format_key}: {e}")
                finally:
                    try:
                        if os.path.exists(pos_file): os.remove(pos_file)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(neg_file): os.remove(neg_file)
                    except Exception:
                        pass


    # 2) Optionally insert tasks (idempotent)
    if not (args.resume_only or args.resume):
        for mutation_type in args.mutations:
            for fmt in (args.formats if args.formats else VALID_FORMATS):
                # Construct DB path, e.g., mutated_files/double_dot.db
                db_name = f"{mutation_type}_{fmt}.db"
                mutation_db_path = os.path.join("mutated_files", db_name)

                if not os.path.exists(mutation_db_path):
                    print(f"[INFO] Skipping, not found: {mutation_db_path}")
                    continue

                print(f"[INFO] Loading samples from {mutation_db_path}")
                samples = load_test_samples_from_db(mutation_db_path)

                total_samples = len(samples)
                if not total_samples:
                    print(f"[INFO] No samples found in '{mutation_db_path}'")
                    continue

                # Use first TRAIN_K for learning grammar (L*), and next TEST_K as test set.
                # Respect an optional numeric cap (from \"--mutations 20\") if provided.
                cap = args.sample_cap
                if cap is not None:
                    budget = min(cap, total_samples)
                else:
                    max_budget = TRAIN_K + TEST_K
                    if max_budget <= 0 or total_samples <= max_budget:
                        budget = total_samples
                    else:
                        budget = max_budget

                limited = samples[:budget]
                train_limit = min(TRAIN_K, len(limited)) if cap is None else min(TRAIN_K, len(limited))
                remaining = len(limited) - train_limit

                if remaining <= 0:
                    train_limit = 0
                    remaining = len(limited)

                if cap is not None:
                    test_limit = remaining
                else:
                    test_limit = min(TEST_K, remaining) if TEST_K > 0 else remaining
                    if test_limit <= 0:
                        test_limit = remaining

                test_samples = limited[train_limit:train_limit + test_limit]

                if test_samples:
                    # Insert each test sample into the 'results' table for each algorithm
                    format_key = f"{mutation_type}_{fmt}"
                    effective_train = train_limit
                    effective_test = len(test_samples)
                    print(f"[INFO] Inserting {effective_test} test samples "
                          f"(train_used={effective_train}, test_used={effective_test}) for {format_key}")
                    insert_test_samples_to_db(db_path, format_key, test_samples)
                else:
                    print(f"[INFO] No samples found in '{mutation_db_path}'")

    # 3) Resume/Run unfinished repairs
    formats_for_rerun = [f"{m}_{f}" for m in args.mutations for f in (args.formats if args.formats else VALID_FORMATS)]

    if LSTAR_MUTATION_POOL:
        LSTAR_MUTATION_POOL.set_mutation_config(
            args.lstar_mutation_count,
            not args.lstar_mutation_deterministic,
            args.lstar_mutation_seed
        )
        LSTAR_MUTATION_POOL.ensure_many(formats_for_rerun)

    rerun_repairs_for_selected_formats(db_path, selected_formats=formats_for_rerun, max_workers=args.max_workers)

    if PAUSE_ON_EXIT:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
