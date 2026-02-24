#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Ensure repo root is importable (when running as tools/<script>.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bm_lstar_mutations import get_mutation_table_name


REGEX_DIR_TO_CATEGORY: dict[str, str] = {
    "date": "Date",
    "iso8601": "ISO8601",
    "time": "Time",
    "url": "URL",
    "isbn": "ISBN",
    "ipv4": "IPv4",
    "ipv6": "IPv6",
}
REGEX_FORMATS = set(REGEX_DIR_TO_CATEGORY.keys())

PROJECT_PATHS: dict[str, str] = {
    "dot": "project/bin/subjects/dot/build/dot_parser",
    "ini": "project/bin/subjects/ini/ini",
    "json": "project/bin/subjects/cjson/cjson",
    "lisp": "project/bin/subjects/sexp-parser/sexp",
    "obj": "project/bin/subjects/obj/build/obj_parser",
    "c": "project/bin/subjects/tiny/tiny",
}


@dataclass(frozen=True)
class PrecomputeCase:
    fmt: str
    category: str
    mutation_db: str
    table: str
    positives_k: int
    negatives_k: int
    engine: str
    learner: str
    mutations: int
    eq_disable_sampling: bool
    eq_max_length: Optional[int]
    eq_samples_per_length: Optional[int]
    eq_max_oracle: Optional[int]
    eq_max_rounds: Optional[int]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


def _pick_mutation_db(fmt: str, preferred: str) -> Optional[str]:
    for src in [preferred, "single", "double", "triple"]:
        p = os.path.join("mutated_files", f"{src}_{fmt}.db")
        if os.path.exists(p):
            return p
    return None


def _load_k_from_db(db_path: str, *, column: str, k: int) -> list[str]:
    if k <= 0:
        return []
    conn = sqlite3.connect(db_path)
    try:
        table = get_mutation_table_name(db_path, conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT {column} FROM {table} ORDER BY LENGTH({column}), id LIMIT ?",
            (int(k),),
        )
        rows = cur.fetchall()
        return [(r[0] or "").rstrip("\n") for r in rows]
    finally:
        conn.close()


def _choose_underlying_oracle_cmd(fmt: str, category: str) -> list[str]:
    if fmt in REGEX_FORMATS:
        cand_bin = os.path.join("validators", f"validate_{fmt}")
        cand_wrap = os.path.join("validators", "regex", f"validate_{fmt}")
        if os.path.exists(cand_bin):
            return [cand_bin]
        if os.path.exists(cand_wrap):
            return [cand_wrap]
        return ["python3", "match.py", category]

    subj = PROJECT_PATHS.get(fmt)
    if not subj:
        raise RuntimeError(f"no subject validator configured for fmt={fmt}")
    return [subj]


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write((line or "").rstrip("\n") + "\n")


def _make_counting_oracle(under_cmd: list[str], *, workdir: Path) -> tuple[Path, Path]:
    """
    Create an oracle wrapper that increments a counter file once per invocation,
    then execs the underlying oracle command with the input file path appended.
    """
    count_path = workdir / "oracle_count.txt"
    wrapper_path = workdir / "oracle_wrapper.py"
    count_path.write_text("0\n", encoding="ascii")

    payload = {
        "under_cmd": under_cmd,
        "count_path": str(count_path),
    }
    cfg_path = workdir / "oracle_cfg.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    wrapper_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json, os, subprocess, sys",
                "cfg_path = os.path.join(os.path.dirname(__file__), 'oracle_cfg.json')",
                "cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))",
                "count_path = cfg['count_path']",
                "under_cmd = cfg['under_cmd']",
                "try:",
                "  raw = open(count_path, 'r', encoding='ascii').read().strip()",
                "  n = int(raw) if raw else 0",
                "except Exception:",
                "  n = 0",
                "try:",
                "  open(count_path, 'w', encoding='ascii').write(str(n+1)+'\\n')",
                "except Exception:",
                "  pass",
                "in_path = sys.argv[1] if len(sys.argv) > 1 else ''",
                "res = subprocess.run(list(under_cmd) + ([in_path] if in_path else []))",
                "raise SystemExit(res.returncode)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)
    return wrapper_path, count_path


def _build_precompute_cmd(
    *,
    engine: str,
    positives: Path,
    negatives: Path,
    category: str,
    cache_path: Path,
    learner: str,
    oracle_validator: Path,
    mutations: int,
    oracle_timeout_ms: Optional[int],
    eq_disable_sampling: bool,
    eq_max_length: Optional[int],
    eq_samples_per_length: Optional[int],
    eq_max_oracle: Optional[int],
    eq_max_rounds: Optional[int],
) -> list[str]:
    eng = (engine or "").strip().lower()
    if eng not in ("cpp", "python"):
        raise RuntimeError(f"unknown engine: {engine}")

    if eng == "python":
        cmd: list[str] = [
            "python3",
            "betamax/app/betamax.py",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--category",
            category,
            "--grammar-cache",
            str(cache_path),
            "--init-cache",
            "--learner",
            learner,
        ]
        cmd += ["--oracle-validator", str(oracle_validator)]
        if eq_disable_sampling:
            cmd += ["--eq-disable-sampling"]
        if eq_max_length is not None:
            cmd += ["--eq-max-length", str(int(eq_max_length))]
        if eq_samples_per_length is not None:
            cmd += ["--eq-samples-per-length", str(int(eq_samples_per_length))]
        if eq_max_oracle is not None:
            cmd += ["--eq-max-oracle", str(int(eq_max_oracle))]
        if mutations > 0:
            cmd += ["--mutations", str(int(mutations))]
        return cmd

    exe = os.environ.get("BM_BETAMAX_CPP_BIN", "betamax_cpp/build/betamax_cpp")
    cmd = [
        exe,
        "--positives",
        str(positives),
        "--negatives",
        str(negatives),
        "--category",
        category,
        "--dfa-cache",
        str(cache_path),
        "--init-cache",
        "--learner",
        learner,
        "--repo-root",
        ".",
        "--oracle-validator",
        str(oracle_validator),
    ]
    if oracle_timeout_ms is not None:
        cmd += ["--oracle-timeout-ms", str(int(oracle_timeout_ms))]
    if eq_disable_sampling:
        cmd += ["--eq-disable-sampling"]
    if eq_max_length is not None:
        cmd += ["--eq-max-length", str(int(eq_max_length))]
    if eq_samples_per_length is not None:
        cmd += ["--eq-samples-per-length", str(int(eq_samples_per_length))]
    if eq_max_oracle is not None:
        cmd += ["--eq-max-oracle", str(int(eq_max_oracle))]
    if eq_max_rounds is not None:
        cmd += ["--eq-max-rounds", str(int(eq_max_rounds))]
    if mutations > 0:
        cmd += ["--mutations", str(int(mutations))]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure oracle-validator invocation count during bm-style precompute (--init-cache) per format."
    )
    ap.add_argument(
        "--engine",
        default=os.environ.get("BM_BETAMAX_ENGINE", "cpp"),
        choices=["cpp", "python"],
        help="betaMax backend engine to measure (default: env BM_BETAMAX_ENGINE or cpp)",
    )
    ap.add_argument(
        "--formats",
        nargs="*",
        help="Formats to measure (default: date time isbn ipv4 url ipv6 json when DBs exist)",
    )
    ap.add_argument(
        "--out",
        default="result4thesis/precompute_oracle_runs.json",
        help="Output JSON report file",
    )
    ap.add_argument(
        "--preferred-source",
        default=os.environ.get("LSTAR_CACHE_SOURCE_MUTATION", "single"),
        help="Preferred mutation DB source prefix (single/double/triple)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=_env_int("LSTAR_PRECOMP_K", _env_int("BM_TRAIN_K", 50)),
        help="Number of examples to load from mutation DB (default: env LSTAR_PRECOMP_K or BM_TRAIN_K or 50)",
    )
    ap.add_argument(
        "--use-db-negatives",
        action="store_true",
        help="Also load negatives from mutated_text (like BM_NEGATIVES_FROM_DB=1)",
    )
    ap.add_argument(
        "--learner",
        default=os.environ.get("LSTAR_CACHE_LEARNER", os.environ.get("LSTAR_LEARNER", "rpni_xover")),
        help="Learner for precompute (default: env LSTAR_CACHE_LEARNER/LSTAR_LEARNER or rpni_xover)",
    )
    ap.add_argument(
        "--mutations",
        type=int,
        default=_env_int("LSTAR_PRECOMPUTE_MUTATIONS", 60),
        help="Mutation augmentation count during precompute (default: env LSTAR_PRECOMPUTE_MUTATIONS or 60; 0 disables)",
    )
    ap.add_argument(
        "--oracle-timeout-ms",
        type=int,
        default=3000,
        help="C++ engine per-oracle call timeout in ms (default: 3000)",
    )
    ap.add_argument(
        "--case-timeout-s",
        type=int,
        default=_env_int("LSTAR_PRECOMPUTE_TIMEOUT", 600),
        help="Wall-clock timeout per format in seconds (default: env LSTAR_PRECOMPUTE_TIMEOUT or 600; 0 disables)",
    )
    args = ap.parse_args()

    formats = list(args.formats or [])
    if not formats:
        formats = ["date", "time", "isbn", "ipv4", "url", "ipv6"]
        # Include json by default when any json mutation DB exists
        if any(os.path.exists(os.path.join("mutated_files", f"{src}_json.db")) for src in ("single", "double", "triple")):
            formats.append("json")

    use_db_neg = bool(args.use_db_negatives) or _env_bool("BM_NEGATIVES_FROM_DB", False)
    eq_disable_sampling = _env_bool("LSTAR_EQ_DISABLE_SAMPLING", False)
    eq_max_length = _env_int("LSTAR_EQ_MAX_LENGTH", -1)
    eq_samples_per_length = _env_int("LSTAR_EQ_SAMPLES_PER_LENGTH", -1)
    eq_max_oracle = _env_int("LSTAR_EQ_MAX_ORACLE", -1)
    eq_max_rounds = _env_int("LSTAR_EQ_MAX_ROUNDS", -1)

    def _maybe(v: int) -> Optional[int]:
        return None if v < 0 else v

    results: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        preferred = str(args.preferred_source)
        db_path = _pick_mutation_db(fmt, preferred)
        if not db_path:
            results.append(
                {
                    "fmt": fmt,
                    "status": "skipped",
                    "reason": "missing mutation DB (mutated_files/<src>_<fmt>.db)",
                }
            )
            continue

        conn = sqlite3.connect(db_path)
        try:
            table = get_mutation_table_name(db_path, conn)
        finally:
            conn.close()

        category = REGEX_DIR_TO_CATEGORY.get(fmt, fmt)
        pos = _load_k_from_db(db_path, column="original_text", k=int(args.k))
        neg = _load_k_from_db(db_path, column="mutated_text", k=int(args.k)) if use_db_neg else []

        under_cmd = _choose_underlying_oracle_cmd(fmt, category)

        with tempfile.TemporaryDirectory(prefix=f"precompute_oracle_{fmt}_") as td:
            workdir = Path(td)
            pos_path = workdir / "positives.txt"
            neg_path = workdir / "negatives.txt"
            cache_ext = "dfa" if args.engine == "cpp" else "json"
            cache_path = workdir / f"cache.{cache_ext}"
            _write_lines(pos_path, pos)
            _write_lines(neg_path, neg)

            wrapper_path, count_path = _make_counting_oracle(under_cmd, workdir=workdir)

            cmd = _build_precompute_cmd(
                engine=args.engine,
                positives=pos_path,
                negatives=neg_path,
                category=category,
                cache_path=cache_path,
                learner=str(args.learner),
                oracle_validator=wrapper_path,
                mutations=int(args.mutations),
                oracle_timeout_ms=(int(args.oracle_timeout_ms) if args.engine == "cpp" else None),
                eq_disable_sampling=eq_disable_sampling,
                eq_max_length=_maybe(eq_max_length),
                eq_samples_per_length=_maybe(eq_samples_per_length),
                eq_max_oracle=_maybe(eq_max_oracle),
                eq_max_rounds=_maybe(eq_max_rounds),
            )

            t0 = time.time()
            timed_out = False
            try:
                timeout = None
                if int(args.case_timeout_s) > 0:
                    timeout = int(args.case_timeout_s)
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as e:
                timed_out = True
                proc = subprocess.CompletedProcess(cmd, returncode=124, stdout=e.stdout or "", stderr=e.stderr or "")
            dt = time.time() - t0

            try:
                oracle_runs = int(count_path.read_text(encoding="ascii").strip() or "0")
            except Exception:
                oracle_runs = None

            case = PrecomputeCase(
                fmt=fmt,
                category=category,
                mutation_db=db_path,
                table=table,
                positives_k=len(pos),
                negatives_k=len(neg),
                engine=str(args.engine),
                learner=str(args.learner),
                mutations=int(args.mutations),
                eq_disable_sampling=eq_disable_sampling,
                eq_max_length=_maybe(eq_max_length),
                eq_samples_per_length=_maybe(eq_samples_per_length),
                eq_max_oracle=_maybe(eq_max_oracle),
                eq_max_rounds=_maybe(eq_max_rounds),
            )

            results.append(
                {
                    "status": "timeout" if timed_out else ("ok" if proc.returncode == 0 else "error"),
                    "oracle_runs": oracle_runs,
                    "exit_code": proc.returncode,
                    "seconds": round(dt, 6),
                    "cmd": " ".join(shlex.quote(x) for x in cmd),
                    "oracle_under_cmd": under_cmd,
                    "case": case.__dict__,
                    "stdout_head": (proc.stdout or "").splitlines()[:30],
                    "stderr_head": (proc.stderr or "").splitlines()[:30],
                }
            )

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print a small table for quick inspection
    print(f"[ok] wrote: {out_path}")
    for row in results:
        fmt = row.get("fmt") or row.get("case", {}).get("fmt")
        st = row.get("status")
        runs = row.get("oracle_runs")
        code = row.get("exit_code")
        print(f"{fmt:8s}  status={st:7s}  oracle_runs={runs!s:6s}  exit={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
