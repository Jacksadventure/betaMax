#!/usr/bin/env python3
"""
bm_ablation.py
-----------------------
Drive `bm_single.py`, `bm_multiple.py` (double), and `bm_triple.py` across several
training-set sizes (K) while keeping the test-set size constant, enabling
ablation experiments on the amount of learning data.

Each (mode, K) run writes to its own SQLite database:
  single_k{K}.db / double_k{K}.db / triple_k{K}.db

Between runs, the LSTAR cache is deleted once to ensure a clean start.

Example:
    python3 bm_ablation.py --formats date time

Pass additional arguments to the underlying benchmark scripts by appending them
after ``--``. For instance:
    python3 bm_ablation.py -- --limit 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shutil
from typing import List

DEFAULT_TRAIN_SIZES = [25, 12, 6]
DEFAULT_TEST_K = 50
DEFAULT_ALGORITHMS = ["betamax"]
DEFAULT_FORMATS = ["date", "iso8601", "time", "url", "isbn", "ipv4", "ipv6"]
DEFAULT_MODES = ["single", "double", "triple"]
DEFAULT_SCRIPTS = {
    "single": "bm_single.py",
    "double": "bm_multiple.py",
    "triple": "bm_triple.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single/double/triple benchmarks across multiple training-set sizes."
    )
    parser.add_argument(
        "--train-sizes", type=int, nargs="+", default=DEFAULT_TRAIN_SIZES,
        help="Training-set sizes K to evaluate (default: %(default)s)."
    )
    parser.add_argument(
        "--test-k", type=int, default=DEFAULT_TEST_K,
        help="Number of test samples to keep per format (default: %(default)s)."
    )
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES,
        help="Which mutation modes to run (default: %(default)s)."
    )
    parser.add_argument(
        "--bm-single-script", default=DEFAULT_SCRIPTS["single"],
        help="Path to bm_single.py (default: %(default)s)."
    )
    parser.add_argument(
        "--bm-double-script", default=DEFAULT_SCRIPTS["double"],
        help="Path to bm_multiple.py for double mutations (default: %(default)s)."
    )
    parser.add_argument(
        "--bm-triple-script", default=DEFAULT_SCRIPTS["triple"],
        help="Path to bm_triple.py (default: %(default)s)."
    )
    parser.add_argument(
        "--formats", nargs="+", default=None,
        help="Subset of formats to run (default: all)."
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Optional override of repair algorithms passed to benchmark scripts."
    )
    parser.add_argument("--resume-only", action="store_true",
                        help="Forward --resume-only to all benchmark scripts.")
    parser.add_argument("--resume", action="store_true",
                        help="Forward --resume to all benchmark scripts.")
    parser.add_argument("--max-workers", type=int, help="Forwarded to benchmark scripts.")
    parser.add_argument("--quiet", action="store_true", help="Forward --quiet.")
    parser.add_argument("--limit", type=int, help="Forward --limit.")
    parser.add_argument("--pause-on-exit", action="store_true",
                        help="Forward --pause-on-exit.")
    parser.add_argument("--lstar-mutation-count", type=int,
                        help="Forward --lstar-mutation-count.")
    parser.add_argument("--lstar-mutation-deterministic", action="store_true",
                        help="Forward --lstar-mutation-deterministic.")
    parser.add_argument("--lstar-mutation-seed", type=int,
                        help="Forward --lstar-mutation-seed.")
    parser.add_argument(
        "--betamax-engine",
        choices=["python", "cpp"],
        default=None,
        help="Forward to bm_* as --betamax-engine (or set env BM_BETAMAX_ENGINE).",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if its DB already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing benchmark scripts.")
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("LSTAR_CACHE_ROOT", "cache"),
        help="LSTAR cache directory to use and delete between runs (default: %(default)s).",
    )
    parser.add_argument(
        "bm_args", nargs=argparse.REMAINDER,
        help="Additional args forwarded verbatim to benchmark scripts (prefix with -- before them)."
    )
    return parser.parse_args()


def _db_path(mode: str, k: int) -> str:
    return f"{mode}_k{k}.db"


def build_command(args: argparse.Namespace, mode: str, db_path: str) -> List[str]:
    script_map = {
        "single": args.bm_single_script,
        "double": args.bm_double_script,
        "triple": args.bm_triple_script,
    }
    bm_script = script_map[mode]
    cmd = ["python3", bm_script, "--db", db_path]
    if args.formats:
        cmd += ["--formats", *args.formats]
    if args.algorithms:
        cmd += ["--algorithms", *args.algorithms]
    if args.resume_only:
        cmd.append("--resume-only")
    if args.resume:
        cmd.append("--resume")
    if args.max_workers is not None:
        cmd += ["--max-workers", str(args.max_workers)]
    if args.quiet:
        cmd.append("--quiet")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.pause_on_exit:
        cmd.append("--pause-on-exit")
    if args.lstar_mutation_count is not None:
        cmd += ["--lstar-mutation-count", str(args.lstar_mutation_count)]
    if args.lstar_mutation_deterministic:
        cmd.append("--lstar-mutation-deterministic")
    if args.lstar_mutation_seed is not None:
        cmd += ["--lstar-mutation-seed", str(args.lstar_mutation_seed)]
    if args.betamax_engine:
        cmd += ["--betamax-engine", args.betamax_engine]
    if args.bm_args:
        cmd += args.bm_args
    return cmd


def _clear_cache(cache_root: str) -> None:
    # Only delete within the repo / relative paths by default.
    # Users can pass an absolute path if they really want, but we avoid
    # following symlinks accidentally.
    if os.path.islink(cache_root):
        raise RuntimeError(f"Refusing to delete symlinked cache dir: {cache_root}")
    shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)


def main():
    args = parse_args()

    train_sizes = []
    for k in args.train_sizes:
        if k <= 0:
            print(f"[WARN] Ignoring non-positive train size K={k}")
            continue
        train_sizes.append(k)
    if not train_sizes:
        print("[ERROR] No valid training sizes provided.", file=sys.stderr)
        return 1

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)

    # Run order: for each K, run modes in the order requested.
    for k in train_sizes:
        for mode in args.modes:
            db_path = _db_path(mode, k)
            if args.skip_existing and os.path.exists(db_path):
                print(f"[INFO] Skipping {mode} K={k} because '{db_path}' already exists.")
                _clear_cache(cache_root)
                continue

            _clear_cache(cache_root)

            cmd = build_command(args, mode, db_path)
            env = os.environ.copy()
            env["BM_TRAIN_K"] = str(k)
            env["BM_TEST_K"] = str(args.test_k)
            env["LSTAR_PRECOMP_K"] = str(k)
            env.setdefault("LSTAR_PRECOMP_K_FALLBACK", str(min(k, 10)))
            env["BM_ABLATION_K"] = str(k)
            env["LSTAR_CACHE_ROOT"] = cache_root
            if args.betamax_engine:
                env["BM_BETAMAX_ENGINE"] = str(args.betamax_engine)

            pretty_cmd = " ".join(cmd)
            print(f"[ABLATION] Running {mode} K={k} -> DB {db_path}")
            print(f"[ABLATION] CMD: {pretty_cmd}")
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] {mode} benchmark failed for K={k} (exit {exc.returncode}).", file=sys.stderr)
                return exc.returncode

            _clear_cache(cache_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
