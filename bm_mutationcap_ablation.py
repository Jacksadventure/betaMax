#!/usr/bin/env python3
"""
bm_mutationcap_ablation.py
--------------------------
Run the single/double/triple benchmark scripts while varying the mutation
augmentation count used by BETAMAX *during precompute only*.

This script sets `LSTAR_PRECOMPUTE_MUTATIONS=<N>` for each run; the bm_* scripts
use it only in the precompute step (grammar cache init), while the actual repair
benchmark continues to use its own hard-coded `--mutations` setting.

Default runs: caps {20, 40, 80} × modes {single, double, triple} = 9 runs.
Each run writes to its own SQLite database: {mode}_m{cap}.db
Between runs, the LSTAR cache directory is deleted once.

Example:
    python3 bm_mutationcap_ablation.py --formats date time --caps 20 40 80

Pass extra args to underlying bm_* scripts after "--":
    python3 bm_mutationcap_ablation.py -- --limit 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List


DEFAULT_CAPS = [0,20,40,80,100]
DEFAULT_MODES = ["single", "double", "triple"]
DEFAULT_TEST_K = 50
DEFAULT_TRAIN_K = 50
DEFAULT_CACHE_LEARNER = "rpni_xover"
DEFAULT_RUNTIME_LEARNER = "rpni"
DEFAULT_SCRIPTS = {
    "single": "bm_single.py",
    "double": "bm_multiple.py",
    "triple": "bm_triple.py",
}
DEFAULT_FORMATS = ["date", "iso8601", "time", "url", "isbn", "ipv4", "ipv6"]
DEFAULT_DB_TEMPLATE = "{mode}_m{cap}.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single/double/triple benchmarks across mutation sample caps."
    )
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=DEFAULT_CAPS,
        help="Values for LSTAR_PRECOMPUTE_MUTATIONS (default: %(default)s).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=DEFAULT_MODES,
        help="Which mutation modes to run (default: %(default)s).",
    )
    parser.add_argument(
        "--train-k",
        type=int,
        default=DEFAULT_TRAIN_K,
        help="BM_TRAIN_K env var (default: %(default)s).",
    )
    parser.add_argument(
        "--test-k",
        type=int,
        default=DEFAULT_TEST_K,
        help="BM_TEST_K env var (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-learner",
        default=DEFAULT_CACHE_LEARNER,
        help="Learner used for precompute cache init (default: %(default)s).",
    )
    parser.add_argument(
        "--runtime-learner",
        default=DEFAULT_RUNTIME_LEARNER,
        help="Learner used during the repair/relearn loop (default: %(default)s).",
    )
    parser.add_argument(
        "--db-template",
        default=DEFAULT_DB_TEMPLATE,
        help="DB template using {mode} and {cap} placeholders (default: %(default)s).",
    )
    parser.add_argument(
        "--bm-single-script",
        default=DEFAULT_SCRIPTS["single"],
        help="Path to bm_single.py (default: %(default)s).",
    )
    parser.add_argument(
        "--bm-double-script",
        default=DEFAULT_SCRIPTS["double"],
        help="Path to bm_multiple.py for double mutations (default: %(default)s).",
    )
    parser.add_argument(
        "--bm-triple-script",
        default=DEFAULT_SCRIPTS["triple"],
        help="Path to bm_triple.py (default: %(default)s).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Subset of formats to run (default: all).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Optional override of repair algorithms passed to benchmark scripts.",
    )
    parser.add_argument("--resume-only", action="store_true", help="Forward --resume-only.")
    parser.add_argument("--resume", action="store_true", help="Forward --resume.")
    parser.add_argument("--max-workers", type=int, help="Forward --max-workers.")
    parser.add_argument("--quiet", action="store_true", help="Forward --quiet.")
    parser.add_argument("--limit", type=int, help="Forward --limit.")
    parser.add_argument("--pause-on-exit", action="store_true", help="Forward --pause-on-exit.")
    parser.add_argument("--lstar-mutation-count", type=int, help="Forward --lstar-mutation-count.")
    parser.add_argument("--lstar-mutation-deterministic", action="store_true",
                        help="Forward --lstar-mutation-deterministic.")
    parser.add_argument("--lstar-mutation-seed", type=int, help="Forward --lstar-mutation-seed.")
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
        "bm_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded verbatim (prefix with -- before them).",
    )
    return parser.parse_args()


def _clear_cache(cache_root: str) -> None:
    if os.path.islink(cache_root):
        raise RuntimeError(f"Refusing to delete symlinked cache dir: {cache_root}")
    shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)


def _render_db_path(template: str, mode: str, cap: int) -> str:
    return template.format(mode=mode, cap=cap, CAP=cap)


def build_command(args: argparse.Namespace, mode: str, db_path: str, cap: int) -> List[str]:
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
    if args.bm_args:
        cmd += args.bm_args
    return cmd


def main() -> int:
    args = parse_args()

    caps: List[int] = []
    for c in args.caps:
        if c <= 0:
            print(f"[WARN] Ignoring non-positive cap={c}")
            continue
        caps.append(c)
    if not caps:
        print("[ERROR] No valid caps provided.", file=sys.stderr)
        return 1

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)

    cache_learner = (args.cache_learner or "").strip()
    runtime_learner = (args.runtime_learner or "").strip()
    if not cache_learner:
        print("[ERROR] --cache-learner is empty.", file=sys.stderr)
        return 1
    if not runtime_learner:
        print("[ERROR] --runtime-learner is empty.", file=sys.stderr)
        return 1

    # Run order: for each cap, run modes in the order requested.
    for cap in caps:
        for mode in args.modes:
            db_path = _render_db_path(args.db_template, mode, cap)
            if args.skip_existing and os.path.exists(db_path):
                print(f"[INFO] Skipping {mode} cap={cap} because '{db_path}' already exists.")
                _clear_cache(cache_root)
                continue

            _clear_cache(cache_root)

            env = os.environ.copy()
            env["BM_TRAIN_K"] = str(args.train_k)
            env["BM_TEST_K"] = str(args.test_k)
            env["LSTAR_CACHE_ROOT"] = cache_root
            env["LSTAR_PRECOMPUTE_MUTATIONS"] = str(cap)
            # Two-stage setup:
            # - bm_* precompute uses LSTAR_CACHE_LEARNER to initialize the grammar cache
            # - per-sample repairs use LSTAR_LEARNER (and BM_BETAMAX_LEARNER) for relearning attempts
            env["LSTAR_CACHE_LEARNER"] = cache_learner
            env["LSTAR_LEARNER"] = runtime_learner
            env["BM_BETAMAX_LEARNER"] = runtime_learner

            cmd = build_command(args, mode, db_path, cap)
            print(f"[ABLATION] Running {mode} cap={cap} -> DB {db_path}")
            print(f"[ABLATION] Learners: cache={cache_learner}, runtime={runtime_learner}")
            print(f"[ABLATION] CMD: {' '.join(cmd)}")
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] {mode} benchmark failed for cap={cap} (exit {exc.returncode}).", file=sys.stderr)
                return exc.returncode
            finally:
                _clear_cache(cache_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
