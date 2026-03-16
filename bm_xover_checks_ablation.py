#!/usr/bin/env python3
"""
bm_xover_checks_ablation.py
---------------------------
Run single/double/triple benchmarks while varying cross-check count used by
`rpni_xover` during merge attempts.

The key ablation variable is:
  - `LSTAR_RPNI_XOVER_CHECKS` (oracle checks per merge attempt)

This script keeps the current benchmark defaults:
  - precompute learner: `rpni_xover`
  - refine-loop learner: `rpni`

Each (mode, checks) run writes to its own SQLite database:
  {mode}_xcheck{checks}.db
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List


DEFAULT_CHECKS = [0, 1, 2, 3]
DEFAULT_PAIRS = 50
DEFAULT_MODES = ["single", "double", "triple"]
DEFAULT_TEST_K = 50
DEFAULT_TRAIN_K = 50
DEFAULT_DB_TEMPLATE = "{mode}_xcheck{checks}.db"
DEFAULT_SCRIPTS = {
    "single": "bm_single.py",
    "double": "bm_multiple.py",
    "triple": "bm_triple.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmarks across different rpni_xover cross-check budgets."
    )
    parser.add_argument(
        "--checks",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKS,
        help="Values for LSTAR_RPNI_XOVER_CHECKS (default: %(default)s).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help="Value for LSTAR_RPNI_XOVER_PAIRS (default: %(default)s).",
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
        "--db-template",
        default=DEFAULT_DB_TEMPLATE,
        help="DB template using {mode}, {checks}, {pairs} placeholders (default: %(default)s).",
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
    parser.add_argument(
        "--lstar-mutation-deterministic",
        action="store_true",
        help="Forward --lstar-mutation-deterministic.",
    )
    parser.add_argument("--lstar-mutation-seed", type=int, help="Forward --lstar-mutation-seed.")
    parser.add_argument(
        "--betamax-engine",
        choices=["python", "cpp"],
        default="cpp",
        help="Forward to bm_* as --betamax-engine (default: %(default)s).",
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


def _render_db_path(template: str, mode: str, checks: int, pairs: int) -> str:
    return template.format(mode=mode, checks=checks, pairs=pairs)


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


def main() -> int:
    args = parse_args()

    checks_values: List[int] = []
    for value in args.checks:
        if value < 0:
            print(f"[WARN] Ignoring negative checks={value}")
            continue
        checks_values.append(value)
    if not checks_values:
        print("[ERROR] No valid checks values provided.", file=sys.stderr)
        return 1
    if args.pairs < 0:
        print("[ERROR] --pairs must be >= 0.", file=sys.stderr)
        return 1

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)

    if args.betamax_engine != "cpp":
        print(
            "[WARN] --betamax-engine is not cpp; "
            "LSTAR_RPNI_XOVER_CHECKS may not affect non-cpp runs."
        )

    for checks in checks_values:
        for mode in args.modes:
            db_path = _render_db_path(args.db_template, mode, checks, args.pairs)
            if args.skip_existing and os.path.exists(db_path):
                print(f"[INFO] Skipping {mode} checks={checks} because '{db_path}' already exists.")
                _clear_cache(cache_root)
                continue

            _clear_cache(cache_root)

            env = os.environ.copy()
            env["BM_TRAIN_K"] = str(args.train_k)
            env["BM_TEST_K"] = str(args.test_k)
            env["LSTAR_PRECOMP_K"] = str(args.train_k)
            env["LSTAR_CACHE_ROOT"] = cache_root
            env["LSTAR_RPNI_XOVER_CHECKS"] = str(checks)
            env["LSTAR_RPNI_XOVER_PAIRS"] = str(args.pairs)
            env["LSTAR_CACHE_LEARNER"] = "rpni_xover"
            env["LSTAR_LEARNER"] = "rpni"
            env["BM_BETAMAX_ENGINE"] = str(args.betamax_engine)
            env["BM_ABLATION_XOVER_CHECKS"] = str(checks)

            cmd = build_command(args, mode, db_path)
            print(f"[ABLATION] Running {mode} checks={checks}, pairs={args.pairs} -> DB {db_path}")
            print("[ABLATION] Learners: cache=rpni_xover, runtime=rpni")
            print(f"[ABLATION] CMD: {' '.join(cmd)}")
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[ERROR] {mode} benchmark failed for checks={checks} "
                    f"(exit {exc.returncode}).",
                    file=sys.stderr,
                )
                return exc.returncode
            finally:
                _clear_cache(cache_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
