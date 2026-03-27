#!/usr/bin/env python3
"""
bm_multiple_learner_ablation.py
-------------------------------
Run bm_multiple.py multiple times while varying the BETAMAX learner
(default: rpni vs rpni_xover) so we can measure the effect of different
grammar learners on the same train/test splits.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

DEFAULT_LEARNERS = ["rpni", "rpni_xover"]
DEFAULT_DB_TEMPLATE = "double_{learner}.db"
DEFAULT_TRAIN_K = 50
DEFAULT_TEST_K = 50
DEFAULT_BENCHMARK_SCRIPT = "bm_multiple.py"
DEFAULT_MUTATION_TYPES = ["double"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bm_multiple.py for a set of BETAMAX learners."
    )
    parser.add_argument(
        "--learners", nargs="+", default=DEFAULT_LEARNERS,
        help="Learner names to evaluate (default: %(default)s)."
    )
    parser.add_argument(
        "--train-k", type=int, default=DEFAULT_TRAIN_K,
        help="Training-set size per format (default: %(default)s)."
    )
    parser.add_argument(
        "--test-k", type=int, default=DEFAULT_TEST_K,
        help="Test-set size per format (default: %(default)s)."
    )
    parser.add_argument(
        "--db-template", default=DEFAULT_DB_TEMPLATE,
        help="Template for DB path. Use {learner} placeholder (default: %(default)s)."
    )
    parser.add_argument(
        "--bm-script", default=DEFAULT_BENCHMARK_SCRIPT,
        help="Path to bm_multiple.py (default: %(default)s)."
    )
    parser.add_argument(
        "--formats", nargs="+",
        help="Optional subset of formats to run (default: all)."
    )
    parser.add_argument(
        "--mutations", nargs="+", default=DEFAULT_MUTATION_TYPES,
        help="Mutation types or caps forwarded to bm_multiple."
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        help="Override bm_multiple --algorithms list."
    )
    parser.add_argument("--resume-only", action="store_true",
                        help="Forward --resume-only to bm_multiple.")
    parser.add_argument("--resume", action="store_true",
                        help="Forward --resume to bm_multiple.")
    parser.add_argument("--max-workers", type=int,
                        help="Forward --max-workers to bm_multiple.")
    parser.add_argument("--quiet", action="store_true",
                        help="Forward --quiet.")
    parser.add_argument("--limit", type=int,
                        help="Forward --limit.")
    parser.add_argument("--pause-on-exit", action="store_true",
                        help="Forward --pause-on-exit.")
    parser.add_argument("--lstar-mutation-count", type=int,
                        help="Forward --lstar-mutation-count.")
    parser.add_argument("--lstar-mutation-deterministic", action="store_true",
                        help="Forward --lstar-mutation-deterministic.")
    parser.add_argument("--lstar-mutation-seed", type=int,
                        help="Forward --lstar-mutation-seed.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose DB already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--cache-root-template",
                        help="Optional template for cache directory (use {learner}).")
    parser.add_argument(
        "bm_args", nargs=argparse.REMAINDER,
        help="Additional args passed verbatim to bm_multiple."
    )
    return parser.parse_args()


def _render_db(template: str, learner: str) -> str:
    if "{learner" not in template:
        return f"{template}_{learner}"
    return template.format(learner=learner)


def build_command(args: argparse.Namespace, db_path: str) -> List[str]:
    cmd = ["python3", args.bm_script, "--db", db_path]
    if args.formats:
        cmd += ["--formats", *args.formats]
    if args.mutations:
        cmd += ["--mutations", *args.mutations]
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


def main():
    args = parse_args()

    learners = [l for l in args.learners if l]
    if not learners:
        print("[ERROR] No learners specified.", file=sys.stderr)
        return 1

    for learner in learners:
        db_path = _render_db(args.db_template, learner)
        if args.skip_existing and os.path.exists(db_path):
            print(f"[INFO] Skipping learner={learner} (DB exists: {db_path})")
            continue
        cmd = build_command(args, db_path)
        env = os.environ.copy()
        env["BM_TRAIN_K"] = str(args.train_k)
        env["BM_TEST_K"] = str(args.test_k)
        env["LSTAR_PRECOMP_K"] = str(args.train_k)
        env["BM_BETAMAX_LEARNER"] = learner
        env["LSTAR_LEARNER"] = learner
        env["LSTAR_CACHE_LEARNER"] = learner
        if args.cache_root_template:
            cache_root = args.cache_root_template.format(learner=learner)
            os.makedirs(cache_root, exist_ok=True)
            env["LSTAR_CACHE_ROOT"] = cache_root

        pretty_cmd = " ".join(cmd)
        print(f"[ABLATION] Learner={learner} -> DB {db_path}")
        print(f"[ABLATION] CMD: {pretty_cmd}")
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] bm_multiple failed for learner={learner} (exit {exc.returncode}).",
                  file=sys.stderr)
            return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
