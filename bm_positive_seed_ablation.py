#!/usr/bin/env python3
"""
bm_positive_seed_ablation.py
----------------------------
Run bm_single/bm_multiple/bm_triple while varying the number of *positive seed*
examples used to learn the initial grammar (K).

This ablation writes DB files named:
  {mode}_k{K}.db
which matches `report.py`'s `BETAMAX_ABLATION_DB_TEMPLATE`.

Example:
  python3 bm_positive_seed_ablation.py --modes double --ks 50 25 12 6

Pass extra args to underlying bm_* scripts after "--":
  python3 bm_positive_seed_ablation.py -- --limit 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


DEFAULT_KS = [25, 12, 6]
DEFAULT_MODES = ["single", "double", "triple"]
DEFAULT_TEST_K = 50
DEFAULT_CACHE_LEARNER = "rpni_xover"
DEFAULT_RUNTIME_LEARNER = "rpni"
DEFAULT_SCRIPTS = {
    "single": "bm_single.py",
    "double": "bm_multiple.py",
    "triple": "bm_triple.py",
}
DEFAULT_DB_TEMPLATE = "{mode}_k{K}.db"
DEFAULT_CACHE_ROOT_TEMPLATE = "cache_{mode}_k{K}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run benchmarks across different positive seed sizes (K)."
    )
    ap.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=DEFAULT_KS,
        help="Training/seed size K values to evaluate (default: %(default)s).",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=["double"],
        help="Which mutation modes to run (default: %(default)s).",
    )
    ap.add_argument(
        "--test-k",
        type=int,
        default=DEFAULT_TEST_K,
        help="BM_TEST_K env var (default: %(default)s).",
    )
    ap.add_argument(
        "--cache-learner",
        default=DEFAULT_CACHE_LEARNER,
        help="Learner used for precompute cache init (default: %(default)s).",
    )
    ap.add_argument(
        "--runtime-learner",
        default=DEFAULT_RUNTIME_LEARNER,
        help="Learner used during the repair/relearn loop (default: %(default)s).",
    )
    ap.add_argument(
        "--db-template",
        default=DEFAULT_DB_TEMPLATE,
        help="DB template using {mode} and {K}/{k} placeholders (default: %(default)s).",
    )
    ap.add_argument(
        "--bm-single-script",
        default=DEFAULT_SCRIPTS["single"],
        help="Path to bm_single.py (default: %(default)s).",
    )
    ap.add_argument(
        "--bm-double-script",
        default=DEFAULT_SCRIPTS["double"],
        help="Path to bm_multiple.py (default: %(default)s).",
    )
    ap.add_argument(
        "--bm-triple-script",
        default=DEFAULT_SCRIPTS["triple"],
        help="Path to bm_triple.py (default: %(default)s).",
    )
    ap.add_argument(
        "--cache-root-template",
        default=DEFAULT_CACHE_ROOT_TEMPLATE,
        help="Cache dir template using {mode} and {K}/{k} (default: %(default)s).",
    )
    ap.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Subset of formats to run (default: all).",
    )
    ap.add_argument(
        "--mutations",
        nargs="+",
        default=None,
        help="Forward --mutations to bm_* (default: use each bm script default).",
    )
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Optional override of repair algorithms passed to benchmark scripts.",
    )
    ap.add_argument("--resume-only", action="store_true", help="Forward --resume-only.")
    ap.add_argument("--resume", action="store_true", help="Forward --resume.")
    ap.add_argument("--max-workers", type=int, help="Forward --max-workers.")
    ap.add_argument("--quiet", action="store_true", help="Forward --quiet.")
    ap.add_argument("--limit", type=int, help="Forward --limit.")
    ap.add_argument("--pause-on-exit", action="store_true", help="Forward --pause-on-exit.")
    ap.add_argument("--lstar-mutation-count", type=int, help="Forward --lstar-mutation-count.")
    ap.add_argument("--lstar-mutation-deterministic", action="store_true",
                    help="Forward --lstar-mutation-deterministic.")
    ap.add_argument("--lstar-mutation-seed", type=int, help="Forward --lstar-mutation-seed.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a run if its DB already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing benchmark scripts.")
    ap.add_argument(
        "bm_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded verbatim (prefix with -- before them).",
    )
    return ap.parse_args()


def _render(template: str, *, mode: str, k: int) -> str:
    return template.format(mode=mode, K=k, k=k)


def build_command(args: argparse.Namespace, *, mode: str, db_path: str) -> List[str]:
    script_map = {
        "single": args.bm_single_script,
        "double": args.bm_double_script,
        "triple": args.bm_triple_script,
    }
    bm_script = script_map[mode]
    cmd = ["python3", bm_script, "--db", db_path]
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


def main() -> int:
    args = parse_args()

    ks: List[int] = []
    for k in args.ks:
        if k <= 0:
            print(f"[WARN] Ignoring non-positive K={k}")
            continue
        ks.append(int(k))
    if not ks:
        print("[ERROR] No valid K values provided.", file=sys.stderr)
        return 1

    cache_learner = (args.cache_learner or "").strip()
    runtime_learner = (args.runtime_learner or "").strip()
    if not cache_learner:
        print("[ERROR] --cache-learner is empty.", file=sys.stderr)
        return 1
    if not runtime_learner:
        print("[ERROR] --runtime-learner is empty.", file=sys.stderr)
        return 1

    for mode in args.modes:
        for k in ks:
            db_path = _render(args.db_template, mode=mode, k=k)
            if args.skip_existing and os.path.exists(db_path):
                print(f"[INFO] Skipping {mode} K={k} (DB exists: {db_path})")
                continue

            cache_root = _render(args.cache_root_template, mode=mode, k=k)
            os.makedirs(cache_root, exist_ok=True)

            env = os.environ.copy()
            env["BM_TRAIN_K"] = str(k)
            env["BM_TEST_K"] = str(args.test_k)
            env["LSTAR_PRECOMP_K"] = str(k)
            env["LSTAR_CACHE_ROOT"] = cache_root
            # Two-stage setup:
            # - bm_* precompute uses LSTAR_CACHE_LEARNER to initialize the grammar cache
            # - per-sample repairs use LSTAR_LEARNER (and BM_BETAMAX_LEARNER) for relearning attempts
            env["LSTAR_CACHE_LEARNER"] = cache_learner
            env["LSTAR_LEARNER"] = runtime_learner
            env["BM_BETAMAX_LEARNER"] = runtime_learner

            cmd = build_command(args, mode=mode, db_path=db_path)
            print(f"[ABLATION] Running {mode} K={k} -> DB {db_path}")
            print(f"[ABLATION] Cache: {cache_root}")
            print(f"[ABLATION] Learners: cache={cache_learner}, runtime={runtime_learner}")
            print(f"[ABLATION] CMD: {' '.join(cmd)}")
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] {mode} benchmark failed for K={k} (exit {exc.returncode}).",
                      file=sys.stderr)
                return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
