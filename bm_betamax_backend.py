#!/usr/bin/env python3
"""
Shared helpers for invoking betaMax from benchmark scripts.

Supported engines:
  - "python": calls betamax/app/betamax.py (original engine)
  - "cpp":    calls betamax_cpp/build/betamax_cpp (C++ engine)
"""

from __future__ import annotations

import os
from typing import Iterable, Optional


def normalize_engine(engine: str | None) -> str:
    e = (engine or "").strip().lower()
    if not e:
        return "python"
    aliases = {
        "py": "python",
        "betamax": "python",
        "c++": "cpp",
        "cc": "cpp",
        "native": "cpp",
    }
    return aliases.get(e, e)


def get_engine(cli_engine: str | None = None, *, default: str = "python") -> str:
    if cli_engine:
        return normalize_engine(cli_engine)
    return normalize_engine(os.environ.get("BM_BETAMAX_ENGINE", default))


def should_precompute_cache(engine: str) -> bool:
    # Both engines support a precompute step:
    # - python: writes a grammar cache JSON via --grammar-cache --init-cache
    # - cpp:    writes a DFA cache via --dfa-cache --init-cache
    return normalize_engine(engine) in ("python", "cpp")


def cpp_bin_path() -> str:
    return os.environ.get("BM_BETAMAX_CPP_BIN", "betamax_cpp/build/betamax_cpp")


def _eq_flags_from_env() -> list[str]:
    flags: list[str] = []
    if os.environ.get("LSTAR_EQ_MAX_LENGTH"):
        flags += ["--eq-max-length", os.environ["LSTAR_EQ_MAX_LENGTH"]]
    if os.environ.get("LSTAR_EQ_SAMPLES_PER_LENGTH"):
        flags += ["--eq-samples-per-length", os.environ["LSTAR_EQ_SAMPLES_PER_LENGTH"]]
    if os.environ.get("LSTAR_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
        flags += ["--eq-disable-sampling"]
    if os.environ.get("LSTAR_EQ_SKIP_NEGATIVES", "").lower() in ("1", "true", "yes"):
        flags += ["--eq-skip-negatives"]
    if os.environ.get("LSTAR_EQ_MAX_ORACLE"):
        flags += ["--eq-max-oracle", os.environ["LSTAR_EQ_MAX_ORACLE"]]
    return flags


def build_cmd_python(
    *,
    positives: str,
    negatives: str,
    cache_path: Optional[str],
    category: str,
    broken_file: str,
    output_file: str,
    attempts: int,
    mutations: int,
    learner: str,
    oracle_cmd: Optional[str],
) -> list[str]:
    cmd = [
        "python3",
        "betamax/app/betamax.py",
        "--positives",
        positives,
        "--negatives",
        negatives,
        "--category",
        category,
        "--broken-file",
        broken_file,
        "--output-file",
        output_file,
        "--max-attempts",
        str(int(attempts)),
        "--mutations",
        str(int(mutations)),
        "--learner",
        learner,
    ]
    if cache_path:
        cmd += ["--grammar-cache", cache_path]
    if oracle_cmd:
        cmd += ["--oracle-validator", oracle_cmd]
    cmd += _eq_flags_from_env()
    return cmd


def build_cmd_cpp(
    *,
    positives: str,
    negatives: str,
    category: str,
    broken_file: str,
    output_file: str,
    attempts: int,
    mutations: int,
    learner: str,
    oracle_cmd: Optional[str],
    cache_path: Optional[str],
) -> list[str]:
    exe = os.environ.get("BM_BETAMAX_CPP_BIN", "betamax_cpp/build/betamax_cpp")
    # Default to unbounded cost search; set BM_CPP_MAX_COST to a non-negative integer to cap.
    max_cost = int(os.environ.get("BM_CPP_MAX_COST", "-1"))
    max_candidates = int(os.environ.get("BM_CPP_MAX_CANDIDATES", "50"))
    attempt_candidates = int(os.environ.get("BM_CPP_ATTEMPT_CANDIDATES", str(max_candidates)))
    cmd = [
        exe,
        "--positives",
        positives,
        "--category",
        category,
        "--broken-file",
        broken_file,
        "--output-file",
        output_file,
        "--learner",
        learner,
        "--max-attempts",
        str(int(attempts)),
        "--attempt-candidates",
        str(int(attempt_candidates)),
        "--max-cost",
        str(max_cost),
        "--max-candidates",
        str(max_candidates),
        "--repo-root",
        ".",
    ]
    # Ensure incremental refine replay is enabled for benchmarks.
    cmd += ["--incremental"]
    if negatives:
        cmd += ["--negatives", negatives]
    if oracle_cmd:
        cmd += ["--oracle-validator", oracle_cmd]
    if cache_path:
        cmd += ["--dfa-cache", cache_path]
    # C++ engine supports mutation-based augmentation via --mutations.
    # Keep behavior consistent with the Python backend: benchmark scripts pass `mutations`.
    _ = attempts  # reserved for future parity options
    if int(mutations) > 0:
        cmd += ["--mutations", str(int(mutations))]
    # Optional sampling knobs are intentionally not propagated by default, to
    # avoid diverging learning behavior unless explicitly enabled.
    if os.environ.get("BM_CPP_EQ_MAX_ORACLE"):
        cmd += ["--eq-max-oracle", os.environ["BM_CPP_EQ_MAX_ORACLE"]]
    if os.environ.get("BM_CPP_EQ_MAX_ROUNDS"):
        cmd += ["--eq-max-rounds", os.environ["BM_CPP_EQ_MAX_ROUNDS"]]
    if os.environ.get("BM_CPP_EQ_MAX_LENGTH"):
        cmd += ["--eq-max-length", os.environ["BM_CPP_EQ_MAX_LENGTH"]]
    if os.environ.get("BM_CPP_EQ_SAMPLES_PER_LENGTH"):
        cmd += ["--eq-samples-per-length", os.environ["BM_CPP_EQ_SAMPLES_PER_LENGTH"]]
    if os.environ.get("BM_CPP_EQ_DISABLE_SAMPLING", "").lower() in ("1", "true", "yes"):
        cmd += ["--eq-disable-sampling"]
    if os.environ.get("BM_CPP_SEED"):
        cmd += ["--seed", os.environ["BM_CPP_SEED"]]
    return cmd


def build_betamax_cmd(
    *,
    engine: str,
    positives: str,
    negatives: str,
    cache_path: Optional[str],
    category: str,
    broken_file: str,
    output_file: str,
    attempts: int,
    mutations: int,
    learner: str,
    oracle_cmd: Optional[str],
) -> list[str]:
    e = normalize_engine(engine)
    if e == "python":
        return build_cmd_python(
            positives=positives,
            negatives=negatives,
            cache_path=cache_path,
            category=category,
            broken_file=broken_file,
            output_file=output_file,
            attempts=attempts,
            mutations=mutations,
            learner=learner,
            oracle_cmd=oracle_cmd,
        )
    if e == "cpp":
        return build_cmd_cpp(
            positives=positives,
            negatives=negatives,
            cache_path=cache_path,
            category=category,
            broken_file=broken_file,
            output_file=output_file,
            attempts=attempts,
            mutations=mutations,
            learner=learner,
            oracle_cmd=oracle_cmd,
        )
    raise ValueError(f"Unknown BM_BETAMAX_ENGINE: {engine!r} (expected 'python' or 'cpp')")
