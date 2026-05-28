#!/usr/bin/env python3
"""
Shared helpers for invoking betaMax from benchmark scripts.

Supported backend:
  - "cpp": calls betamax_cpp/build/betamax_cpp (C++ engine)
"""

from __future__ import annotations

import os
from typing import Optional


def normalize_engine(engine: str | None) -> str:
    e = (engine or "").strip().lower()
    if not e:
        return "cpp"
    aliases = {
        "betamax": "cpp",
        "c++": "cpp",
        "cc": "cpp",
        "native": "cpp",
    }
    normalized = aliases.get(e, e)
    if normalized != "cpp":
        raise ValueError(
            "Legacy Python betaMax backend support has been removed. "
            "Only the C++ backend ('cpp') is supported."
        )
    return "cpp"


def cpp_bin_path() -> str:
    return os.environ.get("BM_BETAMAX_CPP_BIN", "betamax_cpp/build/betamax_cpp")


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
    exe = cpp_bin_path()
    if not os.path.exists(exe):
        raise FileNotFoundError(
            "betaMax C++ binary not found: "
            f"{exe}. Build it with "
            "'cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release' "
            "and 'cmake --build betamax_cpp/build -j', or run ./run_bms.sh."
        )
    # Default to unbounded cost search; set BM_CPP_MAX_COST to a non-negative integer to cap.
    max_cost = int(os.environ.get("BM_CPP_MAX_COST", "-1"))
    max_candidates = int(os.environ.get("BM_CPP_MAX_CANDIDATES", "50"))
    attempt_candidates = int(os.environ.get("BM_CPP_ATTEMPT_CANDIDATES", str(max_candidates)))
    incremental = os.environ.get("BM_CPP_INCREMENTAL", "1").lower() in ("1", "true", "yes")
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
    # Ensure incremental refine replay is enabled (can be disabled via BM_CPP_INCREMENTAL=0).
    cmd += ["--incremental" if incremental else "--no-incremental"]
    if negatives:
        cmd += ["--negatives", negatives]
    if oracle_cmd:
        cmd += ["--oracle-validator", oracle_cmd]
    if cache_path:
        cmd += ["--dfa-cache", cache_path]
    # The C++ engine supports mutation-based augmentation via --mutations.
    mutations = int(os.environ.get("BM_CPP_MUTATIONS", str(mutations)))
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
    assert e == "cpp"
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
