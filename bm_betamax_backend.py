#!/usr/bin/env python3
"""
Shared helpers for invoking betaMax from benchmark scripts.

Supported engines:
  - "python": calls the legacy Python betaMax entrypoint
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


def python_entrypoint_path() -> str:
    return os.environ.get("BM_BETAMAX_PY_ENTRYPOINT", "betamax/app/betamax.py")


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
    entrypoint = python_entrypoint_path()
    if not os.path.exists(entrypoint):
        raise FileNotFoundError(
            "Legacy Python betaMax entrypoint not found: "
            f"{entrypoint}. This repository snapshot is intended to run with "
            "--betamax-engine cpp. If you have the legacy Python backend in a "
            "different checkout, set BM_BETAMAX_PY_ENTRYPOINT to that file."
        )
    cmd = [
        "python3",
        entrypoint,
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
