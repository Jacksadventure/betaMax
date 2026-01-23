#!/usr/bin/env python3
"""
bench_combined_validators.py
---------------------------
Benchmark validator runtime on the 100 samples in data/combined/*.txt.

It runs (when available) the validator variants used in bm_* scripts:
- validators/validate_<fmt>          (native/binary validator)
- validators/regex/validate_<fmt>    (Python match.py wrapper)
- python3 match.py <Category>        (direct)

For each format and validator, prints total time and average time per entry.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORMAT_TO_CATEGORY: dict[str, str] = {
    "date": "Date",
    "iso8601": "ISO8601",
    "time": "Time",
    "url": "URL",
    "isbn": "ISBN",
    "ipv4": "IPv4",
    "ipv6": "IPv6",
}


@dataclass(frozen=True)
class ValidatorSpec:
    name: str
    cmd: list[str]


def _stats_seconds(xs: list[float]) -> tuple[float, float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    sd = math.sqrt(var)
    return mu, sd, min(xs), max(xs)


def _read_samples(path: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < n:
        raise ValueError(f"{path} has only {len(lines)} non-empty lines (<{n})")
    return lines[:n]


def _available_validators(fmt: str, category: str) -> list[ValidatorSpec]:
    specs: list[ValidatorSpec] = []
    bin_path = Path("validators") / f"validate_{fmt}"
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        specs.append(ValidatorSpec(name=f"validators/validate_{fmt}", cmd=[str(bin_path)]))

    # regex_path = Path("validators") / "regex" / f"validate_{fmt}"
    # if regex_path.exists() and os.access(regex_path, os.X_OK):
    #     specs.append(ValidatorSpec(name=f"validators/regex/validate_{fmt}", cmd=[str(regex_path)]))

    specs.append(ValidatorSpec(name="python3 match.py", cmd=["python3", "match.py", category]))
    return specs


def _run_one(cmd_prefix: list[str], file_path: str, timeout_s: float) -> tuple[float, int]:
    t0 = time.perf_counter()
    cp = subprocess.run(
        cmd_prefix + [file_path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
    )
    t1 = time.perf_counter()
    return (t1 - t0), int(cp.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark validators on data/combined (100 samples each).")
    ap.add_argument("--n", type=int, default=100, help="Number of samples per format (default: 100)")
    ap.add_argument("--timeout", type=float, default=5.0, help="Per-validation subprocess timeout in seconds")
    ap.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(FORMAT_TO_CATEGORY.keys()),
        help="Formats to benchmark (default: all present in data/combined)",
    )
    args = ap.parse_args()

    combined_dir = Path("data") / "combined"
    if not combined_dir.is_dir():
        raise SystemExit(f"Missing directory: {combined_dir}")

    formats: Iterable[str]
    if args.formats:
        formats = args.formats
    else:
        present = {p.stem for p in combined_dir.glob("*.txt")}
        formats = [f for f in FORMAT_TO_CATEGORY.keys() if f in present]

    with tempfile.TemporaryDirectory(prefix="earlyrepairer_bench_") as td:
        tmpdir = Path(td)
        for fmt in formats:
            if fmt not in FORMAT_TO_CATEGORY:
                continue
            src = combined_dir / f"{fmt}.txt"
            if not src.is_file():
                print(f"[SKIP] Missing: {src}")
                continue

            category = FORMAT_TO_CATEGORY[fmt]
            samples = _read_samples(src, args.n)

            # Pre-materialize one file per sample so benchmark excludes Python file-write overhead.
            paths: list[str] = []
            for i, s in enumerate(samples):
                p = tmpdir / f"{fmt}_{i}.txt"
                p.write_text(s + "\n", encoding="utf-8")
                paths.append(str(p))

            validators = _available_validators(fmt, category)
            if not validators:
                print(f"[SKIP] No validators found for {fmt}")
                continue

            print(f"\n=== {fmt} (n={args.n}) ===")
            rows: list[tuple[str, float]] = []
            for spec in validators:
                times: list[float] = []
                ok = 0
                timed_out = 0
                for fp in paths:
                    try:
                        dt, rc = _run_one(spec.cmd, fp, args.timeout)
                        times.append(dt)
                        ok += 1 if rc == 0 else 0
                    except subprocess.TimeoutExpired:
                        timed_out += 1
                mu, sd, mn, mx = _stats_seconds(times)
                total = sum(times)
                rows.append((spec.name, mu))
                print(
                    f"{spec.name:<28} total={total:8.3f}s  "
                    f"avg={mu*1000:8.3f}ms  sd={sd*1000:8.3f}ms  "
                    f"min={mn*1000:8.3f}ms  max={mx*1000:8.3f}ms  "
                    f"ok={ok:3d}/{len(paths):3d}  timeout={timed_out:3d}"
                )

            # # Relative speed summary (lower is faster).
            # fastest = min((mu for _, mu in rows), default=None)
            # if fastest and fastest > 0:
            #     print("Speedups vs fastest (avg time):")
            #     for name, mu in sorted(rows, key=lambda x: x[1]):
            #         print(f"  {name:<28} x{(mu/fastest):6.2f}")


if __name__ == "__main__":
    main()
