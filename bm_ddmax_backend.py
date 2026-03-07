#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


def select_regex_oracle_cmd(base_format: str, category: str) -> List[str]:
    """
    Pick an oracle command for regex formats, consistent with bm_* validator selection:
    - prefer native/binary validator under validators/validate_<fmt>
    - else prefer validators/regex/validate_<fmt>
    - else fall back to python3 match.py <Category>
    """
    validator_bin = os.path.join("validators", f"validate_{base_format}")
    wrapper = os.path.join("validators", "regex", f"validate_{base_format}")
    if os.path.exists(validator_bin):
        return [validator_bin]
    if os.path.exists(wrapper):
        return [wrapper]
    return ["python3", "match.py", category]


def oracle_cmd_from_env_or_default(default_cmd: List[str]) -> List[str]:
    """
    Allow users to override the oracle command via LSTAR_ORACLE_VALIDATOR.
    This mirrors the behavior used by betamax for iterative relearn.
    """
    raw = os.environ.get("LSTAR_ORACLE_VALIDATOR")
    if not raw:
        return default_cmd
    try:
        parts = shlex.split(raw)
        return parts if parts else default_cmd
    except Exception:
        return default_cmd


@dataclass(frozen=True)
class DDMaxStats:
    total_runs: int
    accepted_runs: int
    rejected_runs: int
    timed_out: bool


class DeltaSet:
    """
    A set of indices represented as a sorted list of disjoint half-open intervals [lb, ub).

    In our DDMax-by-deletion implementation, a DeltaSet denotes the indices to exclude (delete).
    """

    def __init__(self, intervals: Optional[List[Tuple[int, int]]] = None):
        self.intervals: List[Tuple[int, int]] = []
        if intervals:
            self.intervals = self._normalize(intervals)

    @classmethod
    def full(cls, n: int) -> "DeltaSet":
        return cls([(0, n)])

    @classmethod
    def from_interval(cls, lb: int, ub: int) -> "DeltaSet":
        return cls([(lb, ub)])

    def copy(self) -> "DeltaSet":
        return DeltaSet(list(self.intervals))

    def length(self) -> int:
        return sum(ub - lb for lb, ub in self.intervals)

    def inside(self, i: int) -> bool:
        for lb, ub in self.intervals:
            if i < lb:
                return False
            if lb <= i < ub:
                return True
        return False

    def get_nth_index(self, n: int) -> int:
        for lb, ub in self.intervals:
            seg_len = ub - lb
            if n < seg_len:
                return lb + n
            n -= seg_len
        raise IndexError("nth index out of range")

    def exclude_interval(self, lb: int, ub: int) -> None:
        if lb >= ub:
            return
        new_intervals: List[Tuple[int, int]] = []
        for a, b in self.intervals:
            if ub <= a or b <= lb:
                new_intervals.append((a, b))
                continue
            if a < lb:
                new_intervals.append((a, lb))
            if ub < b:
                new_intervals.append((ub, b))
        self.intervals = self._normalize(new_intervals)

    @staticmethod
    def _normalize(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        xs = [(int(lb), int(ub)) for lb, ub in intervals if int(lb) < int(ub)]
        xs.sort()
        merged: List[Tuple[int, int]] = []
        for lb, ub in xs:
            if not merged:
                merged.append((lb, ub))
                continue
            plb, pub = merged[-1]
            if lb <= pub:
                merged[-1] = (plb, max(pub, ub))
            else:
                merged.append((lb, ub))
        return merged

    def key(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self.intervals)


def _granularity_interval(delta_set: DeltaSet, granularity: int, i: int) -> Tuple[int, int]:
    """
    Equivalent to eRepair's DD.getGranularityInterval for a DeltaSet.
    Returns an interval [lb, ub) within the delta_set.
    """
    total = delta_set.length()
    if total <= 0:
        return (0, 0)
    chunk = total // granularity
    if chunk <= 0:
        chunk = 1
    start_n = i * chunk
    if i != granularity - 1:
        end_n = min(total, (i + 1) * chunk)
    else:
        end_n = total
    if start_n >= total:
        start_n = total - 1
    if end_n <= start_n:
        end_n = min(total, start_n + 1)
    lb = delta_set.get_nth_index(start_n)
    ub = delta_set.get_nth_index(end_n - 1) + 1
    return (lb, ub)


def _apply_exclusion(text: str, excluding: DeltaSet) -> str:
    if not excluding.intervals:
        return text
    out_chars: List[str] = []
    for idx, ch in enumerate(text):
        if not excluding.inside(idx):
            out_chars.append(ch)
    return "".join(out_chars)


def ddmax_repair_by_deletion(
    broken_text: str,
    oracle_cmd_prefix: List[str],
    file_suffix: str,
    timeout_s: float,
    per_call_timeout_s: float = 2.0,
) -> Tuple[str, DDMaxStats]:
    """
    DDMax-style repair that only deletes characters:
    find a (near-)minimal exclusion set such that the remaining string passes the oracle.

    Returns (repaired_text, stats). If no oracle-accepted candidate exists, returns the original text.
    """
    t0 = time.time()
    total_runs = 0
    accepted = 0
    rejected = 0
    timed_out = False
    memo: dict[Tuple[Tuple[int, int], ...], bool] = {}
    last_accepted: Optional[DeltaSet] = None

    # One temp file reused across all oracle calls.
    tmp = tempfile.NamedTemporaryFile(prefix="ddmax_", suffix=f".{file_suffix}", delete=False)
    tmp_path = tmp.name
    tmp.close()

    def remaining_time() -> float:
        return max(0.0, timeout_s - (time.time() - t0))

    def run_oracle(excluding: DeltaSet) -> bool:
        nonlocal total_runs, accepted, rejected, last_accepted, timed_out
        if remaining_time() <= 0:
            timed_out = True
            return False

        k = excluding.key()
        if k in memo:
            return memo[k]

        candidate = _apply_exclusion(broken_text, excluding)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(candidate)
            tmo = min(per_call_timeout_s, max(0.05, remaining_time()))
            cp = subprocess.run(
                oracle_cmd_prefix + [tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=tmo,
            )
            ok = (cp.returncode == 0)
        except subprocess.TimeoutExpired:
            ok = False
        except Exception:
            ok = False

        total_runs += 1
        if ok:
            accepted += 1
            if last_accepted is None or excluding.length() <= last_accepted.length():
                last_accepted = excluding.copy()
        else:
            rejected += 1
        memo[k] = ok
        return ok

    def rec(delta_set: DeltaSet, granularity: int) -> DeltaSet:
        nonlocal timed_out
        if remaining_time() <= 0:
            timed_out = True
            return last_accepted.copy() if last_accepted is not None else delta_set

        inp_len = delta_set.length()
        if inp_len <= 1:
            return delta_set
        if inp_len == 0:
            return last_accepted.copy() if last_accepted is not None else delta_set

        for i in range(granularity):
            if remaining_time() <= 0:
                timed_out = True
                return last_accepted.copy() if last_accepted is not None else delta_set
            lb, ub = _granularity_interval(delta_set, granularity, i)
            new_set = DeltaSet.from_interval(lb, ub)
            if run_oracle(new_set):
                return rec(new_set, 2)

        for i in range(granularity):
            if remaining_time() <= 0:
                timed_out = True
                return last_accepted.copy() if last_accepted is not None else delta_set
            lb, ub = _granularity_interval(delta_set, granularity, i)
            new_set = delta_set.copy()
            new_set.exclude_interval(lb, ub)
            if run_oracle(new_set):
                return rec(new_set, 2)

        if granularity < inp_len:
            return rec(delta_set, min(inp_len, 2 * granularity))
        return delta_set

    try:
        # If already valid, no changes needed.
        if run_oracle(DeltaSet([])):
            repaired = broken_text
        else:
            full = DeltaSet.full(len(broken_text))
            _ = rec(full, 2)
            if last_accepted is None:
                repaired = broken_text
            else:
                repaired = _apply_exclusion(broken_text, last_accepted)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    stats = DDMaxStats(
        total_runs=total_runs,
        accepted_runs=accepted,
        rejected_runs=rejected,
        timed_out=timed_out,
    )
    return repaired, stats

