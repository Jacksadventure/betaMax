#!/usr/bin/env python3
"""
Repairer: L* (validator-backed) or RPNI + Error-Correcting Earley

Pipeline:
1) Learn a DFA via L* (validator-backed membership; seeds table with positives first) or via RPNI from positive/negative samples, then convert to a right-linear CFG.
2) Build a covering grammar and use error-correcting Earley to repair broken inputs.
3) Validate repaired outputs with an oracle (e.g., python3 match.py Date).
4) If oracle fails, incrementally add the failing negative example to Teacher.negatives and relearn (<= max_attempts).

Usage example:
  python3 betamax/app/betamax.py \
    --positives positive/positives.txt \
    --negatives negative/negatives.txt \
    --category Date \
    --limit 10 \
    --max-attempts 5

Notes:
- Does NOT depend on simplefuzzer.
- Requires earleyparser (vendored wheel in betamax/py) and sympy (installed).
- Default learner is passive RPNI (validator-backed). Use --learner lstar_oracle to switch to the L* oracle-backed learner.
"""

import os
import sys
import glob
import argparse
import subprocess
import tempfile
import traceback
import json
import shlex
import time
from typing import List, Set, Tuple, Dict, Any, Optional

# Ensure project root (betamax) on sys.path so 'lstar' and vendored wheels work
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure local vendored wheels in py/ are importable (earleyparser, etc.)
PY_DIR = os.path.join(ROOT_DIR, "py")
if os.path.isdir(PY_DIR):
    if PY_DIR not in sys.path:
        sys.path.insert(0, PY_DIR)
    for whl in glob.glob(os.path.join(PY_DIR, "*.whl")):
        if whl not in sys.path:
            sys.path.append(whl)

# RPNI import (passive NFA-based learning from samples)
try:
    from lstar.rpni_nfa import learn_grammar_from_samples_nfa as rpni_learn_grammar
except Exception:
    # Fallback to DFA-based RPNI if NFA variant unavailable
    from lstar.rpni import learn_grammar_from_samples as rpni_learn_grammar
from lstar.rpni_nfa import learn_grammar_from_samples_nfa as rpni_nfa_learn_grammar
from lstar.rpni_fuzz import learn_grammar_from_samples_fuzz as rpni_fuzz_learn_grammar
from lstar.rpni_xover import learn_grammar_from_samples_xover as rpni_xover_learn_grammar
from lstar.observation_table import ObservationTable
import earleyparser
import cfgrandomsample
import simplefuzzer as fuzzer
import random

# Import error-correcting Earley runtime (no side effects)
try:
    from lstar import ec_runtime as ec
except Exception:
    import ec_runtime as ec

# Types
Grammar = Dict[str, List[List[str]]]

def save_grammar_cache(path: str, g: Grammar, start_sym: str, alphabet: List[str]) -> None:
    data = {
        "start_sym": start_sym,
        "alphabet": alphabet,
        "grammar": g,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_grammar_cache(path: str) -> Tuple[Grammar, str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = data["grammar"]
    start_sym = data["start_sym"]
    alphabet = data["alphabet"]
    return g, start_sym, alphabet


def read_lines(path: str) -> List[str]:
    vals: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            vals.append(line)
    return vals



def derive_alphabet_from_examples(positives: Set[str], negatives: Set[str]) -> List[str]:
    chars = set()
    for s in list(positives) + list(negatives):
        chars.update(list(s))
    return sorted(chars) if chars else list("ab")


def terminals_of_grammar(g: Grammar) -> List[str]:
    syms = set()
    for nt, alts in g.items():
        for alt in alts:
            for t in alt:
                if not ec.is_nt(t):
                    syms.add(t)
    return sorted(syms)

def expand_set_terminals(g: Grammar, alphabet: List[str]) -> Grammar:
    """
    Expand any set/frozenset terminals in grammar productions into multiple alternatives
    over their member characters, so the downstream Earley implementation only sees
    string terminals. Keeps epsilon (empty) productions untouched.
    """
    import itertools as I
    new_g: Grammar = {}
    for nt, alts in g.items():
        new_alts: List[List[str]] = []
        for alt in alts:
            # alt is a sequence of symbols; expand any set-like terminal
            choices: List[List[str]] = []
            for t in alt:
                if isinstance(t, (set, frozenset)):
                    # expand set-like terminal into alternatives of member strings
                    choices.append([str(x) for x in t])
                else:
                    choices.append([t])
            # Cartesian product across choices; if choices is empty, preserve epsilon
            for prod in I.product(*choices) if choices else [()]:
                new_alts.append(list(prod))
        new_g[nt] = new_alts
    return new_g

def sanitize_grammar(g: Grammar) -> Grammar:
    """
    Ensure all production symbols are strings; convert any residual non-strings
    (e.g., ints, tuples) to strings. Should be run after expand_set_terminals.
    """
    new_g: Grammar = {}
    for nt, alts in g.items():
        nt_str = nt if isinstance(nt, str) else str(nt)
        new_alts: List[List[str]] = []
        for alt in alts:
            new_alt: List[str] = []
            for t in alt:
                if isinstance(t, str):
                    new_alt.append(t)
                else:
                    new_alt.append(str(t))
            new_alts.append(new_alt)
        new_g[nt_str] = new_alts
    return new_g

def assert_no_set_tokens(g: Grammar):
    """
    Raise if any set/frozenset remains anywhere in grammar productions.
    """
    for nt, alts in g.items():
        for alt in alts:
            for t in alt:
                if isinstance(t, (set, frozenset)):
                    raise TypeError(f"Grammar contains set terminal {t} in production {nt} -> {alt}")

def debug_count_symbol_types(g: Grammar):
    """
    Print a brief summary of symbol types in grammar (for diagnostics).
    """
    import collections
    cnt = collections.Counter()
    for nt, alts in g.items():
        for alt in alts:
            for t in alt:
                cnt[type(t).__name__] += 1
    print(f"[DEBUG] Grammar symbol types: {dict(cnt)}")


def generate_mutations(positives: Set[str], n: int, alphabet: List[str]) -> List[str]:
    """
    Generate n mutations from THE SHORTEST positive string only, mimicking
    earleyrepairer.cpp's single-edit covering grammar operations:
      - delete one character
      - substitute one character with another from alphabet (different char)
      - insert one character from alphabet at some position
    IMPORTANT: Do NOT introduce new characters beyond 'alphabet'.
    Deterministic enumeration in the order: deletions -> substitutions -> insertions,
    stop when n unique candidates collected.
    """
    if not positives or n <= 0 or not alphabet:
        return []
    # Shortest positive (tie-break lexicographic)
    pos_list = sorted([p for p in positives if isinstance(p, str)], key=lambda s: (len(s), s))
    if not pos_list:
        return []
    s = pos_list[0]
    alpha: List[str] = list(alphabet)
    alpha_set: Set[str] = set(alpha)

    out_list: List[str] = []
    seen: Set[str] = set()

    def add_cand(cand: str) -> bool:
        if cand and cand not in positives and cand != s and all((c in alpha_set) for c in cand):
            if cand not in seen:
                seen.add(cand)
                out_list.append(cand)
                return True
        return False

    # 1) Deletions (strictly shorter)
    for i in range(len(s)):
        if len(out_list) >= n:
            return out_list
        cand = s[:i] + s[i+1:]
        add_cand(cand)

    # 2) Substitutions (same length)
    if len(out_list) < n and len(s) > 0:
        for i in range(len(s)):
            if len(out_list) >= n:
                return out_list
            for ch in alpha:
                if ch == s[i]:
                    continue
                if add_cand(s[:i] + ch + s[i+1:]):
                    if len(out_list) >= n:
                        return out_list

    # 3) Insertions (longer)
    if len(out_list) < n:
        for i in range(len(s) + 1):
            if len(out_list) >= n:
                return out_list
            for ch in alpha:
                if add_cand(s[:i] + ch + s[i:]):
                    if len(out_list) >= n:
                        return out_list

    return out_list[:n]


def generate_mutations_random(positives: Set[str], n: int, alphabet: List[str], seed: Optional[int] = None) -> List[str]:
    """
    Randomly generate up to n mutations from THE SHORTEST positive string only.
    Edits are sampled at random with a bias toward deletions/substitutions:
      - op ∈ {del, sub, ins} with probabilities ~ (0.5, 0.35, 0.15)
      - position chosen uniformly at random
      - substitution/insert character ~ Uniform(alphabet), for sub ensure != original char when possible
    Ensures:
      - no new characters beyond 'alphabet'
      - excludes the original string
      - uniqueness (set)
    """
    if not positives or n <= 0 or not alphabet:
        return []
    # shortest positive (length, then lexicographic)
    base_list = sorted([p for p in positives if isinstance(p, str)], key=lambda s: (len(s), s))
    if not base_list:
        return []
    s = base_list[0]
    import random as _rnd
    if seed is not None:
        try:
            _rnd.seed(int(seed))
        except Exception:
            pass
    alpha = list(alphabet)
    alpha_set = set(alpha)

    out: Set[str] = set()
    tries = 0
    max_tries = max(2000, 40 * n)
    while len(out) < n and tries < max_tries:
        tries += 1
        # choose op with bias (del > sub > ins)
        r = _rnd.random()
        if len(s) == 0:
            op = "ins"
        elif len(s) == 1:
            op = "sub" if r < 0.7 else "ins"
        else:
            op = "del" if r < 0.5 else ("sub" if r < 0.85 else "ins")

        if op == "del":
            i = _rnd.randrange(len(s))
            cand = s[:i] + s[i+1:]
        elif op == "sub":
            i = _rnd.randrange(len(s))
            if len(alpha) > 1:
                choices = [ch for ch in alpha if ch != s[i]]
                if not choices:
                    continue
                ch = _rnd.choice(choices)
            else:
                ch = alpha[0]
            cand = s[:i] + ch + s[i+1:]
        else:  # ins
            i = _rnd.randrange(len(s) + 1)
            ch = _rnd.choice(alpha)
            cand = s[:i] + ch + s[i:]

        if cand != s and cand not in positives and all((c in alpha_set) for c in cand):
            out.add(cand)
    return list(out)[:n]


def learn_grammar(positives: Set[str], negatives: Set[str], unknown_policy: str = "negative") -> Tuple[Grammar, str, List[str]]:
    """
    Learn a right-linear CFG from samples using RPNI (no membership oracle).
    unknown_policy is ignored (kept for CLI compatibility).
    """
    t0 = time.time()
    g, start_sym, alphabet = rpni_learn_grammar(positives, negatives)
    t1 = time.time()
    try:
        print(f"[PROFILE] rpni: {t1 - t0:.2f}s, P={len(positives)}, N={len(negatives)}, |A|={len(alphabet)}")
    except Exception:
        print(f"[PROFILE] rpni: {t1 - t0:.2f}s")
    return g, start_sym, alphabet

def learn_grammar_nfa(positives: Set[str], negatives: Set[str], unknown_policy: str = "negative") -> Tuple[Grammar, str, List[str]]:
    """
    Learn a right-linear CFG from samples using modified RPNI that keeps an NFA.
    unknown_policy is ignored (kept for CLI compatibility).
    """
    t0 = time.time()
    g, start_sym, alphabet = rpni_nfa_learn_grammar(positives, negatives)
    t1 = time.time()
    try:
        print(f"[PROFILE] rpni_nfa: {t1 - t0:.2f}s, P={len(positives)}, N={len(negatives)}, |A|={len(alphabet)}")
    except Exception:
        print(f"[PROFILE] rpni_nfa: {t1 - t0:.2f}s")
    return g, start_sym, alphabet


class ValidatorOracle:
    """
    Oracle that answers membership queries using external validators and
    provides a practical equivalence check combining:
      - all provided positives must be accepted by the grammar
      - provided negatives must be rejected by the grammar
      - random samples generated from the learned grammar must be accepted by the validator
    """
    def __init__(
        self,
        category: str,
        positives: Set[str],
        negatives: Set[str],
        validator_cmd: Optional[List[str]] = None,
        eq_max_length: int = 10,
        eq_samples_per_length: int = 50,
        eq_disable_sampling: bool = False,
        check_negatives: bool = True,
        eq_budget: Optional[int] = None,
    ):
        self.category = category
        self.positives = set(positives)
        self.negatives = set(negatives)
        self.validator_cmd = validator_cmd
        # Equivalence parameters (allow faster/approximate checks)
        self.max_length = int(max(0, eq_max_length))
        self.sample_n = int(max(0, eq_samples_per_length))
        self.eq_disable_sampling = bool(eq_disable_sampling)
        self.check_negatives = bool(check_negatives)
        self.eq_budget = eq_budget if (eq_budget is None or eq_budget >= 0) else None
        self.eq_calls = 0  # counts validate_with_match calls in sampling
        # Membership memoization to reduce external oracle runs
        self.mem_cache: Dict[str, bool] = {}

    def is_member(self, q: str) -> int:
        if q in self.mem_cache:
            ok = self.mem_cache[q]
        else:
            ok = validate_with_match(self.category, q, self.validator_cmd)
            self.mem_cache[q] = ok
        try:
            prev = q if len(q) <= 200 else (q[:200] + "...(truncated)")
        except Exception:
            prev = "<unprintable>"
        print(f"[DEBUG] Membership verdict: {'ACCEPT' if ok else 'REJECT'} for {repr(prev)}")
        return 1 if ok else 0

    def is_equivalent(self, grammar: Dict[str, List[List[str]]], start: str) -> Tuple[bool, Optional[str]]:
        # 1) Positives must be accepted by learned grammar
        parser = earleyparser.EarleyParser(grammar)
        for p in self.positives:
            try:
                list(parser.recognize_on(p, start))
            except Exception:
                return False, p  # positive not accepted by grammar

        # 2) Negatives must be rejected by learned grammar (optional)
        if self.check_negatives:
            for n in self.negatives:
                try:
                    list(parser.recognize_on(n, start))
                    return False, n  # negative wrongly accepted by grammar
                except Exception:
                    pass

        # 3) Optional: random sampling against external validator (no false positives)
        if self.eq_disable_sampling or self.max_length == 0 or self.sample_n == 0:
            print("[DEBUG] Equivalence sampling disabled or parameters set to 0; accepting current hypothesis for speed.")
            return True, None

        try:
            sampler = cfgrandomsample.RandomSampleCFG(grammar)
        except Exception:
            return True, None  # if sampling fails, accept for practicality

        for l in range(1, max(1, self.max_length) + 1):
            # Budget check per equivalence run
            if self.eq_budget is not None and self.eq_calls >= self.eq_budget:
                print(f"[DEBUG] Equivalence sampling budget exhausted ({self.eq_calls}/{self.eq_budget}); accepting hypothesis.")
                return True, None
            try:
                key_node = sampler.key_get_def(start, l)
                cnt = key_node.count
            except Exception:
                continue
            if not cnt:
                continue
            tries = min(self.sample_n, cnt)
            for _ in range(tries):
                if self.eq_budget is not None and self.eq_calls >= self.eq_budget:
                    print(f"[DEBUG] Equivalence sampling budget exhausted ({self.eq_calls}/{self.eq_budget}); accepting hypothesis.")
                    return True, None
                try:
                    at = random.randint(0, max(0, cnt - 1))
                    tree = sampler.key_get_string_at(key_node, at)
                    s = fuzzer.tree_to_string(tree)
                    self.eq_calls += 1
                    if not validate_with_match(self.category, s, self.validator_cmd):
                        return False, s
                except Exception:
                    # ignore sampling issues at this length
                    pass
        return True, None


def lstar_learn_with_oracle(
    positives: Set[str],
    negatives: Set[str],
    category: str,
    validator_cmd: Optional[List[str]] = None,
    eq_max_length: int = 10,
    eq_samples_per_length: int = 50,
    eq_disable_sampling: bool = False,
    check_negatives: bool = True,
    eq_budget: Optional[int] = None,
) -> Tuple[Grammar, str, List[str]]:
    """
    Learn a right-linear CFG using L* where:
      - Observation table membership queries are answered by external validators
      - The table is seeded with positive examples first
      - Equivalence uses a combination of finite checks and sampled conformance
    """
    t0 = time.time()
    alphabet = derive_alphabet_from_examples(positives, negatives)
    oracle = ValidatorOracle(
        category,
        positives,
        negatives,
        validator_cmd=validator_cmd,
        eq_max_length=eq_max_length,
        eq_samples_per_length=eq_samples_per_length,
        eq_disable_sampling=eq_disable_sampling,
        check_negatives=check_negatives,
        eq_budget=eq_budget,
    )
    T = ObservationTable(alphabet)
    # Initialize and seed with positives first
    T.init_table(oracle)
    # Add positive prefixes as candidate access strings to bias early learning
    for p in sorted(positives, key=len):
        if p not in T.P:
            T.add_prefix(p, oracle)

    # L* main loop (closed/consistent) with equivalence via oracle
    while True:
        count = 0
        while True:
            is_closed, unknown_P = T.closed()
            is_consistent, _, unknown_AS = T.consistent()
            if is_closed and is_consistent:
                break
            if not is_closed:
                T.add_prefix(unknown_P, oracle)
            if not is_consistent:
                T.add_suffix(unknown_AS, oracle)
            count += 1
            if count == 1:
                break  
        grammar, start = T.grammar()
        eq, counter = oracle.is_equivalent(grammar, start)
        if eq or counter is None:
            t1 = time.time()
            try:
                print(f"[PROFILE] lstar_oracle: {t1 - t0:.2f}s, P={len(positives)}, N={len(negatives)}, |A|={len(alphabet)}")
            except Exception:
                print(f"[PROFILE] lstar_oracle: {time.time() - t0:.2f}s")
            return grammar, start, alphabet
        # Add prefixes of the counterexample string to refine table
        for i in range(len(counter)):
            T.add_prefix(counter[: i + 1], oracle)


def earley_correct(g: Grammar, start_sym: str, broken: str, symbols: List[str] = None, log: bool = False, penalty: Optional[int] = None, max_penalty: Optional[int] = None) -> str:
    """
    Use the error-correcting Earley parser with a covering grammar to fix 'broken'.
    If 'penalty' is provided, attempt to select a solution with exactly that correction penalty.
    Falls back to lowest-penalty solution if no parse exists with the requested penalty.
    """
    # If symbols not provided, infer from grammar terminals
    if symbols is None:
        symbols = terminals_of_grammar(g)

    covering_grammar, covering_start = ec.augment_grammar_ex(g, start_sym, symbols=symbols)
    parser = ec.ErrorCorrectingEarleyParser(covering_grammar)
    # Max penalty pruning disabled; do not set parser.max_penalty here

    # Parse timeout (seconds), can be overridden via env LSTAR_PARSE_TIMEOUT
    try:
        parse_timeout = float(os.getenv("LSTAR_PARSE_TIMEOUT", "100.0"))
    except Exception:
        parse_timeout = 40.0

    # Single parse attempt without max_penalty budget retries
    se = None
    last_err = None
    # Timeout guard around parse (uses Unix signals; works on macOS/Linux)
    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("parse timeout")

        old_handler = None
        try:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
        except Exception:
            old_handler = None

        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, max(0.0, parse_timeout))
            else:
                # Fallback with integer seconds if setitimer unavailable
                secs = int(parse_timeout) if parse_timeout >= 1 else 1
                signal.alarm(secs)

            se = ec.SimpleExtractorEx(parser, broken, covering_start, penalty=penalty, log=log)
        finally:
            try:
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:
                    signal.alarm(0)
            except Exception:
                pass
            try:
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    except TimeoutError as te:
        last_err = te
        if log:
            try:
                print(f"[WARN] parse_prefix timed out after {parse_timeout:.2f}s")
            except Exception:
                print(f"[WARN] parse_prefix timed out")
    except Exception as e:
        last_err = e
        # If requested penalty is invalid (no parse with that penalty), fall back to minimum-penalty
        if penalty is not None and "Invalid penalty" in str(e):
            if log:
                print(f"[WARN] No solution with penalty={penalty}. Falling back to minimum-penalty solution.")
            try:
                se = ec.SimpleExtractorEx(parser, broken, covering_start, penalty=None, log=log)
            except Exception as e2:
                last_err = e2
        else:
            # propagate other errors
            raise

    if se is None:
        # Final fallback: raise last error if parse never succeeded
        raise last_err if last_err else RuntimeError("parse failed without specific error")
    tree = se.extract_a_tree()
    # Use correction-aware projection that maps covering grammar back to expected terminals
    if hasattr(ec, "tree_to_str_fix_ex"):
        fixed = ec.tree_to_str_fix_ex(tree)
    else:
        fixed = ec.tree_to_str(tree)
    return fixed


def earley_correct_min_penalty(g: Grammar, start_sym: str, broken: str, symbols: List[str] = None, log: bool = False, min_penalty: int = 1, max_penalty: Optional[int] = None) -> Optional[str]:
    """
    Like earley_correct but forces the minimum correction penalty to be >= min_penalty.
    If env LSTAR_RANDOM_MIN_PENALTY is set (e.g., via --random-penalty), choose ONE random penalty
    in [min_penalty..max_penalty] and try only that; otherwise iterate the range.
    Returns first fixed string found, or None if no parse exists.
    """
    if symbols is None:
        symbols = terminals_of_grammar(g)
    cover, start_cov = ec.augment_grammar_ex(g, start_sym, symbols=symbols)
    parser = ec.ErrorCorrectingEarleyParser(cover)

    # Determine bounds
    if max_penalty is None:
        try:
            max_penalty = int(os.getenv("LSTAR_MAX_PENALTY", "32"))
        except Exception:
            max_penalty = 32

    parse_timeout = 600.0

    def try_with_penalty(p: int) -> Optional[str]:
        try:
            # Timeout guard
            import signal
            def _timeout_handler(signum, frame):
                raise TimeoutError("parse timeout")
            old_handler = None
            try:
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
            except Exception:
                old_handler = None
            try:
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, max(0.0, parse_timeout))
                else:
                    secs = int(parse_timeout) if parse_timeout >= 1 else 1
                    signal.alarm(secs)
                se = ec.SimpleExtractorEx(parser, broken, start_cov, penalty=p, log=log)
            finally:
                try:
                    if hasattr(signal, "setitimer"):
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    else:
                        signal.alarm(0)
                except Exception:
                    pass
                try:
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                except Exception:
                    pass
            tree = se.extract_a_tree()
            return ec.tree_to_str_fix_ex(tree) if hasattr(ec, "tree_to_str_fix_ex") else ec.tree_to_str(tree)
        except Exception as e:
            if log:
                try:
                    print(f"[DEBUG] min-penalty probe p={p} failed: {e}")
                except Exception:
                    pass
            return None

    # Random single selection mode
    random_one = os.getenv("LSTAR_RANDOM_MIN_PENALTY", "").lower() in ("1", "true", "yes")
    lo = max(1, int(min_penalty))
    hi = max(1, int(max_penalty))
    if random_one:
        try:
            p = random.randint(lo, hi)
        except Exception:
            import random as _rnd
            p = _rnd.randint(lo, hi)
        if log:
            try:
                print(f"[DEBUG] min-penalty(random-one): p={p} in [{lo}..{hi}]")
            except Exception:
                pass
        return try_with_penalty(p)

    # Default: iterate p = min_penalty..max_penalty
    for p in range(lo, hi + 1):
        res = try_with_penalty(p)
        if res is not None:
            return res
    return None


def enumerate_repairs(g: Grammar, start_sym: str, broken: str, symbols: List[str] = None, log: bool = False, max_penalty: Optional[int] = None, penalties: Optional[List[int]] = None, limit: Optional[int] = None) -> List[str]:
    """
    Enumerate ALL candidate repairs up to max_penalty using EC MultiExtractorEx in one parse.
    Returns a de-duplicated list of candidate fixed strings (order by increasing total penalty).
    """
    # If symbols not provided, infer from grammar terminals
    if symbols is None:
        symbols = terminals_of_grammar(g)
    covering_grammar, covering_start = ec.augment_grammar_ex(g, start_sym, symbols=symbols)
    parser = ec.ErrorCorrectingEarleyParser(covering_grammar)
    # Configure parser pruning threshold: CLI-provided > env > default 32
    if max_penalty is None:
        try:
            max_penalty = int(os.getenv("LSTAR_MAX_PENALTY", "32"))
        except Exception:
            max_penalty = 32
    # Single parse_prefix; enumerate all trees across penalties and forest choices
    mx = ec.MultiExtractorEx(parser, broken, covering_start, penalties=penalties, log=log)
    out: List[str] = []
    seen = set()
    count = 0
    for ntree in mx.trees(limit=limit):
        if hasattr(ec, "tree_to_str_fix_ex"):
            fixed = ec.tree_to_str_fix_ex(ntree)
        else:
            fixed = ec.tree_to_str(ntree)
        if fixed not in seen:
            seen.add(fixed)
            out.append(fixed)
            count += 1
            if limit is not None and count >= limit:
                break
    return out


def validate_with_match(category: str, text: str, validator_cmd: Optional[List[str]] = None) -> bool:
    """
    Validate 'text' using validators/regex/* oracle runners when available, otherwise fallback to match.py.
    Returns True on success (exit code 0).
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
        tf.write(text)
        temp_path = tf.name
    try:
        # Map category to validator basename
        name_map = {
            "Date": "date",
            "Time": "time",
            "URL": "url",
            "ISBN": "isbn",
            "IPv4": "ipv4",
            "IPv6": "ipv6",
            "FilePath": "pathfile",
        }
        base = name_map.get(category, category.lower())

        cmd = None
        if validator_cmd:
            cmd = list(validator_cmd) + [temp_path]
        else:
            candidates = [
                os.path.join("validators", "regex", f"validate_{base}"),
                os.path.join("validators", f"validate_{base}"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    cmd = [c, temp_path]
                    break
            if cmd is None:
                # Fallback to Python validator
                cmd = ["python3", "match.py", category, temp_path]

        # Show input preview and command
        try:
            preview = text if len(text) <= 200 else (text[:200] + "...(truncated)")
        except Exception:
            preview = "<unprintable>"
        print(f"[DEBUG] Oracle in: {repr(preview)} (len={len(text)})")
        print(f"[DEBUG] Oracle cmd: {' '.join(cmd)}")

        # Run oracle with timeout and show outputs; treat hang as reject
        try:
            try:
                oracle_timeout = float(os.getenv("LSTAR_ORACLE_TIMEOUT", "3.0"))
            except Exception:
                oracle_timeout = 3.0
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(0.001, oracle_timeout),
            )
            out = (res.stdout or "").strip()
            err = (res.stderr or "").strip()
            if out:
                print(f"[DEBUG] Oracle out: {out}")
            if err:
                print(f"[DEBUG] Oracle err: {err}")
            print(f"[DEBUG] Oracle rc: {res.returncode}")
            verdict = (res.returncode == 0)
            print(f"[DEBUG] Oracle verdict: {'OK' if verdict else 'FAIL'}")
            return verdict
        except subprocess.TimeoutExpired:
            try:
                print(f"[WARN] Oracle timed out after {oracle_timeout:.2f}s; treating as REJECT and skipping.")
            except Exception:
                print("[WARN] Oracle timed out; treating as REJECT and skipping.")
            return False
        except Exception as e:
            print(f"[WARN] Oracle execution error: {e}; treating as REJECT.")
            return False
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Repair erroneous inputs using RPNI-inferred DFA grammar + Error-Correcting Earley")
    ap.add_argument("--positives", help="Path to positives.txt (one string per line; empty line is epsilon). Optional if --grammar-cache exists.")
    ap.add_argument("--negatives", help="Path to negatives.txt (initial negative set; optional)")
    ap.add_argument("--broken-file", help="Path to a file containing a single broken input; the entire file content is used")
    ap.add_argument("--broken", help="Single broken input string to repair (optional)")
    ap.add_argument("--output-file", help="If given and exactly one broken input is processed, write repaired text here")
    ap.add_argument("--grammar-cache", help="Path to cache JSON for learned grammar. If exists (and no --init-cache), it will be loaded; else a new cache will be saved after learning.")
    ap.add_argument("--init-cache", action="store_true", help="Force re-learn from provided pos/neg and overwrite the cache at --grammar-cache.")
    ap.add_argument("--category", required=True, choices=["Date","Time","URL","ISBN","IPv4","IPv6","FilePath"], help="Oracle category for match.py")
    ap.add_argument("--max-attempts", type=int, default=500, help="Max attempts to relearn with added negatives on oracle failure")
    ap.add_argument("--limit", type=int, default=10, help="Limit number of negatives to process (for quick runs)")
    ap.add_argument("--unknown-policy", default="negative", choices=["negative","positive","error"], help="Unknown membership policy for SampleTeacher")
    ap.add_argument("--log", action="store_true", help="Verbose logs for ErrorCorrectingEarley")
    ap.add_argument("--penalty", type=int, help="Target correction penalty to select (capped at 8). Omit to choose minimum-penalty solution.")
    ap.add_argument("--max-penalty", type=int, default=8, help="Max correction penalty allowed during parsing (higher tolerates longer junk). Overrides env LSTAR_MAX_PENALTY.")
    ap.add_argument("--update-cache-on-relearn", action="store_true", help="If set, overwrite the grammar cache on relearning attempts. Default keeps the original cache intact.")
    ap.add_argument("--results-json", help="Write per-case repair results to this JSON file")
    ap.add_argument("--learner", default="rpni", choices=["lstar_oracle","rpni","rpni_nfa","rpni_fuzz","rpni_xover"], help="Learning algorithm: 'rpni' (default) uses passive RPNI; 'lstar_oracle' uses L* with validator-backed oracle; 'rpni_nfa' uses modified RPNI that keeps an NFA; 'rpni_fuzz' uses RPNI with fuzzing-based DFA consistency checks; 'rpni_xover' uses RPNI with cross-over consistency checks based on positives")
    ap.add_argument("--oracle-validator", help="Path or command for oracle validator; overrides default search under validators/regex or validators")
    # Equivalence/speed knobs (allow approximate acceptance to reduce oracle queries)
    ap.add_argument("--eq-max-length", type=int, default=10, help="Max length to sample in equivalence (default: 10)")
    ap.add_argument("--eq-samples-per-length", type=int, default=50, help="Number of samples per length in equivalence (default: 50)")
    ap.add_argument("--eq-disable-sampling", action="store_true", help="Disable equivalence sampling (accept hypothesis after pos/neg checks)")
    ap.add_argument("--eq-skip-negatives", action="store_true", help="Skip checking negatives in equivalence (fewer grammar parses, faster)")
    ap.add_argument("--eq-max-oracle", type=int, help="Max oracle calls allowed in equivalence sampling per run; accept hypothesis when exhausted")
    # EC enumeration controls
    ap.add_argument("--ec-enumerate", action="store_true", help="Enumerate ALL repair candidates up to max-penalty in a single EC run")
    ap.add_argument("--ec-limit", type=int, help="Optional cap on number of candidates enumerated per input to avoid explosion")
    ap.add_argument("--accumulate-negatives-round", action="store_true", help="Accumulate failing broken inputs across a full pass, then relearn once (batch rounds)")
    ap.add_argument("--mutations", type=int, default=20, help="Number of mutated samples to generate from positives using only existing characters")
    ap.add_argument("--mutations-random", action="store_true", help="Generate mutations randomly (instead of deterministic enumeration)")
    ap.add_argument("--mutations-deterministic", action="store_true", help="Force deterministic mutation enumeration (overrides --mutations-random)")
    ap.add_argument("--mutations-seed", type=int, help="Random seed for mutation generation (optional)")
    ap.add_argument("--random-penalty", action="store_true", help="Randomly choose a penalty in [1..--max-penalty] for EC; falls back to min-penalty on invalid")
    args = ap.parse_args()

    # Prepare optional oracle validator command override
    validator_cmd: Optional[List[str]] = None
    if getattr(args, "oracle_validator", None):
        try:
            validator_cmd = shlex.split(args.oracle_validator)
        except Exception:
            validator_cmd = [args.oracle_validator]

    # Membership callback for fuzzing-based learners (rpni_fuzz)
    def _is_member_oracle(text: str) -> bool:
        return validate_with_match(args.category, text, validator_cmd)

    # Normalize/cap penalty
    penalty_val = None
    if getattr(args, "penalty", None) is not None:
        p = max(0, int(args.penalty))
        if p > 8:
            if args.log:
                print(f"[WARN] --penalty {p} exceeds max of 8; capping to 8.")
            p = 8
        penalty_val = p
    # Optional: random penalty selection if not explicitly provided
    if getattr(args, "random_penalty", False) and penalty_val is None:
        try:
            max_p = int(getattr(args, "max_penalty", 8))
        except Exception:
            max_p = 8
        try:
            penalty_val = random.randint(1, max(1, max_p))
            if args.log:
                print(f"[DEBUG] Randomly selected penalty={penalty_val} in [1..{max_p}]")
        except Exception:
            pass
    # Propagate preference for random penalty selection to min-penalty helper via env
    try:
        if getattr(args, "random_penalty", False):
            os.environ["LSTAR_RANDOM_MIN_PENALTY"] = "1"
    except Exception:
        pass

    pos_lines = read_lines(args.positives) if args.positives and os.path.isfile(args.positives) else []
    neg_lines = read_lines(args.negatives) if args.negatives and os.path.isfile(args.negatives) else []
    broken_inputs: List[str] = []
    # Only one input is allowed: either --broken-file or --broken
    if getattr(args, "broken_file", None) and os.path.isfile(args.broken_file):
        try:
            with open(args.broken_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read --broken-file: {e}")
            return
        # Use entire file content as a single broken input; strip one trailing newline for parity with --broken
        if content.endswith("\n"):
            content = content[:-1]
        broken_inputs.append(content)
    if getattr(args, "broken", None) is not None:
        if broken_inputs:
            print("[ERROR] Provide only one of --broken or --broken-file, not both.")
            return
        broken_inputs.append(args.broken)

    positives = set(pos_lines)

    # Initialize negatives set from provided negatives file (initial hypothesis)
    teacher_negatives: Set[str] = set(neg_lines)

    print(f"[INFO] Loaded positives={len(positives)}, negatives={len(teacher_negatives)}, broken_inputs={len(broken_inputs)}")

    # Optional: generate additional samples via mutation, using ONLY characters from positives
    try:
        mut_n = int(getattr(args, "mutations", 0) or 0)
    except Exception:
        mut_n = 0
    if mut_n > 0 and positives:
        # derive alphabet strictly from positives to avoid introducing new characters
        alph_pos = derive_alphabet_from_examples(positives, set())
        # choose random or deterministic mutation generator
        use_deterministic = bool(getattr(args, "mutations_deterministic", False))
        use_random = bool(getattr(args, "mutations_random", False)) or not use_deterministic
        if use_random:
            muts = generate_mutations_random(positives, mut_n, alph_pos, seed=getattr(args, "mutations_seed", None))
        else:
            muts = generate_mutations(positives, mut_n, alph_pos)
        print(f"[INFO] Generated {len(muts)} mutation(s) from positives (requested {mut_n}). Classifying with oracle ...")
        acc = 0
        rej = 0
        for s in muts:
            ok = validate_with_match(args.category, s, validator_cmd)
            if ok:
                # Add accepted mutants into positives
                positives.add(s)
                acc += 1
            else:
                # Add rejected mutants into negatives
                teacher_negatives.add(s)
                rej += 1
        print(f"[INFO] Mutation classification: accepted={acc}, rejected={rej}. Totals now P={len(positives)}, N={len(teacher_negatives)}")
    # Handle grammar cache: load if available (and not init), otherwise learn and optionally save
    g: Grammar
    start_sym: str
    alphabet: List[str]
    cache_path = args.grammar_cache

    if cache_path and os.path.exists(cache_path) and not args.init_cache:
        print(f"[INFO] Loading grammar cache from {cache_path}")
        g, start_sym, alphabet = load_grammar_cache(cache_path)
        # Basic sanity: ensure strings-only grammar
        assert_no_set_tokens(g)
        try:
            print(f"[INFO] Cache stats: nonterminals={len(g)}, productions={sum(len(v) for v in g.values())}, alphabet={len(alphabet)}, size={os.path.getsize(cache_path)} bytes")
        except Exception:
            pass
    else:
        if not positives and cache_path and os.path.exists(cache_path) and args.init_cache:
            print("[ERROR] --init-cache specified but no positives provided to relearn.")
            return
        if not positives and not cache_path:
            print("[ERROR] No positives provided and no grammar cache to load.")
            return
        print(f"[INFO] Learning initial grammar with provided samples ...")
        t_learn0 = time.time()
        if args.learner == "rpni":
            g_raw, start_sym, alphabet = learn_grammar(positives, teacher_negatives, unknown_policy=args.unknown_policy)
        elif args.learner == "rpni_nfa":
            g_raw, start_sym, alphabet = learn_grammar_nfa(positives, teacher_negatives, unknown_policy=args.unknown_policy)
        elif args.learner == "rpni_fuzz":
            g_raw, start_sym, alphabet = rpni_fuzz_learn_grammar(positives, teacher_negatives, is_member=_is_member_oracle)
        elif args.learner == "rpni_xover":
            g_raw, start_sym, alphabet = rpni_xover_learn_grammar(positives, teacher_negatives, is_member=_is_member_oracle)
        else:
            g_raw, start_sym, alphabet = lstar_learn_with_oracle(
                positives,
                teacher_negatives,
                args.category,
                validator_cmd,
                eq_max_length=int(getattr(args, "eq_max_length", 10)),
                eq_samples_per_length=int(getattr(args, "eq_samples_per_length", 50)),
                eq_disable_sampling=bool(getattr(args, "eq_disable_sampling", False)),
                check_negatives=not bool(getattr(args, "eq_skip_negatives", False)),
                eq_budget=getattr(args, "eq_max_oracle", None),
            )
        t_learn1 = time.time()
        t_prep0 = time.time()
        # Sanitize to make it JSON-serializable and friendly for the parser
        g = sanitize_grammar(expand_set_terminals(g_raw, alphabet))
        assert_no_set_tokens(g)
        t_prep1 = time.time()
        print(f"[PROFILE] learn_grammar(total): {t_learn1 - t_learn0:.2f}s; sanitize+expand: {t_prep1 - t_prep0:.2f}s")
        print(f"[INFO] Learned start symbol: {start_sym}; Nonterminals: {len(g)}; Alphabet(chars): {len(alphabet)}")
        if cache_path:
            try:
                save_grammar_cache(cache_path, g, start_sym, alphabet)
                print(f"[INFO] Saved grammar cache to {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to save grammar cache to {cache_path}: {e}")

    if not broken_inputs:
        print("[INFO] No broken inputs provided. Exiting after grammar learning/caching.")
        return

    processed = 0
    successes = 0
    failures = 0
    results: List[Dict[str, Any]] = []

    for broken in broken_inputs:
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1
        print(f"\n[CASE {processed}] Broken: {repr(broken)}")
        last_fixed: Optional[str] = None
        t0 = time.time()
        if any(isinstance(t, (set, frozenset)) for alts in g.values() for alt in alts for t in alt):
            g_norm = sanitize_grammar(expand_set_terminals(g, alphabet))
        else:
            g_norm = g
        assert_no_set_tokens(g_norm)
        t1 = time.time()
        print(f"[PROFILE] normalize: {t1 - t0:.2f}s")

        debug_count_symbol_types(g_norm)

        # If the broken input is already accepted by the learned grammar but the oracle rejects it,
        # treat it as a counterexample: add to negatives and trigger relearning path
        try:
            parser0 = earleyparser.EarleyParser(g_norm)
            list(parser0.recognize_on(broken, start_sym))
            # Grammar accepts; consult oracle
            t_or0 = time.time()
            ok0 = validate_with_match(args.category, broken, validator_cmd)
            t_or1 = time.time()
            print(f"[PROFILE] oracle_validate(as-is): {t_or1 - t_or0:.2f}s")
            if not ok0:
                print("[INFO] As-is input is accepted by grammar but rejected by oracle; scheduling relearn.")
                last_fixed = None
                raise RuntimeError("oracle-rejects-accepted")
        except Exception as _e:
            # If grammar doesn't accept or membership check raises, carry on to EC
            pass

        # Default path: either single best (earley_correct) or enumerate all candidates (MultiExtractorEx)
        broken_parse = broken
        t2 = time.time()
        # If random penalty requested (and not explicitly set), pick a random penalty per run
        penalty_arg = penalty_val
        if getattr(args, "random_penalty", False) and getattr(args, "penalty", None) is None:
            try:
                max_p = int(getattr(args, "max_penalty", 8))
            except Exception:
                max_p = 8
            try:
                penalty_arg = random.randint(1, max(1, max_p))
                if args.log:
                    print(f"[DEBUG] Randomly selected penalty={penalty_arg} in [1..{max_p}]")
            except Exception:
                penalty_arg = penalty_val
        fixed = earley_correct(g_norm, start_sym, broken_parse, symbols=alphabet, log=args.log, penalty=penalty_arg, max_penalty=int(getattr(args, "max_penalty", 8)))
        t3 = time.time()
        print(f"[PROFILE] ec_earley: {t3 - t2:.2f}s")

        t4 = time.time()
        ok = validate_with_match(args.category, fixed, validator_cmd)
        final_ok = ok
        t5 = time.time()
        print(f"[PROFILE] oracle_validate: {t5 - t4:.2f}s")

        print(f"[ATTEMPT 0] Fixed: {repr(fixed)} | Oracle: {'OK' if ok else 'FAIL'}")
        last_fixed = fixed
        # If an output file is requested (bm_xxx integration), write the repaired text
        if getattr(args, "output_file", None):
            try:
                with open(args.output_file, "w", encoding="utf-8") as outf:
                    outf.write(fixed)
            except Exception:
                pass
        if ok:
            successes += 1
            results.append({"broken": broken, "fixed": last_fixed, "ok": bool(final_ok)})
            continue

        # If oracle failed, add this broken example and the best failed candidate (if any) to Teacher.negatives and relearn up to max-attempts
        attempt = 1
        cur_ok = ok
        while attempt <= args.max_attempts and not cur_ok:
            if last_fixed:
                teacher_negatives.add(last_fixed)
            teacher_negatives.add(broken)
            print(f"[INFO] Re-learning with {len(teacher_negatives)} negative(s) (attempt {attempt}/{args.max_attempts}) ...")
            try:
                t_learn0 = time.time()
                if args.learner == "rpni":
                    g_raw, start_sym, alphabet = learn_grammar(positives, teacher_negatives, unknown_policy=args.unknown_policy)
                elif args.learner == "rpni_nfa":
                    g_raw, start_sym, alphabet = learn_grammar_nfa(positives, teacher_negatives, unknown_policy=args.unknown_policy)
                elif args.learner == "rpni_fuzz":
                    g_raw, start_sym, alphabet = rpni_fuzz_learn_grammar(positives, teacher_negatives, is_member=_is_member_oracle)
                elif args.learner == "rpni_xover":
                    g_raw, start_sym, alphabet = rpni_xover_learn_grammar(positives, teacher_negatives, is_member=_is_member_oracle)
                else:
                    g_raw, start_sym, alphabet = lstar_learn_with_oracle(
                        positives,
                        teacher_negatives,
                        args.category,
                        validator_cmd,
                        eq_max_length=int(getattr(args, "eq_max_length", 10)),
                        eq_samples_per_length=int(getattr(args, "eq_samples_per_length", 50)),
                        eq_disable_sampling=bool(getattr(args, "eq_disable_sampling", False)),
                        check_negatives=not bool(getattr(args, "eq_skip_negatives", False)),
                        eq_budget=getattr(args, "eq_max_oracle", None),
                    )
                t_learn1 = time.time()
                t_prep0 = time.time()
                g = sanitize_grammar(expand_set_terminals(g_raw, alphabet))
                t_prep1 = time.time()
                print(f"[PROFILE] learn_grammar(relearn): {t_learn1 - t_learn0:.2f}s; sanitize+expand(relearn): {t_prep1 - t_prep0:.2f}s")
                # If cache provided, refresh it when relearning only when explicitly requested
                if cache_path and getattr(args, "update_cache_on_relearn", False):
                    try:
                        save_grammar_cache(cache_path, g, start_sym, alphabet)
                        print(f"[INFO] Refreshed grammar cache at {cache_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to refresh grammar cache at {cache_path}: {e}")
            except Exception as e:
                print(f"[ERROR] RPNI learning failed on attempt {attempt}: {e}")
                break

            try:
                t0 = time.time()
                if any(isinstance(t, (set, frozenset)) for alts in g.values() for alt in alts for t in alt):
                    g_norm = sanitize_grammar(expand_set_terminals(g, alphabet))
                else:
                    g_norm = g
                assert_no_set_tokens(g_norm)
                t1 = time.time()
                print(f"[PROFILE] normalize(relearn): {t1 - t0:.2f}s")

                debug_count_symbol_types(g_norm)

                # If the broken input is already accepted by the learned grammar but the oracle rejects it,
                # treat it as a counterexample and continue to next relearn iteration
                try:
                    parser0 = earleyparser.EarleyParser(g_norm)
                    list(parser0.recognize_on(broken, start_sym))
                    t_or0 = time.time()
                    ok0 = validate_with_match(args.category, broken, validator_cmd)
                    t_or1 = time.time()
                    print(f"[PROFILE] oracle_validate(as-is,relearn): {t_or1 - t_or0:.2f}s")
                    if not ok0:
                        print(f"[INFO] [ATTEMPT {attempt}] As-is input accepted by grammar but rejected by oracle; will relearn again.")
                        last_fixed = None
                        raise RuntimeError("oracle-rejects-accepted")
                except Exception as _e:
                    pass

                broken_parse = broken
                t2 = time.time()
                # Random penalty per attempt if requested (and no explicit --penalty)
                penalty_arg = penalty_val
                if getattr(args, "random_penalty", False) and getattr(args, "penalty", None) is None:
                    try:
                        max_p = int(getattr(args, "max_penalty", 8))
                    except Exception:
                        max_p = 8
                    try:
                        penalty_arg = random.randint(1, max(1, max_p))
                        if args.log:
                            print(f"[DEBUG] [ATTEMPT {attempt}] Randomly selected penalty={penalty_arg} in [1..{max_p}]")
                    except Exception:
                        penalty_arg = penalty_val
                fixed = earley_correct(g_norm, start_sym, broken_parse, symbols=alphabet, log=args.log, penalty=penalty_arg, max_penalty=int(getattr(args, "max_penalty", 8)))
                t3 = time.time()
                print(f"[PROFILE] ec_earley(relearn): {t3 - t2:.2f}s")

                t4 = time.time()
                cur_ok = validate_with_match(args.category, fixed, validator_cmd)
                final_ok = cur_ok
                t5 = time.time()
                print(f"[PROFILE] oracle_validate(relearn): {t5 - t4:.2f}s")

                print(f"[ATTEMPT {attempt}] Fixed: {repr(fixed)} | Oracle: {'OK' if cur_ok else 'FAIL'}")
                last_fixed = fixed
                # Update output file with the latest repaired text if requested
                if getattr(args, "output_file", None):
                    try:
                        with open(args.output_file, "w", encoding="utf-8") as outf:
                            outf.write(fixed)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[ATTEMPT {attempt}] Error during correction: {e}")
                print(traceback.format_exc())
                cur_ok = False

            attempt += 1

        if cur_ok:
            successes += 1
        else:
            failures += 1
        try:
            results.append({"broken": broken, "fixed": last_fixed, "ok": bool(final_ok)})
        except Exception:
            pass

    print(f"\n[SUMMARY] Processed={processed}, Successes={successes}, Failures={failures}")
    # Optional: write results JSON for batch runs (useful when stdout isn't captured)
    if getattr(args, "results_json", None):
        try:
            with open(args.results_json, "w", encoding="utf-8") as jf:
                json.dump({"results": results}, jf, ensure_ascii=False, indent=2)
            print(f"[INFO] Wrote results JSON to {args.results_json}")
        except Exception as e:
            print(f"[WARN] Failed to write results JSON: {e}")


if __name__ == "__main__":
    main()
