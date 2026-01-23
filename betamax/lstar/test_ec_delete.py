#!/usr/bin/env python3
import os
import sys
import glob

# Ensure project root (betamax) on sys.path so vendored wheels and lstar import work
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PY_DIR = os.path.join(ROOT_DIR, "py")
if os.path.isdir(PY_DIR):
    if PY_DIR not in sys.path:
        sys.path.insert(0, PY_DIR)
    for whl in glob.glob(os.path.join(PY_DIR, "*.whl")):
        if whl not in sys.path:
            sys.path.append(whl)

from lstar import ec_runtime as ec

# Regression: parse_prefix must tolerate scan-time mutation of '$.' / '!x'
def test_parse_prefix_extended_terminal_mutation() -> bool:
    g = {"<S>": [[ec.Any_term]]}
    parser = ec.ErrorCorrectingEarleyParser(g)
    cursor, states = parser.parse_prefix("a", "<S>")
    ok = (cursor == 1) and any(getattr(s, "finished", lambda: False)() for s in states)
    if not ok:
        print(f"[FAIL] parse_prefix mutation: cursor={cursor}, states={[(s.expr, s.dot) for s in states]}")
    return ok

# Construct a tiny grammar: S -> A B ; A -> 'a' ; B -> 'b'
g = {
    "<S>": [["<A>", "<B>"]],
    "<A>": [["a"]],
    "<B>": [["b"]],
}
start = "<S>"
alphabet = ["a", "b"]

# Build covering grammar with our new <$del[b]> branch available
cover, start_cov = ec.augment_grammar_ex(g, start, symbols=alphabet)
parser = ec.ErrorCorrectingEarleyParser(cover)

def run_case(inp: str, expected_fixed: str = "ab") -> bool:
    print("\nCASE input:", repr(inp))
    try:
        se = ec.SimpleExtractorEx(parser, inp, start_cov, log=True)
        tree = se.extract_a_tree()
        fixed = ec.tree_to_str_fix_ex(tree)
        print("FIXED:", repr(fixed))
        if fixed != expected_fixed:
            print(f"[FAIL] Expected {expected_fixed!r}, got {fixed!r}")
            return False
        return True
    except Exception as e:
        print("[FAIL] ERROR:", e)
        return False

# 1) Missing 'b': should be able to delete grammar position 'b' producing 'a'
ok = True
ok &= test_parse_prefix_extended_terminal_mutation()
ok &= run_case("a")

# 2) Exact match 'ab': fixed stays 'ab'
ok &= run_case("ab")

# 3) Extra junk 'abx': Any_plus should drop 'x' and keep 'ab'
ok &= run_case("abx")

# 4) Wrong char at B position: 'ax' -> replace x->b or delete grammar 'b' depending on cost; projection should end as 'ab' or 'a'
ok &= run_case("ax")

sys.exit(0 if ok else 1)
