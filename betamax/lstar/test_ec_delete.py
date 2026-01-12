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

def run_case(inp: str):
    print("\nCASE input:", repr(inp))
    try:
        se = ec.SimpleExtractorEx(parser, inp, start_cov, log=True)
        tree = se.extract_a_tree()
        fixed = ec.tree_to_str_fix_ex(tree)
        print("FIXED:", repr(fixed))
    except Exception as e:
        print("ERROR:", e)

# 1) Missing 'b': should be able to delete grammar position 'b' producing 'a'
run_case("a")

# 2) Exact match 'ab': fixed stays 'ab'
run_case("ab")

# 3) Extra junk 'abx': Any_plus should drop 'x' and keep 'ab'
run_case("abx")

# 4) Wrong char at B position: 'ax' -> replace x->b or delete grammar 'b' depending on cost; projection should end as 'ab' or 'a'
run_case("ax")
