#!/usr/bin/env python3
"""
Lightweight runtime for Error-Correcting Earley utilities extracted from
2021-02-22-error-correcting-earley-parser.py without top-level side effects.

Exposes:
- is_nt, format_parsetree, rem_terminals, tree_to_str (from earleyparser)
- Constants: Any_one, Any_plus, Empty, Any_term, Any_not_term,
             This_sym_str, Any_not_str
- Functions: This_sym, Any_not, translate_terminals, add_start, augment_grammar_ex,
             nullable_ex
- Classes: ECState, ECColumn, ErrorCorrectingEarleyParser, SimpleExtractor, SimpleExtractorEx
"""

from typing import List, Dict, Tuple, Any
import random
import os
import time

import earleyparser

# Re-export helpers from earleyparser
is_nt = earleyparser.is_nt
format_parsetree = earleyparser.format_parsetree
rem_terminals = earleyparser.rem_terminals
tree_to_str = earleyparser.tree_to_str

# Symbols and helper nonterminals/terminals
This_sym_str = '<$[%s]>'
def This_sym(t: str) -> str:
    return This_sym_str % t

Any_one = '<$.>'
Any_plus = '<$.+>'
Star_one = '<$*>'
Star_plus = '<$*+>'
Empty = '<$>'

Any_not_str = '<$![%s]>'
def Any_not(t: str) -> str:
    return Any_not_str % t

# Optimized terminal symbols used by augment_grammar_ex
Any_term = '$.'
Any_not_term = '!%s'
Del_sym_str = '<$del[%s]>'
def Del_sym(t: str) -> str:
    return Del_sym_str % t

def translate_terminal(t) -> str:
    # Coerce to string to avoid passing non-strings (e.g., set) into is_nt
    if not isinstance(t, str):
        t = str(t)
    if is_nt(t):
        return t
    return This_sym(t)

def translate_terminals(g: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
    return {k: [[translate_terminal(t) for t in alt] for alt in g[k]] for k in g}

def corrupt_start(old_start: str) -> str:
    return '<@# %s>' % old_start[1:-1]

def add_start(old_start: str) -> Tuple[Dict[str, List[List[str]]], str]:
    """
    <$corrupt_start> -> <start>
    """
    g_: Dict[str, List[List[str]]] = {}
    c_start = corrupt_start(old_start)
    # Also allow consuming trailing junk in the broken input (via Any_plus -> '$.'+).
    # This keeps EC parsing total (so parse_prefix can reach len(text)) while
    # SimpleExtractorEx prefers the length-preserving alternative when possible.
    g_[c_start] = [[old_start], [old_start, Any_plus]]
    return g_, c_start

def augment_grammar_ex(g: Dict[str, List[List[str]]], start: str, symbols: List[str] = None) -> Tuple[Dict[str, List[List[str]]], str]:
    """
    Build covering grammar that allows minimal error-correction.
    This version uses special terminals '$.' (any) and '!a' (any except 'a')
    to avoid T^2/T^3 grammar blow-up.
    """
    if symbols is None:
        # Safely derive terminal alphabet by coercing tokens to strings before is_nt
        syms = set()
        for k in g:
            for alt in g[k]:
                for t in alt:
                    t_str = t if isinstance(t, str) else str(t)
                    if not is_nt(t_str):
                        syms.add(t_str)
        symbols = sorted(syms)

    # Match any single symbol
    Match_any_sym = {Any_one: [[Any_term]]}
    # Kleene-plus over Any_one
    Match_any_sym_plus = {Any_plus: [[Any_one], [Any_plus, Any_one]]}
    # Match '*' literal (single)
    Match_star_sym = {Star_one: [['*']]}

    # Match any symbol except given terminal
    Match_any_sym_except: Dict[str, List[List[str]]] = {}
    for kk in symbols:
        Match_any_sym_except[Any_not(kk)] = [[Any_not_term % kk]]

    # Empty
    Match_empty = {Empty: [[]]}

    # Match delete: delete this grammar position (epsilon)
    Match_del_sym: Dict[str, List[List[str]]] = {}
    for kk in symbols:
        Match_del_sym[Del_sym(kk)] = [[]]

    # For each terminal 'kk' in original grammar
    # <$ [kk]> -> kk | (optionally <$*> kk for delimiters/digits) | <$![kk]> | <$>
    Match_a_sym: Dict[str, List[List[str]]] = {}
    for kk in symbols:
        Match_a_sym[This_sym(kk)] = [
            [kk],
            [Star_one, kk],
            [Any_one, kk],
            [Any_not(kk)],
            [Empty],
        ]

    start_g, start_s = add_start(start)
    covering = {
        **start_g,
        **g,
        **translate_terminals(g),
        **Match_any_sym,
        **Match_any_sym_plus,
        **Match_star_sym,
        **Match_a_sym,
        **Match_any_sym_except,
        **Match_del_sym,
        **Match_empty,
    }
    return covering, start_s

def nullable_ex(g: Dict[str, List[List[str]]]) -> Dict[str, int]:
    """
    Compute nullable nonterminals and their penalties.
    Penalty 1 for Empty, Any_one, Any_not(...) flows through.
    """
    nullable_keys = {}
    for k in g:
        if [] in g[k]:
            if k == Empty:
                nullable_keys[k] = 1
            elif isinstance(k, str) and k.startswith('<$del['):
                nullable_keys[k] = 2
            else:
                nullable_keys[k] = 0
    unprocessed = list(nullable_keys.keys())

    g_cur_ = rem_terminals(g)
    g_cur: Dict[str, List[Tuple[List[str], int]]] = {k: [(alt, 0) for alt in g_cur_[k]] for k in g_cur_}
    while unprocessed:
        nxt, *unprocessed = unprocessed
        g_nxt: Dict[str, List[Tuple[List[str], int]]] = {}
        for k in g_cur:
            if k in nullable_keys:
                continue
            g_alts: List[Tuple[List[str], int]] = []
            for alt, penalty in g_cur[k]:
                penalty_ = len([t for t in alt if t == nxt]) * nullable_keys[nxt]
                alt_ = [t for t in alt if t != nxt]
                if not alt_:
                    nullable_keys[k] = penalty + penalty_
                    unprocessed.append(k)
                    break
                else:
                    g_alts.append((alt_, penalty + penalty_))
            if g_alts:
                g_nxt[k] = g_alts
        g_cur = g_nxt
    return nullable_keys

class ErrorCorrectingEarleyParser(earleyparser.EarleyParser):
    def __init__(self, grammar: Dict[str, List[List[str]]], log: bool = False, **kwargs):
        self._grammar = grammar
        self.log = log
        # Initialize base parser first (it may assign its own epsilon)
        super().__init__(grammar, **kwargs)
        # Now override with our penalty-aware nullable map
        self.epsilon = nullable_ex(grammar)
        if isinstance(self.epsilon, set):
            self.epsilon = {k: 1 for k in self.epsilon}

    # complete: propagate penalty from completed child to parent
    def complete(self, col, state):
        parent_states = [st for st in state.s_col.states if st.at_dot() == state.name]
        for st in parent_states:
            s = st.advance()
            s.penalty += state.penalty
            col.add(s)

    # predict: include nullable transitions with penalty
    def predict(self, col, sym, state):
        for alt in self._grammar[sym]:
            col.add(self.create_state(sym, tuple(alt), 0, col))
        # Coerce symbol to string for lookup safety
        sy = sym if isinstance(sym, str) else str(sym)
        if sy in self.epsilon:
            s = state.advance()
            s.penalty += self.epsilon.get(sy, 0)
            col.add(s)

    def match_terminal(self, rex: str, input_term: str) -> bool:
        # Extended terminals
        if isinstance(rex, str) and len(rex) > 1:
            if rex == Any_term:
                return True
            if rex[0] == Any_not_term[0]:
                return rex[1] != input_term  # Any-not
            return False
        # Normal terminals: single-char string
        return rex == input_term

    def scan(self, col, state, letter):
        # Note: base Earley expects to check match against the grammar terminal at the dot vs current input column letter
        cur = state.expr[state.dot]
        if self.match_terminal(cur, col.letter):
            my_expr = list(state.expr)
            if cur == Any_term:
                my_expr[state.dot] = col.letter
            elif isinstance(cur, str) and cur.startswith('!') and len(cur) > 1:
                my_expr[state.dot] = col.letter
            else:
                assert cur == col.letter
            s = state.advance()
            s.expr = tuple(my_expr)
            col.add(s)

class ECState(earleyparser.State):
    def __init__(self, name, expr, dot, s_col, e_col=None):
        self.name, self.expr, self.dot = name, expr, dot
        self.s_col, self.e_col = s_col, e_col
        if self.name == Empty:
            self.penalty = 1
        elif self.name == Any_one:
            self.penalty = 1
        elif isinstance(self.name, str) and self.name.startswith(Any_not_str[0:4]):  # '<$![...]>'
            self.penalty = 1
        elif isinstance(self.name, str) and self.name.startswith('<$del['):
            self.penalty = 2
        else:
            self.penalty = 0

    def copy(self):
        s = ECState(self.name, self.expr, self.dot, self.s_col, self.e_col)
        s.penalty = self.penalty
        return s

    def advance(self):
        s = ECState(self.name, self.expr, self.dot + 1, self.s_col, self.e_col)
        s.penalty = self.penalty
        return s

class ECColumn(earleyparser.Column):
    def add(self, state):
        if state in self._unique:
            if self._unique[state].penalty > state.penalty:
                self._unique[state] = state
                self.states.append(state)
                state.e_col = self
            return self._unique[state]
        self._unique[state] = state
        self.states.append(state)
        state.e_col = self
        return self._unique[state]

# Hook custom column/state into parser
class ErrorCorrectingEarleyParser(ErrorCorrectingEarleyParser):
    def create_column(self, i, tok):
        col = ECColumn(i, tok)
        return col

    def create_state(self, sym, alt, num, col):
        return ECState(sym, alt, num, col)

    def parse_prefix(self, text, start_symbol):
        """
        Like earleyparser.EarleyParser.parse_prefix(), but tolerant of our scan-time
        mutation of extended terminals (e.g., '$.' / '!x') into concrete symbols.
        """
        alts = [tuple(alt) for alt in self._grammar[start_symbol]]
        self.table = self.chart_parse(text, start_symbol, alts)

        def _compatible(expr, alt):
            if len(expr) != len(alt):
                return False
            for alt_sym, expr_sym in zip(alt, expr):
                if alt_sym == expr_sym:
                    continue
                if alt_sym == Any_term:
                    continue
                if isinstance(alt_sym, str) and len(alt_sym) > 1 and alt_sym[0] == Any_not_term[0]:
                    if len(alt_sym) > 1 and expr_sym != alt_sym[1]:
                        continue
                return False
            return True

        for col in reversed(self.table):
            states = [
                st for st in col.states
                if st.name == start_symbol
                and st.s_col.index == 0
                and any(_compatible(st.expr, alt) for alt in alts)
            ]
            if states:
                return col.index, states
        return -1, []

class SimpleExtractor:
    """
    Build a parse forest for fully parsed input, then allow extraction of a tree.
    """
    def __init__(self, parser: ErrorCorrectingEarleyParser, text: str, start_symbol: str):
        self.parser = parser
        cursor, states = parser.parse_prefix(text, start_symbol)
        starts = [s for s in states if s.finished()]
        if cursor < len(text) or not starts:
            raise SyntaxError("at " + repr(cursor))
        self.my_forest = parser.parse_forest(parser.table, starts)

    def extract_a_node(self, forest_node):
        name, paths = forest_node
        if not paths:
            return ((name, 0, 1), []), (name, [])
        cur_path, i, l = self.choose_path(paths)
        child_nodes = []
        pos_nodes = []
        for s, kind, chart in cur_path:
            f = self.parser.forest(s, kind, chart)
            postree, ntree = self.extract_a_node(f)
            child_nodes.append(ntree)
            pos_nodes.append(postree)
        return ((name, i, l), pos_nodes), (name, child_nodes)

    def choose_path(self, arr):
        l = len(arr)
        i = 0
        return arr[i], i, l

    def extract_a_tree(self):
        pos_tree, parse_tree = self.extract_a_node(self.my_forest)
        return parse_tree

class SimpleExtractorEx(SimpleExtractor):
    """
    Choose lowest-penalty start state and lowest-cost path at each forest choice.
    """
    def __init__(self, parser: ErrorCorrectingEarleyParser, text: str, start_symbol: str, penalty: int = None, log: bool = False):
        self.parser = parser
        self.log = log
        t0 = time.time()
        cursor, states = parser.parse_prefix(text, start_symbol)
        t1 = time.time()
        if self.log:
            try:
                ncols = len(parser.table)
                nstates = sum(len(c.states) for c in parser.table)
                print(f"[PROFILE] parse_prefix: {t1 - t0:.2f}s, cols={ncols}, states={nstates}")
            except Exception:
                print(f"[PROFILE] parse_prefix: {t1 - t0:.2f}s")
        starts = [s for s in states if s.finished()]
        if cursor < len(text) or not starts:
            raise SyntaxError("at " + repr(cursor))
        if self.log:
            for start in starts:
                print(start.expr, "correction length:", start.penalty)
        if penalty is not None:
            my_starts = [s for s in starts if s.penalty == penalty]
        else:
            # Prefer start states that do NOT use trailing Any_plus (i.e., c_start -> start <$.+>)
            # when penalties tie, to encourage length-preserving fixes (e.g., substitutions)
            def _has_trailing_any_plus(st):
                try:
                    expr = getattr(st, "expr", ())
                    return Any_plus in expr
                except Exception:
                    return False
            my_starts = sorted(starts, key=lambda x: (x.penalty, 1 if _has_trailing_any_plus(x) else 0))
        if not my_starts:
            raise Exception("Invalid penalty", penalty)
        if self.log:
            print("Choosing first state with penalty:", my_starts[0].penalty, "out of", len(my_starts))
        t2 = time.time()
        self.my_forest = parser.parse_forest(parser.table, [my_starts[0]])
        t3 = time.time()
        if self.log:
            print(f"[PROFILE] parse_forest: {t3 - t2:.2f}s")

    def choose_path(self, arr):
        res = sorted([(self.cost_of_path(a), a) for a in arr], key=lambda a: a[0])
        cost = res[0][0]
        low_res = [c for c in res if c[0] == cost]
        if self.log:
            print("Choices:<%s> for:" % len(low_res), str(arr[0][0][0]))
        v = random.choice(low_res)
        return v[1], None, None

    def cost_of_path(self, p):
        states = [s for s, kind, chart in p if kind == 'n']
        return sum([s.penalty for s in states])

class MultiExtractorEx:
    """
    Enumerate ALL candidate parse trees (and thus repair strings) for the given text
    in a single EC run.

    Usage:
      parser = ErrorCorrectingEarleyParser(covering_grammar)
      mx = MultiExtractorEx(parser, text, start_symbol, penalties=None, log=False)
      for tree in mx.trees(limit=K):
          ...

    Notes:
    - This enumerates across:
        (a) all finished start states returned by parse_prefix (i.e., different total penalties),
        (b) all path choices in the parse forest for each start state.
    - 'penalties' can restrict which exact total penalties to enumerate.
    - Use betamax/app/betamax.py's projection (tree_to_str_fix_ex, via the lstar backend)
      to map covering grammar trees back to terminal strings.
    """
    def __init__(self, parser: ErrorCorrectingEarleyParser, text: str, start_symbol: str, penalties=None, log: bool = False):
        self.parser = parser
        self.text = text
        self.start_symbol = start_symbol
        self.log = log
        t0 = time.time()
        cursor, states = parser.parse_prefix(text, start_symbol)
        t1 = time.time()
        if self.log:
            try:
                ncols = len(parser.table)
                nstates = sum(len(c.states) for c in parser.table)
                print(f"[PROFILE] parse_prefix: {t1 - t0:.2f}s, cols={ncols}, states={nstates}")
            except Exception:
                print(f"[PROFILE] parse_prefix: {t1 - t0:.2f}s")
        starts = [s for s in states if s.finished()]
        if cursor < len(text) or not starts:
            raise SyntaxError("at " + repr(cursor))
        if penalties is not None:
            try:
                pen_set = set(int(p) for p in penalties)
            except Exception:
                pen_set = set(penalties)
            starts = [s for s in starts if s.penalty in pen_set]
        # Sort by increasing total correction penalty, then by insertion order
        self.starts = sorted(starts, key=lambda x: x.penalty)

    def _enum_ntree(self, forest_node, seen=None):
        """
        Recursively enumerate parse trees (nonterminal-only) with cycle guards.
        'seen' tracks (id(state), kind, id(chart)) visited along the current expansion path
        to avoid infinite recursion in cyclic forests (e.g., due to nullable/recursive rules).
        forest_node is expected to be a tuple: (name, paths), where each path is a list of
        (state, kind, chart) triples to expand.
        """
        if seen is None:
            seen = set()
        name, paths = forest_node
        # If no alternative paths, yield leaf
        if not paths:
            yield (name, [])
            return
        import itertools as I
        for path in paths:
            # Maintain per-branch visited set to prevent cycles
            local_seen = set(seen)
            child_iters = []
            skip = False
            for s, kind, chart in path:
                key = (id(s), kind, id(chart))
                if key in local_seen:
                    # Cycle detected on this branch; skip this path
                    skip = True
                    break
                local_seen.add(key)
                subnode = self.parser.forest(s, kind, chart)
                child_iters.append(self._enum_ntree(subnode, local_seen))
            if skip:
                continue
            # Cartesian product across child expansions
            for prod in I.product(*child_iters) if child_iters else [()]:
                yield (name, list(prod))

    def trees(self, limit: int = None):
        """
        Yield parse trees (nonterminal-only structure) across all finished start states,
        traversing all forest path choices. If 'limit' is provided, stop after emitting
        that many trees.
        """
        count = 0
        for st in self.starts:
            forest = self.parser.parse_forest(self.parser.table, [st])
            for ntree in self._enum_ntree(forest, set()):
                yield ntree
                count += 1
                if limit is not None and count >= limit:
                    return


def tree_to_str_fix_ex(tree) -> str:
    """
    Build the corrected string (projecting covering grammar back to original grammar):
    - For This_sym(kk) nonterminals (like '<$[k]>'), always emit the expected terminal 'kk'
      regardless of which covering alternative matched (Any_plus/Empty/Any_not).
    - Drop Any_plus (junk) and other <$...> machinery from the output.
    - Recurse through other nonterminals.
    """
    out = []

    def visit(node):
        key, children, *rest = node
        # Nonterminal
        if is_nt(key):
            # This_sym format is like '<$[x]>'
            if key.startswith('<$[') and key.endswith(']>'):
                # extract expected symbol between [...]
                i1 = key.find('[')
                i2 = key.rfind(']')
                expected = key[i1 + 1:i2]
                # Always emit expected symbol; treat Empty/substitution paths as insert/replace fixes
                out.append(expected)
                # Do not recurse into children; they encode corrections/junk
                return
            # Skip Any_plus/Empty machinery nodes entirely
            if key in (Any_plus, Empty) or key.startswith('<$!['):
                return
            # Otherwise recurse into children
            for ch in children:
                visit(ch)
        else:
            # Terminals outside This_sym should generally not appear;
            # to be safe we do not emit them to avoid duplications of junk.
            return

    visit(tree)
    return ''.join(out)
