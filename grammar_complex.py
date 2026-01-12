#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Tuple, Iterable, Dict, Set, List, Union

# ============================================================
# 1) Token alphabets (small, thesis-friendly)
# ============================================================

# Common tokens
DIGIT = "DIGIT"
HEX = "HEX"
OTHER = "OTHER"
DASH = "DASH"
COLON = "COLON"
DOT = "DOT"
SEP = "SEP"          # '-' or ' ' for ISBN
X = "X"

# URL tokens (coarse, but deterministic and reproducible)
H = "H"; T = "T"; P = "P"; S = "S"
URL_COLON = "URL_COLON"   # ':'
SLASH = "SLASH"           # '/'
W = "W"                   # 'w'
DOMCHAR = "DOMCHAR"       # domain body class (letters/digits/some symbols)
TLDCHAR = "TLDCHAR"       # TLD class
PATHCHAR = "PATHCHAR"     # path/query fragment class
URL_DOT = "URL_DOT"       # '.'

ALPHABETS = {
    "Date":  (DIGIT, DASH, OTHER),
    "Time":  (DIGIT, COLON, OTHER),
    "IPv4":  (DIGIT, DOT, OTHER),
    "IPv6":  (HEX, COLON, OTHER),
    "ISBN":  (DIGIT, SEP, X, OTHER),
    "URL":   (H, T, P, S, URL_COLON, SLASH, URL_DOT, W, DOMCHAR, TLDCHAR, PATHCHAR, OTHER),
}

# ============================================================
# 2) Regular-expression AST over tokens
#    Using derivatives to build DFA.
# ============================================================

class RE:
    def nullable(self) -> bool:
        raise NotImplementedError
    def deriv(self, a: str) -> "RE":
        raise NotImplementedError
    def simplify(self) -> "RE":
        return self

@dataclass(frozen=True)
class Empty(RE):  # matches nothing
    def nullable(self) -> bool: return False
    def deriv(self, a: str) -> RE: return self
    def __repr__(self) -> str: return "∅"

@dataclass(frozen=True)
class Eps(RE):    # matches empty string
    def nullable(self) -> bool: return True
    def deriv(self, a: str) -> RE: return Empty()
    def __repr__(self) -> str: return "ε"

@dataclass(frozen=True)
class Lit(RE):    # single token
    tok: str
    def nullable(self) -> bool: return False
    def deriv(self, a: str) -> RE: return Eps() if a == self.tok else Empty()
    def __repr__(self) -> str: return self.tok

@dataclass(frozen=True)
class CharClass(RE):  # one-of tokens (union of literals)
    toks: FrozenSet[str]
    def nullable(self) -> bool: return False
    def deriv(self, a: str) -> RE: return Eps() if a in self.toks else Empty()
    def __repr__(self) -> str: return f"[{','.join(sorted(self.toks))}]"

@dataclass(frozen=True)
class Alt(RE):
    parts: Tuple[RE, ...]
    def nullable(self) -> bool:
        return any(p.nullable() for p in self.parts)
    def deriv(self, a: str) -> RE:
        return alt(*(p.deriv(a) for p in self.parts)).simplify()
    def simplify(self) -> RE:
        flat: List[RE] = []
        for p in self.parts:
            p = p.simplify()
            if isinstance(p, Alt):
                flat.extend(p.parts)
            elif isinstance(p, Empty):
                continue
            else:
                flat.append(p)
        if not flat: return Empty()
        # deduplicate
        flat = sorted(set(flat), key=repr)
        if len(flat) == 1: return flat[0]
        return Alt(tuple(flat))
    def __repr__(self) -> str:
        return "(" + " | ".join(map(repr, self.parts)) + ")"

@dataclass(frozen=True)
class Concat(RE):
    parts: Tuple[RE, ...]
    def nullable(self) -> bool:
        return all(p.nullable() for p in self.parts)
    def deriv(self, a: str) -> RE:
        # d_a(r1 r2 ... rn) = d_a(r1) r2...rn  OR (if r1 nullable) d_a(r2...) etc.
        out: List[RE] = []
        # iterative derivative over concatenation (standard rule)
        prefix_nullable = True
        for i, p in enumerate(self.parts):
            if not prefix_nullable:
                break
            dp = p.deriv(a)
            suffix = self.parts[i+1:]
            out.append(concat(dp, *suffix))
            prefix_nullable = prefix_nullable and p.nullable()
        return alt(*out).simplify()
    def simplify(self) -> RE:
        flat: List[RE] = []
        for p in self.parts:
            p = p.simplify()
            if isinstance(p, Empty):
                return Empty()
            if isinstance(p, Eps):
                continue
            if isinstance(p, Concat):
                flat.extend(p.parts)
            else:
                flat.append(p)
        if not flat: return Eps()
        if len(flat) == 1: return flat[0]
        return Concat(tuple(flat))
    def __repr__(self) -> str:
        return "".join(map(repr, self.parts))

@dataclass(frozen=True)
class Star(RE):
    inner: RE
    def nullable(self) -> bool: return True
    def deriv(self, a: str) -> RE:
        return concat(self.inner.deriv(a), self).simplify()
    def simplify(self) -> RE:
        inner = self.inner.simplify()
        if isinstance(inner, Empty) or isinstance(inner, Eps):
            return Eps()
        if isinstance(inner, Star):
            return inner
        return Star(inner)
    def __repr__(self) -> str:
        return f"({repr(self.inner)})*"

def alt(*xs: RE) -> RE:
    return Alt(tuple(xs)).simplify()

def concat(*xs: RE) -> RE:
    return Concat(tuple(xs)).simplify()

def star(x: RE) -> RE:
    return Star(x).simplify()

def opt(x: RE) -> RE:
    return alt(Eps(), x)

def rep_exact(x: RE, n: int) -> RE:
    if n == 0: return Eps()
    return concat(*([x] * n))

def rep_range(x: RE, lo: int, hi: int) -> RE:
    # union_{k=lo..hi} x^k
    return alt(*[rep_exact(x, k) for k in range(lo, hi + 1)])

# ============================================================
# 3) Build token-level regex ASTs for your 6 regexes
# ============================================================

RAW_REGEX = {
    "Date": r"^\d{4}-\d{2}-\d{2}$",
    "Time": r"^\d{2}:\d{2}:\d{2}$",
    "URL":  r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
    "ISBN": r"^(?:\d[- ]?){9}[\dX]$",
    "IPv4": r"^(\d{1,3}\.){3}\d{1,3}$",
    "IPv6": r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$",
}

def build_date() -> RE:
    return concat(
        rep_exact(Lit(DIGIT), 4),
        Lit(DASH),
        rep_exact(Lit(DIGIT), 2),
        Lit(DASH),
        rep_exact(Lit(DIGIT), 2),
    )

def build_time() -> RE:
    return concat(
        rep_exact(Lit(DIGIT), 2),
        Lit(COLON),
        rep_exact(Lit(DIGIT), 2),
        Lit(COLON),
        rep_exact(Lit(DIGIT), 2),
    )

def build_ipv4() -> RE:
    # (\d{1,3}\.){3}\d{1,3}
    block = concat(rep_range(Lit(DIGIT), 1, 3), Lit(DOT))
    return concat(rep_exact(block, 3), rep_range(Lit(DIGIT), 1, 3))

def build_ipv6() -> RE:
    # ([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}
    group = concat(rep_range(Lit(HEX), 1, 4), Lit(COLON))
    return concat(rep_exact(group, 7), rep_range(Lit(HEX), 1, 4))

def build_isbn() -> RE:
    # (?:\d[- ]?){9}[\dX]
    sep_opt = opt(Lit(SEP))
    digit_then_sepopt = concat(Lit(DIGIT), sep_opt)
    return concat(rep_exact(digit_then_sepopt, 9), alt(Lit(DIGIT), Lit(X)))

def build_url() -> RE:
    # We force whole-string match: ^...$
    # Tokenized approximation of your URL regex:
    #
    # https? :// (www.)? DOMCHAR{1,256} '.' TLDCHAR{1,6}  (word boundary ignored at token level)  PATHCHAR*
    #
    scheme = concat(Lit(H), Lit(T), Lit(T), Lit(P), opt(Lit(S)))
    colon_slash_slash = concat(Lit(URL_COLON), Lit(SLASH), Lit(SLASH))
    www_opt = opt(concat(Lit(W), Lit(W), Lit(W), Lit(URL_DOT)))
    domain = rep_range(Lit(DOMCHAR), 1, 256)
    dot = Lit(URL_DOT)
    tld = rep_range(Lit(TLDCHAR), 1, 6)
    path = star(Lit(PATHCHAR))
    return concat(scheme, colon_slash_slash, www_opt, domain, dot, tld, path)

BUILDERS = {
    "Date": build_date,
    "Time": build_time,
    "IPv4": build_ipv4,
    "IPv6": build_ipv6,
    "ISBN": build_isbn,
    "URL":  build_url,
}

# ============================================================
# 4) DFA construction via derivatives
# ============================================================

def build_dfa(start: RE, alphabet: Tuple[str, ...]) -> Tuple[List[RE], Dict[Tuple[int, str], int], Set[int]]:
    # BFS over distinct derivatives
    states: List[RE] = []
    idx: Dict[RE, int] = {}
    trans: Dict[Tuple[int, str], int] = {}
    accepting: Set[int] = set()

    def get_id(r: RE) -> int:
        r = r.simplify()
        if r in idx: return idx[r]
        i = len(states)
        states.append(r)
        idx[r] = i
        return i

    q0 = get_id(start)
    work = [q0]
    seen = {q0}

    while work:
        q = work.pop()
        r = states[q]
        if r.nullable():
            accepting.add(q)
        for a in alphabet:
            dr = r.deriv(a).simplify()
            q2 = get_id(dr)
            trans[(q, a)] = q2
            if q2 not in seen:
                seen.add(q2)
                work.append(q2)

    return states, trans, accepting

def grammar_production_count(num_states: int, alphabet_size: int, num_accepting: int) -> int:
    # Complete DFA => |delta| = |Q|*|Sigma|, then + |F| epsilon productions
    return num_states * alphabet_size + num_accepting

# ============================================================
# 5) Run
# ============================================================

def main() -> None:
    print("=== Equivalent Grammar Production Counts (token-level DFA -> right-linear grammar) ===\n")
    for name in ["Date", "Time", "URL", "ISBN", "IPv4", "IPv6"]:
        raw = RAW_REGEX[name]
        alphabet = ALPHABETS[name]
        re_ast = BUILDERS[name]().simplify()
        states, trans, accepting = build_dfa(re_ast, alphabet)

        Q = len(states)
        Sigma = len(alphabet)
        F = len(accepting)
        delta = Q * Sigma  # complete by construction
        P = grammar_production_count(Q, Sigma, F)

        print(f"{name}")
        print(f"  raw regex: {raw}")
        print(f"  alphabet tokens (|Σ|={Sigma}): {alphabet}")
        print(f"  DFA states |Q|={Q}, accepting |F|={F}, transitions |δ|={delta}")
        print(f"  productions |P| = |δ| + |F| = {delta} + {F} = {P}")
        print()

if __name__ == "__main__":
    main()