#!/usr/bin/env python3
"""\
RPNI variant with cross-over based consistency check.

This learner starts from the PTA (as in `rpni.RPNI`) but plugs in the
*Can Merge Check* cross-over test from Algorithm 2 in the supplied note:

  - accumulate, for every PTA state, the positive strings (prefix +
    suffix) and negative strings that pass through that state;
  - when considering a merge of states `s_i` and `s_j`, generate the
    cross-over candidates `p(x_i)·s(x_j)` and `p(x_j)·s(x_i)` for
    positives that reach each state, and reject the merge if any such
    candidate is rejected by the external membership oracle;
  - also reject if the oracle classifies any negative string that passes
    through either state as *positive*.

In other words, we reproduce the behaviour of Algorithm 2: merging is
allowed only when all cross-over candidates remain in the target
language and the negative samples touching either state stay negative.

This module mirrors the structure of `rpni_fuzz`: we subclass RPNI and
override `_try_merge` so that every candidate merge is first filtered by
the cross-over rule from Algorithm 2 before falling back to the stock
negative-sample check.

A budget of cross-over oracle queries is enforced, so at most *n* new
membership checks are performed per merge attempt.
Existing learners remain untouched; betamax.py can import this as an
additional --learner option (e.g. 'rpni_xover').
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple, Optional, Set
from collections import defaultdict
import os
import random

from .rpni import RPNI, DFA, dfa_to_right_linear_grammar

# External membership oracle: True if string is in the target language.
MembershipOracle = Callable[[str], bool]


class XoverRPNI(RPNI):
    """RPNI learner whose merge consistency is checked via cross-over.

    Behaviour relative to base RPNI:
      - still enforces that no known negative is accepted (original
        _consistent_with_negatives in the parent class);
      - additionally, generates cross-over examples from positive
        samples under the current DFA hypothesis and rejects merges
        that introduce oracle-rejected strings;
      - also rejects merges if any negative string that passes through
        either candidate state is classified as positive by the oracle.

    A per-merge budget of oracle queries is enforced: at most
    ``max_checks`` membership calls are performed during a single
    merge attempt (cross-over + negatives combined). If the budget
    is exhausted before all candidates are checked, the remaining
    candidates are simply not queried (the merge is decided based
    on the checks performed so far).
    """

    def __init__(
        self,
        positives: Iterable[str],
        negatives: Iterable[str],
        is_member: MembershipOracle,
        max_pairs: int = 50,
        max_checks: int = 10,
    ) -> None:
        # Parent constructor builds the PTA and stores positives/negatives.
        super().__init__(positives, negatives)

        self._is_member: Optional[MembershipOracle] = is_member

        # Snapshot positives as a list (for deterministic iteration).
        self._positives: List[str] = [p for p in positives if isinstance(p, str)]

        # Normalise limits.
        try:
            max_pairs = int(max_pairs)
        except Exception:
            max_pairs = 1
        try:
            max_checks = int(max_checks)
        except Exception:
            max_checks = 2

        # max_pairs: per-merge cap on |pos_r| × |pos_b| combinations
        # (0 => disable cross-over checks entirely).
        self._max_pairs: int = max(0, max_pairs)
        # max_checks: per-merge cap on oracle membership calls
        # (0 => disable both cross-over and negative checks).
        self._max_cross_checks: int = max(0, max_checks)

        # Pre-compute PTA-derived metadata so that Algorithm 2 can be
        # evaluated quickly during every merge consideration.
        self._node_prefix: List[str] = self._compute_node_prefixes()
        self._pos_suffixes: Dict[int, List[str]] = self._index_positive_suffixes()
        self._negatives_by_node: Dict[int, List[str]] = self._index_negative_strings()
        self._oracle_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # PTA-derived metadata
    # ------------------------------------------------------------------

    def _compute_node_prefixes(self) -> List[str]:
        """Compute the input prefix (root → node) for each PTA node."""
        prefixes = [""] * len(self.pta.nodes)
        for node_id in range(1, len(self.pta.nodes)):
            parent = self.pta.nodes[node_id].parent
            if parent >= 0:
                prefixes[node_id] = prefixes[parent] + self.pta.nodes[node_id].via
        return prefixes

    def _index_positive_suffixes(self) -> Dict[int, List[str]]:
        """For each PTA node, collect suffixes of positive strings that pass through it.

        For a positive word w = a_1…a_k, we walk it through the PTA:

          - at the root, we record the full string w as a suffix;
          - at each subsequent node reached after consuming a_1…a_i,
            we record the suffix w[i:].

        This lets us reconstruct any positive x = p(x)·s(x) at any node.
        """
        suffix_map: Dict[int, List[str]] = defaultdict(list)
        for w in self._positives:
            if not isinstance(w, str):
                continue
            node = 0
            suffix_map[node].append(w)
            consumed = 0
            for ch in w:
                nxt = self.pta.nodes[node].next.get(ch)
                if nxt is None:
                    break
                node = nxt
                consumed += 1
                suffix_map[node].append(w[consumed:])
        return suffix_map

    def _index_negative_strings(self) -> Dict[int, List[str]]:
        """For each PTA node, collect all negative strings that pass through it.

        We record the *entire* negative string w for every node on the
        path taken by w in the PTA (until the walk fails).
        """
        neg_map: Dict[int, List[str]] = defaultdict(list)
        for w in self.negatives:
            if not isinstance(w, str):
                continue
            node = 0
            neg_map[node].append(w)
            for ch in w:
                nxt = self.pta.nodes[node].next.get(ch)
                if nxt is None:
                    break
                node = nxt
                neg_map[node].append(w)
        return neg_map

    # ------------------------------------------------------------------
    # Helpers for Algorithm 2
    # ------------------------------------------------------------------

    def _nodes_for_roots(
        self,
        rep: List[int],
        roots: Tuple[int, int],
    ) -> Dict[int, List[int]]:
        """Group PTA nodes according to their current DFA representative.

        Given the union-find representative array `rep` and two roots
        (red/blue), return a mapping {root -> [nodes]} where each list
        contains all PTA nodes whose representative is that root.
        """
        groups: Dict[int, List[int]] = {root: [] for root in roots}
        for node in range(len(rep)):
            root = self._find(rep, node)
            if root in groups:
                groups[root].append(node)
        return groups

    def _collect_positive_pairs(self, nodes: List[int]) -> List[Tuple[str, str]]:
        """Collect distinct (prefix, suffix) pairs for positives through `nodes`."""
        acc: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for node in nodes:
            suffixes = self._pos_suffixes.get(node)
            if not suffixes:
                continue
            prefix = self._node_prefix[node]
            for suffix in suffixes:
                pair = (prefix, suffix)
                if pair in seen:
                    continue
                seen.add(pair)
                acc.append(pair)
        return acc

    def _collect_negative_candidates(self, nodes: List[int]) -> List[str]:
        """Collect distinct negative strings that pass through any of `nodes`."""
        acc: List[str] = []
        seen: Set[str] = set()
        for node in nodes:
            entries = self._negatives_by_node.get(node)
            if not entries:
                continue
            for w in entries:
                if w in seen:
                    continue
                seen.add(w)
                acc.append(w)
        return acc

    def _oracle_accepts(self, word: str) -> Optional[bool]:
        """Query the external oracle with caching.

        Returns:
          - True  if oracle says `word` is in the language;
          - False if oracle says `word` is not in the language;
          - None  if the oracle is unavailable or raised an exception.
        """
        if self._is_member is None:
            return None
        if word in self._oracle_cache:
            return self._oracle_cache[word]
        try:
            verdict = bool(self._is_member(word))
        except Exception:
            return None
        self._oracle_cache[word] = verdict
        return verdict

    # ------------------------------------------------------------------
    # Algorithm 2: cross-over + negatives for a given merge
    # ------------------------------------------------------------------

    def _cross_over_merge_check(self, rep: List[int], qr: int, qb: int) -> bool:
        """Implement Algorithm 2 for a proposed merge (qr, qb).

        For the two DFA states with representatives `qr` and `qb`:

          1. Collect all PTA nodes currently mapped to each root;
          2. From those nodes, extract (prefix, suffix) pairs for
             positive samples and build cross-over candidates:
                p(x_i)·s(x_j) and p(x_j)·s(x_i);
             Reject the merge if the oracle says any candidate is
             *outside* the target language;
          3. Collect all negative samples that pass through any of the
             nodes mapped to either root and reject the merge if the
             oracle classifies any of them as positive.

        A per-merge budget of `self._max_cross_checks` oracle calls is
        enforced; once exhausted, no further candidates are queried and
        the decision is based on checks seen so far.

        Returns:
          True  if the merge passes the cross-over and negative checks
          False if it violates Algorithm 2's conditions
        """
        # If no oracle or no budget, Algorithm 2 is effectively disabled.
        if (
            self._is_member is None
            or self._max_pairs <= 0
            or self._max_cross_checks <= 0
        ):
            return True

        budget = self._max_cross_checks

        root_r = self._find(rep, qr)
        root_b = self._find(rep, qb)
        # Already in the same class: nothing to check.
        if root_r == root_b:
            return True

        groups = self._nodes_for_roots(rep, (root_r, root_b))
        nodes_r = groups.get(root_r, [])
        nodes_b = groups.get(root_b, [])

        # ------------------------------
        # 1) Cross-over from positives
        # ------------------------------
        pos_r = self._collect_positive_pairs(nodes_r)
        pos_b = self._collect_positive_pairs(nodes_b)

        if pos_r and pos_b and budget > 0:
            seen_words: Set[str] = set()
            total_pairs = len(pos_r) * len(pos_b)
            if self._max_pairs > 0:
                sample_size = min(self._max_pairs, total_pairs)
                pair_indices = random.sample(range(total_pairs), sample_size)
            else:
                pair_indices = range(total_pairs)

            width = len(pos_b)

            for idx in pair_indices:
                if budget <= 0:
                    break
                i, j = divmod(idx, width)
                pref_r, suf_r = pos_r[i]
                pref_b, suf_b = pos_b[j]

                # Two cross-over candidates: p_r·s_b and p_b·s_r.
                for cand in (pref_r + suf_b, pref_b + suf_r):
                    if budget <= 0:
                        break
                    if cand in seen_words:
                        continue
                    seen_words.add(cand)

                    verdict = self._oracle_accepts(cand)
                    budget -= 1

                    # If the oracle explicitly says "not in language",
                    # this merge violates Algorithm 2.
                    if verdict is False:
                        return False

        # ------------------------------
        # 2) Negatives must stay negative
        # ------------------------------
        neg_candidates = self._collect_negative_candidates(nodes_r)
        neg_candidates += self._collect_negative_candidates(nodes_b)

        if neg_candidates and budget > 0:
            seen_neg: Set[str] = set()
            for cand in neg_candidates:
                if budget <= 0:
                    break
                if cand in seen_neg:
                    continue
                seen_neg.add(cand)

                verdict = self._oracle_accepts(cand)
                budget -= 1

                # If the oracle says this negative is actually in the language,
                # we reject the merge.
                if verdict is True:
                    return False

        # No violation observed within the budget.
        return True

    # ------------------------------------------------------------------
    # RPNI hook
    # ------------------------------------------------------------------

    def _try_merge(
        self,
        rep_in: List[int],
        qr: int,
        qb: int,
    ) -> Optional[List[int]]:  # type: ignore[override]
        """Override RPNI's merge hook to inject Algorithm 2.

        If the cross-over + negatives check passes, we delegate to the
        base class which will still perform its stock negative-sample
        consistency check on the DFA hypothesis.
        """
        if not self._cross_over_merge_check(rep_in, qr, qb):
            return None
        return super()._try_merge(rep_in, qr, qb)


# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------

def learn_grammar_from_samples_xover(
    positives: Iterable[str],
    negatives: Iterable[str],
    is_member: MembershipOracle,
    max_pairs: Optional[int] = None,
    max_checks: Optional[int] = None,
) -> Tuple[Dict[str, List[List[str]]], str, List[str]]:
    """Convenience wrapper:

        - Learn a DFA using XoverRPNI
        - Convert it to a right-linear grammar

    Cross-over exploration limits can be controlled via arguments or
    environment variables:

        LSTAR_RPNI_XOVER_PAIRS   (default: 50, 0 disables checks)
        LSTAR_RPNI_XOVER_CHECKS  (default: 10 oracle queries per merge)
    """
    if max_pairs is None:
        try:
            max_pairs = int(os.getenv("LSTAR_RPNI_XOVER_PAIRS", "50"))
        except Exception:
            max_pairs = 1
    if max_checks is None:
        try:
            max_checks = int(os.getenv("LSTAR_RPNI_XOVER_CHECKS", "10"))
        except Exception:
            max_checks = 1

    learner = XoverRPNI(
        positives=positives,
        negatives=negatives,
        is_member=is_member,
        max_pairs=max_pairs,
        max_checks=max_checks,
    )
    dfa: DFA = learner.learn()
    return dfa_to_right_linear_grammar(dfa)
