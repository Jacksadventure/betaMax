from __future__ import annotations

import os
import shlex
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from betamax.lstar import betamax as _lstar_mod
except ImportError:
    _lstar_mod = None


_MUTATION_TABLE_CACHE: Dict[str, str] = {}


def get_mutation_table_name(mutation_db_path: str, conn: sqlite3.Connection | None = None) -> str:
    """
    Returns the name of the table that stores mutation samples.
    Accepts either the standard 'mutations' or legacy 'mutations_triple' tables.
    """
    if mutation_db_path in _MUTATION_TABLE_CACHE:
        return _MUTATION_TABLE_CACHE[mutation_db_path]

    if not os.path.exists(mutation_db_path):
        raise FileNotFoundError(f"Mutation database not found: {mutation_db_path}")

    close_conn = False
    target_conn = conn
    if target_conn is None:
        target_conn = sqlite3.connect(mutation_db_path)
        close_conn = True

    cursor = target_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    if close_conn:
        target_conn.close()

    for candidate in ("mutations", "mutations_triple"):
        if candidate in tables:
            _MUTATION_TABLE_CACHE[mutation_db_path] = candidate
            return candidate

    raise sqlite3.OperationalError(
        f"No table named 'mutations' or 'mutations_triple' in {mutation_db_path}"
    )


class LStarMutationPool:
    """
    Pre-loads positives/negatives from mutation databases and optionally augments them
    with additional oracle-classified mutations (mirroring betamax's logic).
    """

    def __init__(self, train_k: int, regex_formats: Iterable[str], dir_to_category: Dict[str, str]):
        self.train_k = max(0, int(train_k))
        self.regex_formats = set(regex_formats)
        self.dir_to_category = dict(dir_to_category)
        self.seed_pools: Dict[str, Dict[str, List[str]]] = {}
        self.mutation_pools: Dict[str, Dict[str, List[str]]] = {}
        self._mutation_cfg: Tuple[int, bool, Optional[int]] = (0, False, None)
        self._validator_cache_key: Optional[str] = None
        self._validator_cache: Optional[List[str]] = None
        self._warned_missing_lstar = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_mutation_config(self, count: int, use_random: bool = True, seed: Optional[int] = None):
        norm_count = max(0, int(count or 0))
        cfg = (norm_count, bool(use_random), seed if (use_random and seed is not None) else None)
        if cfg != self._mutation_cfg:
            self.mutation_pools.clear()
            self._mutation_cfg = cfg

    def ensure_many(self, format_keys: Iterable[str]):
        for fmt in format_keys:
            self.ensure_format(fmt)

    def ensure_format(self, format_key: str):
        self._ensure_seed_pool(format_key)
        if self._mutation_cfg[0] > 0:
            self._ensure_mutation_pool(format_key)

    def get_seed_positives(self, format_key: str) -> List[str]:
        return list(self.seed_pools.get(format_key, {}).get("positives", []))

    def get_seed_negatives(self, format_key: str) -> List[str]:
        return list(self.seed_pools.get(format_key, {}).get("negatives", []))

    def get_mutation_positives(self, format_key: str) -> List[str]:
        return list(self.mutation_pools.get(format_key, {}).get("positives", []))

    def get_mutation_negatives(self, format_key: str) -> List[str]:
        return list(self.mutation_pools.get(format_key, {}).get("negatives", []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_seed_pool(self, format_key: str):
        if format_key in self.seed_pools:
            return
        mdb_path = os.path.join("mutated_files", f"{format_key}.db")
        positives: List[str] = []
        negatives: List[str] = []
        use_db_negatives = os.environ.get("BM_NEGATIVES_FROM_DB", "0").lower() in ("1", "true", "yes")
        if not os.path.exists(mdb_path):
            print(f"[INFO] Mutation DB missing for {format_key}, skipping seed pool.")
            self.seed_pools[format_key] = {"positives": positives, "negatives": negatives}
            return
        conn = None
        try:
            conn = sqlite3.connect(mdb_path)
            table_name = get_mutation_table_name(mdb_path, conn)
            cur = conn.cursor()
            cur.execute(f"""
                SELECT original_text
                FROM {table_name}
                ORDER BY LENGTH(original_text), id
                LIMIT ?
            """, (self.train_k,))
            positives = [(row[0] or "") for row in cur.fetchall()]
            if use_db_negatives:
                cur.execute(f"""
                    SELECT mutated_text
                    FROM {table_name}
                    ORDER BY LENGTH(mutated_text), id
                    LIMIT ?
                """, (self.train_k,))
                negatives = [(row[0] or "") for row in cur.fetchall()]
            print(f"[INFO] Seed pool for {format_key}: +{len(positives)} / -{len(negatives)}")
        except Exception as exc:
            print(f"[WARN] Failed to load mutation seed pool for {format_key}: {exc}")
        finally:
            if conn:
                conn.close()
        self.seed_pools[format_key] = {"positives": positives, "negatives": negatives}

    def _ensure_mutation_pool(self, format_key: str):
        if format_key in self.mutation_pools:
            return
        count, use_random, seed = self._mutation_cfg
        if count <= 0:
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return
        seeds = self.seed_pools.get(format_key)
        if not seeds or not seeds.get("positives"):
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return

        if _lstar_mod is None:
            if not self._warned_missing_lstar:
                print("[WARN] betamax.lstar not available; skipping mutation augmentation.")
                self._warned_missing_lstar = True
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return

        base_format = format_key.split("_")[-1]
        if base_format not in self.regex_formats:
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return

        category = self.dir_to_category.get(base_format, base_format)
        pos_set = {p for p in seeds.get("positives", []) if p}
        if not pos_set:
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return

        alphabet = _lstar_mod.derive_alphabet_from_examples(pos_set, set())
        if not alphabet:
            self.mutation_pools[format_key] = {"positives": [], "negatives": []}
            return

        if use_random:
            candidates = _lstar_mod.generate_mutations_random(pos_set, count, alphabet, seed=seed)
        else:
            candidates = _lstar_mod.generate_mutations(pos_set, count, alphabet)

        validator_cmd = self._validator_cmd()
        aug_pos: List[str] = []
        aug_neg: List[str] = []
        seen_pos = set(pos_set)
        seen_neg = {n for n in seeds.get("negatives", []) if n}

        for cand in candidates:
            if not cand:
                continue
            try:
                verdict = _lstar_mod.validate_with_match(category, cand, validator_cmd)
            except Exception as exc:
                print(f"[WARN] Oracle classification failed for {format_key}: {exc}")
                continue
            if verdict:
                if cand not in seen_pos:
                    seen_pos.add(cand)
                    aug_pos.append(cand)
            else:
                if cand not in seen_neg:
                    seen_neg.add(cand)
                    aug_neg.append(cand)

        print(f"[INFO] Mutation augmentation for {format_key}: requested={count} -> +pos={len(aug_pos)}, +neg={len(aug_neg)}")
        self.mutation_pools[format_key] = {"positives": aug_pos, "negatives": aug_neg}

    def _validator_cmd(self) -> Optional[List[str]]:
        raw = os.environ.get("LSTAR_ORACLE_VALIDATOR")
        if raw == self._validator_cache_key:
            return self._validator_cache
        self._validator_cache_key = raw
        if not raw:
            self._validator_cache = None
            return None
        try:
            self._validator_cache = shlex.split(raw)
        except Exception:
            self._validator_cache = [raw]
        return self._validator_cache


__all__ = ["get_mutation_table_name", "LStarMutationPool"]
