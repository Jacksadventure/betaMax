#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# NOTE: these must be exported; otherwise bm_* subprocesses won't see them and
# will fall back to their internal defaults (e.g., LSTAR_PARSE_TIMEOUT=100s).
export LSTAR_PRECOMPUTE_TIMEOUT=1800

# Two-stage strategy:
# 1) Precompute a strong cache grammar with `rpni_xover` (expensive, done once per format).
# 2) For each test sample, start from that cached grammar; if oracle fails and we need to relearn,
#    fall back to cheaper `rpni` for the iterative loop.
#
# Valid values (see `betamax/lstar/betamax.py`): rpni, rpni_nfa, rpni_fuzz, rpni_xover, lstar_oracle
export LSTAR_CACHE_LEARNER="${LSTAR_CACHE_LEARNER:-rpni_xover}"
export LSTAR_LEARNER="${LSTAR_LEARNER:-rpni}"

export LSTAR_PARSE_TIMEOUT=600
export LSTAR_EC_TIMEOUT=600
export LSTAR_PRECOMPUTE_MUTATIONS=40
export BM_NEGATIVES_FROM_DB="${BM_NEGATIVES_FROM_DB:-0}"
export BETAMAX_DEBUG_ORACLE="${BETAMAX_DEBUG_ORACLE:-0}"

# If >0, benchmarks will pre-generate oracle-classified mutations to augment the pos/neg pool.
BM_LSTAR_MUTATION_COUNT="${BM_LSTAR_MUTATION_COUNT:-${LSTAR_PRECOMPUTE_MUTATIONS:-0}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${MAX_WORKERS:-3}"

if [[ ! -d mutated_files ]]; then
  echo "[ERROR] Missing required directory: mutated_files/" >&2
  exit 1
fi

# Benchmarks expect mutation DBs to exist; fail early with a clear message instead of silently doing no work.
missing_db=0
formats=(date time isbn ipv4 url)
for fmt in "${formats[@]}"; do
  for mt in single double triple; do
    db="mutated_files/${mt}_${fmt}.db"
    if [[ ! -f "$db" ]]; then
      echo "[ERROR] Missing mutation DB: $db" >&2
      missing_db=1
    fi
  done
done
if [[ "$missing_db" != "0" && "${BM_ALLOW_MISSING_MUTATED:-0}" != "1" ]]; then
  echo "[ERROR] Required mutation DBs are missing. Set BM_ALLOW_MISSING_MUTATED=1 to run anyway." >&2
  exit 1
fi

"$PYTHON_BIN" "bm_single.py"   --max-workers "$MAX_WORKERS" --algorithms betamax --lstar-mutation-count "$BM_LSTAR_MUTATION_COUNT"
"$PYTHON_BIN" "bm_multiple.py" --max-workers "$MAX_WORKERS" --algorithms betamax --lstar-mutation-count "$BM_LSTAR_MUTATION_COUNT"
"$PYTHON_BIN" "bm_triple.py"   --max-workers "$MAX_WORKERS" --algorithms betamax --lstar-mutation-count "$BM_LSTAR_MUTATION_COUNT"
