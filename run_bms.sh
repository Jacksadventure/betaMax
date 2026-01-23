#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# NOTE: these must be exported; otherwise bm_* subprocesses won't see them and
# will fall back to their internal defaults (e.g., LSTAR_PARSE_TIMEOUT=100s).
export LSTAR_PRECOMPUTE_TIMEOUT=1800
export LSTAR_CACHE_LEARNER=xover_rpni
export LSTAR_PARSE_TIMEOUT=600
export LSTAR_EC_TIMEOUT=600
export LSTAR_PRECOMPUTE_MUTATIONS=40
python3 "/Users/jack/EarlyRepairer/bm_single.py" --max-workers 6  --algorithm betamax erepair
python3 "/Users/jack/EarlyRepairer/bm_multiple.py" --max-workers 6 --algorithm betamax erepair
python3 "/Users/jack/EarlyRepairer/bm_triple.py" --max-workers 6 --algorithm betamax erepair
