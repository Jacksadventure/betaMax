#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python3 "/Users/jack/EarlyRepairer/bm_single.py" --max-workers 10 --db single_non_mut.db --format ipv6
python3 "/Users/jack/EarlyRepairer/bm_multiple.py" --max-workers 10 --db double_non_mut.db --format ipv6
python3 "/Users/jack/EarlyRepairer/bm_triple.py" --max-workers 10 --db triple_non_mut.db --format ipv6
