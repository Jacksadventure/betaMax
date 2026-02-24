#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${MAX_WORKERS:-1}"

# DDMax does not need training examples; keep defaults small and overridable.
export BM_TRAIN_K="${BM_TRAIN_K:-0}"
export BM_TEST_K="${BM_TEST_K:-50}"

# Choose which format family to run:
# - regex: existing URL/date/time/etc line-based validators
# - subjects: grammar-based eRepair subjects (JSON/INI/DOT/...)
DDMAX_FORMAT_SET="${DDMAX_FORMAT_SET:-regex}"
if [[ -n "${DDMAX_FORMATS:-}" ]]; then
  read -r -a formats <<< "${DDMAX_FORMATS}"
elif [[ "${DDMAX_FORMAT_SET}" == "subjects" ]]; then
  formats=(json ini dot obj lisp c)
else
  formats=(date time isbn ipv4 ipv6 url)
fi

if [[ "${DDMAX_FORMAT_SET}" == "subjects" ]]; then
  if [[ ! -f project/bin/erepair.jar ]]; then
    echo "[ERROR] Missing: project/bin/erepair.jar" >&2
    exit 1
  fi
fi

# Benchmarks require mutation DBs like mutated_files/single_url.db.
missing_db=0
for fmt in "${formats[@]}"; do
  for mt in single double triple; do
    db="mutated_files/${mt}_${fmt}.db"
    if [[ ! -f "$db" ]]; then
      echo "[ERROR] Missing mutation DB: $db" >&2
      missing_db=1
    fi
  done
done

if [[ "$missing_db" != "0" ]]; then
  cat >&2 <<'EOF'
[HINT] Generate subject-format mutation DBs using:
  python3 mutation_single.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/single_<fmt>.db
  python3 mutation_double.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/double_<fmt>.db
  python3 mutation_triple.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/triple_<fmt>.db

Validators typically live under:
  project/bin/subjects/
EOF
  exit 1
fi

"$PYTHON_BIN" bm_single.py   --db ddmax_single.db   --algorithms ddmax --formats "${formats[@]}" --max-workers "$MAX_WORKERS"
"$PYTHON_BIN" bm_multiple.py --db ddmax_double.db   --algorithms ddmax --formats "${formats[@]}" --max-workers "$MAX_WORKERS"
"$PYTHON_BIN" bm_triple.py   --db ddmax_triple.db   --algorithms ddmax --formats "${formats[@]}" --max-workers "$MAX_WORKERS"
