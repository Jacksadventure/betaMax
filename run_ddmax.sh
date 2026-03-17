#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MODE="${1:-${DDMAX_FORMAT_SET:-regex}}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

# DDMax does not learn from positives; keep the benchmark split explicit and overridable.
export BM_TRAIN_K="${BM_TRAIN_K:-0}"
export BM_TEST_K="${BM_TEST_K:-50}"

DEFAULT_REGEX_FORMATS=(date time isbn ipv4 ipv6 url)
DEFAULT_SUBJECT_FORMATS=(json ini dot obj lisp c)

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

has_flag() {
  local needle="$1"
  shift || true
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

print_cmd() {
  printf '[RUN]'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

require_python_module() {
  local module="$1"
  if ! "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1; then
    die "Missing Python module '${module}'. Install dependencies with '${PYTHON_BIN} -m pip install -r requirements.txt'."
  fi
}

default_db_path() {
  local stem="$1"
  if [[ -n "${DB_PREFIX:-}" ]]; then
    printf '%s_%s.db\n' "$DB_PREFIX" "$stem"
  else
    printf '%s.db\n' "$stem"
  fi
}

resolve_formats() {
  local mode="$1"
  local raw="${DDMAX_FORMATS:-}"
  local -a values=()
  if [[ -n "$raw" ]]; then
    read -r -a values <<< "$raw"
  elif [[ "$mode" == "subjects" ]]; then
    values=("${DEFAULT_SUBJECT_FORMATS[@]}")
  else
    values=("${DEFAULT_REGEX_FORMATS[@]}")
  fi
  printf '%s\n' "${values[@]}"
}

ensure_mutation_dbs() {
  local mut_type="$1"
  shift || true
  local fmt
  local missing=0
  for fmt in "$@"; do
    local db_path="mutated_files/${mut_type}_${fmt}.db"
    if [[ ! -f "$db_path" ]]; then
      echo "[ERROR] Missing mutation DB: $db_path" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    if [[ "$MODE" == "subjects" ]]; then
      cat >&2 <<'EOF'
[HINT] Subject-format mutation DBs can be regenerated with:
  python3 mutation_single.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/single_<fmt>.db
  python3 mutation_double.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/double_<fmt>.db
  python3 mutation_triple.py  --folder original_files/<fmt>_data --validator <validator> --database mutated_files/triple_<fmt>.db
EOF
    fi
    die "Required mutation DBs are missing under mutated_files/."
  fi
}

ensure_subject_prereqs() {
  if [[ ! -f "project/bin/erepair.jar" ]]; then
    die "Missing project/bin/erepair.jar. Build subject artifacts with './build_all.sh' before running DDMax subject benchmarks."
  fi
}

run_suite() {
  local mode="$1"
  if has_flag "--db" "${EXTRA_ARGS[@]}"; then
    die "Do not pass --db when a full DDMax suite is run. Use DB_PREFIX=<name> ./run_ddmax.sh ${mode} instead."
  fi

  local -a formats=()
  while IFS= read -r fmt; do
    formats+=("$fmt")
  done < <(resolve_formats "$mode")

  if [[ "${#formats[@]}" -eq 0 ]]; then
    die "No formats selected for mode '$mode'."
  fi

  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    ensure_mutation_dbs single "${formats[@]}"
    ensure_mutation_dbs double "${formats[@]}"
    ensure_mutation_dbs triple "${formats[@]}"
  fi

  if [[ "$mode" == "subjects" ]]; then
    ensure_subject_prereqs
  else
    require_python_module regex
  fi

  local single_db double_db triple_db
  single_db="$(default_db_path ddmax_single)"
  double_db="$(default_db_path ddmax_double)"
  triple_db="$(default_db_path ddmax_triple)"

  local -a common_args=()
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    common_args+=(--formats "${formats[@]}")
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    common_args+=(--algorithms ddmax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    common_args+=(--max-workers "$MAX_WORKERS")
  fi
  common_args+=("${EXTRA_ARGS[@]}")

  local -a cmd_single=("$PYTHON_BIN" "bm_single.py" --db "$single_db")
  local -a cmd_double=("$PYTHON_BIN" "bm_multiple.py" --db "$double_db")
  local -a cmd_triple=("$PYTHON_BIN" "bm_triple.py" --db "$triple_db")
  cmd_single+=("${common_args[@]}")
  cmd_double+=("${common_args[@]}")
  cmd_triple+=("${common_args[@]}")

  print_cmd "${cmd_single[@]}"
  "${cmd_single[@]}"
  print_cmd "${cmd_double[@]}"
  "${cmd_double[@]}"
  print_cmd "${cmd_triple[@]}"
  "${cmd_triple[@]}"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_ddmax.sh <mode> [extra bm_* args]

Modes:
  regex     Run DDMax on the regex benchmarks (default)
  subjects  Run DDMax on JSON/INI/DOT/OBJ/LISP/C subjects
  help      Show this message

Examples:
  ./run_ddmax.sh regex
  DB_PREFIX=trial2 ./run_ddmax.sh regex --limit 10
  ./run_ddmax.sh subjects

Useful env vars:
  PYTHON_BIN        Python interpreter to use (default: python3)
  MAX_WORKERS       Passed to bm_*.py if --max-workers is omitted
  BM_TRAIN_K        Defaults to 0 for DDMax
  BM_TEST_K         Defaults to 50
  DDMAX_FORMATS     Override default formats, e.g. "date url"
  DB_PREFIX         Prefix per-mode DB names, e.g. trial2_ddmax_single.db

Notes:
  - Extra args are forwarded to bm_single.py, bm_multiple.py, and bm_triple.py.
  - For subject mode you need project/bin/erepair.jar and the corresponding subject mutation DBs.
EOF
}

case "$MODE" in
  regex)
    run_suite regex
    ;;
  subjects)
    run_suite subjects
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "Unknown mode '$MODE'."
    ;;
esac
