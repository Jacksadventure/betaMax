#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${MAX_WORKERS:-3}"
MODE="${1:-quick}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

# Benchmark defaults used by the betaMax runners.
export LSTAR_PRECOMPUTE_TIMEOUT="${LSTAR_PRECOMPUTE_TIMEOUT:-18000}"
export LSTAR_CACHE_LEARNER="${LSTAR_CACHE_LEARNER:-rpni_xover}"
export LSTAR_LEARNER="${LSTAR_LEARNER:-rpni}"
export BM_BETAMAX_ENGINE="${BM_BETAMAX_ENGINE:-cpp}"
export LSTAR_PARSE_TIMEOUT="${LSTAR_PARSE_TIMEOUT:-600}"
export LSTAR_EC_TIMEOUT="${LSTAR_EC_TIMEOUT:-600}"
export LSTAR_PRECOMPUTE_MUTATIONS="${LSTAR_PRECOMPUTE_MUTATIONS:-60}"
export BM_NEGATIVES_FROM_DB="${BM_NEGATIVES_FROM_DB:-0}"
export BETAMAX_DEBUG_ORACLE="${BETAMAX_DEBUG_ORACLE:-0}"
export BM_LSTAR_MUTATION_COUNT="${BM_LSTAR_MUTATION_COUNT:-${LSTAR_PRECOMPUTE_MUTATIONS}}"

DEFAULT_REGEX_FORMATS=(date time isbn ipv4 url ipv6)

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

require_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    die "Missing command '$cmd'. $hint"
  fi
}

require_python_module() {
  local module="$1"
  local pkg_hint="$2"
  if ! "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1; then
    die "Missing Python module '${module}'. Install dependencies with '${PYTHON_BIN} -m pip install -r requirements.txt'${pkg_hint:+ (needs ${pkg_hint})}."
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

ensure_cpp_engine() {
  if [[ "$BM_BETAMAX_ENGINE" != "cpp" ]]; then
    return 0
  fi
  if [[ -x "betamax_cpp/build/betamax_cpp" ]]; then
    return 0
  fi
  require_command cmake "Install CMake or build betamax_cpp manually."
  echo "[INFO] betamax_cpp/build/betamax_cpp not found. Building the C++ backend..."
  print_cmd cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
  print_cmd cmake --build betamax_cpp/build -j
  cmake --build betamax_cpp/build -j
}

ensure_python_backend() {
  if [[ "$BM_BETAMAX_ENGINE" != "python" ]]; then
    return 0
  fi
  local entrypoint="${BM_BETAMAX_PY_ENTRYPOINT:-betamax/app/betamax.py}"
  if [[ ! -f "$entrypoint" ]]; then
    die "Legacy Python betaMax backend not found at '$entrypoint'. Use 'BM_BETAMAX_ENGINE=cpp' or point BM_BETAMAX_PY_ENTRYPOINT to the legacy checkout."
  fi
}

ensure_betamax_backend() {
  require_python_module regex regex
  case "$BM_BETAMAX_ENGINE" in
    cpp) ensure_cpp_engine ;;
    python) ensure_python_backend ;;
    *) die "Unsupported BM_BETAMAX_ENGINE='$BM_BETAMAX_ENGINE'. Expected 'cpp' or 'python'." ;;
  esac
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
    die "Required mutation DBs are missing under mutated_files/."
  fi
}

ensure_truncation_subjects() {
  if has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    return 0
  fi
  if has_flag "--truncated-json-validator" "${EXTRA_ARGS[@]}"; then
    return 0
  fi
  if [[ -n "${BM_TRUNCATED_JSON_VALIDATOR:-}" ]]; then
    return 0
  fi
  local validator="project/bin/subjects/cjson/cjson"
  if [[ ! -x "$validator" ]]; then
    die "Missing JSON subject validator '$validator'. Build subject artifacts with './build_all.sh' before running truncation benchmarks."
  fi
}

run_single() {
  ensure_betamax_backend
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    ensure_mutation_dbs single "${DEFAULT_REGEX_FORMATS[@]}"
  fi
  local db_name
  db_name="$(default_db_path single)"
  local -a cmd=("$PYTHON_BIN" "bm_single.py")
  if ! has_flag "--db" "${EXTRA_ARGS[@]}"; then
    cmd+=(--db "$db_name")
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    cmd+=(--formats "${DEFAULT_REGEX_FORMATS[@]}")
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    cmd+=(--algorithms betamax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    cmd+=(--max-workers "$MAX_WORKERS")
  fi
  if ! has_flag "--betamax-engine" "${EXTRA_ARGS[@]}"; then
    cmd+=(--betamax-engine "$BM_BETAMAX_ENGINE")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  print_cmd "${cmd[@]}"
  "${cmd[@]}"
}

run_double() {
  ensure_betamax_backend
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    ensure_mutation_dbs double "${DEFAULT_REGEX_FORMATS[@]}"
  fi
  local db_name
  db_name="$(default_db_path double)"
  local -a cmd=("$PYTHON_BIN" "bm_multiple.py")
  if ! has_flag "--db" "${EXTRA_ARGS[@]}"; then
    cmd+=(--db "$db_name")
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    cmd+=(--formats "${DEFAULT_REGEX_FORMATS[@]}")
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    cmd+=(--algorithms betamax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    cmd+=(--max-workers "$MAX_WORKERS")
  fi
  if ! has_flag "--betamax-engine" "${EXTRA_ARGS[@]}"; then
    cmd+=(--betamax-engine "$BM_BETAMAX_ENGINE")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  print_cmd "${cmd[@]}"
  "${cmd[@]}"
}

run_triple() {
  ensure_betamax_backend
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    ensure_mutation_dbs triple "${DEFAULT_REGEX_FORMATS[@]}"
  fi
  local db_name
  db_name="$(default_db_path triple)"
  local -a cmd=("$PYTHON_BIN" "bm_triple.py")
  if ! has_flag "--db" "${EXTRA_ARGS[@]}"; then
    cmd+=(--db "$db_name")
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    cmd+=(--formats "${DEFAULT_REGEX_FORMATS[@]}")
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    cmd+=(--algorithms betamax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    cmd+=(--max-workers "$MAX_WORKERS")
  fi
  if ! has_flag "--betamax-engine" "${EXTRA_ARGS[@]}"; then
    cmd+=(--betamax-engine "$BM_BETAMAX_ENGINE")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  print_cmd "${cmd[@]}"
  "${cmd[@]}"
}

run_quick() {
  ensure_betamax_backend
  local quick_limit="${BM_QUICK_LIMIT:-3}"
  local quick_precompute="${BM_QUICK_PRECOMPUTE_MUTATIONS:-10}"
  local quick_formats_raw="${BM_QUICK_FORMATS:-date}"
  local -a quick_formats=()
  read -r -a quick_formats <<< "$quick_formats_raw"
  if [[ "${#quick_formats[@]}" -eq 0 ]]; then
    quick_formats=(date)
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    ensure_mutation_dbs single "${quick_formats[@]}"
  fi
  local db_name
  db_name="$(default_db_path smoke_single)"
  local -a cmd=("$PYTHON_BIN" "bm_single.py")
  if ! has_flag "--db" "${EXTRA_ARGS[@]}"; then
    cmd+=(--db "$db_name")
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    cmd+=(--formats "${quick_formats[@]}")
  fi
  if ! has_flag "--limit" "${EXTRA_ARGS[@]}"; then
    cmd+=(--limit "$quick_limit")
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    cmd+=(--algorithms betamax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    cmd+=(--max-workers "$MAX_WORKERS")
  fi
  if ! has_flag "--betamax-engine" "${EXTRA_ARGS[@]}"; then
    cmd+=(--betamax-engine "$BM_BETAMAX_ENGINE")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  printf '[RUN] %s=%q %s=%q' \
    "LSTAR_PRECOMPUTE_MUTATIONS" "$quick_precompute" \
    "BM_LSTAR_MUTATION_COUNT" "$quick_precompute"
  local arg
  for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  LSTAR_PRECOMPUTE_MUTATIONS="$quick_precompute" \
  BM_LSTAR_MUTATION_COUNT="$quick_precompute" \
    "${cmd[@]}"
}

run_regex_suite() {
  if has_flag "--db" "${EXTRA_ARGS[@]}"; then
    die "Do not pass --db with mode 'regex'. Use DB_PREFIX=<name> ./run_bms.sh regex or run single/double/triple separately."
  fi
  run_single
  run_double
  run_triple
}

run_truncation() {
  ensure_betamax_backend
  ensure_truncation_subjects
  local db_name
  db_name="$(default_db_path truncation)"
  local -a cmd=("$PYTHON_BIN" "bm_truncation.py")
  if ! has_flag "--db" "${EXTRA_ARGS[@]}"; then
    cmd+=(--db "$db_name")
  fi
  if ! has_flag "--formats" "${EXTRA_ARGS[@]}"; then
    cmd+=(--formats json)
  fi
  if ! has_flag "--algorithms" "${EXTRA_ARGS[@]}"; then
    cmd+=(--algorithms betamax)
  fi
  if ! has_flag "--max-workers" "${EXTRA_ARGS[@]}"; then
    cmd+=(--max-workers "$MAX_WORKERS")
  fi
  if ! has_flag "--betamax-engine" "${EXTRA_ARGS[@]}"; then
    cmd+=(--betamax-engine "$BM_BETAMAX_ENGINE")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  print_cmd "${cmd[@]}"
  "${cmd[@]}"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_bms.sh <mode> [extra bm_* args]

Modes:
  quick       Smoke test: small single-mutation run (default DB: smoke_single.db)
  single      Run bm_single.py on the main regex benchmarks
  double      Run bm_multiple.py on the main regex benchmarks
  triple      Run bm_triple.py on the main regex benchmarks
  regex       Run single + double + triple regex benchmarks
  truncation  Run the JSON truncation benchmark (requires subject binaries)
  all         Run regex + truncation benchmarks
  help        Show this message

Examples:
  ./run_bms.sh quick
  ./run_bms.sh single --formats date url --limit 10
  DB_PREFIX=trial1 ./run_bms.sh regex
  ./run_bms.sh truncation --limit 10

Useful env vars:
  PYTHON_BIN                   Python interpreter to use (default: python3)
  MAX_WORKERS                  Passed to bm_*.py if --max-workers is omitted
  BM_BETAMAX_ENGINE            cpp (default) or python
  BM_BETAMAX_PY_ENTRYPOINT     Legacy Python backend entrypoint when BM_BETAMAX_ENGINE=python
  BM_QUICK_FORMATS             Formats used by quick mode (default: "date")
  BM_QUICK_LIMIT               Sample limit for quick mode (default: 3)
  BM_QUICK_PRECOMPUTE_MUTATIONS Precompute mutation count for quick mode (default: 10)
  DB_PREFIX                    Prefix per-mode DB names, e.g. trial1_single.db

Notes:
  - Extra args are forwarded to the underlying bm_*.py runner(s).
  - For mode 'regex', use DB_PREFIX instead of --db because three scripts are run.
  - The supported quickstart path in this checkout uses the C++ backend under betamax_cpp/.
EOF
}

case "$MODE" in
  quick)
    run_quick
    ;;
  single)
    run_single
    ;;
  double|multiple)
    run_double
    ;;
  triple)
    run_triple
    ;;
  regex|main)
    run_regex_suite
    ;;
  truncation|json)
    run_truncation
    ;;
  all)
    if has_flag "--db" "${EXTRA_ARGS[@]}"; then
      die "Do not pass --db with mode 'all'. Use DB_PREFIX=<name> ./run_bms.sh all."
    fi
    run_regex_suite
    run_truncation
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "Unknown mode '$MODE'."
    ;;
esac
