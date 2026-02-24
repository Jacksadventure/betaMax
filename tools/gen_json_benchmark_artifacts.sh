#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALIDATOR="project/bin/subjects/cjson/cjson"
CORPUS_DIR="original_files/json_small_data"

# Build JSON validator (oracle)
make -C project/bin/subjects/cjson

# Optional: build native betaMax engine (used when BM_BETAMAX_ENGINE=cpp)
cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build betamax_cpp/build -j

# Generate small valid JSON corpus (10..15 bytes per file)
python3 tools/gen_tiny_json_corpus.py --out-dir "$CORPUS_DIR" --count 100 --min-bytes 10 --max-bytes 15

# Build mutation DBs (broken inputs) for bm_single/bm_multiple/bm_triple
rm -f mutated_files/single_json.db mutated_files/double_json.db mutated_files/triple_json.db

python3 mutation_single.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/single_json.db \
  --max-attempts 80 \
  --max-per-file 1 \
  --seed 1

python3 mutation_double.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/double_json.db \
  --max-attempts 80 \
  --seed 1

python3 mutation_triple.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/triple_json.db \
  --max-attempts 120 \
  --seed 1

echo "[ok] artifacts ready:"
echo "  corpus: $CORPUS_DIR"
echo "  db: mutated_files/single_json.db"
echo "  db: mutated_files/double_json.db"
echo "  db: mutated_files/triple_json.db"
