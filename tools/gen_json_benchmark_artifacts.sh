#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALIDATOR="project/bin/subjects/cjson/cjson"
CORPUS_DIR="original_files/json_tiny_data"

make -C project/bin/subjects/cjson

python3 tools/gen_tiny_json_corpus.py --out-dir "$CORPUS_DIR" --count 100

python3 mutation_single.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/single_json.db \
  --max-attempts 50 \
  --max-per-file 1 \
  --seed 1

python3 mutation_double.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/double_json.db \
  --max-attempts 50 \
  --seed 1

python3 mutation_triple.py \
  --folder "$CORPUS_DIR" \
  --validator "$VALIDATOR" \
  --database mutated_files/triple_json.db \
  --max-attempts 80 \
  --seed 1

echo "[ok] artifacts ready:"
echo "  corpus: $CORPUS_DIR"
echo "  db: mutated_files/single_json.db"
echo "  db: mutated_files/double_json.db"
echo "  db: mutated_files/triple_json.db"

