#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATORS_DIR="$ROOT_DIR/validators"

CXX="${CXX:-clang++}"
STD_FLAGS=(-std=c++17)
OPT_FLAGS=(-O3 -DNDEBUG)

RE2_CFLAGS=()
RE2_LIBDIRS=()
RE2_LIBS=(-lre2)
RE2_OTHER=()

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists re2; then
  # Use pkg-config for include/lib dirs (but keep libs minimal: just -lre2).
  # Homebrew's libre2 dylib already depends on abseil dylibs, so linking to -lre2 is enough.
  read -r -a RE2_CFLAGS <<<"$(pkg-config --cflags re2)"
  read -r -a RE2_LIBDIRS <<<"$(pkg-config --libs-only-L re2)"
  read -r -a RE2_OTHER <<<"$(pkg-config --libs-only-other re2)"
else
  RE2_CFLAGS=(-I/opt/homebrew/opt/re2/include -I/opt/homebrew/opt/abseil/include)
  RE2_LIBDIRS=(-L/opt/homebrew/opt/re2/lib)
fi

echo "[INFO] Building validators with:"
echo "  CXX=$CXX"
echo "  FLAGS=${STD_FLAGS[*]} ${OPT_FLAGS[*]}"
echo "  RE2_CFLAGS=${RE2_CFLAGS[*]}"
echo "  RE2_LIBDIRS=${RE2_LIBDIRS[*]}"
echo "  RE2_LIBS=${RE2_LIBS[*]} ${RE2_OTHER[*]}"

shopt -s nullglob
SOURCES=("$VALIDATORS_DIR"/*.cpp)
if ((${#SOURCES[@]} == 0)); then
  echo "[ERROR] No C++ sources found in $VALIDATORS_DIR"
  exit 1
fi

for src in "${SOURCES[@]}"; do
  out="${src%.cpp}"
  echo "[BUILD] $(basename "$src") -> $(basename "$out")"
  "$CXX" "${STD_FLAGS[@]}" "${OPT_FLAGS[@]}" "${RE2_CFLAGS[@]}" "$src" "${RE2_LIBDIRS[@]}" "${RE2_LIBS[@]}" "${RE2_OTHER[@]}" -o "$out"
done

echo "[OK] Built ${#SOURCES[@]} validators."

