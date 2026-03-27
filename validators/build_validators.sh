#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATORS_DIR="$ROOT_DIR/validators"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

: "${CXX:=}"
if [[ -z "$CXX" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
    CXX="$(xcrun --sdk macosx --find clang++ 2>/dev/null || true)"
  fi
  CXX="${CXX:-clang++}"
fi
STD_FLAGS=(-std=c++17)
OPT_FLAGS=(-O3 -DNDEBUG)

RE2_CFLAGS=()
RE2_LIBDIRS=()
RE2_LIBS=(-lre2)
RE2_OTHER=()

find_re2_prefix() {
  local prefix
  for prefix in "${RE2_PREFIX:-}" /opt/homebrew/opt/re2 /usr/local/opt/re2 /usr/local /usr; do
    [[ -n "$prefix" ]] || continue
    if [[ -f "$prefix/include/re2/re2.h" ]] && compgen -G "$prefix/lib/libre2*" >/dev/null; then
      printf '%s\n' "$prefix"
      return 0
    fi
  done
  return 1
}

have_re2() {
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists re2; then
    return 0
  fi
  [[ -n "$(find_re2_prefix || true)" ]]
}

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return $?
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return $?
  fi
  return 1
}

install_re2() {
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || die "RE2 is missing and Homebrew is not available. Install Homebrew or install RE2 manually."
      echo "[INFO] RE2 not found. Installing 're2' and 'pkg-config' with Homebrew..."
      HOMEBREW_NO_AUTO_UPDATE="${HOMEBREW_NO_AUTO_UPDATE:-1}" brew install re2 pkg-config
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] RE2 not found. Installing 'libre2-dev' with apt-get..."
        run_as_root apt-get update || die "RE2 is missing and apt-get update requires root access."
        run_as_root apt-get install -y libre2-dev pkg-config || die "Failed to install libre2-dev/pkg-config with apt-get."
      elif command -v dnf >/dev/null 2>&1; then
        echo "[INFO] RE2 not found. Installing 're2-devel' with dnf..."
        run_as_root dnf install -y re2-devel pkgconf-pkg-config || die "Failed to install re2-devel/pkgconf-pkg-config with dnf."
      elif command -v yum >/dev/null 2>&1; then
        echo "[INFO] RE2 not found. Installing 're2-devel' with yum..."
        run_as_root yum install -y re2-devel pkgconfig || die "Failed to install re2-devel/pkgconfig with yum."
      else
        die "RE2 is missing and no supported package manager was detected. Install RE2 manually or set RE2_PREFIX."
      fi
      ;;
    *)
      die "RE2 is missing and this platform is not supported for automatic installation. Install RE2 manually or set RE2_PREFIX."
      ;;
  esac
}

ensure_re2_installed() {
  if have_re2; then
    return 0
  fi
  install_re2
  have_re2 || die "RE2 installation finished but the library is still not discoverable. Set RE2_PREFIX if it was installed to a non-standard location."
}

ensure_re2_installed

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists re2; then
  # Use pkg-config for include/lib dirs (but keep libs minimal: just -lre2).
  # Homebrew's libre2 dylib already depends on abseil dylibs, so linking to -lre2 is enough.
  read -r -a RE2_CFLAGS <<<"$(pkg-config --cflags re2)"
  read -r -a RE2_LIBDIRS <<<"$(pkg-config --libs-only-L re2)"
  read -r -a RE2_OTHER <<<"$(pkg-config --libs-only-other re2)"
else
  RE2_PREFIX_FOUND="$(find_re2_prefix || true)"
  if [[ -z "$RE2_PREFIX_FOUND" ]]; then
    die "RE2 is required to build the regex validators. Install RE2 first (for example: 'brew install re2 pkg-config')."
  fi
  RE2_CFLAGS=(-I"$RE2_PREFIX_FOUND/include")
  if [[ -d "${RE2_PREFIX_FOUND%/re2}/abseil/include" ]]; then
    RE2_CFLAGS+=(-I"${RE2_PREFIX_FOUND%/re2}/abseil/include")
  fi
  RE2_LIBDIRS=(-L"$RE2_PREFIX_FOUND/lib")
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
