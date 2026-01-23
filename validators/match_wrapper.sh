#!/usr/bin/env bash
set -euo pipefail

# This wrapper dispatches to match.py based on its own name.
# Create symlinks named like: validate_date, validate_time, validate_url, etc.
# Usage: validators/validate_date <file_path>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$(basename "$0")"

case "$BASE" in
  validate_date*) CATEGORY="Date" ;;
  validate_iso8601*) CATEGORY="ISO8601" ;;
  validate_time*) CATEGORY="Time" ;;
  validate_url*) CATEGORY="URL" ;;
  validate_isbn*) CATEGORY="ISBN" ;;
  validate_ipv4*) CATEGORY="IPv4" ;;
  validate_ipv6*) CATEGORY="IPv6" ;;
  validate_pathfile*) CATEGORY="FilePath" ;;
  *)
    echo "Unknown validator name: $BASE" >&2
    exit 2
    ;;
esac

if [ $# -ne 1 ]; then
  echo "Usage: $BASE <file_path>" >&2
  exit 2
fi

# Strict mode for repair oracles: require full match (exit 0), partial (-1→255) is NOT success.
if [ "$CATEGORY" != "ISO8601" ] && [ -x "$SCRIPT_DIR/match_re2" ]; then
  exec "$SCRIPT_DIR/match_re2" "$CATEGORY" "$1"
else
  exec python3 "$SCRIPT_DIR/../match.py" "$CATEGORY" "$1"
fi
