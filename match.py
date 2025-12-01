import sys
import argparse

# Prefer Google RE2 bindings if available for safe/linear-time regex.
try:
    import re2 as re  # pyre2
    USING_RE2 = True
except Exception:
    import re  # fallback to stdlib if RE2 not installed
    USING_RE2 = False

# Define regular expressions for the formats (anchored for full-string match)
patterns = {
    "Date": r"^\d{4}-\d{2}-\d{2}$",  # Format: YYYY-MM-DD
    "Time": r"^\d{2}:\d{2}:\d{2}$",  # Format: HH:MM:SS
    "URL":  r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",  # Anchored for full match
    "ISBN": r"^(?:\d[- ]?){9}[\dX]$",  # ISBN-10
    "IPv4": r"^(\d{1,3}\.){3}\d{1,3}$",
    "IPv6": r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$",
    "FilePath": r"^[a-zA-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*$"
}

def validate_data_full(pattern, data):
    """Validate the data requiring a full match. Returns 0 on full match, 1 otherwise."""
    # Using anchors in patterns ensures match() implies full-string match.
    match = re.match(pattern, data)
    return 0 if match else 1

def main():
    parser = argparse.ArgumentParser(description="Validate file content using RE2 full-match semantics.")
    parser.add_argument("category", choices=patterns.keys(), help="Category to validate against.")
    parser.add_argument("file_path", help="Path to the input file.")
    args = parser.parse_args()

    # Read the file
    try:
        with open(args.file_path, "r") as file:
            data = file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Inform about fallback if RE2 is not available
    if not USING_RE2:
        # Stderr only to avoid affecting pipelines that read stdout
        print("Warning: Python re2 module not found; falling back to stdlib 're'. Install RE2 with: pip install re2", file=sys.stderr)

    # Validate data (full match required)
    pattern = patterns[args.category]
    result = validate_data_full(pattern, data)
    sys.exit(result)

if __name__ == "__main__":
    main()
