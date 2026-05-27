#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_MAP = {
    "broken_repaired": "mean_success_broken_repaired_distance",
    "original_repaired": "mean_success_original_repaired_distance",
    "input": "mean_input_edit_distance",
}

MODE_ORDER = {"single": 1, "double": 2, "triple": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Levenshtein distances across corruption levels (single/double/triple) "
            "from one or more sweep summary.csv files."
        )
    )
    parser.add_argument(
        "--summary-files",
        nargs="+",
        required=True,
        help="One or more sweep summary.csv files.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_MAP.keys()),
        default="broken_repaired",
        help="Which distance metric to compare across corruption levels.",
    )
    parser.add_argument(
        "--format",
        default="ALL",
        help="Format filter from summary.csv (default: ALL).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout to evaluate. If omitted, uses the largest available timeout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("time_limit_sweep_results"),
        help="Directory for generated table and plot outputs.",
    )
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def to_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def pick_timeout(rows: list[dict[str, str]], requested: int | None) -> int:
    timeouts = sorted({to_int(r.get("timeout_s", "0")) for r in rows})
    if not timeouts:
        raise SystemExit("[ERROR] No timeout values found in provided summaries.")
    if requested is None:
        return timeouts[-1]
    if requested not in timeouts:
        raise SystemExit(f"[ERROR] Requested timeout {requested}s not found. Available: {timeouts}")
    return requested


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, matrix: dict[str, dict[str, float]], title: str, ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Plot skipped: {exc}")
        return

    levels = ["single", "double", "triple"]
    x = [MODE_ORDER[level] for level in levels]

    plt.figure(figsize=(8, 5))
    for algorithm in sorted(matrix.keys()):
        y = [matrix[algorithm].get(level, float("nan")) for level in levels]
        plt.plot(x, y, marker="o", linewidth=2, label=algorithm)

    plt.xticks(x, levels)
    plt.title(title)
    plt.xlabel("Corruption level")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_paths = [Path(p).resolve() for p in args.summary_files]
    for p in summary_paths:
        if not p.exists():
            raise SystemExit(f"[ERROR] Missing summary file: {p}")

    metric_col = METRIC_MAP[args.metric]
    rows = read_rows(summary_paths)

    # Keep only rows usable for level comparison.
    filtered = [
        r
        for r in rows
        if r.get("mode") in MODE_ORDER
        and r.get("format") == args.format
        and r.get("algorithm")
        and to_int(r.get("total", "0")) > 0
    ]

    if not filtered:
        raise SystemExit("[ERROR] No matching rows after filtering by mode/format.")

    timeout_s = pick_timeout(filtered, args.timeout)
    filtered = [r for r in filtered if to_int(r.get("timeout_s", "0")) == timeout_s]

    # matrix[algorithm][mode] = metric
    matrix: dict[str, dict[str, float]] = {}
    for r in filtered:
        algorithm = str(r["algorithm"])
        mode = str(r["mode"])
        matrix.setdefault(algorithm, {})[mode] = to_float(r.get(metric_col, "0"))

    header = ["algorithm", "single", "double", "triple"]
    table_rows: list[list[object]] = []
    for algorithm in sorted(matrix.keys()):
        row: list[object] = [algorithm]
        for level in ["single", "double", "triple"]:
            val = matrix[algorithm].get(level)
            row.append("" if val is None else f"{val:.4f}")
        table_rows.append(row)

    out_dir = args.output_dir.resolve()
    stem = f"levenshtein_{args.metric}_by_corruption_t{timeout_s}"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    png_path = out_dir / f"{stem}.png"

    write_csv(csv_path, header, table_rows)
    write_md(md_path, header, table_rows)
    write_plot(
        png_path,
        matrix,
        title=f"Levenshtein across corruption levels ({args.metric}, {args.format}, t={timeout_s}s)",
        ylabel=metric_col,
    )

    print(csv_path)
    print(md_path)
    print(png_path)


if __name__ == "__main__":
    main()
