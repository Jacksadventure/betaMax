from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    base = Path("time_limit_sweep_results")
    sources = [
        base / "double_url_full" / "summary.csv",
        base / "double_url_400_600" / "summary.csv",
    ]

    rows: list[dict[str, str]] = []
    for src in sources:
        rows.extend(read_summary(src))

    # Keep one row per (algorithm, timeout). If duplicated, keep the latest source row encountered.
    merged: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        algorithm = row.get("algorithm", "")
        timeout = int(row.get("timeout_s", "0"))
        merged[(algorithm, timeout)] = row

    algorithms = ["betamax", "erepair"]
    colors = {"betamax": "#1f77b4", "erepair": "#d62728"}

    plt.figure(figsize=(9, 5.5))
    for algorithm in algorithms:
        points = sorted(
            (
                timeout,
                float(row["accuracy"]),
            )
            for (alg, timeout), row in merged.items()
            if alg == algorithm
        )
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=2.2, color=colors[algorithm], label=algorithm)

    plt.title("URL Accuracy vs Timeout")
    plt.xlabel("Timeout (s)")
    plt.ylabel("Accuracy (fixed/total)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out = base / "url_accuracy_5_600.png"
    plt.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
