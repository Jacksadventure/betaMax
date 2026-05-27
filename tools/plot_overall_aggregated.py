from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    base = Path("time_limit_sweep_results")
    categories = ["date", "time", "isbn", "ipv4", "url", "ipv6"]
    algorithms = ["betamax", "erepair"]

    # Sum fixed/total across categories for each (algorithm, timeout).
    agg: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"fixed": 0, "total": 0})

    for category in categories:
        summary_path = base / f"double_{category}_full" / "summary.csv"
        if not summary_path.exists():
            continue

        with summary_path.open() as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            algorithm = row["algorithm"]
            if algorithm not in algorithms:
                continue
            timeout = int(row["timeout_s"])
            agg[(algorithm, timeout)]["fixed"] += int(row["fixed"])
            agg[(algorithm, timeout)]["total"] += int(row["total"])

    plt.figure(figsize=(9, 5.2))

    for algorithm in algorithms:
        points = sorted(
            (
                timeout,
                vals["fixed"] / vals["total"] if vals["total"] else 0.0,
            )
            for (alg, timeout), vals in agg.items()
            if alg == algorithm
        )
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=2.2, label=algorithm)

    plt.title("Overall Accuracy Across All Categories (Double Corruption)")
    plt.xlabel("Timeout (s)")
    plt.ylabel("Aggregated Accuracy (sum(fixed)/sum(total))")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out = base / "overall_aggregated_accuracy.png"
    plt.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
