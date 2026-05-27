from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_series(base: Path, categories: list[str], algorithms: list[str]) -> dict[str, dict[str, tuple[list[int], list[float]]]]:
    series: dict[str, dict[str, tuple[list[int], list[float]]]] = {alg: {} for alg in algorithms}

    for category in categories:
        summary_path = base / f"double_{category}_full" / "summary.csv"
        if not summary_path.exists():
            continue

        with summary_path.open() as f:
            rows = list(csv.DictReader(f))

        for algorithm in algorithms:
            data = [r for r in rows if r["algorithm"] == algorithm]
            data.sort(key=lambda r: int(r["timeout_s"]))
            x = [int(r["timeout_s"]) for r in data]
            y = [float(r["accuracy"]) for r in data]
            series[algorithm][category] = (x, y)

    return series


def main() -> None:
    base = Path("time_limit_sweep_results")
    categories = ["date", "time", "isbn", "ipv4", "url", "ipv6"]
    algorithms = ["betamax", "erepair"]

    series = load_series(base, categories, algorithms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, algorithm in enumerate(algorithms):
        ax = axes[i]
        for category in categories:
            if category not in series[algorithm]:
                continue
            x, y = series[algorithm][category]
            ax.plot(x, y, marker="o", linewidth=2, label=category)

        ax.set_title(f"{algorithm} Accuracy vs Timeout")
        ax.set_xlabel("Timeout (s)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Accuracy (fixed/total)")
    axes[1].legend(title="Category", loc="lower right", fontsize=9)
    fig.suptitle("Double Corruption: All Categories in One Figure", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = base / "all_categories_accuracy.png"
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
