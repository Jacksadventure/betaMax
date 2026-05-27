from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    base = Path("time_limit_sweep_results")
    categories = ["date", "time", "isbn", "ipv4", "url", "ipv6"]
    algorithms = ["betamax", "erepair"]
    timeouts = [5, 10, 30, 60, 120, 180, 240, 300]

    sums: dict[tuple[int, str], dict[str, int]] = {
        (t, a): {"fixed": 0, "total": 0} for t in timeouts for a in algorithms
    }

    for category in categories:
        summary_path = base / f"double_{category}_full" / "summary.csv"
        if not summary_path.exists():
            continue

        with summary_path.open() as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            timeout = int(row["timeout_s"])
            algorithm = row["algorithm"]
            if timeout not in timeouts or algorithm not in algorithms:
                continue
            sums[(timeout, algorithm)]["fixed"] += int(row["fixed"])
            sums[(timeout, algorithm)]["total"] += int(row["total"])

    out_csv = base / "overall_aggregated_accuracy_table.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timeout_s",
                "betamax_fixed",
                "betamax_total",
                "betamax_acc",
                "erepair_fixed",
                "erepair_total",
                "erepair_acc",
            ]
        )
        for timeout in timeouts:
            b = sums[(timeout, "betamax")]
            e = sums[(timeout, "erepair")]
            bacc = b["fixed"] / b["total"] if b["total"] else 0.0
            eacc = e["fixed"] / e["total"] if e["total"] else 0.0
            writer.writerow(
                [
                    timeout,
                    b["fixed"],
                    b["total"],
                    f"{bacc:.4f}",
                    e["fixed"],
                    e["total"],
                    f"{eacc:.4f}",
                ]
            )

    out_md = base / "overall_aggregated_accuracy_table.md"
    with out_md.open("w") as f:
        f.write(
            "| Timeout(s) | betaMax fixed | betaMax total | betaMax acc | eRepair fixed | eRepair total | eRepair acc |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for timeout in timeouts:
            b = sums[(timeout, "betamax")]
            e = sums[(timeout, "erepair")]
            bacc = b["fixed"] / b["total"] if b["total"] else 0.0
            eacc = e["fixed"] / e["total"] if e["total"] else 0.0
            f.write(
                f"| {timeout} | {b['fixed']} | {b['total']} | {bacc:.4f} | {e['fixed']} | {e['total']} | {eacc:.4f} |\n"
            )

    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
