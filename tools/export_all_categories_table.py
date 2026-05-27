from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    base = Path("time_limit_sweep_results")
    categories = ["date", "time", "isbn", "ipv4", "url", "ipv6"]
    timeouts = [5, 10, 30, 60, 120, 180, 240, 300]

    # category -> timeout -> algorithm -> accuracy
    acc: dict[str, dict[int, dict[str, float]]] = {
        c: {t: {} for t in timeouts} for c in categories
    }

    for category in categories:
        summary_path = base / f"double_{category}_full" / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            timeout = int(row["timeout_s"])
            if timeout not in acc[category]:
                continue
            acc[category][timeout][row["algorithm"]] = float(row["accuracy"])

    out_csv = base / "all_categories_accuracy_table.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["timeout_s"]
        for category in categories:
            header += [f"{category}_betamax_acc", f"{category}_erepair_acc"]
        writer.writerow(header)

        for timeout in timeouts:
            row: list[object] = [timeout]
            for category in categories:
                row.append(acc[category][timeout].get("betamax", ""))
                row.append(acc[category][timeout].get("erepair", ""))
            writer.writerow(row)

    out_md = base / "all_categories_accuracy_table.md"
    with out_md.open("w") as f:
        cols = ["Timeout(s)"]
        for category in categories:
            cols += [f"{category} betaMax", f"{category} eRepair"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

        for timeout in timeouts:
            vals = [str(timeout)]
            for category in categories:
                b = acc[category][timeout].get("betamax")
                e = acc[category][timeout].get("erepair")
                vals.append("" if b is None else f"{b:.2f}")
                vals.append("" if e is None else f"{e:.2f}")
            f.write("| " + " | ".join(vals) + " |\n")

    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
