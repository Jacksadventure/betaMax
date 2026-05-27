#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    base = Path("time_limit_sweep_results")
    sources = [
        base / "double_url_full" / "summary.csv",
        base / "double_url_400_600" / "summary.csv",
    ]

    rows: list[dict[str, str]] = []
    for path in sources:
        if not path.exists():
            continue
        with path.open() as f:
            rows.extend(list(csv.DictReader(f)))

    # URL summaries are already format=ALL in those files; keep mode=double and dedupe by (timeout, algorithm).
    filtered = [r for r in rows if r.get("mode") == "double" and r.get("format") == "ALL"]
    latest: dict[tuple[int, str], dict[str, str]] = {}
    for r in filtered:
        key = (int(r["timeout_s"]), str(r["algorithm"]))
        latest[key] = r

    keys = sorted(latest.keys(), key=lambda x: (x[0], x[1]))

    csv_out = base / "url_timeout_distance_table.csv"
    md_out = base / "url_timeout_distance_table.md"

    header = [
        "timeout_s",
        "algorithm",
        "fixed",
        "total",
        "accuracy",
        "timeout_rate",
        "mean_input_edit_distance",
        "mean_success_broken_repaired_distance",
        "mean_success_original_repaired_distance",
    ]

    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t, alg in keys:
            r = latest[(t, alg)]
            writer.writerow(
                [
                    t,
                    alg,
                    r["fixed"],
                    r["total"],
                    r["accuracy"],
                    r["timeout_rate"],
                    r["mean_input_edit_distance"],
                    r["mean_success_broken_repaired_distance"],
                    r["mean_success_original_repaired_distance"],
                ]
            )

    with md_out.open("w") as f:
        f.write(
            "| timeout_s | algorithm | fixed/total | acc | timeout_rate | input_ed | broken_repaired_ed | original_repaired_ed |\n"
        )
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for t, alg in keys:
            r = latest[(t, alg)]
            f.write(
                "| {t} | {alg} | {fixed}/{total} | {acc:.4f} | {to:.4f} | {inp:.4f} | {br:.4f} | {or_: .4f} |\n".format(
                    t=t,
                    alg=alg,
                    fixed=int(r["fixed"]),
                    total=int(r["total"]),
                    acc=float(r["accuracy"]),
                    to=float(r["timeout_rate"]),
                    inp=float(r["mean_input_edit_distance"]),
                    br=float(r["mean_success_broken_repaired_distance"]),
                    or_=float(r["mean_success_original_repaired_distance"]),
                )
            )

    print(csv_out.resolve())
    print(md_out.resolve())


if __name__ == "__main__":
    main()
