#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


REPO_ROOT = Path(__file__).resolve().parents[1]

MODE_TO_SCRIPT = {
    "single": "bm_single.py",
    "double": "bm_multiple.py",
    "triple": "bm_triple.py",
}

DEFAULT_FORMATS = ["date", "time", "isbn", "ipv4", "ipv6", "url"]
DEFAULT_TIMEOUTS = [5, 10, 30, 60, 120, 300]
DEFAULT_ALGORITHMS = ["betamax", "erepair"]


@dataclass(frozen=True)
class RunSpec:
    mode: str
    timeout_s: int
    db_path: Path
    log_path: Path


def _default_output_dir() -> Path:
    if Path("/results").is_dir():
        return Path("/results/time_limit_sweep")
    return Path("time_limit_sweep_results")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _nonnegative(values: list[int]) -> list[int]:
    return [value for value in values if value >= 0]


def _aggregate(mode: str, timeout_s: int, fmt: str, algorithm: str, values: list[tuple[int, float, int, int, int, int]]) -> dict[str, object]:
    total = len(values)
    fixed = sum(v[0] for v in values)
    timed_out = sum(v[2] for v in values)
    times = [v[1] for v in values]
    success_times = [v[1] for v in values if v[0]]
    input_distances = _nonnegative([v[3] for v in values])
    success_broken_repaired = _nonnegative([v[4] for v in values if v[0]])
    success_original_repaired = _nonnegative([v[5] for v in values if v[0]])
    return {
        "mode": mode,
        "timeout_s": timeout_s,
        "format": fmt,
        "algorithm": algorithm,
        "total": total,
        "fixed": fixed,
        "accuracy": fixed / total if total else 0.0,
        "timed_out": timed_out,
        "timeout_rate": timed_out / total if total else 0.0,
        "mean_repair_time_s": mean(times) if times else 0.0,
        "median_repair_time_s": median(times) if times else 0.0,
        "p95_repair_time_s": _quantile(times, 0.95),
        "mean_success_time_s": mean(success_times) if success_times else 0.0,
        "mean_input_edit_distance": mean(input_distances) if input_distances else 0.0,
        "mean_success_broken_repaired_distance": mean(success_broken_repaired) if success_broken_repaired else 0.0,
        "mean_success_original_repaired_distance": mean(success_original_repaired) if success_original_repaired else 0.0,
    }


def _summarize_db(mode: str, timeout_s: int, db_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conn = sqlite3.connect(db_path)
    try:
        timed_out_expr = "timed_out" if _has_column(conn, "results", "timed_out") else "0 AS timed_out"
        return_code_expr = "return_code" if _has_column(conn, "results", "return_code") else "NULL AS return_code"
        rows = conn.execute(
            f"""
            SELECT format, algorithm, fixed, repair_time, {timed_out_expr}, iterations, {return_code_expr},
                   distance_original_broken, distance_broken_repaired, distance_original_repaired
            FROM results
            """
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[tuple[str, str], list[tuple[int, float, int, int, int, int]]] = {}
    by_format_grouped: dict[tuple[str, str, str], list[tuple[int, float, int, int, int, int]]] = {}
    for fmt, algorithm, fixed, repair_time, timed_out, iterations, return_code, dist_ob, dist_br, dist_or in rows:
        was_attempted = bool(fixed) or bool(timed_out) or _safe_float(repair_time) > 0.0 or int(iterations or 0) > 0 or return_code is not None
        if not was_attempted:
            continue
        item = (
            int(fixed or 0),
            _safe_float(repair_time),
            int(timed_out or 0),
            int(dist_ob if dist_ob is not None else -1),
            int(dist_br if dist_br is not None else -1),
            int(dist_or if dist_or is not None else -1),
        )
        grouped.setdefault((mode, str(algorithm)), []).append(item)
        by_format_grouped.setdefault((mode, str(fmt), str(algorithm)), []).append(item)

    summary_rows = [
        _aggregate(mode, timeout_s, "ALL", algorithm, values)
        for (_mode, algorithm), values in sorted(grouped.items())
    ]
    format_rows = [
        _aggregate(mode, timeout_s, fmt, algorithm, values)
        for (_mode, fmt, algorithm), values in sorted(by_format_grouped.items())
    ]
    return summary_rows, format_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "timeout_s",
        "format",
        "algorithm",
        "total",
        "fixed",
        "accuracy",
        "timed_out",
        "timeout_rate",
        "mean_repair_time_s",
        "median_repair_time_s",
        "p95_repair_time_s",
        "mean_success_time_s",
        "mean_input_edit_distance",
        "mean_success_broken_repaired_distance",
        "mean_success_original_repaired_distance",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_float(value: object) -> str:
    return f"{float(value):.4f}"


def _write_markdown(path: Path, rows: list[dict[str, object]], *, output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Time Limit Sweep Summary",
        "",
        "This experiment sweeps the per-repair timeout and compares betaMax with eRepair on the same selected samples.",
        "",
        "## Configuration",
        "",
        f"- Modes: `{', '.join(args.modes)}`",
        f"- Formats: `{', '.join(args.formats)}`",
        f"- Algorithms: `{', '.join(args.algorithms)}`",
        f"- Timeouts: `{', '.join(str(t) for t in args.timeouts)}` seconds",
        f"- Sample limit per mode: `{args.limit if args.limit is not None else 'all available'}`",
        f"- Max workers: `{args.max_workers}`",
        "",
        "Accuracy is `fixed / total`, where `total` counts attempted rows only. Timeout rate is `timed_out / total`.",
        "",
        "## Overall Accuracy",
        "",
        "| mode | timeout_s | algorithm | total | fixed | accuracy | timeout_rate | mean_time_s | p95_time_s | input_ed | fix_ed | output_ed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {timeout_s} | {algorithm} | {total} | {fixed} | {accuracy} | {timeout_rate} | {mean} | {p95} | {input_ed} | {fix_ed} | {output_ed} |".format(
                mode=row["mode"],
                timeout_s=row["timeout_s"],
                algorithm=row["algorithm"],
                total=row["total"],
                fixed=row["fixed"],
                accuracy=_fmt_float(row["accuracy"]),
                timeout_rate=_fmt_float(row["timeout_rate"]),
                mean=_fmt_float(row["mean_repair_time_s"]),
                p95=_fmt_float(row["p95_repair_time_s"]),
                input_ed=_fmt_float(row["mean_input_edit_distance"]),
                fix_ed=_fmt_float(row["mean_success_broken_repaired_distance"]),
                output_ed=_fmt_float(row["mean_success_original_repaired_distance"]),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary CSV: `{output_dir / 'summary.csv'}`",
            f"- Per-format CSV: `{output_dir / 'by_format.csv'}`",
            f"- Accuracy plot: `{output_dir / 'accuracy_vs_timeout.png'}`",
            f"- Logs: `{output_dir / 'logs'}`",
            f"- SQLite DBs: `{output_dir / 'db'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not create plot: {exc}", file=sys.stderr)
        return

    by_series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        by_series.setdefault((str(row["mode"]), str(row["algorithm"])), []).append(
            (int(row["timeout_s"]), float(row["accuracy"]))
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for (mode, algorithm), points in sorted(by_series.items()):
        points = sorted(points)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=f"{mode}/{algorithm}",
        )
    ax.set_xlabel("Per-repair timeout (seconds)")
    ax.set_ylabel("Accuracy (fixed / total)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _run_spec(spec: RunSpec, args: argparse.Namespace) -> None:
    script = MODE_TO_SCRIPT[spec.mode]
    cmd = [
        args.python,
        script,
        "--db",
        str(spec.db_path),
        "--algorithms",
        *args.algorithms,
        "--formats",
        *args.formats,
        "--max-workers",
        str(args.max_workers),
        "--quiet",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit * len(args.algorithms))]

    env = dict(os.environ)
    env["BM_REPAIR_TIMEOUT"] = str(spec.timeout_s)
    env["LSTAR_EC_TIMEOUT"] = str(spec.timeout_s)
    env["PYTHONUNBUFFERED"] = "1"

    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] mode={spec.mode} timeout={spec.timeout_s}s db={spec.db_path}")
    with open(spec.log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n")
        log_handle.write(f"BM_REPAIR_TIMEOUT={spec.timeout_s}\n")
        log_handle.write(f"LSTAR_EC_TIMEOUT={spec.timeout_s}\n\n")
        log_handle.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=True)


def _check_prereqs(args: argparse.Namespace) -> None:
    erepair_path = REPO_ROOT / "erepair"
    if "erepair" in args.algorithms and not os.access(erepair_path, os.X_OK):
        raise SystemExit("[ERROR] ./erepair is required for --algorithms erepair. In Docker, rebuild the image from Dockerfile.")
    if "betamax" in args.algorithms and not (REPO_ROOT / "betamax_cpp/build/betamax_cpp").exists():
        raise SystemExit("[ERROR] betamax_cpp/build/betamax_cpp is required. Build it first or run inside the Docker image.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep per-repair time limits for betaMax vs eRepair.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--timeouts", nargs="+", type=int, default=DEFAULT_TIMEOUTS)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_TO_SCRIPT), default=["single", "double", "triple"])
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument("--algorithms", nargs="+", choices=["betamax", "erepair"], default=DEFAULT_ALGORITHMS)
    parser.add_argument("--limit", type=int, help="Optional sample cap per bm_* invocation for pilot runs.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1")))
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true", help="Remove the output directory before running.")
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing DBs in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    db_dir = output_dir / "db"
    log_dir = output_dir / "logs"

    if args.force and output_dir.exists() and not args.skip_runs:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        _check_prereqs(args)
        for mode in args.modes:
            for timeout_s in args.timeouts:
                spec = RunSpec(
                    mode=mode,
                    timeout_s=timeout_s,
                    db_path=db_dir / f"{mode}_t{timeout_s}.db",
                    log_path=log_dir / f"{mode}_t{timeout_s}.log",
                )
                _run_spec(spec, args)

    summary_rows: list[dict[str, object]] = []
    format_rows: list[dict[str, object]] = []
    for mode in args.modes:
        for timeout_s in args.timeouts:
            db_path = db_dir / f"{mode}_t{timeout_s}.db"
            if not db_path.exists():
                print(f"[WARN] Missing DB, skipping summary: {db_path}", file=sys.stderr)
                continue
            mode_summary, mode_by_format = _summarize_db(mode, timeout_s, db_path)
            summary_rows.extend(mode_summary)
            format_rows.extend(mode_by_format)

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "by_format.csv", format_rows)
    _write_markdown(output_dir / "summary.md", summary_rows, output_dir=output_dir, args=args)
    _plot_accuracy(output_dir / "accuracy_vs_timeout.png", summary_rows)
    print(f"[OK] Wrote sweep results to {output_dir}")


if __name__ == "__main__":
    main()