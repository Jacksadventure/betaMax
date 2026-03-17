# betaMax

Benchmark runners and experiment scripts for betaMax / eRepair-style repair experiments.

## What This Checkout Supports

This repository snapshot is organized around the **C++ betaMax backend** in [betamax_cpp](./betamax_cpp).

The easiest path after cloning is:

1. Install the small Python runtime from [requirements.txt](./requirements.txt).
2. Run [run_bms.sh](./run_bms.sh).
3. Let the script build `betamax_cpp/build/betamax_cpp` automatically if it is missing.

What works cleanly in this checkout:

- Main regex benchmarks: `date`, `time`, `isbn`, `ipv4`, `ipv6`, `url`

This README focuses on the regex benchmark workflow.

Important:

- The benchmark scripts still expose `--betamax-engine python` for legacy compatibility.
- The legacy Python backend entrypoint `betamax/app/betamax.py` is **not bundled** in this checkout.
- Use `cpp` unless you have a separate legacy checkout and set `BM_BETAMAX_PY_ENTRYPOINT`.

## Quick Start

Clone the repository, create a virtual environment, and install the runtime:

```bash
git clone <repo-url> betaMax
cd betaMax

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a small smoke benchmark:

```bash
./run_bms.sh quick
```

This default smoke run:

- builds `betamax_cpp/build/betamax_cpp` automatically if needed
- runs a small single-mutation benchmark
- writes results to `smoke_single.db`

Run the main regex benchmark suite:

```bash
./run_bms.sh regex
```

This runs:

- `bm_single.py`
- `bm_multiple.py`
- `bm_triple.py`

with the default regex formats and writes:

- `single.db`
- `double.db`
- `triple.db`

If you want separate DB names for a fresh run, use `DB_PREFIX`:

```bash
DB_PREFIX=trial1 ./run_bms.sh regex
```

That produces:

- `trial1_single.db`
- `trial1_double.db`
- `trial1_triple.db`

## Main Entry Points

### betaMax benchmarks

Use [run_bms.sh](./run_bms.sh):

```bash
./run_bms.sh quick
./run_bms.sh single
./run_bms.sh double
./run_bms.sh triple
./run_bms.sh regex
```

Examples:

```bash
./run_bms.sh single --formats date url --limit 10
DB_PREFIX=trial2 ./run_bms.sh regex
```

Notes:

- `regex` runs the main single/double/triple regex suite.
- For `regex`, prefer `DB_PREFIX=...` instead of `--db`, because multiple benchmark scripts are executed.

## Makefile Shortcuts

If you prefer `make`, the top-level [Makefile](./Makefile) exposes the common commands:

```bash
make help
make install
make quick
make single
make double
make triple
make regex
```

Useful variables:

```bash
PYTHON=python3 make install
DB_PREFIX=trial3 make regex
MAX_WORKERS=4 make quick
```

## Optional Native Validators

For regex benchmarks, native validators under [validators](./validators) are optional.

If you want them:

```bash
./validators/build_validators.sh
```

Without these binaries, the benchmark scripts fall back to the Python regex oracle in [match.py](./match.py).

## Outputs

During runs, the repository writes to:

- `single.db`, `double.db`, `triple.db`
- `repair_results/` for per-sample outputs and jar materialization
- `cache/` for learned caches / DFA artifacts

The mutation corpora consumed by the benchmark runners live in:

- `mutated_files/`

Source corpora and original files live in:

- `data/`
- `original_files/`

## Reporting

The default summary script is [report.py](./report.py):

```bash
python report.py
```

It prints summary tables based on the default DB names such as `single.db`, `double.db`, and `triple.db`.

If you ran with `DB_PREFIX`, either rename/copy the DBs you want to inspect or query them directly with SQLite / your own analysis script.

## Repository Map

Key files and directories:

- [run_bms.sh](./run_bms.sh): main betaMax benchmark launcher
- [Makefile](./Makefile): common shortcuts
- [bm_single.py](./bm_single.py): single-mutation runner
- [bm_multiple.py](./bm_multiple.py): double-mutation runner
- [bm_triple.py](./bm_triple.py): triple-mutation runner
- [betamax_cpp](./betamax_cpp): C++ betaMax backend
- [validators](./validators): optional native regex validators
- [mutated_files](./mutated_files): mutation databases used as benchmark input
- [repair_results](./repair_results): per-run outputs
- [cache](./cache): precompute cache artifacts

## Troubleshooting

### `Missing Python module 'regex'`

Install the runtime:

```bash
python -m pip install -r requirements.txt
```

### `betaMax C++ binary not found`

Either run:

```bash
./run_bms.sh quick
```

or build it explicitly:

```bash
cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build betamax_cpp/build -j
```

### I want custom benchmark arguments

`run_bms.sh` forwards extra arguments to the underlying `bm_*.py` scripts.

Examples:

```bash
./run_bms.sh single --formats date --limit 5 --resume
```
