# betaMax

Benchmark runners and experiment scripts for betaMax / eRepair-style repair experiments.

## What This Checkout Supports

This repository snapshot is organized around the **C++ betaMax backend** in [betamax_cpp](./betamax_cpp).

The easiest path after cloning is:

1. Install the small Python runtime from [requirements.txt](./requirements.txt).
2. Run [run_bms.sh](./run_bms.sh).
3. Let the script bootstrap `clang++`/RE2 when possible, then build `betamax_cpp/build/betamax_cpp` and `validators/validate_*` automatically if they are missing.

What works cleanly in this checkout:

- Main regex benchmarks: `date`, `time`, `isbn`, `ipv4`, `ipv6`, `url`

This README focuses on the regex benchmark workflow.

Important:

- This checkout supports the bundled C++ betaMax backend only.

## Dependencies

For the main regex benchmark workflow (`run_bms.sh quick|single|double|triple|regex`), the required system dependencies are:

- Python 3.10+
- `pip` and `venv` for the recommended isolated environment setup
- CMake 3.16+ to build the bundled C++ backend in [betamax_cpp](./betamax_cpp)
- A C++17 compiler such as `clang++`
- RE2 headers and libraries, because the benchmark automation uses native RE2-backed validators under [validators](./validators)
- The pre-generated mutation databases under [mutated_files](./mutated_files), which the benchmark runners expect to exist

The Python packages in [requirements.txt](./requirements.txt) are split by purpose:

- `regex`: required by the Python regex helper oracles in [match.py](./match.py) and [match_partial.py](./match_partial.py)
- `matplotlib`: only needed for reporting via [report.py](./report.py)
- `requests`: only needed for corpus/data download via [data_fetch.py](./data_fetch.py)

Optional or workflow-specific dependencies:

- `make`: only needed if you want the shortcuts in [Makefile](./Makefile)
- `pkg-config`: optional, but helps [validators/build_validators.sh](./validators/build_validators.sh) locate RE2 cleanly

The standard regex quickstart in this checkout does **not** require Java, Gradle, or the optional subject-build toolchain.

For `./run_bms.sh quick`, the launcher now tries to bootstrap missing native dependencies automatically:

- missing `clang++`:
  on macOS it prefers `brew install llvm`; if Homebrew is unavailable but `xcode-select` exists, it requests Xcode Command Line Tools installation and asks you to rerun after that completes
- missing RE2:
  on macOS it runs `brew install re2 pkg-config`
  on Linux it uses `apt-get`, `dnf`, or `yum` when available

Automatic installation still depends on a usable package manager and, on Linux, sufficient privileges.

## Quick Start

After the dependencies above are available, clone the repository, create a virtual environment, and install the Python runtime:

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

- installs `clang++` and RE2 automatically when supported package-manager access is available
- builds `betamax_cpp/build/betamax_cpp` automatically if needed
- builds all native regex validators automatically
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

## Regex Validators

For regex benchmarks, native validators under [validators](./validators) are required. The benchmark automation passes `validators/validate_*` into betaMax as the oracle, so RE2 must be installed.

Build them manually if you want to prepare the environment ahead of time:

```bash
./validators/build_validators.sh
```

If they are missing, [run_bms.sh](./run_bms.sh) and [run_ddmax.sh](./run_ddmax.sh) try to build them automatically before starting the regex benchmarks. `run_bms.sh quick` now rebuilds the full validator set up front, and [validators/build_validators.sh](./validators/build_validators.sh) will also try to install missing `clang++` / RE2 dependencies when a supported package manager is available.

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
- [validators](./validators): native RE2 regex validators used by benchmark automation
- [mutated_files](./mutated_files): mutation databases used as benchmark input
- [repair_results](./repair_results): per-run outputs
- [cache](./cache): precompute cache artifacts

## Troubleshooting

### `Missing Python module 'regex'`

Install the runtime:

```bash
python -m pip install -r requirements.txt
```

### `Missing native regex validator 'validators/validate_*'`

Either rerun:

```bash
./run_bms.sh quick
```

to let the launcher try automatic installation/build, or install RE2 yourself and build the validators explicitly:

```bash
./validators/build_validators.sh
```

If automatic installation is unavailable, check that Homebrew (`brew`) is installed on macOS, or that `apt-get` / `dnf` / `yum` plus the required privileges are available on Linux.

### `clang++ not found`

Either rerun:

```bash
./run_bms.sh quick
```

to let the launcher try automatic installation, or install the compiler toolchain yourself:

- macOS: `brew install llvm`, or install Xcode Command Line Tools
- Linux: install `clang` with your system package manager

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
