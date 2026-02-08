# betaMax

---

## 1. High‑level overview

betaMax provides:

1. **Repair engine (`betamax`)**
   - Learns a lightweight acceptor from examples (default: RPNI family learners) to guide repair.
   - Uses external validators (C++ binaries under `validators/`) as an oracle to confirm candidate repairs.

2. **Repair / validation experiments on real‑world like data**
   - Focus domains include:
     - Date/time
     - IPv4 / IPv6
     - URL
     - File paths
     - ISBN
   - Uses external validators (C++ binaries under `validators/`) as "oracles" to judge whether a candidate string is valid.

3. **Benchmark and mutation tooling**
   - Scripts to generate mutated (broken) inputs from known valid data.
   - Benchmarks to evaluate how well the learned automata support repair.

The core idea is to **learn an approximate model from examples** and then explore small edits to broken inputs, validating promising candidates with the oracle.

---

## 2. Repository layout (top‑level)

Important top‑level files and directories:

- `README.md`  
  This file.

- `requirements.txt`  
  Top‑level Python requirements. It currently delegates to:
  - `betamax/requirements-local.txt`  
    which installs local wheels from `betamax/py/`.

- **Benchmark / experiment scripts**
  - `bm_single.py` – run benchmarks on single‑error cases.
  - `bm_multiple.py` – benchmarks for multiple‑error settings.
  - `bm_triple.py` – benchmarks for triple‑error settings.
  - `bm_iso8601_mixed500.py` – dedicated ISO8601 mixed benchmark (train=400/test=100) with **no precompute timeout**.
  - `warmup.py` – warm‑up runs and cache precomputation.
  - `mutation_single.py`, `mutation_double.py`, `mutation_triple.py`, `mutation_truncated.py` – mutation generators for building broken inputs from seed data.
  - `report.py` – reporting/aggregation utilities for benchmark outputs.

- **Data & results**
  - `data/` – combined and per‑domain positive data (e.g. `data/date`, `data/ipv4`, `data/url`, ...).
  - `data/iso8601_500_mixed.txt` – shuffled ISO8601/time mixed dataset used by `bm_iso8601_mixed500.py`.
  - `mutated_files/` – mutated DB files (single/double/triple/truncated variants) used as broken inputs.
  - `repair_results/` – repair output and logs produced by experiments.

- **Core engine and learning utilities**
  - `betamax/`
    - `pyproject.toml`, `requirements-local.txt`
    - `app/betamax.py` – CLI entrypoint (forwards into the implementation under `betamax/lstar/`).
    - `py/` – vendored wheels used by the runtime (installed via `betamax/requirements-local.txt`).
    - `lstar/`
      - `betamax.py` – core repair implementation (invoked via `betamax/app/betamax.py`).
      - `rpni.py`, `rpni_xover.py`, `rpni_fuzz.py`, `rpni_nfa.py` – example-driven learner variants used by the engine.
      - `ec_runtime.py` – runtime support for experiments (data loading, logging, helpers).
  - `betamax_cpp/`
    - Standalone **pure C++** engine (RPNI / rpni_xover + edit-distance repair + validator oracle).
    - Build with CMake; see `betamax_cpp/README.md`.

- **Validators**
  - `validators/` – C++ validators and wrappers used as domain‑specific oracles:
    - `validate_date`, `validate_ipv4`, `validate_ipv6`, `validate_url`, `validate_isbn`, `validate_pathfile`, `validate_time`, ...
    - `validate_iso8601_mixed` – wrapper oracle for the mixed ISO8601/time dataset.
    - `match_re2`, `re2_server`, `match_wrapper.sh`, etc.

- **Legacy / experimental**
  - `fuzzer.cpp` and other C++ utilities.
  - Older C++ repair engine `earlyrepairer.cpp` and related binaries (kept for reference; primary development is now in the betaMax engine implemented in `betamax/app/betamax.py`).

---

## 3. Installation

### 3.1. Prerequisites

- Python **3.11+**
- A C++17 compiler (for building validators or legacy components if needed)
- CMake **3.16+** (only needed for `betamax_cpp/`)
- On macOS/Linux, typical build tools (`make`, `g++` or `clang++`) are useful.

### 3.2. Clone and install Python dependencies

```bash
git clone git@github.com:Jacksadventure/EarlyRepairer.git
cd EarlyRepairer

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# On Windows (PowerShell): .\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To leave the virtual environment, run:

```bash
deactivate
```

`requirements.txt` includes:

```text
-r betamax/requirements-local.txt
```

and that file installs a set of local wheels from `betamax/py/`, including:

- `simplefuzzer`
- `rxfuzzer`
- `earleyparser`
- `cfgrandomsample`
- `cfgremoveepsilon`
- plus helper libraries (`graphviz`, `pyparsing`, `pydot`, `sympy`, `mpmath`) pinned to versions matching the wheels.

---

## 4. Quick start with betaMax

Below is a typical workflow for running experiments with the betaMax engine.

### 4.1. Warm up and precompute caches

Many scripts rely on cached data under `cache/` (e.g. learned automata, negative samples).  
It is recommended to run:

```bash
python warmup.py
```

This will:

- Precompute caches for several domains (date, time, IPv4/IPv6, URL, ISBN, pathfile, etc.).
- Exercise the validators to ensure they are callable in the current environment.

### 4.2. Run single‑error benchmarks

```bash
python bm_single.py
```

This script typically:

- Loads seed data/DBs (e.g. from `data/` or `mutated_files/`).
- Runs betaMax‑based repair experiments on single‑error inputs.
- Stores outputs and logs under `repair_results/`.

### 4.3. Run multiple/triple‑error benchmarks

```bash
python bm_multiple.py
python bm_triple.py
```

These evaluate the robustness of the learned automata and repair strategy when the inputs are further from the training distribution (multiple edits/broken positions).

### 4.4. Inspecting results

Results are written as text files under:

- `repair_results/`
- or nested `validators/repair_results/`, depending on the script.

You'll typically see files like:

- `repair_XXX_output.date`
- `repair_XXX_output.triple_ipv4`
- `repair_XXX_output.triple_ipv6`
- etc.

Each file contains:

- The original broken input.
- The proposed repaired candidate(s).
- Validator outcomes and statistics (e.g., number of oracle calls).

### 4.5. Using `betamax.py` directly

The benchmark scripts (`bm_single.py`, `bm_multiple.py`, `bm_triple.py`) are thin wrappers around the core engine in `betamax/app/betamax.py`.  
You can also call the engine directly for ad‑hoc experiments.

#### Switching benchmarks to the C++ engine

By default, `bm_single.py` / `bm_multiple.py` / `bm_triple.py` use the **C++** engine (under `betamax_cpp/`) for `algorithm=betamax`.

Build it first:

```bash
cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build betamax_cpp/build -j
```

To explicitly select the C++ engine:

```bash
python bm_single.py --betamax-engine cpp
python bm_multiple.py --betamax-engine cpp
python bm_triple.py --betamax-engine cpp
```

Or via environment variable:

```bash
BM_BETAMAX_ENGINE=cpp python bm_single.py
```

Useful C++ knobs:
- `BM_BETAMAX_CPP_BIN` (default: `betamax_cpp/build/betamax_cpp`)
- `BM_CPP_MAX_COST` (default: `3`)
- `BM_CPP_MAX_CANDIDATES` (default: `50`)
- Note: benchmark scripts pass their existing betaMax `--mutations` value to the C++ engine as well (mutation-based augmentation is supported).

To switch back to the original Python engine:

```bash
python bm_single.py --betamax-engine python
```

#### Basic single‑string repair

```bash
python betamax/app/betamax.py \
  --positives positive/positives.txt \
  --category Date \
  --broken "2025-13-1a" \
  --output-file repaired.txt
```

This will:

- Learn a grammar from `positive/positives.txt` using the selected learner (default: `rpni`).
- Use the Date validator (`validators/validate_date` or `validators/regex/validate_date`) as the oracle.
- Attempt to repair the single broken input `2025-13-1a`.
- Write the repaired string into `repaired.txt` (and print logs to stdout).

You can alternatively pass a file:

```bash
python betamax/app/betamax.py \
  --positives positive/positives.txt \
  --category Date \
  --broken-file path/to/broken_input.txt \
  --output-file repaired.txt
```

Use only one of `--broken` or `--broken-file`.

#### Reusing a learned grammar (cache)

For repeated runs on many broken inputs, use a cache:

```bash
# First time: learn grammar and save cache
python betamax/app/betamax.py \
  --positives positive/positives.txt \
  --category Date \
  --grammar-cache cache/date_grammar.json \
  --init-cache

# Later runs: reuse cached grammar, only repair inputs
python betamax/app/betamax.py \
  --grammar-cache cache/date_grammar.json \
  --category Date \
  --broken "2025-13-1a" \
  --output-file repaired.txt
```

#### Switching learners and controlling repair behavior

- Choose learner:

  ```bash
  --learner rpni           # default (passive RPNI)
  --learner rpni_nfa       # NFA-based RPNI
  --learner rpni_fuzz      # RPNI + fuzz consistency
  --learner rpni_xover     # RPNI + cross-over consistency
  ```

- Control how aggressive repairs are:

  ```bash
  --penalty 3        # target correction penalty (0–8, capped at 8)
  ```

- Log more detail from the error-correcting Earley parser:

  ```bash
  --log
  ```

See `betamax/app/betamax.py` docstring and `--help` output for the full list of options.

---

## 5. The betaMax engine in more detail

The core engine lives in `betamax/app/betamax.py` and uses several supporting modules:

- `rpni.py` / `rpni_xover.py` / `rpni_fuzz.py` / `rpni_nfa.py` – construct and generalise automata from example sets.
- `ec_runtime.py` – glue code for experiments (loading data, running teachers/oracles, logging).

A typical high‑level flow is:

1. **Data preparation**
   - Read positive examples from `positive/*.txt` (e.g. dates, URLs).
   - Optionally read negative examples from `negative/*.txt` (if present) or construct them via mutation.

2. **Learning a DFA**
   - Construct an automaton from positive examples (and optional negatives), using the selected learner.
   - Optionally apply generalisation/consistency checks to reduce overfitting to the finite sample set.

3. **Repair / validation**
   - For each broken input (e.g. from `mutated_files/*.db`):
     - Propose edits or candidate strings.
     - Filter them using the learned DFA (cheap, approximate check).
     - Confirm promising candidates with the true oracle (validators/).
   - Record the successful repairs and statistics.

For further details, consult the docstrings and comments inside:

- `betamax/app/betamax.py`
- `betamax/lstar/betamax.py`

---

## 6. Development notes

- If you change or add validators under `validators/`, ensure they are built and executable:
  - e.g. `g++ -std=c++17 -O2 -o validators/validate_date validators/validate_date.cpp`
- New domains can be added by:
  1. Providing a validator (oracle) for the new language.
  2. Supplying positive examples (and optionally negatives).
  3. Extending the benchmark scripts or adding new ones following the existing patterns.
