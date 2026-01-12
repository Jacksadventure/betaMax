# betaMax

---

## 1. High‑level overview

betaMax provides:

1. **L\* / RPNI‑based learning core (`betaMax`)**
   - Uses observation tables and membership/equivalence queries to learn a DFA for a target regular language.
   - Can combine L\* with RPNI‑style generalisation (see `rpni.py`, `rpni_xover.py`, `rpni_fuzz.py`).

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

The core idea is to **approximate the target language from examples** (via DFA learning) and then explore small edits to broken inputs that move them into the learned language and pass the oracle.

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
  - `warmup.py` – warm‑up runs and cache precomputation.
  - `mutation_single.py`, `mutation_double.py`, `mutation_triple.py`, `mutation_truncated.py` – mutation generators for building broken inputs from seed data.
  - `report.py` – reporting/aggregation utilities for benchmark outputs.

- **Data & results**
  - `data/` – combined and per‑domain positive data (e.g. `data/date`, `data/ipv4`, `data/url`, ...).
  - `mutated_files/` – mutated DB files (single/double/triple/truncated variants) used as broken inputs.
  - `repair_results/` – repair output and logs produced by experiments.

- **L\* learner and core repair engine**
  - `betamax/`
    - `pyproject.toml`, `requirements-local.txt`
    - `lstar/`
      - `betamax.py` – **main L\*‑based repair engine (betaMax core)**.
      - `observation_table.py` – observation table implementation used by L\*.
      - `rpni.py`, `rpni_xover.py`, `rpni_fuzz.py` – RPNI and related generalisation / fuzzing utilities.
      - `ec_runtime.py` – runtime support for experiments.
      - `README.md` – additional documentation for the L\* library itself.

- **Validators**
  - `validators/` – C++ validators and wrappers used as domain‑specific oracles:
    - `validate_date`, `validate_ipv4`, `validate_ipv6`, `validate_url`, `validate_isbn`, `validate_pathfile`, `validate_time`, ...
    - `match_re2`, `re2_server`, `match_wrapper.sh`, etc.

- **Legacy / experimental**
  - `fuzzer.cpp` and other C++ utilities.
  - Older C++ repair engine `earlyrepairer.cpp` and related binaries (kept for reference; primary development is now in the betaMax engine implemented in `betamax/app/betamax.py`).

---

## 3. Installation

### 3.1. Prerequisites

- Python **3.11+**
- A C++17 compiler (for building validators or legacy components if needed)
- On macOS/Linux, typical build tools (`make`, `g++` or `clang++`) are useful.

### 3.2. Clone and install Python dependencies

```bash
git clone git@github.com:Jacksadventure/EarlyRepairer.git
cd EarlyRepairer

# Recommended: create a virtualenv here

pip install -r requirements.txt
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

Below is a typical workflow for running experiments with the main L\*‑based engine. 

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
  --learner lstar_oracle   # L* + oracle
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

- `observation_table.py` – maintains the L\* observation table (rows = prefixes, columns = suffixes).
- `rpni.py` / `rpni_xover.py` – construct and generalise automata from example sets.
- `ec_runtime.py` – glue code for experiments (loading data, running teachers/oracles, logging).

A typical high‑level flow is:

1. **Data preparation**
   - Read positive examples from `positive/*.txt` (e.g. dates, URLs).
   - Optionally read negative examples from `negative/*.txt` (if present) or construct them via mutation.

2. **Learning a DFA**
   - Construct an observation table from membership queries to an oracle (e.g. the C++ validators).
   - Repeatedly refine the table until a consistent, closed DFA hypothesis is formed.
   - Optionally apply RPNI/generalisation to reduce overfitting to the finite sample set.

3. **Repair / validation**
   - For each broken input (e.g. from `mutated_files/*.db`):
     - Propose edits or candidate strings.
     - Filter them using the learned DFA (cheap, approximate check).
     - Confirm promising candidates with the true oracle (validators/).
   - Record the successful repairs and statistics.

For further details, consult the docstrings and comments inside:

- `betamax/app/betamax.py`
- `betamax/lstar/README.md`

---

## 6. Development notes

- If you change or add validators under `validators/`, ensure they are built and executable:
  - e.g. `g++ -std=c++17 -O2 -o validators/validate_date validators/validate_date.cpp`
- New domains can be added by:
  1. Providing a validator (oracle) for the new language.
  2. Supplying positive examples (and optionally negatives).
  3. Extending the benchmark scripts or adding new ones following the existing patterns.
