# betaMax (pure C++ implementation)

This folder contains a **standalone C++** betaMax-style implementation (separate from the original Python engine; no existing code is modified).

High-level flow:
- Learn a DFA from positive/negative examples (RPNI / Blue-Fringe).
- Search for candidate repairs with minimum edit cost that are accepted by the DFA.
- Validate candidates with the repository’s C++ validators (oracle) and return the first oracle-accepted repair.

---

## 1. Build

```bash
cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build betamax_cpp/build -j
```

Binary: `betamax_cpp/build/betamax_cpp`

---

## 2. Usage

### 2.1 Basic repair

Learn a DFA from `positives` and repair a single `broken` input:

```bash
./betamax_cpp/build/betamax_cpp \
  --positives positive/positives.txt \
  --category Date \
  --broken "2025-13-1a" \
  --max-cost 3
```

The program prints:
- Learned DFA statistics
- The first repair that passes both the DFA and the oracle (if found)

### 2.2 Provide negatives (optional)

```bash
./betamax_cpp/build/betamax_cpp \
  --positives /tmp/ipv4_pos.txt \
  --negatives negative/accum_ipv4.txt \
  --category IPv4 \
  --broken "179.44.157"
```

### 2.3 Choose / override the oracle (optional)

By default, based on `--category`, it searches in order:
- `validators/validate_<name>`
- `validators/regex/validate_<name>`
- If neither exists, it falls back to `python3 match.py <Category> <tempfile>` (only if `python3` is available)

You can override the oracle command (arguments supported). The temporary input file path is appended as the last argument:

```bash
./betamax_cpp/build/betamax_cpp \
  --positives positive/positives.txt \
  --category IPv4 \
  --oracle-validator "validators/validate_ipv4" \
  --broken "999.1.2.3"
```

---

## 3. Options

- `--positives <path>`: positives file (one string per line; empty line means epsilon)
- `--negatives <path>`: negatives file (optional)
- `--broken <string>` / `--broken-file <path>`: exactly one must be provided
- `--category <Date|ISO8601|Time|URL|ISBN|IPv4|IPv6|FilePath>`: selects the default oracle
- `--oracle-validator <cmd>`: overrides the oracle command (temp file path is appended)
- `--learner <rpni|rpni_xover>`: learner algorithm (default: `rpni`)
- `--xover-pairs <int>`: `rpni_xover` control (default: 50; `0` disables cross-over checks)
- `--xover-checks <int>`: per-merge oracle query budget for `rpni_xover` (default: 10; `0` disables)
- `--repo-root <path>`: repository root (default: `.`, used to locate `validators/`)
- `--max-cost <int>`: max edit cost (default: 3)
- `--max-candidates <int>`: max DFA-accepted candidates to try (default: 50)
- `--oracle-timeout-ms <int>`: oracle timeout (default: 3000ms)
- `--max-attempts <int>`: active-learning attempts (default: 500). If the closest repair is rejected by the oracle, it is added to negatives and the DFA is re-learned.
- `--attempt-candidates <int>`: max candidates to validate per attempt (default: 1)
- `--mutations <int>`: mutation augmentation count (default: 0). Generates mutated strings from positives, labels them with the oracle, adds accepted ones to positives and rejected ones to negatives.
- `--mutations-edits <int>`: number of random edits per mutated sample (default: 1)
- `--mutations-random`: random mutation generation (default)
- `--mutations-deterministic`: deterministic single-edit neighborhood enumeration (capped by `--mutations`)
- `--mutations-seed <uint64>`: RNG seed for random mutations (optional)
- `--eq-max-oracle <int>` / `--eq-max-rounds <int>`: optional oracle-guided sampling during learning (both default to `0`, i.e. disabled)
- `--eq-disable-sampling`: disables sampling (useful if you enabled it via `--eq-max-*`)
- `--eq-max-length <int>`: max sampled length (default: 10)
- `--eq-samples-per-length <int>`: samples per length (default: 20)
- `--seed <uint64>`: random seed (reproducible runs)
