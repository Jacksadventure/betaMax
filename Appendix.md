ϵREPAIR on program inputs defined by context-free grammar (e.g. JSON)
```
———— general ———————————————

Metrics for single.db
-------------------------------------------------------------------------------------------------------------------------------------------
Format   Alg        Avg BR     σ BR   Avg OR     σ OR Avg RecO   σ RecO Avg RecB   σ RecB    Avg t      σ t   Avg t*     σ t*   Succ    Tot
-------------------------------------------------------------------------------------------------------------------------------------------
single_json betamax      2.12     1.42     1.42     1.50     0.97     0.06     0.97     0.06     4.79     9.80     4.79     9.80     50     50

Metrics for double.db
-------------------------------------------------------------------------------------------------------------------------------------------
Format   Alg        Avg BR     σ BR   Avg OR     σ OR Avg RecO   σ RecO Avg RecB   σ RecB    Avg t      σ t   Avg t*     σ t*   Succ    Tot
-------------------------------------------------------------------------------------------------------------------------------------------
double_json betamax      3.49     1.13     2.85     1.70     0.94     0.07     0.94     0.07    15.22    28.84    77.87   120.69     39     50

Metrics for triple.db
-------------------------------------------------------------------------------------------------------------------------------------------
Format   Alg        Avg BR     σ BR   Avg OR     σ OR Avg RecO   σ RecO Avg RecB   σ RecB    Avg t      σ t   Avg t*     σ t*   Succ    Tot
-------------------------------------------------------------------------------------------------------------------------------------------
triple_json betamax      3.94     1.04     3.43     1.71     0.92     0.09     0.94     0.07    20.29    25.58   104.20   129.95     35     50

Combined Metrics Across All Databases
-------------------------------------------------------------------------------------------------------------------------------------------
Format   Alg        Avg BR     σ BR   Avg OR     σ OR Avg RecO   σ RecO Avg RecB   σ RecB    Avg t      σ t   Avg t*     σ t*   Succ    Tot
-------------------------------------------------------------------------------------------------------------------------------------------
double_json betamax      3.49     1.13     2.85     1.70     0.94     0.07     0.94     0.07    15.22    28.84    77.87   120.69     39     50
single_json betamax      2.12     1.42     1.42     1.50     0.97     0.06     0.97     0.06     4.79     9.80     4.79     9.80     50     50
triple_json betamax      3.94     1.04     3.43     1.71     0.92     0.09     0.94     0.07    20.29    25.58   104.20   129.95     35     50
———— Levenshtein distances ————

Overall Distance Metrics Across All Databases
Alg        Avg BR     σ BR   Avg OR     σ OR  Avg RecO    σ RecO  Avg RecB    σ RecB
--------------------------------------------------------------------------------------
betamax      3.06     1.47     2.44     1.84      0.95      0.08      0.95      0.07

———— count repaired —————————

Repaired counts by format in single.db (1-mutation, repair_time ≤300s)
------------------------------------
Format       Alg            Repaired
------------------------------------
single_json  betamax              50

Repaired counts by format in double.db (2-mutations, repair_time ≤300s)
------------------------------------
Format       Alg            Repaired
------------------------------------
double_json  betamax              39

Repaired counts by format in triple.db (3-mutations, repair_time ≤300s)
------------------------------------
Format       Alg            Repaired
------------------------------------
triple_json  betamax              35

Overall repaired counts by format across DBs (repair_time ≤300s)
------------------------------------
Format       Alg            Repaired
------------------------------------
double_json  betamax              39
single_json  betamax              50
triple_json  betamax              35

Overall repaired counts by algorithm across DBs (repair_time ≤300s)
  betamax             124
———— perfect repairs —————————

Perfect repairs in single.db
  single_json  betamax          17

Perfect repairs in double.db
  double_json  betamax           1

Perfect repairs in triple.db
  None

Overall perfect repairs by format across DBs
----------------------------------
Format       Alg           Perfect
----------------------------------
double_json  betamax             1
single_json  betamax            17

Overall perfect repairs by algorithm across DBs
  betamax            18
```