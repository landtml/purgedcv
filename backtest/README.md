# purgedcv-backtest

An overfitting-aware backtesting framework, built bottom-up on `purgedcv`.
**Status: layers 1-2 of 6** (see [DESIGN.md](DESIGN.md)). This is a separate package: it never changes
`purgedcv`, which stays a splitter and path map with no return computation.

Most backtesters answer "how good does this look?". This one is built to
answer "should I believe it?", so the statistics come first and everything
else is built to feed them.

## Layer 1: statistics and trial accounting

```python
import purgedcv_backtest as pb

log = pb.TrialLog()
for lookback in range(5, 200, 5):             # every configuration you try...
    log.record(oos_returns(lookback), lookback=lookback)   # ...gets recorded

log.summary()      # best trial: Sharpe, PSR, the luck bar, DSR, MinTRL
log.pbo()          # probability the in-sample winner is overfit (CSCV)
```

| Statistic | Question it answers | Function |
|---|---|---|
| Probabilistic Sharpe ratio | Is this Sharpe ratio above a benchmark, given sample length, skew and fat tails? | `probabilistic_sharpe_ratio` |
| Expected max Sharpe | What would the best of N skill-less trials reach by luck? | `expected_max_sharpe` |
| Deflated Sharpe ratio | Is the *selected* strategy real, given every trial tried to find it? | `deflated_sharpe_ratio`, `TrialLog.deflated_sharpe` |
| Minimum track record length | How many observations does this Sharpe ratio need to be significant? | `min_track_record_length` |
| Probability of backtest overfitting | How often does the in-sample winner land in the bottom half out of sample? | `probability_of_backtest_overfitting`, `TrialLog.pbo` |

All Sharpe ratios are **per period**, the scale the formulas are derived on.

### Design choices that keep it honest

- **Counting trials is the default path.** `deflated_sharpe_ratio` has no
  default for `n_trials`, so you can't silently deflate by one.
  `TrialLog` supplies the count and the Sharpe dispersion from its record.
- **The log is append-only.** There is no way to remove, replace or edit a
  trial, and recorded returns and parameters are frozen copies. Abandoned
  configurations still count, because they were still tried.
- **Correlated trials are handled conservatively.** The raw count overstates
  the effective number of independent trials, which can only make DSR
  harsher.

### How it's verified

- **Published value:** reproduces the numerical example in Bailey & López de
  Prado (2014), DSR = 0.9004.
- **Calibration under the null:**
  - On pure noise, PSR > 0.95 happens at the nominal ~5% rate.
  - The best of 50 skill-less strategies passes naive PSR > 0.95 in over 60%
    of experiments, and passes DSR from the trial log in under 8%. This is the
    trial-counting claim, tested.
- **Power:** a real edge hidden among 49 noise trials is still recognised
  (DSR > 0.99), and PBO is near 0 for a genuinely better trial but about 0.5
  for noise.
- **Consistency checks:**
  - MinTRL is exactly where PSR crosses 1 - alpha.
  - The expected-max-Sharpe formula matches simulated maxima to within 3%.
  - The fast CSCV Sharpe ratios, built from block sums, match a direct
    computation.

```bash
pip install -e backtest
pytest backtest/tests
```

## Layer 2: engine core and falsification harness

```python
w = pb.SignSizer(gross=1.0)(predictions)            # or ProbabilitySizer (AFML ch. 10)
costs = pb.CompositeCost(
    pb.LinearCost.from_instruments(instruments),      # fees per instrument
    pb.SquareRootImpact(vol, adv, capital=5e6, coef=0.7),
)
verdict = pb.falsify(w, returns, costs=costs, calendar=pb.CRYPTO_DAILY,
                     carry=pb.FundingCarry(funding_rates))
print(verdict)                                      # illustrative output:
# Verdict: SURVIVED
#   [PASS] F1 execution timing: positions held from 1 bar(s) after the decision
#   [PASS] F3 permutation test: p = 0.0010 over 1000 time rotations ...
#   [PASS] F2 lag cliff: Sharpe decays smoothly with execution delay
#   [PASS] F5 net of costs: net per-period Sharpe 0.0412; flat-cost break-even 14.2 bps ...
```

| Piece | Guards against |
|---|---|
| `simulate` refuses `lag=0`, requires a cost model and calendar, never reindexes or fills silently | F1 same-bar look-ahead, free trading by default |
| `permutation_test`: rotates the signal against returns, keeping both series' autocorrelation | F3 spurious edge |
| `lag_profile`: flags a significant Sharpe that collapses with one more bar of delay | F2 leaky features |
| `cost_sensitivity` + findings for `ZeroCost` and unfunded shorts | F5 unrealistic costs |
| `cpcv_path_returns`: positions assembled per path, returns and costs simulated per path | F6 path luck, return double-counting |

Asset classes differ only in the models plugged in: `Calendar` (24/7 crypto,
exchange sessions), fees per `Instrument`, `SquareRootImpact` for capacity,
`FundingCarry` for perpetuals, `BorrowCarry` for shorts.

Each guard has a planted-flaw test and a clean counterpart, on synthetic
markets where the truth is known:

- A feature containing next bar's return is flagged. A persistent, honestly
  timed signal keeps over 80% of its Sharpe ratio one bar later and is not
  flagged.
- 60 noise strategies get permutation p-values averaging about 0.5, with at
  most 15% below 0.05. A real edge gets p < 0.01.
- A real but thin, high-turnover edge breaks even near its theoretical 5 bps,
  and is falsified at 10 bps.
- Each path's returns and costs equal a direct re-simulation of that path's
  positions. Training rows filled with NaN are never read.
- The tests were mutation-checked: disabling each guard makes its test fail.

## Roadmap (bottom-up)

Depth before breadth; the full failure-pattern catalogue is in
[DESIGN.md](DESIGN.md).

| Layer | What | Failure patterns |
|---|---|---|
| 1 | Statistics + trial log (**done**) | F4 selection bias, F7 short track record |
| 2 | Engine core, cost/carry/sizing models, CPCV path returns, falsification harness (**done**) | F1, F2, F3, F5, F6 |
| 3 | Research-session integrity: persistent ledger, sealed holdout vault, point-in-time universe | F8 iterating until it works, F9 survivorship |
| 4 | Path and regime analytics | F10 regime dependence |
| 5 | Event-driven engine with vectorised parity; verdict-first report and dashboard | F11 engine error |
| 6 | Breadth: asset-class model packs, data adapters, Rust kernels for hot paths | |
