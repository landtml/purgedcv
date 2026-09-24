# purgedcv-backtest

An overfitting-aware backtesting framework, built bottom-up on `purgedcv`.
**Status: layer 1 of 5.** This is a separate package: it never changes
`purgedcv`, which stays a splitter and path map with no return computation.

Most backtesters answer "how good does this look?". This one is built to
answer "should I believe it?", so the statistics come first and everything
else is built to feed them.

## Layer 1: statistics and trial accounting (this release)

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

## Roadmap (bottom-up)

| Layer | What | Feeds |
|---|---|---|
| 1 | Statistics + trial log | everything above |
| 2 | Returns: predictions to positions to per-path returns, with costs, slippage and turnover, from `CPCVPaths` | the trial log |
| 3 | Fit harness: fit every CPCV split (Rust engine, parallel) and record each configuration as a trial automatically | layer 2 |
| 4 | Path-level analytics: Sharpe distribution across CPCV paths (with the caveat that paths are correlated, not independent draws), drawdowns, performance by regime | the report |
| 5 | Dashboard: one report per research run, verdict first (DSR, PBO, MinTRL vs track record), equity curves after | people |
