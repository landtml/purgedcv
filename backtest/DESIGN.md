# purgedcv-backtest: design charter

A backtesting framework whose foundation is the ways backtests fail.
Overfitting and falsification rigour are built into the core abstractions,
not added as a report at the end. Breadth (asset classes, venues, data
vendors) comes later and plugs into the interfaces defined here.

Every layer is checked against this document. A change that weakens an
invariant below needs to change this document first, with a reason.

## 1. Principles

1. **A backtest is a hypothesis test, not a measurement.** The primary
   output is a verdict (did the strategy survive falsification?). Equity
   curves are supporting evidence.
2. **Guards are structural.** The honest path is the default and the easy
   path; the dishonest path requires explicit, visible opt-outs that show up
   in the report.
3. **Every guard is proven by a planted flaw.** For each failure pattern,
   the test suite plants that flaw in synthetic data and asserts the engine
   catches it, and plants a clean equivalent and asserts it passes. A guard
   without both tests does not count.
4. **Asset-class specifics live in models, not in the engine.** The engine
   knows weights, returns, costs and carry. Crypto, equities and futures
   differ in which cost, carry, calendar and instrument models they plug in.
5. **Two engines, one set of models.** A fast vectorised engine for
   research and an event-driven engine for realism share every model, and a
   parity test between them is itself a falsification check.

## 2. Failure-pattern catalogue

| # | Failure pattern | Structural guard | Planted-flaw test | Layer |
|---|---|---|---|---|
| F1 | Same-bar look-ahead (trading on the bar whose return you earn) | Engine execution lag must be >= 1; `lag=0` raises | lag 0 rejected | 2 |
| F2 | Leaky feature (future information in the data itself) | Lag-cliff test: Sharpe that collapses with one extra bar of delay is flagged | a feature built from the next bar's return is flagged; a slow, genuinely predictive signal is not | 2 |
| F3 | Spurious edge (the result is what noise produces) | Circular-shift permutation test: the signal is shifted against returns, keeping both series' autocorrelation and breaking only their alignment | noise strategy gets p ~ U(0,1); a real edge gets p < 0.01 | 2 |
| F4 | Selection bias across trials | Append-only `TrialLog`; DSR has no default trial count; PBO by CSCV | best of 50 noise strategies passes naive PSR > 60% of the time, DSR < 8% | 1 (done) |
| F5 | Unrealistic costs | Cost model is a required argument; `ZeroCost` exists but is reported as a finding; cost-sensitivity sweep reports the break-even cost | zero-cost run is flagged; a turnover-heavy edge is shown to die at realistic cost | 2 |
| F6 | Path luck (one historical path) | CPCV path distribution; paths assembled from positions, with returns and costs computed per path, never reused across paths | per-path returns equal a direct re-simulation of that path | 2 |
| F7 | Short track record | MinTRL vs actual track record | done (layer 1) | 1 (done) |
| F8 | Iterating until it works | Persistent ledger across sessions; a sealed holdout ("vault") that can be opened once, with the opening recorded | second vault opening refused | 3 |
| F9 | Survivorship bias | Point-in-time universe membership; delisting returns | survivor-only universe flagged | 3 |
| F10 | Regime dependence | Performance by regime; stability across CPCV paths and sub-periods | edge confined to one regime flagged | 4 |
| F11 | Engine error | Vectorised/event-driven parity | engines disagree -> failure | 5 |

## 3. Core conventions (the invariants)

- **Time.** Row `t` of every panel is the bar ending at time `t`. A target
  weight decided with information up to the end of bar `t` is held over
  bars `t + lag, ...`, with `lag >= 1`. The return earned in bar `t` is the
  one ending at `t`. (F1)
- **Units.** Positions are weights: signed fraction of equity per
  instrument. Returns are simple per-bar returns of each instrument
  (futures and perps as return on notional). Every cost and carry model
  returns a charge in the same units, so the net return is
  `sum(held * returns) - costs - carry`.
- **Rebalancing.** Held weights are reset to target every bar, and turnover
  is `|held_t - held_{t-1}|` summed over instruments. Weight drift between
  rebalances is not modelled in the vectorised engine; the event-driven
  engine is the reference for that.
- **Sharpe ratios are per period**, annualised only for display with the
  calendar's `periods_per_year`.
- **Nothing is free by default.** No cost model, calendar or sizing rule has
  a silent default in the engine API.

## 4. Interfaces (layer 2)

| Interface | Contract | Shipped implementations |
|---|---|---|
| `Calendar` | `periods_per_year`, name | equity daily (252), crypto daily (365), crypto hourly (8760), futures daily (252) |
| `Instrument` | symbol, asset class, currency, calendar, fee rates | plain dataclass |
| `CostModel` | `cost(trades, context) -> per-bar charge` from the traded weights | `ZeroCost`, `LinearCost` (per-instrument bps), `SquareRootImpact` (half-spread + `k * sigma * sqrt(participation)`), `CompositeCost` |
| `CarryModel` | `carry(held, context) -> per-bar charge` | `FundingCarry` (perpetual funding rates), `BorrowCarry` (short borrow) |
| `Sizer` | predictions -> target weights | `SignSizer`, `ProbabilitySizer` (AFML ch. 10 bet sizing) |
| `simulate` | weights, returns, costs, carry, lag -> `SimResult` | vectorised engine |
| `cpcv_path_returns` | `CPCVPaths` + per-simulation weights -> per-path `SimResult`s | F6 |
| `falsify` | runs F2, F3 and F5 checks and returns findings | permutation, lag-cliff, cost-sensitivity |

## 5. Roadmap

Depth before breadth. Each layer ships with its planted-flaw tests.

1. Statistics + trial log (F4, F7). **Done.**
2. Engine core: conventions, cost/carry/sizing interfaces, vectorised engine,
   CPCV path returns, falsification harness (F1, F2, F3, F5, F6).
3. Research-session integrity: persistent ledger, vault, point-in-time
   universe (F8, F9).
4. Path and regime analytics (F10).
5. Event-driven engine and parity (F11); the verdict-first report and
   dashboard.
6. Breadth: asset-class model packs, data adapters, Rust kernels for the
   hot paths.
