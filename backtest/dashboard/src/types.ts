// Mirror of purgedcv_backtest.report (schema "purgedcv-backtest/report@1").
// Non-finite floats arrive as null.

export const SCHEMA = "purgedcv-backtest/report@1";

export type Num = number | null;
export type Status = "pass" | "warn" | "fail";

export interface Finding {
  check: string;
  status: Status;
  detail: string;
}

export interface Verdict {
  survived: boolean;
  findings: Finding[];
  permutation: { observed: Num; p_value: Num; null: Num[] };
  lags: { lags: number[]; sharpes: Num[]; base_psr: Num; cliff_ratio: Num; flagged: boolean };
  cost_curve: { bps: Num[]; sharpes: Num[]; break_even_bps: Num; break_even_is_beyond_grid: boolean };
}

export interface Performance {
  n_obs: number;
  sharpe: Num;
  annualized_sharpe: Num;
  skew: Num;
  kurtosis: Num;
  psr: Num;
  min_track_record_length: Num;
  total_return: Num;
  max_drawdown: Num;
  mean_turnover: Num;
  total_costs: Num;
  total_carry: Num;
}

export interface TrialRow {
  number: number;
  name: string;
  params: Record<string, string | number | boolean | null>;
  sharpe: Num;
  n_obs: number;
  psr: Num;
}

export interface Trials {
  n_trials: number;
  sharpe_variance: Num;
  expected_max_sharpe: Num;
  best: number;
  dsr: Num;
  trials: TrialRow[];
  pbo: { pbo: Num; prob_oos_loss: Num; logits: Num[]; n_splits: number } | null;
  pbo_unavailable?: string;
}

export interface Paths {
  summary: Record<string, Num>;
  sharpes: Num[];
  index: string[];
  equity: Num[][];
}

export interface Report {
  schema: string;
  title: string;
  generated_at: string;
  calendar: { name: string; periods_per_year: number } | null;
  assumptions: Record<string, unknown> | null;
  verdict: Verdict | null;
  performance: Performance | null;
  series: { index: string[]; equity: Num[]; gross_equity: Num[]; drawdown: Num[] } | null;
  trials: Trials | null;
  paths: Paths | null;
}
