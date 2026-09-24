"""Report export: one JSON document the dashboard reads.

The report is the contract between the Python engine and any viewer. It is
verdict-first (findings before evidence), carries every assumption the
numbers depend on, and holds only plain JSON: non-finite floats become
``null`` and are named in ``notes`` rather than silently dropped.

    report = build_report(verdict=v, trials=log, paths=paths, title="Momentum")
    write_report("dashboard/public/report.json", report)
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .engine import PathResults, SimResult
from .falsify import Verdict
from .stats import min_track_record_length, probabilistic_sharpe_ratio
from .trials import TrialLog

__all__ = ["SCHEMA", "build_report", "write_report"]

SCHEMA = "purgedcv-backtest/report@1"
_MAX_POINTS = 1500


def _num(x: Any) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _nums(xs) -> list[float | None]:
    """Arrays for charts, rounded to 6 significant digits to keep files small."""
    return [None if v is None else float(f"{v:.6g}")
            for v in (_num(x) for x in np.asarray(xs, dtype=np.float64).ravel())]


def _stride(n: int) -> int:
    return max(1, math.ceil(n / _MAX_POINTS))


def _thin(index: pd.Index, *series: np.ndarray) -> tuple[list[str], list[list[float | None]]]:
    """Every ``k``-th point, always keeping the last one."""
    n = len(index)
    keep = np.arange(0, n, _stride(n))
    if n and keep[-1] != n - 1:
        keep = np.append(keep, n - 1)
    labels = [str(pd.Timestamp(i).isoformat()) if isinstance(index, pd.DatetimeIndex) else str(i)
              for i in index[keep]]
    return labels, [_nums(np.asarray(s)[keep]) for s in series]


def _performance(result: SimResult) -> dict[str, Any]:
    net = result.net.to_numpy()
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    drawdown = equity / peak - 1.0
    s = result.stats
    return {
        "n_obs": s.n_obs,
        "sharpe": _num(s.sharpe),
        "annualized_sharpe": _num(result.annualized_sharpe),
        "skew": _num(s.skew),
        "kurtosis": _num(s.kurtosis),
        "psr": _num(probabilistic_sharpe_ratio(s)) if net.std() > 0 else None,
        "min_track_record_length": _num(min_track_record_length(s)),
        "total_return": _num(equity[-1] - 1.0) if len(equity) else None,
        "max_drawdown": _num(drawdown.min()) if len(drawdown) else None,
        "mean_turnover": _num(result.turnover.mean()),
        "total_costs": _num(result.costs.sum()),
        "total_carry": _num(result.carry.sum()),
    }


def _series(result: SimResult) -> dict[str, Any]:
    net = result.net.to_numpy()
    gross = result.gross.to_numpy()
    equity = np.cumprod(1.0 + net)
    gross_equity = np.cumprod(1.0 + gross)
    drawdown = equity / np.maximum.accumulate(np.maximum(equity, 1.0)) - 1.0
    labels, (eq, geq, dd) = _thin(result.net.index, equity, gross_equity, drawdown)
    return {"index": labels, "equity": eq, "gross_equity": geq, "drawdown": dd}


def _verdict(v: Verdict) -> dict[str, Any]:
    return {
        "survived": v.survived,
        "findings": [
            {"check": f.check, "status": f.status.value, "detail": f.detail} for f in v.findings
        ],
        "permutation": {
            "observed": _num(v.permutation.observed),
            "p_value": _num(v.permutation.p_value),
            "null": _nums(v.permutation.null),
        },
        "lags": {
            "lags": list(v.lags.lags),
            "sharpes": _nums(v.lags.sharpes),
            "base_psr": _num(v.lags.base_psr),
            "cliff_ratio": _num(v.lags.cliff_ratio),
            "flagged": v.lags.flagged,
        },
        "cost_curve": {
            "bps": _nums(v.cost_curve.bps),
            "sharpes": _nums(v.cost_curve.sharpes),
            "break_even_bps": _num(v.cost_curve.break_even_bps),
            "break_even_is_beyond_grid": math.isinf(v.cost_curve.break_even_bps),
        },
    }


def _trials(log: TrialLog, pbo_splits: int) -> dict[str, Any]:
    best = log.best()
    out: dict[str, Any] = {
        "n_trials": log.n_trials,
        "sharpe_variance": _num(log.sharpe_variance()),
        "expected_max_sharpe": _num(log.expected_max_sharpe()),
        "best": best.number,
        "dsr": _num(log.deflated_sharpe()),
        "trials": [
            {
                "number": t.number,
                "name": t.name,
                "params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else repr(v))
                           for k, v in t.params.items()},
                "sharpe": _num(t.sharpe),
                "n_obs": t.stats.n_obs,
                "psr": _num(probabilistic_sharpe_ratio(t.stats)) if t.returns.std() > 0 else None,
            }
            for t in log
        ],
        "pbo": None,
    }
    if log.n_trials >= 2:
        try:
            p = log.pbo(n_splits=pbo_splits)
        except ValueError as exc:  # unequal lengths, too few observations
            out["pbo_unavailable"] = str(exc)
        else:
            out["pbo"] = {
                "pbo": _num(p.pbo),
                "prob_oos_loss": _num(p.prob_oos_loss),
                "logits": _nums(p.logits),
                "n_splits": pbo_splits,
            }
    return out


def _paths(paths: PathResults) -> dict[str, Any]:
    first = paths.paths[0]
    equities = [np.cumprod(1.0 + r.net.to_numpy()) for r in paths.paths]
    labels, thinned = _thin(first.net.index, *equities)
    return {
        "summary": {k: _num(v) for k, v in paths.summary().items()},
        "sharpes": _nums(paths.sharpes),
        "index": labels,
        "equity": thinned,
    }


def build_report(
    *,
    verdict: Verdict | None = None,
    trials: TrialLog | None = None,
    paths: PathResults | None = None,
    title: str = "Backtest report",
    pbo_splits: int = 16,
) -> dict[str, Any]:
    """Collect a strategy's verdict, trial accounting and CPCV paths into one
    JSON-ready dict. Every section is optional; the dashboard shows what is
    present and says what is missing."""
    if verdict is None and trials is None and paths is None:
        raise ValueError("build_report needs at least one of verdict, trials or paths.")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "title": title,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "calendar": None,
        "assumptions": None,
        "verdict": None,
        "performance": None,
        "series": None,
        "trials": None,
        "paths": None,
    }
    result = verdict.result if verdict is not None else (paths.paths[0] if paths else None)
    if result is not None:
        report["calendar"] = {
            "name": result.calendar.name,
            "periods_per_year": result.calendar.periods_per_year,
        }
    if verdict is not None:
        report["verdict"] = _verdict(verdict)
        report["performance"] = _performance(verdict.result)
        report["series"] = _series(verdict.result)
        report["assumptions"] = dict(verdict.result.assumptions)
    if trials is not None and len(trials):
        report["trials"] = _trials(trials, pbo_splits)
    if paths is not None and paths.paths:
        report["paths"] = _paths(paths)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write ``report`` as strict JSON (no NaN/Infinity tokens)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")))
    return path
