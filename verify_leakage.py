"""Large-scale empirical leakage verifier for the CPCV splitter.

Downloads decades of real market data across many tickers and, for a grid of
CV configurations / label horizons / embargoes / anchors, checks the properties
that *define* a correct purged-embargoed combinatorial scheme:

    D  disjoint        train and test never share an observation
    L  no leakage      no kept training label overlaps any test label
                       - certified two independent ways:
                         (cert)  kept-set vs the test label-union ranges
                         (brute) kept-set vs every individual test obs (n<=cap)
    E  embargo         the embargo band after each test block holds no train obs
    Q  equivalence     library purge == per-observation brute force (embargo=0)
    C  coverage        every bar is tested in exactly n_paths simulations

It prints a certificate with the totals behind the verdict so the numbers are
auditable. Exit code is non-zero if any check fails.

Run:  python verify_leakage.py
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cpcv import (
    CombinatorialPurgedCV,
    make_t1,
    _contiguous_blocks,
    _end_positions,
    _make_group_labels,
)

# Long-history, liquid names (period="max" pulls their full available history).
TICKERS = ["^GSPC", "^DJI", "IBM", "KO", "GE", "XOM", "JNJ", "PG", "MSFT", "AAPL"]
CONFIGS = [(6, 2), (8, 2), (10, 3), (8, 3)]
HORIZONS = [1, 5, 21, 63]
EMBARGOS = [0.0, 0.02]
ANCHORS = ["label_end", "test_end"]
BRUTE_CAP = 4000  # full per-observation brute force only below this many bars


@dataclass
class Tally:
    datasets: int = 0
    total_bars: int = 0
    scenarios: int = 0
    splits: int = 0
    brute_pairs: int = 0          # individual train x test interval comparisons
    cert_assertions: int = 0      # kept-obs x test-block disjointness assertions
    equivalence_checks: int = 0
    failures: list[str] = field(default_factory=list)
    spans: list[tuple] = field(default_factory=list)


def _download(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf

    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df.dropna()
    # Guard the splitter's contract up front.
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _brute_overlap(train_idx, test_idx, start, end, chunk=512) -> bool:
    """True iff ANY kept train interval overlaps ANY test interval (chunked)."""
    tr0, tr1 = start[train_idx][:, None], end[train_idx][:, None]
    for c0 in range(0, len(test_idx), chunk):
        sl = test_idx[c0 : c0 + chunk]
        if ((tr1 >= start[sl][None, :]) & (tr0 <= end[sl][None, :])).any():
            return True
    return False


def verify_dataset(name: str, df: pd.DataFrame, tally: Tally) -> None:
    n = len(df)
    index = df.index
    start = np.arange(n)
    tally.datasets += 1
    tally.total_bars += n
    tally.spans.append((name, n, index[0].date(), index[-1].date()))
    do_brute = n <= BRUTE_CAP

    for N, k in CONFIGS:
        labels = _make_group_labels(n, N)
        combos = list(itertools.combinations(range(N), k))
        n_paths = math.comb(N, k) * k // N

        for horizon in HORIZONS:
            t1 = make_t1(index, horizon)
            end = _end_positions(index, t1)

            for embargo_pct in EMBARGOS:
                embargo = math.ceil(embargo_pct * n) if embargo_pct else 0
                for anchor in ANCHORS:
                    tally.scenarios += 1
                    cv = CombinatorialPurgedCV(
                        N, k, embargo_pct=embargo_pct, t1=t1, embargo_anchor=anchor
                    )
                    n_test_seen = np.zeros(n, dtype=int)

                    for (train_idx, test_idx), combo in zip(cv.split(df), combos):
                        tally.splits += 1
                        tag = f"{name} N={N} k={k} h={horizon} emb={embargo_pct} {anchor} {combo}"

                        # D: disjoint
                        if np.intersect1d(train_idx, test_idx).size:
                            tally.failures.append(f"DISJOINT {tag}")

                        n_test_seen[test_idx] += 1
                        blocks = _contiguous_blocks(labels, combo)

                        # L (cert): kept train vs each test-block label-union range.
                        for b0, b1 in blocks:
                            label_end = int(end[b0 : b1 + 1].max())
                            leak = (end[train_idx] >= b0) & (start[train_idx] <= label_end)
                            tally.cert_assertions += train_idx.size
                            if leak.any():
                                tally.failures.append(f"LEAK-CERT {tag} block=({b0},{b1})")
                            # E: embargo band empty of train obs
                            if embargo:
                                anchor_pos = label_end if anchor == "label_end" else b1
                                band = (start[train_idx] > anchor_pos) & (
                                    start[train_idx] <= anchor_pos + embargo
                                )
                                if band.any():
                                    tally.failures.append(f"EMBARGO {tag} block=({b0},{b1})")

                        # L (brute): independent per-observation corroboration.
                        if do_brute:
                            tally.brute_pairs += train_idx.size * test_idx.size
                            if _brute_overlap(train_idx, test_idx, start, end):
                                tally.failures.append(f"LEAK-BRUTE {tag}")

                        # Q: envelope purge == brute-force purge (embargo-free only).
                        if do_brute and embargo == 0 and anchor == "label_end":
                            tally.equivalence_checks += 1
                            keep = np.ones(n, dtype=bool)
                            test_mask = np.isin(labels, combo)
                            ts0, te0 = start[test_mask], end[test_mask]
                            for i in range(n):
                                if test_mask[i] or (
                                    (end[i] >= ts0) & (start[i] <= te0)
                                ).any():
                                    keep[i] = False
                            if not np.array_equal(np.flatnonzero(keep), train_idx):
                                tally.failures.append(f"EQUIV {tag}")

                    # C: full path coverage
                    if not (n_test_seen == n_paths).all():
                        tally.failures.append(
                            f"COVERAGE {name} N={N} k={k} h={horizon} "
                            f"emb={embargo_pct} {anchor}"
                        )


def main() -> int:
    try:
        import yfinance  # noqa: F401
    except Exception:
        print("yfinance not installed -> cannot run live verification.")
        return 2

    tally = Tally()
    for ticker in TICKERS:
        df = _download(ticker)
        if df is None:
            print(f"  - {ticker:<6} no data (offline?), skipping")
            continue
        print(f"  - {ticker:<6} {len(df):>6} bars  "
              f"{df.index[0].date()} -> {df.index[-1].date()}  verifying...")
        verify_dataset(ticker, df, tally)
        # Also verify a sub-BRUTE_CAP recent window so the independent
        # per-observation brute force and purge-equivalence checks run on real data.
        window = df.tail(BRUTE_CAP - 100)
        if len(window) < len(df):
            verify_dataset(f"{ticker}~recent", window, tally)

    if tally.datasets == 0:
        print("No data downloaded (offline?). Nothing verified.")
        return 2

    print("\n" + "=" * 70)
    print("CPCV LEAKAGE VERIFICATION CERTIFICATE")
    print("=" * 70)
    earliest = min(s[2] for s in tally.spans)
    latest = max(s[3] for s in tally.spans)
    print(f"  datasets verified      : {tally.datasets} tickers")
    print(f"  calendar span          : {earliest} -> {latest}")
    print(f"  total bars (obs)        : {tally.total_bars:,}")
    print(f"  scenarios (cfg x h x e) : {tally.scenarios:,}")
    print(f"  train/test splits       : {tally.splits:,}")
    print(f"  envelope assertions     : {tally.cert_assertions:,}")
    print(f"  brute-force pair checks : {tally.brute_pairs:,}")
    print(f"  purge-equivalence checks: {tally.equivalence_checks:,}")
    print("-" * 70)
    if tally.failures:
        print(f"  RESULT: FAIL  ({len(tally.failures)} violations)")
        for f in tally.failures[:25]:
            print(f"    x {f}")
        if len(tally.failures) > 25:
            print(f"    ... and {len(tally.failures) - 25} more")
        return 1
    print("  RESULT: PASS")
    print("  D disjoint | L no-leakage (cert+brute) | E embargo | "
          "Q purge==bruteforce | C coverage")
    print("  Zero leakage detected across every split, scenario, and dataset.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
