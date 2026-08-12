"""Generate the animated README banner from the real splitter output.

The frames are not hand-drawn: each one is an actual (train, test) split from
CombinatorialPurgedCV, so the picture cannot drift from what the library does.
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from purgedcv import CombinatorialPurgedCV, make_t1  # noqa: E402

N_BARS, N_GROUPS, K = 36, 6, 2
LIGHT = {"train": "#8199b3", "purge": "#e8a21c", "test": "#2f6fed"}
HORIZON, EMBARGO = 3, 0.04

# --- geometry -------------------------------------------------------------- #
BAR_W, GAP, BAR_H = 11.0, 2.0, 26.0
X0, Y0 = 26.0, 66.0
W = X0 * 2 + N_BARS * BAR_W + (N_BARS - 1) * GAP
H = 152.0
SECS_PER_FRAME = 0.62

idx = pd.bdate_range("2020-01-01", periods=N_BARS)
X = pd.DataFrame({"f": np.zeros(N_BARS)}, index=idx)
cv = CombinatorialPurgedCV(N_GROUPS, K, embargo_pct=EMBARGO, t1=make_t1(idx, HORIZON))

frames: list[list[str]] = []
for train_idx, test_idx in cv.split(X):
    state = ["train"] * N_BARS
    keep = set(int(i) for i in train_idx)
    for j in test_idx:
        state[int(j)] = "test"
    for j in range(N_BARS):
        if state[j] != "test" and j not in keep:
            state[j] = "purge"
    frames.append(state)

n_frames = len(frames)
total = n_frames * SECS_PER_FRAME

# Each bar animates its own fill through the whole cycle via discrete keyframes.
times = ";".join(f"{i / n_frames:.3f}".rstrip("0").rstrip(".") or "0" for i in range(n_frames + 1))


def bar_svg(j: int) -> str:
    x = X0 + j * (BAR_W + GAP)
    seq = [frames[f][j] for f in range(n_frames)]
    vals = ";".join(LIGHT[s] for s in seq + [seq[0]])
    return (
        f'<rect x="{x:g}" y="{Y0:g}" width="{BAR_W:g}" height="{BAR_H:g}" rx="2.5" '
        f'class="bar" fill="{LIGHT[seq[0]]}">'
        f'<animate attributeName="fill" values="{vals}" keyTimes="{times}" '
        f'dur="{total:g}s" calcMode="discrete" repeatCount="indefinite"/>'
        f"</rect>"
    )


# Sweep line tracks the leading edge of the test block.
sweep_x = []
for f in range(n_frames + 1):
    st = frames[f % n_frames]
    first = st.index("test")
    sweep_x.append(f"{X0 + first * (BAR_W + GAP) - 1.5:g}")

bars = "\n    ".join(bar_svg(j) for j in range(N_BARS))
counter = ";".join(str(i + 1) for i in range(n_frames)) + ";1"

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}"
     width="{W:.0f}" height="{H:.0f}" role="img"
     aria-label="purgedcv: combinatorial purged cross-validation, animated">
  <title>purgedcv</title>
  <desc>Each frame is a real split from CombinatorialPurgedCV(6, 2): the test
  block moves across the timeline, and the bars beside it are purged and
  embargoed so no training label overlaps a test label.</desc>
  <style>
    .wm {{ font: 700 30px ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;
           fill: #16202b; letter-spacing: -0.6px; }}
    .sub {{ font: 400 13px ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;
            fill: #5b6b7f; }}
    .key {{ font: 500 11px ui-monospace,SFMono-Regular,Menlo,monospace; fill: #5b6b7f; }}
    .rule {{ stroke: #dde5ee; }}
    .sweep {{ fill: #2f6fed; }}
    .k-test {{ fill: #2f6fed; }} .k-purge {{ fill: #e8a21c; }} .k-train {{ fill: #8199b3; }}
    .bar {{ shape-rendering: crispEdges; }}
    .bg {{ fill: #ffffff; }}
    @media (prefers-color-scheme: dark) {{
      .bg {{ fill: #0d1117; }}
      .wm {{ fill: #e9eff7; }} .sub, .key {{ fill: #93a3b8; }}
      .rule {{ stroke: #2b3644; }}
    }}
  </style>

  <rect class="bg" x="0" y="0" width="{W:g}" height="{H:g}" rx="10"/>

  <text x="{X0:g}" y="30" class="wm">purgedcv</text>
  <text x="{X0:g}" y="49" class="sub">combinatorial purged cross-validation, with embargo</text>

  <g>
    {bars}
  </g>

  <line x1="{X0:.0f}" y1="{Y0 + BAR_H + 9:.0f}" x2="{W - X0:.0f}" y2="{Y0 + BAR_H + 9:.0f}"
        class="rule" stroke-width="1"/>

  <rect y="{Y0 - 7:.0f}" width="2" height="{BAR_H + 14:.0f}" rx="1" class="sweep" opacity="0.30">
    <animate attributeName="x" values="{";".join(sweep_x)}" keyTimes="{times}"
             dur="{total:.2f}s" calcMode="spline"
             keySplines="{" ".join(["0.65 0 0.35 1"] * n_frames)}" repeatCount="indefinite"/>
  </rect>

  <g transform="translate({X0:.0f},{Y0 + BAR_H + 27:.0f})">
    <rect x="0" y="-8" width="10" height="10" rx="2" class="k-test"/>
    <text x="16" y="0" class="key">test</text>
    <rect x="62" y="-8" width="10" height="10" rx="2" class="k-purge"/>
    <text x="78" y="0" class="key">purged + embargoed</text>
    <rect x="212" y="-8" width="10" height="10" rx="2" class="k-train"/>
    <text x="228" y="0" class="key">train</text>
  </g>
  <text x="{W - X0:.0f}" y="{Y0 + BAR_H + 27:.0f}" class="key" text-anchor="end">C(6,2) = {n_frames} splits, 5 paths</text>
</svg>
"""

out = str(pathlib.Path(__file__).resolve().parent / "banner.svg")
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write(svg)
print(f"wrote {out}  ({len(svg)} bytes, {n_frames} frames, {total:.1f}s loop)")
