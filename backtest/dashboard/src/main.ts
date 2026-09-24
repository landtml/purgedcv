import "./style.css";
import { barChart, histogram, lineChart, niceTicks } from "./charts";
import { count, date, escapeHtml, fixed, pct, prob, signedPct } from "./format";
import { SCHEMA, type Num, type Report, type Status, type TrialRow } from "./types";

const C = {
  s1: "var(--series-1)",
  s2: "var(--series-2)",
  s3: "var(--series-3)",
  muted: "var(--series-muted)",
  critical: "var(--status-critical)",
};

const app = document.querySelector<HTMLDivElement>("#app")!;
let current: Report | null = null;

// --------------------------------------------------------------------------- //
// Loading                                                                     //
// --------------------------------------------------------------------------- //
function validate(data: unknown): Report {
  const r = data as Partial<Report> | null;
  if (!r || typeof r !== "object" || r.schema !== SCHEMA) {
    throw new Error(
      `This file is not a purgedcv-backtest report (expected schema "${SCHEMA}", ` +
        `found ${r && typeof r === "object" ? JSON.stringify(r.schema) : "no schema"}). ` +
        `Create one with purgedcv_backtest.write_report().`,
    );
  }
  return r as Report;
}

async function boot(): Promise<void> {
  const embedded = document.getElementById("report-data");
  try {
    if (embedded?.textContent?.trim()) {
      show(validate(JSON.parse(embedded.textContent)));
      return;
    }
    const res = await fetch(`${import.meta.env.BASE_URL}report.json`, { cache: "no-store" });
    if (!res.ok) throw new Error(`No report.json next to the dashboard (HTTP ${res.status}).`);
    show(validate(await res.json()));
  } catch (err) {
    showEmpty(err instanceof Error ? err.message : String(err));
  }
}

function openFile(file: File): void {
  file
    .text()
    .then((t) => show(validate(JSON.parse(t))))
    .catch((err: unknown) => showEmpty(err instanceof Error ? err.message : String(err)));
}

document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files[0];
  if (f) openFile(f);
});

function header(r: Report | null): string {
  const meta = r
    ? [
        r.calendar ? `${escapeHtml(r.calendar.name)} calendar, ${r.calendar.periods_per_year} bars a year` : "",
        r.performance ? `${count(r.performance.n_obs)} bars` : "",
        `generated ${escapeHtml(r.generated_at.replace("T", " ").replace("+00:00", " UTC"))}`,
      ].filter(Boolean).join(" · ")
    : "No report loaded";
  return `
    <header class="top">
      <div>
        <p class="eyebrow">purgedcv-backtest report</p>
        <h1>${escapeHtml(r?.title ?? "Backtest dashboard")}</h1>
        <p class="meta">${meta}</p>
      </div>
      <label class="button" for="file-input">Open report…
        <input id="file-input" type="file" accept="application/json,.json">
      </label>
    </header>`;
}

function wireHeader(): void {
  document.querySelector<HTMLInputElement>("#file-input")?.addEventListener("change", (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) openFile(f);
  });
}

function showEmpty(message: string): void {
  current = null;
  app.innerHTML = `${header(null)}
    <section class="panel empty">
      <h2>Nothing to show yet</h2>
      <p>${escapeHtml(message)}</p>
      <p>Write a report from Python, then reload this page or open the file here:</p>
      <pre><code>python backtest/examples/demo_report.py</code></pre>
      <p>or, from your own code: <code>pb.write_report("backtest/dashboard/public/report.json", pb.build_report(verdict=v, trials=log, paths=paths))</code></p>
    </section>`;
  wireHeader();
}

// --------------------------------------------------------------------------- //
// Rendering                                                                   //
// --------------------------------------------------------------------------- //
const ICON: Record<Status, string> = { pass: "✓", warn: "!", fail: "✕" };
const WORD: Record<Status, string> = { pass: "Pass", warn: "Warn", fail: "Fail" };

function chip(s: Status): string {
  return `<span class="chip chip-${s}"><span aria-hidden="true">${ICON[s]}</span>${WORD[s]}</span>`;
}

function tile(label: string, value: string, sub: string, title = ""): string {
  return `<div class="tile" ${title ? `title="${escapeHtml(title)}"` : ""}>
    <span class="tile-label">${label}</span>
    <span class="tile-value">${value}</span>
    <span class="tile-sub">${sub}</span>
  </div>`;
}

function missing(what: string, how: string): string {
  return `<p class="missing">${what} is not in this report. ${how}</p>`;
}

function show(r: Report): void {
  current = r;
  const v = r.verdict;
  const p = r.performance;
  const t = r.trials;
  const perBar = (x: Num) => (x == null ? "–" : `${x.toFixed(3)} per bar`);

  const verdictHtml = v
    ? (() => {
        const worst: Status = v.findings.some((f) => f.status === "fail")
          ? "fail"
          : v.findings.some((f) => f.status === "warn") ? "warn" : "pass";
        return `
      <section class="verdict verdict-${v.survived ? "survived" : "falsified"}" aria-label="Verdict">
        <div class="verdict-head">
          <span class="verdict-icon" aria-hidden="true">${v.survived ? "✓" : "✕"}</span>
          <div>
            <p class="eyebrow">Verdict</p>
            <p class="verdict-word">${v.survived ? "Survived" : "Falsified"}</p>
            <p class="verdict-note">${
              v.survived
                ? worst === "warn"
                  ? "No check failed, but the warnings below need an answer before you trust it."
                  : "Every falsification check passed. Selection across trials is judged below."
                : "At least one check failed. Treat the performance numbers as unproven."
            }</p>
          </div>
        </div>
        <ul class="findings">
          ${v.findings.map((f) => `<li>${chip(f.status)}<span class="check">${escapeHtml(f.check)}</span><span class="detail">${escapeHtml(f.detail)}</span></li>`).join("")}
        </ul>
      </section>`;
      })()
    : `<section class="panel">${missing("A single-strategy verdict", "Pass verdict=falsify(...) to build_report.")}</section>`;

  const beGrid = v?.cost_curve.break_even_is_beyond_grid;
  const tiles = [
    p ? tile("Sharpe, annualized", fixed(p.annualized_sharpe), perBar(p.sharpe)) : "",
    p ? tile("Probabilistic Sharpe", prob(p.psr), "P(true Sharpe > 0), one trial") : "",
    t ? tile("Deflated Sharpe", prob(t.dsr), `best of ${t.n_trials} trials, luck removed`,
      "Probability the best trial's Sharpe beats the best you would expect from luck across every recorded trial.") : "",
    t?.pbo ? tile("Overfitting prob. (PBO)", pct(t.pbo.pbo), `OOS loss in ${pct(t.pbo.prob_oos_loss, 0)} of splits`) : "",
    v ? tile("Break-even cost", beGrid ? "> " + fixed(v.cost_curve.bps[v.cost_curve.bps.length - 1] ?? null, 0) + " bps" : `${fixed(v.cost_curve.break_even_bps, 1)} bps`, "per unit of turnover") : "",
    p ? tile("Max drawdown", pct(p.max_drawdown), `total return ${signedPct(p.total_return, 0)}`) : "",
    p ? tile("Min. track record", `${count(p.min_track_record_length)} bars`, `have ${count(p.n_obs)} (95% confidence)`) : "",
  ].filter(Boolean).join("");

  app.innerHTML = `
    ${header(r)}
    ${verdictHtml}
    <section class="tiles" aria-label="Headline numbers">${tiles}</section>

    <section class="panel wide">
      <div class="panel-head">
        <h2>Equity, net of costs</h2>
        <div class="legend"><span><i style="background:${C.s1}"></i>Net</span><span><i class="dash" style="border-color:${C.s2}"></i>Before costs and carry</span></div>
      </div>
      ${r.series ? `<div class="chart" id="c-equity"></div><h3>Drawdown</h3><div class="chart" id="c-dd"></div>` : missing("The equity curve", "It comes with the verdict.")}
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head"><h2>Permutation test</h2><span class="tag">F3 spurious edge</span></div>
        ${v ? `<p class="lede">Per-bar Sharpe when the signal is rotated in time against the returns. A real edge sits far right of this cloud; p = ${prob(v.permutation.p_value)}.</p><div class="chart" id="c-perm"></div>` : missing("The permutation test", "It comes with the verdict.")}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>Execution delay</h2><span class="tag">F2 look-ahead</span></div>
        ${v ? `<p class="lede">Sharpe by bars between decision and trade. ${
          v.lags.cliff_ratio == null ? "" : `One extra bar keeps ${pct(v.lags.cliff_ratio, 0)} of it${v.lags.flagged ? ", a cliff: check that every input was known in time" : ""}.`
        }</p><div class="chart" id="c-lag"></div>` : missing("The lag profile", "It comes with the verdict.")}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>Cost sensitivity</h2><span class="tag">F5 costs</span></div>
        ${v ? `<p class="lede">Per-bar Sharpe as a flat cost per unit of turnover rises. Carry stays as modelled.</p><div class="chart" id="c-cost"></div>` : missing("The cost curve", "It comes with the verdict.")}
      </section>
      <section class="panel">
        <div class="panel-head"><h2>CPCV path Sharpes</h2><span class="tag">F6 path dependence</span></div>
        ${r.paths ? `<p class="lede">${r.paths.sharpes.length} backtest paths, ${pct(r.paths.summary.share_positive ?? null, 0)} positive. Paths share training data, so this spread is not a confidence interval.</p><div class="chart" id="c-paths"></div>` : missing("CPCV paths", "Pass paths=cpcv_path_returns(...) to build_report.")}
      </section>
    </div>

    ${r.paths ? `<section class="panel wide">
      <div class="panel-head">
        <h2>CPCV path equity</h2>
        <div class="legend"><span><i style="background:${C.s1};opacity:.45"></i>Each path</span><span><i style="background:${C.s2}"></i>Median path</span></div>
      </div>
      <div class="chart" id="c-fan"></div>
    </section>` : ""}

    <section class="panel wide">
      <div class="panel-head"><h2>Trial search</h2><span class="tag">F4 selection bias</span></div>
      ${t ? `
        <p class="lede">Every configuration tried, not just the winner. The best trial's Sharpe must clear what the best of ${t.n_trials} would reach by luck alone.</p>
        <div class="split">
          <div><h3>Sharpe of every trial</h3><div class="chart" id="c-trials"></div></div>
          <div><h3>PBO: rank logits of the in-sample winner out of sample</h3>
            ${t.pbo ? `<div class="chart" id="c-pbo"></div>` : missing("PBO", escapeHtml(t.pbo_unavailable ?? "It needs at least two trials on the same observations."))}
          </div>
        </div>
        <div class="table-wrap"><table id="trials-table">
          <thead><tr>
            <th scope="col" data-sort="number">#</th>
            <th scope="col" data-sort="name">Trial</th>
            <th scope="col">Parameters</th>
            <th scope="col" data-sort="sharpe" class="num">Sharpe / bar</th>
            <th scope="col" data-sort="psr" class="num">PSR</th>
          </tr></thead>
          <tbody></tbody>
        </table></div>` : missing("The trial log", "Pass trials=TrialLog to build_report; without it the deflated Sharpe cannot be computed.")}
    </section>

    ${r.assumptions ? `<section class="panel wide assumptions">
      <h2>Assumptions</h2>
      <dl>${Object.entries(r.assumptions).map(([k, val]) => `<div><dt>${escapeHtml(k)}</dt><dd><code>${escapeHtml(String(val))}</code></dd></div>`).join("")}</dl>
    </section>` : ""}
  `;
  wireHeader();
  if (t) renderTable(t.trials, t.best, "sharpe", -1);
  drawCharts();
}

// --------------------------------------------------------------------------- //
// Charts                                                                      //
// --------------------------------------------------------------------------- //
// Axis tick labels: three significant digits, so 0.025 never prints as 0.03.
const sig = (v: number) => String(Number(v.toPrecision(3)));
const pctTick = (v: number) => `${sig(v * 100)}%`;

function timeTicks(labels: string[]) {
  return (x0: number, x1: number) => {
    const n = Math.min(app.clientWidth < 600 ? 3 : 6, labels.length);
    return Array.from({ length: n }, (_, k) => {
      const i = Math.round(x0 + ((x1 - x0) * k) / Math.max(1, n - 1));
      return { at: i, label: date(labels[i] ?? "").slice(0, 7) };
    });
  };
}

function numTicks(fmt: (v: number) => string) {
  return (x0: number, x1: number) => niceTicks(x0, x1, 5).filter((v) => v >= x0 && v <= x1).map((v) => ({ at: v, label: fmt(v) }));
}

function row(color: string, name: string, value: string): string {
  return `<div class="tip-row"><i style="background:${color}"></i>${escapeHtml(name)}<b>${value}</b></div>`;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function drawCharts(): void {
  const r = current;
  if (!r) return;
  const get = (id: string) => document.getElementById(id);
  const mult = (v: number) => `${v.toFixed(2)}×`;

  if (r.series && get("c-equity")) {
    const s = r.series;
    const xs = s.index.map((_, i) => i);
    lineChart(get("c-equity")!, {
      x: xs, height: 280, yFormat: mult, xTicks: timeTicks(s.index), refY: [1], label: "Equity curve, net and before costs",
      series: [
        { name: "Before costs and carry", values: s.gross_equity, color: C.s2, dash: "5 4", width: 1.5 },
        { name: "Net", values: s.equity, color: C.s1 },
      ],
      tooltip: (i) => `<div class="tip-head">${date(s.index[i]!)}</div>` +
        row(C.s1, "Net", mult(s.equity[i] ?? NaN)) + row(C.s2, "Before costs", mult(s.gross_equity[i] ?? NaN)),
    });
    lineChart(get("c-dd")!, {
      x: xs, height: 140, yFormat: pctTick, xTicks: timeTicks(s.index), yInclude: [0], label: "Drawdown",
      series: [{ name: "Drawdown", values: s.drawdown, color: C.critical, width: 1.5, area: true }],
      tooltip: (i) => `<div class="tip-head">${date(s.index[i]!)}</div>` + row(C.critical, "Drawdown", pct(s.drawdown[i] ?? null)),
    });
  }

  const v = r.verdict;
  if (v && get("c-perm")) {
    histogram(get("c-perm")!, {
      values: v.permutation.null, height: 220, xFormat: sig, color: C.muted, unit: "rotations",
      label: "Null distribution of Sharpe under time rotation",
      markers: v.permutation.observed == null ? [] : [{ at: v.permutation.observed, label: `Observed ${v.permutation.observed.toFixed(3)}`, color: C.s1 }],
    });
  }
  if (v && get("c-lag")) {
    const lags = v.lags;
    barChart(get("c-lag")!, {
      labels: lags.lags.map((l) => `${l}`), values: lags.sharpes, height: 220, yFormat: sig,
      color: C.muted, highlight: 0, highlightColor: C.s1, showEveryLabel: true, label: "Sharpe by execution lag",
      tooltip: (i) => `<div class="tip-head">Lag ${lags.lags[i]} bar${lags.lags[i] === 1 ? "" : "s"}${i === 0 ? " (as tested)" : ""}</div>` +
        row(i === 0 ? C.s1 : C.muted, "Sharpe / bar", fixed(lags.sharpes[i] ?? null, 4)),
    });
  }
  if (v && get("c-cost")) {
    const cc = v.cost_curve;
    const be = cc.break_even_bps;
    lineChart(get("c-cost")!, {
      x: cc.bps.map((b) => b ?? 0), height: 220, yFormat: sig, refY: [0],
      xTicks: numTicks((b) => `${b} bps`), label: "Sharpe against flat cost",
      refX: be != null && !cc.break_even_is_beyond_grid ? { at: be, label: `Break-even ${be.toFixed(1)} bps` } : undefined,
      series: [{ name: "Sharpe", values: cc.sharpes, color: C.s1, markers: true }],
      tooltip: (i) => `<div class="tip-head">${cc.bps[i]} bps per unit turnover</div>` + row(C.s1, "Sharpe / bar", fixed(cc.sharpes[i] ?? null, 4)),
    });
  }

  const P = r.paths;
  if (P && get("c-paths")) {
    const order = P.sharpes.map((s, i) => ({ s, i })).sort((a, b) => (b.s ?? -Infinity) - (a.s ?? -Infinity));
    const mean = P.summary.sharpe_mean ?? null;
    barChart(get("c-paths")!, {
      labels: order.map((o) => `${o.i}`), values: order.map((o) => o.s), height: 220, yFormat: sig,
      color: C.s1, label: "Sharpe of each CPCV path, sorted",
      refY: mean == null ? undefined : { at: mean, label: `Mean ${mean.toFixed(3)}` },
      tooltip: (k) => `<div class="tip-head">Path ${order[k]!.i}</div>` + row(C.s1, "Sharpe / bar", fixed(order[k]!.s, 4)),
    });
  }
  if (P && get("c-fan")) {
    const xs = P.index.map((_, i) => i);
    const med = xs.map((i) => {
      const col = P.equity.map((e) => e[i]).filter((x): x is number => x != null);
      return col.length ? median(col) : null;
    });
    lineChart(get("c-fan")!, {
      x: xs, height: 260, yFormat: mult, xTicks: timeTicks(P.index), refY: [1], label: "Equity of every CPCV path",
      series: [
        ...P.equity.map((e, k) => ({ name: `Path ${k}`, values: e, color: C.s1, width: 1, opacity: 0.3 })),
        { name: "Median path", values: med, color: C.s2, width: 2 },
      ],
      tooltip: (i) => {
        const col = P.equity.map((e) => e[i]).filter((x): x is number => x != null);
        return `<div class="tip-head">${date(P.index[i]!)}</div>` +
          row(C.s2, "Median path", mult(med[i] ?? NaN)) +
          row(C.s1, "Worst path", mult(Math.min(...col))) + row(C.s1, "Best path", mult(Math.max(...col)));
      },
    });
  }

  const t = r.trials;
  if (t && get("c-trials")) {
    const best = t.trials.find((x) => x.number === t.best);
    histogram(get("c-trials")!, {
      values: t.trials.map((x) => x.sharpe), height: 220, xFormat: sig, color: C.muted, unit: "trials",
      label: "Sharpe of every recorded trial",
      markers: [
        ...(best?.sharpe != null ? [{ at: best.sharpe, label: `Best ${best.sharpe.toFixed(3)}`, color: C.s1 }] : []),
        ...(t.expected_max_sharpe != null ? [{ at: t.expected_max_sharpe, label: `Luck alone ${t.expected_max_sharpe.toFixed(3)}`, color: C.s2 }] : []),
      ],
    });
  }
  if (t?.pbo && get("c-pbo")) {
    histogram(get("c-pbo")!, {
      values: t.pbo.logits, height: 220, xFormat: sig, color: C.muted, unit: "splits",
      label: "Out-of-sample rank logits of the in-sample best",
      markers: [{ at: 0, label: `Median rank · PBO ${pct(t.pbo.pbo, 0)} left of here`, color: C.s2 }],
    });
  }
}

// --------------------------------------------------------------------------- //
// Trials table                                                                //
// --------------------------------------------------------------------------- //
type SortKey = "number" | "name" | "sharpe" | "psr";

function renderTable(rows: TrialRow[], best: number, key: SortKey, dir: 1 | -1): void {
  const table = document.querySelector<HTMLTableElement>("#trials-table");
  if (!table) return;
  const val = (r: TrialRow) => r[key] ?? -Infinity;
  const sorted = [...rows].sort((a, b) => {
    const x = val(a);
    const y = val(b);
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });
  table.tBodies[0]!.innerHTML = sorted
    .map((r) => `<tr class="${r.number === best ? "best" : ""}">
      <td class="num">${r.number}</td>
      <td>${escapeHtml(r.name)}${r.number === best ? ' <span class="pill">selected</span>' : ""}</td>
      <td class="params">${Object.entries(r.params).map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(String(v))}`).join(", ")}</td>
      <td class="num">${fixed(r.sharpe, 4)}</td>
      <td class="num">${prob(r.psr)}</td>
    </tr>`)
    .join("");
  table.querySelectorAll<HTMLTableCellElement>("th[data-sort]").forEach((th) => {
    const k = th.dataset.sort as SortKey;
    th.setAttribute("aria-sort", k === key ? (dir === 1 ? "ascending" : "descending") : "none");
    th.tabIndex = 0;
    th.onclick = th.onkeydown = (e: Event) => {
      if (e instanceof KeyboardEvent && e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      renderTable(rows, best, k, k === key ? ((-dir) as 1 | -1) : k === "name" || k === "number" ? 1 : -1);
    };
  });
}

// Redraw charts when the layout width changes.
let lastWidth = 0;
let pending = 0;
new ResizeObserver(([entry]) => {
  const w = Math.round(entry?.contentRect.width ?? 0);
  if (w === lastWidth) return;
  lastWidth = w;
  cancelAnimationFrame(pending);
  pending = requestAnimationFrame(drawCharts);
}).observe(app);

void boot();
