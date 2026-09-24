// Small dependency-free SVG charts: line, histogram and bar. Colours are CSS
// custom properties so both themes restyle every chart without a redraw.

import type { Num } from "./types";
import { escapeHtml } from "./format";

const SVG = "http://www.w3.org/2000/svg";
const M = { t: 14, r: 18, b: 30, l: 56 };

type Fmt = (v: number) => string;

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  parent?: Element,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent?.appendChild(node);
  return node;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) {
    const pad = Math.abs(lo) * 0.1 || 1;
    lo -= pad;
    hi += pad;
  }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

interface Frame {
  svg: SVGSVGElement;
  w: number;
  h: number;
  x: (v: number) => number;
  y: (v: number) => number;
  plot: SVGGElement;
}

/** Axes, grid and scales shared by every chart. */
function frame(
  host: HTMLElement,
  height: number,
  xDomain: [number, number],
  yDomain: [number, number],
  yFormat: Fmt,
  xTicks: { at: number; label: string }[],
  ariaLabel: string,
): Frame {
  host.replaceChildren();
  const w = Math.max(280, host.clientWidth);
  const h = height;
  const ticks = niceTicks(yDomain[0], yDomain[1], height < 200 ? 3 : 5);
  const y0 = Math.min(yDomain[0], ticks[0] ?? yDomain[0]);
  const y1 = Math.max(yDomain[1], ticks[ticks.length - 1] ?? yDomain[1]);
  const [x0, x1] = xDomain[1] > xDomain[0] ? xDomain : [xDomain[0] - 1, xDomain[0] + 1];
  const x = (v: number) => M.l + ((v - x0) / (x1 - x0)) * (w - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - y0) / (y1 - y0)) * (h - M.t - M.b);

  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img", "aria-label": ariaLabel });
  const axis = el("g", { class: "axis" }, svg);
  for (const t of ticks) {
    el("line", { x1: M.l, x2: w - M.r, y1: y(t), y2: y(t), class: t === 0 ? "baseline" : "grid" }, axis);
    el("text", { x: M.l - 8, y: y(t), "text-anchor": "end", "dominant-baseline": "middle" }, axis).textContent =
      yFormat(t);
  }
  for (const t of xTicks) {
    el("text", { x: x(t.at), y: h - 8, "text-anchor": "middle" }, axis).textContent = t.label;
  }
  const plot = el("g", {}, svg);
  host.appendChild(svg);
  return { svg, w, h, x, y, plot };
}

function tooltip(host: HTMLElement): HTMLDivElement {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  host.appendChild(tip);
  return tip;
}

function place(tip: HTMLDivElement, host: HTMLElement, px: number, py: number): void {
  tip.hidden = false;
  const room = host.clientWidth - px;
  tip.style.left = room < tip.offsetWidth + 24 ? `${px - tip.offsetWidth - 12}px` : `${px + 12}px`;
  tip.style.top = `${Math.max(0, py - tip.offsetHeight / 2)}px`;
}

function finiteRange(arrays: Num[][], extra: number[] = []): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) for (const v of a) if (v != null) ((lo = Math.min(lo, v)), (hi = Math.max(hi, v)));
  for (const v of extra) ((lo = Math.min(lo, v)), (hi = Math.max(hi, v)));
  return Number.isFinite(lo) ? [lo, hi] : [0, 1];
}

function pathD(xs: number[], ys: Num[], x: (v: number) => number, y: (v: number) => number): string {
  let d = "";
  let pen = false;
  ys.forEach((v, i) => {
    if (v == null) return void (pen = false);
    d += `${pen ? "L" : "M"}${x(xs[i]!).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

// --------------------------------------------------------------------------- //
// Line chart                                                                  //
// --------------------------------------------------------------------------- //
export interface LineSeries {
  name: string;
  values: Num[];
  color: string;
  width?: number;
  opacity?: number;
  dash?: string;
  area?: boolean;
  markers?: boolean;
}

export interface LineOpts {
  x: number[];
  series: LineSeries[];
  height: number;
  yFormat: Fmt;
  xTicks: (x0: number, x1: number) => { at: number; label: string }[];
  tooltip: (i: number) => string;
  refY?: number[];
  refX?: { at: number; label: string };
  yInclude?: number[];
  label: string;
}

export function lineChart(host: HTMLElement, o: LineOpts): void {
  const x0 = o.x[0] ?? 0;
  const x1 = o.x[o.x.length - 1] ?? 1;
  const f = frame(
    host,
    o.height,
    [x0, x1],
    finiteRange(o.series.map((s) => s.values), [...(o.refY ?? []), ...(o.yInclude ?? [])]),
    o.yFormat,
    o.xTicks(x0, x1),
    o.label,
  );
  for (const r of o.refY ?? []) {
    el("line", { x1: M.l, x2: f.w - M.r, y1: f.y(r), y2: f.y(r), class: "ref" }, f.plot);
  }
  if (o.refX) {
    const rx = f.x(Math.min(Math.max(o.refX.at, x0), x1));
    el("line", { x1: rx, x2: rx, y1: M.t, y2: f.h - M.b, class: "ref" }, f.plot);
    el("text", { x: rx + 6, y: M.t + 10, class: "ref-label" }, f.plot).textContent = o.refX.label;
  }
  for (const s of o.series) {
    const d = pathD(o.x, s.values, f.x, f.y);
    if (s.area) {
      const firstI = s.values.findIndex((v) => v != null);
      const lastI = s.values.length - 1 - [...s.values].reverse().findIndex((v) => v != null);
      if (firstI >= 0) {
        const base = f.y(0);
        el("path", {
          d: `${d}L${f.x(o.x[lastI]!).toFixed(1)},${base}L${f.x(o.x[firstI]!).toFixed(1)},${base}Z`,
          style: `fill:${s.color};opacity:0.14`,
        }, f.plot);
      }
    }
    el("path", {
      d,
      style: `fill:none;stroke:${s.color};stroke-width:${s.width ?? 2};opacity:${s.opacity ?? 1};` +
        `stroke-linejoin:round;stroke-linecap:round${s.dash ? `;stroke-dasharray:${s.dash}` : ""}`,
    }, f.plot);
    if (s.markers) {
      s.values.forEach((v, i) => {
        if (v == null) return;
        el("circle", { cx: f.x(o.x[i]!), cy: f.y(v), r: 4, class: "marker", style: `fill:${s.color}` }, f.plot);
      });
    }
  }

  // Hover layer: crosshair, dots on the hovered series, tooltip.
  const tip = tooltip(host);
  const cross = el("line", { y1: M.t, y2: f.h - M.b, class: "crosshair", visibility: "hidden" }, f.svg);
  const dots = el("g", { visibility: "hidden" }, f.svg);
  const hot = o.series.filter((s) => (s.opacity ?? 1) >= 0.9);
  const dotEls = hot.map((s) => el("circle", { r: 4.5, class: "hover-dot", style: `fill:${s.color}` }, dots));
  const hit = el("rect", { x: M.l, y: 0, width: f.w - M.l - M.r, height: f.h, fill: "transparent" }, f.svg);
  const nearest = (px: number) => {
    let best = 0;
    let dist = Infinity;
    o.x.forEach((v, i) => {
      const d = Math.abs(f.x(v) - px);
      if (d < dist) ((dist = d), (best = i));
    });
    return best;
  };
  const scale = () => f.svg.getBoundingClientRect().width / f.w || 1;
  hit.addEventListener("pointermove", (ev) => {
    const px = ev.offsetX / scale();
    const i = nearest(px);
    const cx = f.x(o.x[i]!);
    cross.setAttribute("x1", String(cx));
    cross.setAttribute("x2", String(cx));
    cross.setAttribute("visibility", "visible");
    dots.setAttribute("visibility", "visible");
    hot.forEach((s, k) => {
      const v = s.values[i];
      const dot = dotEls[k]!;
      dot.setAttribute("visibility", v == null ? "hidden" : "visible");
      if (v != null) (dot.setAttribute("cx", String(cx)), dot.setAttribute("cy", String(f.y(v))));
    });
    tip.innerHTML = o.tooltip(i);
    place(tip, host, cx * scale(), ev.offsetY);
  });
  hit.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.setAttribute("visibility", "hidden");
  });
}

// --------------------------------------------------------------------------- //
// Histogram with reference markers                                            //
// --------------------------------------------------------------------------- //
export interface HistMarker {
  at: number;
  label: string;
  color: string;
}

export interface HistOpts {
  values: Num[];
  bins?: number;
  height: number;
  xFormat: Fmt;
  markers: HistMarker[];
  color: string;
  unit: string;
  label: string;
}

export function histogram(host: HTMLElement, o: HistOpts): void {
  const vals = o.values.filter((v): v is number => v != null);
  const [lo0, hi0] = finiteRange([vals], o.markers.map((m) => m.at));
  const span = hi0 - lo0 || 1;
  const lo = lo0 - span * 0.04;
  const hi = hi0 + span * 0.04;
  const nb = o.bins ?? Math.min(40, Math.max(8, Math.round(Math.sqrt(vals.length) * 1.5)));
  const width = (hi - lo) / nb;
  const counts = new Array<number>(nb).fill(0);
  for (const v of vals) counts[Math.min(nb - 1, Math.floor((v - lo) / width))]!++;
  const xt = niceTicks(lo, hi, 5).filter((t) => t >= lo && t <= hi);
  const f = frame(host, o.height, [lo, hi], [0, Math.max(...counts, 1)], (v) => String(Math.round(v)),
    xt.map((t) => ({ at: t, label: o.xFormat(t) })), o.label);

  const tip = tooltip(host);
  const scale = () => f.svg.getBoundingClientRect().width / f.w || 1;
  counts.forEach((c, i) => {
    const a = lo + i * width;
    const xa = f.x(a) + 1;
    const bw = Math.max(1, f.x(a + width) - f.x(a) - 2);
    if (c > 0) {
      const top = f.y(c);
      const base = f.y(0);
      const r = Math.min(4, bw / 2, base - top);
      el("path", {
        d: `M${xa},${base}V${top + r}Q${xa},${top} ${xa + r},${top}H${xa + bw - r}Q${xa + bw},${top} ${xa + bw},${top + r}V${base}Z`,
        style: `fill:${o.color}`,
      }, f.plot);
    }
    const hit = el("rect", { x: xa - 1, y: M.t, width: bw + 2, height: f.h - M.t - M.b, fill: "transparent" }, f.svg);
    hit.addEventListener("pointermove", (ev) => {
      tip.innerHTML = `<b>${c}</b> ${escapeHtml(o.unit)}<br><span>${o.xFormat(a)} to ${o.xFormat(a + width)}</span>`;
      place(tip, host, (xa + bw / 2) * scale(), ev.offsetY);
    });
    hit.addEventListener("pointerleave", () => (tip.hidden = true));
  });
  o.markers.forEach((m, k) => {
    const mx = f.x(m.at);
    el("line", { x1: mx, x2: mx, y1: M.t, y2: f.h - M.b, class: "marker-line", style: `stroke:${m.color}` }, f.plot);
    const anchor = mx > f.w - 140 ? "end" : "start";
    el("text", {
      x: anchor === "end" ? mx - 6 : mx + 6,
      y: M.t + 10 + k * 16,
      "text-anchor": anchor,
      class: "ref-label",
    }, f.plot).textContent = m.label;
  });
}

// --------------------------------------------------------------------------- //
// Bar chart (categorical x)                                                   //
// --------------------------------------------------------------------------- //
export interface BarOpts {
  labels: string[];
  values: Num[];
  height: number;
  yFormat: Fmt;
  color: string;
  highlight?: number;
  highlightColor?: string;
  refY?: { at: number; label: string };
  tooltip: (i: number) => string;
  label: string;
  showEveryLabel?: boolean;
}

export function barChart(host: HTMLElement, o: BarOpts): void {
  const n = o.values.length;
  const yr = finiteRange([o.values], [0, ...(o.refY ? [o.refY.at] : [])]);
  const every = o.showEveryLabel ? 1 : Math.max(1, Math.ceil(n / 12));
  const f = frame(host, o.height, [-0.5, n - 0.5], yr, o.yFormat,
    o.labels.map((l, i) => ({ at: i, label: i % every === 0 ? l : "" })), o.label);
  const tip = tooltip(host);
  const scale = () => f.svg.getBoundingClientRect().width / f.w || 1;
  const slot = f.x(1) - f.x(0);
  const bw = Math.max(2, Math.min(48, slot * 0.7));
  o.values.forEach((v, i) => {
    const cx = f.x(i);
    if (v != null) {
      const base = f.y(0);
      const top = f.y(v);
      const up = v >= 0;
      const hgt = Math.abs(base - top);
      const r = Math.min(4, bw / 2, hgt);
      const xa = cx - bw / 2;
      const d = up
        ? `M${xa},${base}V${top + r}Q${xa},${top} ${xa + r},${top}H${xa + bw - r}Q${xa + bw},${top} ${xa + bw},${top + r}V${base}Z`
        : `M${xa},${base}V${top - r}Q${xa},${top} ${xa + r},${top}H${xa + bw - r}Q${xa + bw},${top} ${xa + bw},${top - r}V${base}Z`;
      const color = i === o.highlight ? (o.highlightColor ?? o.color) : o.color;
      el("path", { d, style: `fill:${color}` }, f.plot);
    }
    const hit = el("rect", { x: cx - slot / 2, y: M.t, width: slot, height: f.h - M.t - M.b, fill: "transparent" }, f.svg);
    hit.addEventListener("pointermove", (ev) => {
      tip.innerHTML = o.tooltip(i);
      place(tip, host, cx * scale(), ev.offsetY);
    });
    hit.addEventListener("pointerleave", () => (tip.hidden = true));
  });
  if (o.refY) {
    const ry = f.y(o.refY.at);
    el("line", { x1: M.l, x2: f.w - M.r, y1: ry, y2: ry, class: "ref" }, f.plot);
    el("text", { x: f.w - M.r, y: ry - 6, "text-anchor": "end", class: "ref-label" }, f.plot).textContent = o.refY.label;
  }
}
