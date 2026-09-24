import type { Num } from "./types";

const DASH = "–";

export function fixed(x: Num | undefined, digits = 2): string {
  return x == null ? DASH : x.toFixed(digits);
}

export function pct(x: Num | undefined, digits = 1): string {
  return x == null ? DASH : `${(x * 100).toFixed(digits)}%`;
}

export function signedPct(x: Num | undefined, digits = 1): string {
  if (x == null) return DASH;
  const s = (x * 100).toFixed(digits);
  return x > 0 ? `+${s}%` : `${s}%`;
}

export function prob(x: Num | undefined): string {
  if (x == null) return DASH;
  if (x > 0.9999) return "> 0.9999";
  if (x < 0.0001) return "< 0.0001";
  return x.toFixed(4);
}

export function count(x: Num | undefined): string {
  if (x == null) return "∞";
  return Math.round(x).toLocaleString("en-US");
}

export function date(label: string): string {
  const d = new Date(label);
  return Number.isNaN(d.getTime()) ? label : d.toISOString().slice(0, 10);
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}
