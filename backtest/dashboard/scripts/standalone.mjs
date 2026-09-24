// Inline the built dashboard and a report into one self-contained HTML file
// that opens from disk (no server) and can be shared as a single file.
//
//   node scripts/standalone.mjs [report.json] [out.html] [--fragment]
//
// --fragment drops the <html>/<head>/<body> wrapper, for hosts that add
// their own.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const fragment = args.includes("--fragment");
const [reportPath = join(root, "public", "report.json"), outPath = join(root, "dist", "report.html")] =
  args.filter((a) => !a.startsWith("--"));

const dist = join(root, "dist");
let html = readFileSync(join(dist, "index.html"), "utf8");
const asset = (href) => readFileSync(join(dist, href.replace(/^.*?assets\//, "assets/")), "utf8");
const safe = (s) => s.replace(/<\/(script)/gi, "<\\/$1");

let css = "";
let js = "";
html = html
  .replace(/<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_, href) => ((css += asset(href)), ""))
  .replace(/<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g, (_, src) => ((js += asset(src)), ""));
if (!css || !js) throw new Error("dist/index.html has no bundled assets; run `vite build` first.");

const report = JSON.parse(readFileSync(reportPath, "utf8"));
const data = `<script type="application/json" id="report-data">${safe(JSON.stringify(report))}</script>`;
const title = /<title>.*?<\/title>/.exec(html)?.[0] ?? "<title>Backtest Verdict</title>";
const body = `<div id="app"></div>\n${data}\n<script type="module">${safe(js)}</script>`;

const out = fragment
  ? `${title}\n<style>${css}</style>\n${body}\n`
  : html.replace("</head>", `<style>${css}</style>\n</head>`).replace('<div id="app"></div>', body);
writeFileSync(outPath, out);
console.log(`wrote ${outPath} (${(out.length / 1024).toFixed(0)} KiB) from ${reportPath}`);
