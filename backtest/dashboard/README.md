# purgedcv-backtest dashboard

A TypeScript viewer for `purgedcv-backtest` reports. It computes nothing:
every number comes from the JSON that `purgedcv_backtest.write_report`
writes, so the dashboard can never disagree with the Python statistics.

```bash
python ../examples/demo_report.py   # writes public/report.json
npm install
npm run dev                         # http://localhost:5173/dashboard/
```

| Command | What it does |
|---|---|
| `npm run dev` | Dev server at `http://localhost:5173/dashboard/`, reloading on changes |
| `npm run build` | Type-check and build into `dist/` |
| `npm run export` | Build, then write `dist/report.html`: one file with the report inlined, no server needed |
| `node scripts/standalone.mjs <report.json> <out.html>` | Inline any report into a standalone page |

To look at a different report, press **Open report…** or drop a JSON file
on the page. Sections missing from a report are shown as missing, with
what to pass to `build_report` to fill them.

Requires Node 20.19 or newer (Vite 7).
