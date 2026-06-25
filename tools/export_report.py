"""Export factor analysis report as a static HTML for GitHub Pages.

Reads the bundled `Japan_5_Factors.csv` (Ken French style monthly Japan
data, July 1990 onwards), produces four interactive plotly charts +
descriptive statistics, and writes the whole thing as a single self-contained
HTML to `docs/index.html`.

Wired into `.github/workflows/build-report.yml` so the report is regenerated
on every push to main. The result is served via GitHub Pages at
`https://shihuiarenjinba-png.github.io/factor-simulator/`.

Run locally:
    python tools/export_report.py

No API keys / external services required — the script is fully reproducible
from the data already in the repo.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "Japan_5_Factors.csv"
OUT = ROOT / "docs" / "index.html"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def load_factors() -> pd.DataFrame:
    """Load Ken French style Japan 5-factor monthly file.

    The raw file has 4 metadata lines then a single block of monthly rows
    keyed by `YYYYMM`. Yearly aggregate rows (4-digit keys) are dropped.
    Sentinel `-99.99` is converted to NaN.
    """
    df = pd.read_csv(DATA, skiprows=5, sep=",", skipinitialspace=True)
    df.columns = ["YearMonth", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df["YearMonth"] = df["YearMonth"].astype(str).str.strip()
    df = df[df["YearMonth"].str.len() == 6]
    df["Date"] = pd.to_datetime(df["YearMonth"], format="%Y%m")
    df = df.set_index("Date").drop(columns="YearMonth")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace(-99.99, pd.NA).dropna(how="all")


def fig_cumulative(df: pd.DataFrame) -> go.Figure:
    cum = (1 + df[FACTORS] / 100).cumprod()
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[f], mode="lines", name=f))
    fig.update_layout(
        title="Cumulative Wealth Index per Factor (1990-07 = 1.0)",
        xaxis_title="Date", yaxis_title="Wealth",
        template="plotly_white", height=420,
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    return fig


def fig_correlation(df: pd.DataFrame) -> go.Figure:
    corr = df[FACTORS].corr().round(3)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Pairwise Correlation of Monthly Factor Returns",
    )
    fig.update_layout(template="plotly_white", height=380)
    return fig


def fig_distribution(df: pd.DataFrame) -> go.Figure:
    long = df[FACTORS].melt(var_name="Factor", value_name="MonthlyReturn(%)")
    fig = px.box(
        long, x="Factor", y="MonthlyReturn(%)", color="Factor",
        title="Monthly Return Distribution per Factor (full history)",
        template="plotly_white",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


def fig_recent_36m(df: pd.DataFrame) -> go.Figure:
    tail = df[FACTORS].tail(36)
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Bar(x=tail.index, y=tail[f], name=f))
    fig.update_layout(
        barmode="group",
        title="Last 36 Months — Monthly Factor Returns (%)",
        template="plotly_white", height=420,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


HTML_HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>Japan 5 Factor — Live Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Japan 5 factor (Mkt-RF / SMB / HML / RMW / CMA) live analysis, auto-rebuilt on push.">
<meta property="og:title" content="Japan 5 Factor — Live Report">
<meta property="og:description" content="Cumulative performance, correlations, distribution, last 36 months — interactive plotly.">
<style>
*{box-sizing:border-box}
html,body{margin:0;background:#f8fafc;color:#0f172a;
  font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic","Noto Sans CJK JP",sans-serif;
  -webkit-text-size-adjust:100%}
.wrap{max-width:960px;margin:0 auto;padding:24px 18px 64px}
header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:24px;margin:0;font-weight:700;letter-spacing:-.01em}
.sub{color:#64748b;margin:0;font-size:14px}
.badges{margin:6px 0 18px;display:flex;gap:6px;flex-wrap:wrap}
.badge{display:inline-block;background:#e0f2fe;color:#0369a1;
  padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge.alt{background:#f1f5f9;color:#334155}
section{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 3px rgb(0 0 0 /.04)}
section h2{font-size:15px;margin:0 0 12px;font-weight:700;color:#0f172a}
.stats{font-size:12.5px;border-collapse:collapse;width:100%}
.stats th,.stats td{padding:6px 10px;text-align:right;border-bottom:1px solid #e2e8f0;
  font-variant-numeric:tabular-nums}
.stats th{background:#f1f5f9;font-weight:600;text-align:left}
footer{text-align:center;color:#94a3b8;font-size:12px;margin-top:32px;line-height:1.7}
footer a{color:#0ea5e9}
@media(prefers-color-scheme:dark){
  body{background:#0a0f1a;color:#f1f5f9}
  .sub{color:#94a3b8}
  .badge{background:#0c4a6e;color:#7dd3fc}
  .badge.alt{background:#1e293b;color:#cbd5e1}
  section{background:#0f172a;border-color:#1f2a3a;box-shadow:0 1px 3px rgb(0 0 0 /.3)}
  section h2{color:#f1f5f9}
  .stats th{background:#1e293b;color:#cbd5e1}
  .stats td{border-color:#1f2a3a}
  footer a{color:#38bdf8}
}
@media(max-width:480px){
  .wrap{padding:16px 12px 48px}
  h1{font-size:20px}
}
</style></head><body><div class="wrap">
"""

HTML_FOOTER = """
<footer>
Auto-generated by <a href="https://github.com/shihuiarenjinba-png/factor-simulator/actions" target="_blank" rel="noopener">GitHub Actions</a>
 ・ <a href="https://github.com/shihuiarenjinba-png/factor-simulator" target="_blank" rel="noopener">Source on GitHub</a><br>
Data source: Ken French Library, Japan 5 factors (bundled `Japan_5_Factors.csv`)
</footer>
</div></body></html>
"""


def render_html(figs: dict[str, go.Figure], df: pd.DataFrame) -> str:
    start = df.index.min().strftime("%Y-%m")
    end = df.index.max().strftime("%Y-%m")
    build_ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    summary = df[FACTORS + ["RF"]].describe().round(2).to_html(classes="stats", border=0)

    parts = [HTML_HEAD]
    parts.append(f'''<header>
<h1>📊 Japan 5 Factor — Live Report</h1>
</header>
<p class="sub">Bundled Ken French style monthly data — period: <b>{start}</b> 〜 <b>{end}</b> ({len(df)} rows)</p>
<div class="badges">
  <span class="badge">Mkt-RF · SMB · HML · RMW · CMA</span>
  <span class="badge alt">Monthly</span>
  <span class="badge alt">No API key</span>
  <span class="badge alt">Build: {build_ts}</span>
</div>''')
    for title, fig in figs.items():
        snippet = fig.to_html(include_plotlyjs="cdn", full_html=False,
                              config={"displaylogo": False, "responsive": True})
        parts.append(f'<section><h2>{title}</h2>{snippet}</section>')
    parts.append(f'<section><h2>Descriptive Statistics (%)</h2>{summary}</section>')
    parts.append(HTML_FOOTER)
    return "".join(parts)


def main() -> None:
    df = load_factors()
    figs = {
        "Cumulative Performance": fig_cumulative(df),
        "Pairwise Correlation": fig_correlation(df),
        "Monthly Distribution": fig_distribution(df),
        "Last 36 Months": fig_recent_36m(df),
    }
    html = render_html(figs, df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"WROTE {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
