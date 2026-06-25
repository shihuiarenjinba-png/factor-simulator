"""Export factor analysis report as a static HTML for GitHub Pages.

Reads two bundled Ken French Library files —
`Japan_5_Factors.csv` (1990-07 onwards) and `US_5_Factors.csv` (1963-07
onwards) — and produces an interactive single-page report with:

  - JP / US tab switcher: switch which region's charts you see
  - 4 plotly charts per region: cumulative wealth, pairwise correlation,
    monthly distribution, last 36 months
  - Factor glossary section explaining what each factor (Mkt-RF / SMB /
    HML / RMW / CMA / RF) represents — plain Japanese
  - Descriptive statistics tables for the active region

No API keys, no network calls. Fully reproducible from the bundled CSVs.

Local rebuild:
    python tools/export_report.py
"""
from __future__ import annotations

import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parents[1]
JP_PATH = ROOT / "Japan_5_Factors.csv"
US_PATH = ROOT / "US_5_Factors.csv"
OUT = ROOT / "docs" / "index.html"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

GLOSSARY = [
    ("Mkt-RF",
     "市場プレミアム",
     "市場全体のリターンから安全資産(短期国債)を引いたもの。'株式に投資したご褒美'。"),
    ("SMB",
     "Small Minus Big",
     "小型株(時価総額が小さい)のリターン − 大型株のリターン。小型ほど期待リターンが高い傾向(サイズ効果)。"),
    ("HML",
     "High Minus Low",
     "高 Book-to-Market 株(割安・バリュー)のリターン − 低い株(割高・グロース)のリターン。バリュー効果。"),
    ("RMW",
     "Robust Minus Weak",
     "営業収益性が高い銘柄のリターン − 収益性が低い銘柄のリターン。クオリティ要因。"),
    ("CMA",
     "Conservative Minus Aggressive",
     "投資が保守的な企業のリターン − 投資が積極的な企業のリターン。'設備投資の控えめさ'のプレミアム。"),
    ("RF",
     "Risk-Free Rate",
     "1ヶ月物 TBill (短期国債) のリターン。他のファクターはここからの超過リターンで定義される。"),
]


def load_factors(path: pathlib.Path) -> pd.DataFrame:
    """Read a Ken French style 5-factor CSV (Japan or US).

    Skips the 4-line header (description + blank + column names) and the
    yearly aggregate block at the end, returns a DataFrame indexed by
    monthly Date with columns Mkt-RF / SMB / HML / RMW / CMA / RF.
    """
    # Find the header line dynamically (first line starting with ",Mkt-RF")
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(",Mkt-RF"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Header row not found in {path.name}")

    df = pd.read_csv(path, skiprows=header_idx, sep=",", skipinitialspace=True)
    df.columns = ["YearMonth"] + FACTORS + ["RF"]
    df["YearMonth"] = df["YearMonth"].astype(str).str.strip()
    # Monthly rows are YYYYMM (6 digits); yearly aggregates are YYYY (4 digits).
    df = df[df["YearMonth"].str.len() == 6]
    df["Date"] = pd.to_datetime(df["YearMonth"], format="%Y%m")
    df = df.set_index("Date").drop(columns="YearMonth")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace(-99.99, pd.NA).dropna(how="all")


def fig_cumulative(df: pd.DataFrame, region: str) -> go.Figure:
    cum = (1 + df[FACTORS] / 100).cumprod()
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[f], mode="lines", name=f))
    fig.update_layout(
        title=f"{region} 5 Factors — Cumulative Wealth Index (start = 1.0)",
        xaxis_title="Date", yaxis_title="Wealth",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=60),
    )
    return fig


def fig_correlation(df: pd.DataFrame, region: str) -> go.Figure:
    corr = df[FACTORS].corr().round(3)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title=f"{region} — Pairwise Correlation of Monthly Factor Returns",
    )
    fig.update_layout(template="plotly_white", height=360,
                      margin=dict(l=50, r=20, t=50, b=20))
    return fig


def fig_distribution(df: pd.DataFrame, region: str) -> go.Figure:
    long = df[FACTORS].melt(var_name="Factor", value_name="MonthlyReturn(%)")
    fig = px.box(
        long, x="Factor", y="MonthlyReturn(%)", color="Factor",
        title=f"{region} — Monthly Return Distribution per Factor",
        template="plotly_white",
    )
    fig.update_layout(height=360, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=40))
    return fig


def fig_recent_36m(df: pd.DataFrame, region: str) -> go.Figure:
    tail = df[FACTORS].tail(36)
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Bar(x=tail.index, y=tail[f], name=f))
    fig.update_layout(
        barmode="group",
        title=f"{region} — Last 36 Months (Monthly Returns, %)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=50, r=20, t=50, b=60),
    )
    return fig


def make_section(name: str, fig: go.Figure) -> str:
    snippet = fig.to_html(include_plotlyjs="cdn", full_html=False,
                          config={"displaylogo": False, "responsive": True})
    return f'<section><h2>{name}</h2>{snippet}</section>'


HTML_HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>Fama-French 5 Factor — Live Report (JP / US)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Japan / US Fama-French 5 factor (Mkt-RF / SMB / HML / RMW / CMA) live analysis with tab switcher and factor glossary.">
<meta property="og:title" content="Fama-French 5 Factor Live Report">
<meta property="og:description" content="Cumulative, correlation, distribution, last 36 months — toggle JP / US. Each factor explained in plain Japanese.">
<style>
*{box-sizing:border-box}
html,body{margin:0;background:#f8fafc;color:#0f172a;
  font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic","Noto Sans CJK JP",sans-serif;
  -webkit-text-size-adjust:100%}
.wrap{max-width:960px;margin:0 auto;padding:24px 18px 64px}
header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px}
h1{font-size:24px;margin:0;font-weight:700;letter-spacing:-.01em}
.sub{color:#64748b;margin:0;font-size:14px;line-height:1.6}
.badges{margin:6px 0 14px;display:flex;gap:6px;flex-wrap:wrap}
.badge{display:inline-block;background:#e0f2fe;color:#0369a1;
  padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge.alt{background:#f1f5f9;color:#334155}
.tabs{display:flex;gap:6px;margin:18px 0 14px;border-bottom:1px solid #e2e8f0}
.tab-btn{padding:9px 18px;border:none;background:none;cursor:pointer;font-size:14px;font-weight:600;
  color:#64748b;border-bottom:2px solid transparent;font-family:inherit;
  margin-bottom:-1px;transition:.12s}
.tab-btn:hover{color:#0f172a}
.tab-btn.active{color:#0ea5e9;border-bottom-color:#0ea5e9}
.tab-content{display:none}
.tab-content.active{display:block}
section{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 3px rgb(0 0 0 /.04)}
section h2{font-size:15px;margin:0 0 12px;font-weight:700;color:#0f172a}
section h3{font-size:14px;margin:0 0 4px;font-weight:700;color:#0f172a}
.glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:4px}
.gloss-item{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:11px 14px}
.gloss-item .code{display:inline-block;background:#0ea5e9;color:#fff;font-weight:700;
  padding:1px 8px;border-radius:6px;font-size:12px;letter-spacing:.01em;
  font-family:ui-monospace,Consolas,Menlo,monospace}
.gloss-item .ja{font-size:13px;color:#64748b;margin:4px 0 6px}
.gloss-item .desc{font-size:13px;color:#334155;line-height:1.55;margin:0}
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
  section h2,section h3{color:#f1f5f9}
  .tabs{border-bottom-color:#1f2a3a}
  .tab-btn{color:#94a3b8}
  .tab-btn:hover{color:#f1f5f9}
  .tab-btn.active{color:#38bdf8;border-bottom-color:#38bdf8}
  .gloss-item{background:#1e293b;border-color:#1f2a3a}
  .gloss-item .ja{color:#94a3b8}
  .gloss-item .desc{color:#cbd5e1}
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
Data: <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html" target="_blank" rel="noopener">Kenneth R. French Data Library</a><br>
<a href="https://github.com/shihuiarenjinba-png/factor-simulator" target="_blank" rel="noopener">Source on GitHub</a>
 ・ Regenerate locally: <code>python tools/export_report.py</code>
</footer>
</div></body></html>
"""


def glossary_section() -> str:
    items = "".join(
        f'<div class="gloss-item">'
        f'<span class="code">{code}</span>'
        f'<div class="ja">{ja}</div>'
        f'<p class="desc">{desc}</p>'
        f'</div>'
        for code, ja, desc in GLOSSARY
    )
    return f'''<section>
<h2>ファクターの意味 — Factor Glossary</h2>
<p style="font-size:13px;color:#64748b;margin:0 0 12px;line-height:1.6">
Fama-French 5 ファクターは、株式リターンを説明するために実証研究で広く使われる
リスクファクター群です。各因子は「ロング・ショート」ポートフォリオの月次リターン (%) として
計算されており、累積で長期トレンド、相関で同時に効くかどうか、分布で振れ幅を読み取ります。
</p>
<div class="glossary">{items}</div>
</section>'''


def tab_charts(df: pd.DataFrame, region: str) -> str:
    figs = [
        ("Cumulative Performance", fig_cumulative(df, region)),
        ("Pairwise Correlation", fig_correlation(df, region)),
        ("Monthly Distribution", fig_distribution(df, region)),
        ("Last 36 Months", fig_recent_36m(df, region)),
    ]
    parts = [make_section(n, f) for n, f in figs]
    summary = df[FACTORS + ["RF"]].describe().round(2).to_html(classes="stats", border=0)
    parts.append(f'<section><h2>{region} — Descriptive Statistics (%)</h2>{summary}</section>')
    return "".join(parts)


def main() -> None:
    jp = load_factors(JP_PATH)
    us = load_factors(US_PATH)
    build_ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")

    parts = [HTML_HEAD]
    parts.append(f'''<header><h1>📊 Fama-French 5 Factor — Live Report</h1></header>
<p class="sub">日本 ({jp.index.min():%Y-%m} 〜 {jp.index.max():%Y-%m}, {len(jp)} 行) と
米国 ({us.index.min():%Y-%m} 〜 {us.index.max():%Y-%m}, {len(us)} 行) のファクターリターンを並べ、
タブで切り替えながら累積パフォーマンス・相関・分布・直近を比較できます。</p>
<div class="badges">
  <span class="badge">Mkt-RF · SMB · HML · RMW · CMA</span>
  <span class="badge alt">月次</span>
  <span class="badge alt">JP + US</span>
  <span class="badge alt">No API key</span>
  <span class="badge alt">Build: {build_ts}</span>
</div>''')

    parts.append('''<div class="tabs" role="tablist">
  <button class="tab-btn active" data-tab="jp" aria-selected="true">🇯🇵 Japan</button>
  <button class="tab-btn" data-tab="us" aria-selected="false">🇺🇸 US</button>
</div>''')

    parts.append('<div class="tab-content active" id="tab-jp">')
    parts.append(tab_charts(jp, "Japan"))
    parts.append('</div>')

    parts.append('<div class="tab-content" id="tab-us">')
    parts.append(tab_charts(us, "US"))
    parts.append('</div>')

    parts.append(glossary_section())

    parts.append('''<script>
document.querySelectorAll(".tab-btn").forEach(b => b.addEventListener("click", e => {
  const t = b.dataset.tab;
  document.querySelectorAll(".tab-btn").forEach(x => {
    x.classList.toggle("active", x.dataset.tab === t);
    x.setAttribute("aria-selected", x.dataset.tab === t ? "true" : "false");
  });
  document.querySelectorAll(".tab-content").forEach(x =>
    x.classList.toggle("active", x.id === "tab-" + t));
  // Trigger plotly resize so it adapts on first reveal
  window.dispatchEvent(new Event("resize"));
}));
</script>''')

    parts.append(HTML_FOOTER)
    html = "".join(parts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"WROTE {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"  JP: {len(jp)} rows ({jp.index.min().date()} → {jp.index.max().date()})")
    print(f"  US: {len(us)} rows ({us.index.min().date()} → {us.index.max().date()})")


if __name__ == "__main__":
    main()
