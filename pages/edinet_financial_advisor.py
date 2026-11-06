from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from edinet_store import (
    EdinetStoreError,
    benchmark_rows,
    dataset_summary,
    db_path,
    get_filing,
    list_filings,
    search_companies,
)
from industry_benchmark import BENCHMARKS, calculate_metrics, diagnose

st.set_page_config(page_title="EDINET DB 同業診断", page_icon="🏢", layout="wide")

st.title("🏢 EDINET DB 同業診断")
st.caption("事前に整形・検証したEDINET財務データをSQLiteから読み込みます。本番画面から外部APIは呼びません。")

db = db_path()
if not db.is_file():
    st.error(f"診断用SQLiteがまだ配置されていません: {db}")
    st.code(
        "python tools/build_edinet_sqlite.py data/normalized_edinet.csv "
        "--db data/edinet_financials.sqlite3 --dataset-version YYYY-MM-DD"
    )
    st.stop()

try:
    summary = dataset_summary()
except EdinetStoreError as exc:
    st.error(str(exc))
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("企業数", f"{summary['companies']:,}")
m2.metric("有報レコード", f"{summary['filings']:,}")
m3.metric("データ版", summary["metadata"].get("dataset_version", "未設定"))
st.caption(
    f"収録決算期: {summary['min_year'] or '-'} ～ {summary['max_year'] or '-'} ｜ "
    f"更新: {summary['metadata'].get('updated_at', '未設定')}"
)

st.markdown("### 会社を探す")
query = st.text_input("会社名 / EDINETコード / 証券コード", placeholder="例：トヨタ / E02144 / 7203")
companies = search_companies(query, limit=100)

if not companies:
    st.info("該当企業がありません。検索条件を変えてください。")
    st.stop()

labels = [
    f"{row['company_name']}｜{row.get('sec_code') or '-'}｜{row['edinet_code']}｜{row['industry_label']}"
    for row in companies
]
company_idx = st.selectbox("企業", range(len(companies)), format_func=lambda i: labels[i])
company = companies[company_idx]

filings = list_filings(company["edinet_code"])
filing_labels = [
    f"{row['fiscal_year_end']}｜{row['doc_id']}"
    for row in filings
]
filing_idx = st.selectbox("決算期", range(len(filings)), format_func=lambda i: filing_labels[i])
filing_meta = filings[filing_idx]

try:
    row = get_filing(filing_meta["doc_id"])
except EdinetStoreError as exc:
    st.error(str(exc))
    st.stop()

st.markdown("### 収録データ")
st.json({
    "会社名": row["company_name"],
    "EDINETコード": row["edinet_code"],
    "証券コード": row.get("sec_code") or "",
    "決算期": row["fiscal_year_end"],
    "会計基準": row["accounting_standard"],
    "業種": row["industry_label"],
    "docID": row["doc_id"],
    "原典URL": row["source_url"],
    "原典SHA-256": row["source_sha256"],
})

metrics = calculate_metrics(
    revenue=row["revenue"],
    operating_profit=row["operating_profit"],
    ordinary_profit=row["ordinary_profit"],
    net_profit=row["net_profit"] or 0.0,
    current_total_assets=row["total_assets"],
    prior_total_assets=row["prior_total_assets"],
    equity=row["equity"],
    current_assets=row["current_assets"],
    current_liabilities=row["current_liabilities"],
    current_inventory=row["inventory"],
    prior_inventory=row["prior_inventory"],
    current_receivables=row["receivables"],
    prior_receivables=row["prior_receivables"],
)

metric_labels = {
    "operating_margin": ("売上高営業利益率", "%"),
    "ordinary_margin": ("売上高経常利益率", "%"),
    "ordinary_roa": ("総資本経常利益率", "%"),
    "equity_ratio": ("自己資本比率", "%"),
    "current_ratio": ("流動比率", "%"),
    "inventory_months": ("棚卸資産回転期間", "か月"),
    "receivable_months": ("売上債権回転期間", "か月"),
}

st.markdown("### 計算済み財務指標")
metric_rows = []
for key, (label, unit) in metric_labels.items():
    value = metrics.get(key)
    if value is not None:
        metric_rows.append({"指標": label, "値": round(float(value), 2), "単位": unit})
st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

st.markdown("### EDINET収録企業の同業分布")
fiscal_year = int(str(row["fiscal_year_end"])[:4])
peer = benchmark_rows(row["industry_code"], fiscal_year, accounting_standard=row["accounting_standard"])
peer_map = {r["metric"]: r for r in peer}

peer_rows = []
for key, (label, unit) in metric_labels.items():
    value = metrics.get(key)
    b = peer_map.get(key)
    if value is None or not b:
        continue
    peer_rows.append({
        "指標": label,
        "あなた": round(float(value), 2),
        "Q25": round(float(b["q25"]), 2) if b["q25"] is not None else None,
        "中央値": round(float(b["median"]), 2) if b["median"] is not None else None,
        "Q75": round(float(b["q75"]), 2) if b["q75"] is not None else None,
        "標本数": b["sample_count"],
        "単位": unit,
    })

if peer_rows:
    st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)
    st.caption("同業分布はSQLiteへ収録済みの同一業種・同一決算年・J-GAAP企業だけから事前計算しています。")
else:
    st.info("この業種・年度の同業分布はまだ十分に収録されていません。")

st.markdown("### 公的基準との診断")
public_industry = row["industry_code"] if row["industry_code"] in BENCHMARKS else "all"
diagnosis = diagnose(public_industry, metrics)

for guide in diagnosis["guidance_rows"]:
    text = (
        f"**{guide['title']} {guide['value']:.1f}{guide['unit']}** ｜ "
        f"一般的な目安 {guide['threshold']:.1f}{guide['unit']}"
    )
    if guide["status"] == "healthy":
        st.success(text)
    else:
        st.warning(text)

for item in diagnosis["rows"]:
    text = (
        f"**{item['title']} {item['value']:.2f}{item['unit']}** ｜ "
        f"公的な業種参考 {item['reference']:.2f}{item['unit']}"
    )
    if item["status"] == "healthy":
        st.success(text)
    else:
        st.info(text)

st.markdown("### 次に確認すること")
for i, advice in enumerate(diagnosis["advice"], 1):
    st.write(f"{i}. {advice}")

with st.expander("データの入り方"):
    st.write("1. EDINET等から取得した原票は、本番Webとは別の投入工程で正規化します。")
    st.write("2. 正規化CSVを tools/build_edinet_sqlite.py でSQLiteへ投入します。")
    st.write("3. 同時に業種・年度別のQ10/Q25/中央値/Q75/Q90を再計算します。")
    st.write("4. 本番WebはSQLiteを読み取るだけで、EDINET API・APIキーを必要としません。")
    st.write("5. 各レコードにdocID・原典URL・原典SHA-256を保存し、出典を追跡できます。")
