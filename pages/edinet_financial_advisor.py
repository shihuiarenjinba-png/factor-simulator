from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from edinet_api import EdinetError, download_csv_zip, extract_jgaap_metrics, list_annual_reports, parse_csv_zip, provenance
from industry_benchmark import BENCHMARKS, calculate_metrics, diagnose

st.set_page_config(page_title="EDINET 有報診断", page_icon="🏢", layout="wide")

st.title("🏢 EDINET 有報診断（J-GAAP MVP）")
st.caption("EDINET APIから有価証券報告書のCSVを取得し、主要財務数値の候補を抽出して診断します。")
st.warning("このページは抽出結果の確認を前提にします。APIキーは保存しません。投資・融資・税務判断の代替ではありません。")

api_key = st.text_input("EDINET APIキー", type="password", help="金融庁EDINET APIで発行したキー。画面外へ保存しません。")
submit_date = st.date_input("提出日", value=date.today())

if st.button("この日の有価証券報告書を検索", use_container_width=True):
    try:
        docs = list_annual_reports(submit_date.isoformat(), api_key)
    except EdinetError as exc:
        st.error(str(exc))
    else:
        st.session_state["edinet_docs"] = docs
        if not docs:
            st.info("CSV取得可能な有価証券報告書が見つかりませんでした。")
        else:
            st.success(f"{len(docs)}件見つかりました。")

docs = st.session_state.get("edinet_docs", [])
if docs:
    labels = [
        f"{d['filer_name']}｜{d['period_end']}｜{d['doc_id']}"
        for d in docs
    ]
    choice = st.selectbox("診断する有報", range(len(docs)), format_func=lambda i: labels[i])
    doc = docs[choice]

    st.markdown("### 書類メタデータ")
    st.json({
        "会社名": doc["filer_name"],
        "EDINETコード": doc["edinet_code"],
        "証券コード": doc["sec_code"],
        "docID": doc["doc_id"],
        "対象期間": f"{doc['period_start']} ～ {doc['period_end']}",
        "提出日時": doc["submitted_at"],
    })

    if st.button("EDINET CSVを取得して抽出", type="primary", use_container_width=True):
        try:
            blob = download_csv_zip(doc["doc_id"], api_key)
            rows = parse_csv_zip(blob)
            extracted = extract_jgaap_metrics(rows)
        except EdinetError as exc:
            st.error(str(exc))
        else:
            st.session_state["edinet_result"] = {
                "doc": doc,
                "provenance": provenance(blob, doc),
                "extracted": extracted,
            }

result = st.session_state.get("edinet_result")
if result:
    extracted = result["extracted"]
    values = extracted["values"]
    evidence = extracted["evidence"]

    st.markdown("### 抽出結果の確認")
    st.caption("下の値と要素IDを確認してから診断してください。誤っている場合は診断しないでください。")

    rows = []
    labels = {
        "revenue": "売上高",
        "operating_profit": "営業利益",
        "ordinary_profit": "経常利益",
        "net_profit": "当期純利益",
        "total_assets": "総資産",
        "prior_total_assets": "前期総資産",
        "current_assets": "流動資産",
        "current_liabilities": "流動負債",
        "equity": "純資産候補",
        "inventory": "棚卸資産",
        "prior_inventory": "前期棚卸資産",
        "receivables": "売上債権",
        "prior_receivables": "前期売上債権",
    }
    for key, title in labels.items():
        ev = evidence.get(key, {})
        rows.append({
            "項目": title,
            "値": values.get(key),
            "要素ID": ev.get("element_id", ""),
            "コンテキスト": ev.get("context_id", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 出典証跡")
    st.json(result["provenance"])

    industry = st.selectbox(
        "比較する業種",
        options=["manufacturing", "all"],
        format_func=lambda key: BENCHMARKS[key]["label"],
        key="edinet_industry",
    )
    confirmed = st.checkbox("抽出値と会社・決算期を確認しました")

    if st.button("確認済みの値で診断", type="primary", disabled=not confirmed, use_container_width=True):
        required = ("revenue", "operating_profit", "ordinary_profit")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            st.error("診断に必要な主要値が不足しています: " + ", ".join(missing))
        else:
            metrics = calculate_metrics(
                revenue=values["revenue"],
                operating_profit=values["operating_profit"],
                ordinary_profit=values["ordinary_profit"],
                net_profit=values.get("net_profit") or 0.0,
                current_total_assets=values.get("total_assets"),
                prior_total_assets=values.get("prior_total_assets"),
                equity=values.get("equity"),
                current_assets=values.get("current_assets"),
                current_liabilities=values.get("current_liabilities"),
                current_inventory=values.get("inventory"),
                prior_inventory=values.get("prior_inventory"),
                current_receivables=values.get("receivables"),
                prior_receivables=values.get("prior_receivables"),
            )
            diagnosis = diagnose(industry, metrics)
            st.subheader(f"診断結果：{diagnosis['industry_label']}")
            st.metric("簡易スコア", f"{diagnosis['score']} / 100")

            st.markdown("#### 一般的な財務目安")
            for row in diagnosis["guidance_rows"]:
                msg = f"{row['title']} {row['value']:.1f}{row['unit']} ｜ 目安 {row['threshold']:.1f}{row['unit']}"
                if row["status"] == "healthy":
                    st.success(msg)
                else:
                    st.warning(msg)

            st.markdown("#### 公的な業種参考値との比較")
            for row in diagnosis["rows"]:
                msg = f"{row['title']} {row['value']:.2f}{row['unit']} ｜ 業種参考 {row['reference']:.2f}{row['unit']}"
                if row["status"] == "healthy":
                    st.success(msg)
                else:
                    st.info(msg)

            st.markdown("#### 次に確認すること")
            for i, item in enumerate(diagnosis["advice"], 1):
                st.write(f"{i}. {item}")

with st.expander("EDINET版のデータ方針"):
    st.write("・EDINET API v2のCSV（type=5）を利用します。")
    st.write("・初期版は日本基準（J-GAAP）の有価証券報告書だけを対象にします。")
    st.write("・抽出値は要素ID・コンテキストID・docID・CSV ZIP SHA-256と一緒に表示します。")
    st.write("・IFRS/US GAAPはJ-GAAPと同じロジックに混ぜません。")
    st.write("・APIキーは保存しません。")
