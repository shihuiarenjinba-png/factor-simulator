from __future__ import annotations

import streamlit as st

from industry_benchmark import BENCHMARKS, SOURCE_NOTES, calculate_metrics, diagnose

st.set_page_config(page_title="日本企業 業種別 数字診断", page_icon="🧭", layout="wide")

st.title("🧭 日本企業 業種別 数字診断")
st.caption("決算書の数字を入れると、公的な業種参考値と市場で一般的に使われる財務目安を並べて表示します。")
st.info("日本市場向け。企業別EDINETデータは本番基準に使用せず、財務省の公的集計と公的機関の一般的な財務目安を使用します。")

industry = st.selectbox(
    "業種",
    options=["manufacturing", "all"],
    format_func=lambda key: BENCHMARKS[key]["label"],
)
subindustry = None
if industry == "manufacturing":
    subindustry = st.selectbox(
        "製造業の内訳（分かる場合）",
        options=["", "food", "chemical"],
        format_func=lambda key: {
            "": "製造業全体",
            "food": "食料品製造",
            "chemical": "化学",
        }[key],
    ) or None

st.markdown("### 1. 損益計算書")
c1, c2, c3 = st.columns(3)
with c1:
    revenue = st.number_input("売上高（年額・円）", min_value=0.0, value=100_000_000.0, step=1_000_000.0)
    operating_profit = st.number_input("営業利益（円）", value=5_000_000.0, step=500_000.0)
with c2:
    ordinary_profit = st.number_input("経常利益（円）", value=7_000_000.0, step=500_000.0)
    net_profit = st.number_input("当期純利益（円）", value=5_000_000.0, step=500_000.0)
with c3:
    st.markdown("**入力の目安**")
    st.caption("同一年度・同一会計単位の数字を入力してください。金額単位は円で統一します。")

with st.expander("2. 貸借対照表も入力する", expanded=True):
    a1, a2, a3 = st.columns(3)
    with a1:
        current_assets_total = st.number_input("総資産（円、0なら未入力）", min_value=0.0, value=120_000_000.0, step=1_000_000.0)
        prior_assets_total = st.number_input("前期 総資産（円、0なら当期値で近似）", min_value=0.0, value=115_000_000.0, step=1_000_000.0)
        equity = st.number_input("自己資本（円）", value=60_000_000.0, step=1_000_000.0)
    with a2:
        current_assets = st.number_input("流動資産（円）", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
        current_liabilities = st.number_input("流動負債（円）", min_value=0.0, value=40_000_000.0, step=1_000_000.0)
    with a3:
        inv_current = st.number_input("当期 棚卸資産（円）", min_value=0.0, value=14_000_000.0, step=500_000.0)
        inv_prior = st.number_input("前期 棚卸資産（円）", min_value=0.0, value=13_000_000.0, step=500_000.0)
        rec_current = st.number_input("当期 売上債権（円）", min_value=0.0, value=18_000_000.0, step=500_000.0)
        rec_prior = st.number_input("前期 売上債権（円）", min_value=0.0, value=17_000_000.0, step=500_000.0)

st.markdown("### いつでも見える基準")
g1, g2 = st.columns(2)
g1.info("**流動比率：100％**\n\n日本政策金融公庫は、100％未満では資金繰り悪化が考えられると説明しています。")
g2.info("**自己資本比率：30％**\n\n日本政策金融公庫は、業種差はあるものの30％超を一つの目安としています。")

if st.button("この会社を診断", type="primary", use_container_width=True):
    def optional(v):
        return None if v == 0 else v

    try:
        metrics = calculate_metrics(
            revenue=revenue,
            operating_profit=operating_profit,
            ordinary_profit=ordinary_profit,
            net_profit=net_profit,
            current_total_assets=optional(current_assets_total),
            prior_total_assets=optional(prior_assets_total),
            equity=optional(equity),
            current_assets=optional(current_assets),
            current_liabilities=optional(current_liabilities),
            current_inventory=optional(inv_current),
            prior_inventory=optional(inv_prior),
            current_receivables=optional(rec_current),
            prior_receivables=optional(rec_prior),
        )
        result = diagnose(industry, metrics, subindustry)
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
    else:
        st.subheader(f"診断結果：{result['industry_label']}")
        st.caption(f"ベンチマーク版: {result['benchmark_version']} / 対象: {result['period']}")
        st.metric("簡易スコア", f"{result['score']} / 100")

        st.markdown("### 一般的な財務目安")
        for row in result["guidance_rows"]:
            line = f"**{row['title']}：{row['value']:.1f}{row['unit']}** ｜ 目安 {row['threshold']:.1f}{row['unit']} — {row['message']}"
            if row["status"] == "healthy":
                st.success(line)
            else:
                st.warning(line)

        st.markdown("### 業種別の参考値")
        st.caption("こちらは合否ラインではなく、公的な業種集計との比較です。")
        for row in result["rows"]:
            line = (
                f"**{row['title']}：{row['value']:.2f}{row['unit']}** ｜ "
                f"業種参考 {row['reference']:.2f}{row['unit']} — {row['message']}"
            )
            if row["status"] == "healthy":
                st.success(line)
            else:
                st.info(line)

        st.markdown("### 次に確認すること")
        for i, item in enumerate(result["advice"], 1):
            st.write(f"{i}. {item}")

        st.markdown("### この診断で使った出典")
        for src in result["sources"]:
            st.write(f"- **{src['publisher']}**「{src['title']}」")
            st.caption(src["url"])
        st.caption(result["threshold_note"])

with st.expander("データ方針"):
    st.write(
        "商用版の主基準は、公的な業種集計と公的機関が明示する一般的な財務目安です。"
        "企業別EDINETデータは本番診断の基準値には使用しません。"
    )
    st.write("基準値は data/ の版管理JSONで管理し、出典・対象年度を追跡できる状態にします。")

with st.expander("基準値の一次ソース"):
    for note in SOURCE_NOTES:
        st.write("• " + note)
    st.markdown(
        "- 財務省 法人企業統計: https://www.mof.go.jp/pri/reference/ssc/results/data.htm\n"
        "- 日本政策金融公庫 財務診断指標: https://www.jfc.go.jp/n/zaimushindan/shihyo.html\n"
        "- 日本政策金融公庫 決算書の見方: https://www.jfc.go.jp/n/finance/keiei/support-plus/detail/column.html?id=c55_392"
    )
