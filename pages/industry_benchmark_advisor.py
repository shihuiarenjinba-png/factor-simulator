from __future__ import annotations

import streamlit as st

from industry_benchmark import BENCHMARKS, SOURCE_NOTES, calculate_metrics, diagnose

st.set_page_config(page_title="日本企業 業種別 数字診断", page_icon="🧭", layout="wide")

st.title("🧭 日本企業 業種別 数字診断")
st.caption("有報・決算書の数字を入れると、日本の公的統計と比較して「どの数字を先に見るか」を返します。")
st.info("日本市場向け。現在の基準は財務省「法人企業統計」。注意/危険ラインは公式認定基準ではなく、業界参考値からの乖離を分かりやすくする診断ルールです。")

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
    st.caption("まず有価証券報告書・決算書の連結数値を同一年度で入力してください。金額単位は円で統一します。")

with st.expander("2. 貸借対照表も入力する（ROA・自己資本比率・在庫/売上債権の診断）", expanded=True):
    a1, a2, a3 = st.columns(3)
    with a1:
        current_assets_total = st.number_input("当期 総資産（円、0なら未入力）", min_value=0.0, value=120_000_000.0, step=1_000_000.0)
        prior_assets_total = st.number_input("前期 総資産（円、0なら当期値で近似）", min_value=0.0, value=115_000_000.0, step=1_000_000.0)
        equity = st.number_input("自己資本（純資産−新株予約権、円）", value=60_000_000.0, step=1_000_000.0)
    with a2:
        inv_current = st.number_input("当期 棚卸資産（円）", min_value=0.0, value=14_000_000.0, step=500_000.0)
        inv_prior = st.number_input("前期 棚卸資産（円、0なら当期値で近似）", min_value=0.0, value=13_000_000.0, step=500_000.0)
    with a3:
        rec_current = st.number_input("当期 売上債権（受取手形+売掛金、円）", min_value=0.0, value=18_000_000.0, step=500_000.0)
        rec_prior = st.number_input("前期 売上債権（円、0なら当期値で近似）", min_value=0.0, value=17_000_000.0, step=500_000.0)

st.markdown("### 判定の見方")
l1, l2, l3, l4 = st.columns(4)
l1.success("緑：業界参考値以上 / 以下（良い方向）")
l2.info("青：参考値から少し外れる")
l3.warning("黄：参考値から25%以上悪化")
l4.error("赤：参考値から50%以上悪化")
st.caption("25%/50%は本ツールの見やすさのためのヒューリスティックで、財務省・金融庁が定めた合否基準ではありません。")

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

        st.markdown("### 指標とマジックナンバー")
        for row in result["rows"]:
            if row["direction"] == "higher":
                line = (
                    f"**{row['title']}：{row['value']:.2f}{row['unit']}** ｜ "
                    f"業界参考 {row['reference']:.2f} ｜ 注意 < {row['caution']:.2f} ｜ 危険 < {row['danger']:.2f}"
                )
            else:
                line = (
                    f"**{row['title']}：{row['value']:.2f}{row['unit']}** ｜ "
                    f"業界参考 {row['reference']:.2f} ｜ 注意 > {row['caution']:.2f} ｜ 危険 > {row['danger']:.2f}"
                )
            line += f" — {row['message']}"
            if row["status"] == "healthy":
                st.success(line)
            elif row["status"] == "watch":
                st.info(line)
            elif row["status"] == "warning":
                st.warning(line)
            else:
                st.error(line)

        st.markdown("### 次に確認すること")
        for i, item in enumerate(result["advice"], 1):
            st.write(f"{i}. {item}")

        st.markdown("### この診断で使った出典")
        for src in result["sources"]:
            st.write(f"- **{src['publisher']}**「{src['title']}」")
            st.caption(src["url"])
        st.caption(result["threshold_note"])

with st.expander("有報データの管理方針"):
    st.write(
        "今後EDINETから企業別データを追加する際は、EDINETコード・docID・会社名・決算期・会計基準・"
        "タクソノミ年・業種・取得URL・取得日時・原本SHA-256を保存します。日本基準(J-GAAP)とIFRSは混ぜず、"
        "業界分布は別系列で集計します。"
    )
    st.write("基準値そのものはコードではなく data/ の版管理JSONで更新します。")

with st.expander("基準値の一次ソース"):
    for note in SOURCE_NOTES:
        st.write("• " + note)
    st.markdown(
        "- 財務省 法人企業統計: https://www.mof.go.jp/pri/reference/ssc/results/data.htm\n"
        "- 財務指標と算式: https://www.mof.go.jp/pri/reference/ssc/results/calculation.htm\n"
        "- 金融庁 EDINETタクソノミ: https://www.fsa.go.jp/search/EDINET_Taxonomy_All.html"
    )
