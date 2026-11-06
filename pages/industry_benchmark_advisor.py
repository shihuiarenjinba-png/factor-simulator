from __future__ import annotations

import streamlit as st

from industry_benchmark import BENCHMARKS, SOURCE_NOTES, calculate_metrics, diagnose

st.set_page_config(page_title="業種別 数字診断", page_icon="🧭", layout="wide")

st.title("🧭 業種別 数字診断")
st.caption("売上・原価・利益・人件費を入れると、業種別の参考レンジと比較して『次に見る数字』を3つまで返します。")
st.info("参考診断です。税務・融資・投資判断の代替ではありません。ベンチマークは会社規模・地域・成長段階で大きく変わります。")

industry = st.selectbox(
    "業種",
    options=list(BENCHMARKS),
    format_func=lambda key: BENCHMARKS[key]["label"],
)

left, right = st.columns(2)
with left:
    revenue = st.number_input("月間売上", min_value=0.0, value=1_000_000.0, step=50_000.0)
    cogs = st.number_input("月間売上原価（仕入・材料など）", min_value=0.0, value=400_000.0, step=25_000.0)
    net_profit = st.number_input("月間純利益（赤字はマイナス可）", value=80_000.0, step=10_000.0)
with right:
    labor_cost = st.number_input("月間人件費", min_value=0.0, value=250_000.0, step=25_000.0)
    fixed_costs = st.number_input("月間固定費（家賃・固定契約等）", min_value=0.0, value=250_000.0, step=25_000.0)

if st.button("この数字を診断", type="primary", use_container_width=True):
    try:
        metrics = calculate_metrics(
            revenue=revenue,
            cogs=cogs,
            net_profit=net_profit,
            labor_cost=labor_cost,
            fixed_costs=fixed_costs,
        )
        result = diagnose(industry, metrics)
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
    else:
        st.subheader(f"{result['industry_label']} の診断")
        cols = st.columns(4)
        cols[0].metric("簡易スコア", f"{result['score']} / 100")
        cols[1].metric("粗利率", f"{metrics['gross_margin']:.1f}%")
        cols[2].metric("純利益率", f"{metrics['net_margin']:.1f}%")
        cols[3].metric("人件費率", f"{metrics['labor_ratio']:.1f}%")

        if metrics["break_even_revenue"] is not None:
            st.metric("概算 損益分岐売上", f"{metrics['break_even_revenue']:,.0f}")

        st.markdown("### 業種の目安との比較")
        for row in result["rows"]:
            text = (
                f"**{row['title']} {row['value']:.1f}%** — "
                f"{row['message']}（参考: {row['benchmark_low']:.1f}–{row['benchmark_high']:.1f}%）"
            )
            if row["status"] in ("healthy", "strong"):
                st.success(text)
            elif row["status"] == "watch":
                st.info(text)
            else:
                st.warning(text)

        st.markdown("### 次に見る数字")
        for i, item in enumerate(result["advice"], start=1):
            st.write(f"{i}. {item}")

        st.markdown("### このページの考え方")
        st.write(
            "単一の『正解値』を出すのではなく、業種ごとの参考レンジと比較し、"
            "大きく外れている数字から確認する順番を示します。"
        )

with st.expander("ベンチマークの出典・注意事項"):
    for note in SOURCE_NOTES:
        st.write("• " + note)
    st.markdown(
        "- Xero profitability ratios: https://www.xero.com/us/guides/profitability-ratios/\n"
        "- WorkCalcs labor cost percentage: https://workcalcs.com/labor-cost-percentage-calculator/\n"
        "- 数値は参考レンジです。今後、一次データへ置き換え・業種細分化を行う前提です。"
    )
