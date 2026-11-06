"""Industry benchmark advisor: pure calculation and interpretation logic.

Benchmarks are starter reference ranges, not guarantees or professional advice.
They are deliberately kept separate from the Streamlit UI so formulas can be tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    direction: str = "higher"  # "higher" for margins, "band" for ratios.


BENCHMARKS = {
    "restaurant": {
        "label": "飲食・レストラン",
        "gross_margin": Range(55.0, 65.0, "higher"),
        "net_margin": Range(3.0, 9.0, "higher"),
        "labor_ratio": Range(28.0, 35.0, "band"),
    },
    "retail": {
        "label": "小売・EC",
        "gross_margin": Range(20.0, 50.0, "higher"),
        "net_margin": Range(2.0, 5.0, "higher"),
        "labor_ratio": Range(10.0, 20.0, "band"),
    },
    "manufacturing": {
        "label": "製造業",
        "gross_margin": Range(25.0, 35.0, "higher"),
        "net_margin": Range(5.0, 10.0, "higher"),
        "labor_ratio": Range(15.0, 30.0, "band"),
    },
    "professional_services": {
        "label": "コンサル・士業・専門サービス",
        "gross_margin": Range(50.0, 70.0, "higher"),
        "net_margin": Range(15.0, 25.0, "higher"),
        "labor_ratio": Range(30.0, 50.0, "band"),
    },
    "saas": {
        "label": "SaaS・ソフトウェア",
        "gross_margin": Range(60.0, 80.0, "higher"),
        "net_margin": Range(10.0, 20.0, "higher"),
        "labor_ratio": None,
    },
}

SOURCE_NOTES = [
    "Xero (2026): profitability ratio examples by industry — gross/net margin reference ranges.",
    "WorkCalcs (2026): labor-cost percentage examples by industry.",
    "These are broad reference ranges. Company size, country, growth stage and accounting policy can materially change a healthy value.",
]


def _require_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def calculate_metrics(
    *,
    revenue: float,
    cogs: float,
    net_profit: float,
    labor_cost: float,
    fixed_costs: float,
) -> dict:
    revenue = float(revenue)
    if revenue <= 0:
        raise ValueError("revenue must be greater than zero")
    cogs = _require_nonnegative("cogs", cogs)
    labor_cost = _require_nonnegative("labor_cost", labor_cost)
    fixed_costs = _require_nonnegative("fixed_costs", fixed_costs)
    net_profit = float(net_profit)

    gross_profit = revenue - cogs
    gross_margin = gross_profit / revenue * 100.0
    net_margin = net_profit / revenue * 100.0
    labor_ratio = labor_cost / revenue * 100.0

    contribution_ratio = gross_margin / 100.0
    break_even_revenue: Optional[float]
    if contribution_ratio > 0:
        break_even_revenue = fixed_costs / contribution_ratio
    else:
        break_even_revenue = None

    return {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "labor_ratio": labor_ratio,
        "break_even_revenue": break_even_revenue,
    }


def classify(value: float, benchmark: Range) -> tuple[str, str, int]:
    """Return status, plain-language interpretation, and 0-100 subscore."""
    low, high = benchmark.low, benchmark.high
    if benchmark.direction == "higher":
        if value < low:
            score = max(0, min(99, round(100 * value / low))) if low > 0 else 0
            return "warning", f"目安の下限 {low:.1f}% を下回っています", score
        if value <= high:
            return "healthy", f"目安 {low:.1f}–{high:.1f}% の範囲です", 100
        return "strong", f"目安上限 {high:.1f}% を上回っています", 100

    if low <= value <= high:
        return "healthy", f"目安 {low:.1f}–{high:.1f}% の範囲です", 100
    if value > high:
        excess = value - high
        score = max(0, round(100 - excess * 5))
        return "warning", f"目安上限 {high:.1f}% を {excess:.1f}pt 上回っています", score

    gap = low - value
    score = max(0, round(90 - gap * 2))
    return "watch", f"目安下限 {low:.1f}% を {gap:.1f}pt 下回っています。省人化か人員不足かを確認してください", score


def diagnose(industry: str, metrics: dict) -> dict:
    if industry not in BENCHMARKS:
        raise KeyError(f"unknown industry: {industry}")
    spec = BENCHMARKS[industry]
    rows = []
    advice = []

    for key, title in (
        ("gross_margin", "粗利率"),
        ("net_margin", "純利益率"),
        ("labor_ratio", "人件費率"),
    ):
        benchmark = spec.get(key)
        if benchmark is None:
            continue
        status, message, score = classify(float(metrics[key]), benchmark)
        rows.append({
            "metric": key,
            "title": title,
            "value": float(metrics[key]),
            "benchmark_low": benchmark.low,
            "benchmark_high": benchmark.high,
            "status": status,
            "message": message,
            "score": score,
        })

        if key == "gross_margin" and status == "warning":
            advice.append("粗利率が低めです。値上げ余地、仕入原価、原価の高い商品・案件を順番に確認してください。")
        elif key == "net_margin" and status == "warning":
            advice.append("純利益率が低めです。固定費と販管費を金額順に並べ、売上を落とさず削れる費用から確認してください。")
        elif key == "labor_ratio" and status == "warning":
            advice.append("人件費率が高めです。売上の弱い時間帯・案件とシフト/工数の対応を確認してください。")
        elif key == "labor_ratio" and status == "watch":
            advice.append("人件費率は低めです。効率が良い可能性もありますが、欠員・過重労働・外注費への付け替えがないか確認してください。")

    overall_score = round(sum(row["score"] for row in rows) / len(rows)) if rows else 0
    if not advice:
        advice.append("主要比率は参考レンジ内です。次は月次推移を見て、悪化し始めた指標を先に確認してください。")

    be = metrics.get("break_even_revenue")
    if be is None:
        advice.insert(0, "粗利がプラスでないため、固定費を回収する損益分岐売上を計算できません。価格・原価を先に見直してください。")
    elif metrics["revenue"] < be:
        advice.insert(0, f"現在売上は概算損益分岐売上を下回ります。まず約 {be - metrics['revenue']:,.0f} の売上差を埋めるか固定費を下げる必要があります。")

    return {
        "industry": industry,
        "industry_label": spec["label"],
        "score": overall_score,
        "rows": rows,
        "advice": advice[:3],
    }
