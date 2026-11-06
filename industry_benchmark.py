"""Japan-market industry benchmark advisor.

Official Japan-wide reference values are versioned under data/.
EDINET company-level observations are intentionally kept in a separate registry.
The product's caution/danger lines are transparent heuristics, not government cutoffs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data" / "jp_benchmarks_2024.json").read_text(encoding="utf-8"))
BENCHMARKS = DATA["industries"]
SOURCE_NOTES = [
    "日本市場向け。業界参考値は財務省「法人企業統計」を主な一次ソースとします。",
    "注意・危険ラインは政府の認定基準ではなく、業界参考値からの乖離を可視化する本ツール独自ルールです。",
    "将来の企業別分布はEDINET有価証券報告書から日本基準(J-GAAP)提出企業を優先して別系列で集計します。",
]


def _num(name: str, value: Optional[float], *, positive: bool = False) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} は0より大きい値が必要です")
    return value


def _avg(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None:
        return None
    if prior is None or prior <= 0:
        return current
    return (current + prior) / 2.0


def calculate_metrics(
    *,
    revenue: float,
    operating_profit: float,
    ordinary_profit: float,
    net_profit: float = 0.0,
    current_total_assets: Optional[float] = None,
    prior_total_assets: Optional[float] = None,
    equity: Optional[float] = None,
    current_inventory: Optional[float] = None,
    prior_inventory: Optional[float] = None,
    current_receivables: Optional[float] = None,
    prior_receivables: Optional[float] = None,
) -> dict:
    revenue = _num("売上高", revenue, positive=True)
    operating_profit = _num("営業利益", operating_profit)
    ordinary_profit = _num("経常利益", ordinary_profit)
    net_profit = _num("当期純利益", net_profit)

    current_total_assets = _num("総資産", current_total_assets)
    prior_total_assets = _num("前期総資産", prior_total_assets)
    equity = _num("自己資本", equity)
    current_inventory = _num("棚卸資産", current_inventory)
    prior_inventory = _num("前期棚卸資産", prior_inventory)
    current_receivables = _num("売上債権", current_receivables)
    prior_receivables = _num("前期売上債権", prior_receivables)

    avg_assets = _avg(current_total_assets, prior_total_assets)
    avg_inventory = _avg(current_inventory, prior_inventory)
    avg_receivables = _avg(current_receivables, prior_receivables)

    metrics = {
        "operating_margin": operating_profit / revenue * 100.0,
        "ordinary_margin": ordinary_profit / revenue * 100.0,
        "net_margin": net_profit / revenue * 100.0,
        "ordinary_roa": None,
        "equity_ratio": None,
        "inventory_months": None,
        "receivable_months": None,
    }
    if avg_assets and avg_assets > 0:
        metrics["ordinary_roa"] = ordinary_profit / avg_assets * 100.0
    if current_total_assets and current_total_assets > 0 and equity is not None:
        metrics["equity_ratio"] = equity / current_total_assets * 100.0
    if avg_inventory is not None:
        metrics["inventory_months"] = avg_inventory / (revenue / 12.0)
    if avg_receivables is not None:
        metrics["receivable_months"] = avg_receivables / (revenue / 12.0)
    return metrics


def industry_spec(industry: str, subindustry: Optional[str] = None) -> dict:
    if industry not in BENCHMARKS:
        raise KeyError(f"未対応の業種です: {industry}")
    base = json.loads(json.dumps(BENCHMARKS[industry]))
    if subindustry and subindustry in base.get("subindustry_overrides", {}):
        override = base["subindustry_overrides"][subindustry]
        base["label"] = override["label"]
        for metric, values in override.get("metrics", {}).items():
            base["metrics"][metric].update(values)
    return base


def thresholds(reference: float, direction: str) -> dict:
    policy = DATA["threshold_policy"]
    if direction == "higher":
        return {
            "caution": reference * policy["higher_is_better"]["caution_below_reference_ratio"],
            "danger": reference * policy["higher_is_better"]["danger_below_reference_ratio"],
        }
    return {
        "caution": reference * policy["lower_is_better"]["caution_above_reference_ratio"],
        "danger": reference * policy["lower_is_better"]["danger_above_reference_ratio"],
    }


def classify(value: float, metric: dict) -> tuple[str, str, int, dict]:
    reference = float(metric["reference"])
    direction = metric["direction"]
    cut = thresholds(reference, direction)

    if direction == "higher":
        if value < cut["danger"]:
            return "danger", f"危険ライン {cut['danger']:.2f}{metric['unit']} を下回っています", 20, cut
        if value < cut["caution"]:
            score = max(21, min(69, round(value / cut["caution"] * 70))) if cut["caution"] else 21
            return "warning", f"注意ライン {cut['caution']:.2f}{metric['unit']} を下回っています", score, cut
        if value < reference:
            return "watch", f"業界参考値 {reference:.2f}{metric['unit']} をやや下回っています", 80, cut
        return "healthy", f"業界参考値 {reference:.2f}{metric['unit']} 以上です", 100, cut

    if value > cut["danger"]:
        return "danger", f"危険ライン {cut['danger']:.2f}{metric['unit']} を上回っています", 20, cut
    if value > cut["caution"]:
        return "warning", f"注意ライン {cut['caution']:.2f}{metric['unit']} を上回っています", 60, cut
    if value > reference:
        return "watch", f"業界参考値 {reference:.2f}{metric['unit']} より長めです", 80, cut
    return "healthy", f"業界参考値 {reference:.2f}{metric['unit']} 以下です", 100, cut


ADVICE = {
    "operating_margin": "本業の利益率です。低い場合は価格、製品ミックス、原価、販管費を順に分解してください。",
    "ordinary_margin": "営業外損益を含む通常の収益力です。営業利益との差が大きい場合は支払利息などを確認してください。",
    "ordinary_roa": "総資産に対する通常の収益力です。利益率と資産回転のどちらが弱いか分解してください。",
    "equity_ratio": "財務の耐久力を見る指標です。低い場合は借入依存と利益蓄積の推移を確認してください。",
    "inventory_months": "在庫が売上の何か月分あるかを見る指標です。長い場合は滞留在庫・生産計画・販売速度を確認してください。",
    "receivable_months": "売上債権の回収期間の目安です。長い場合は回収条件・滞留債権・顧客構成を確認してください。",
}


def diagnose(industry: str, metrics: dict, subindustry: Optional[str] = None) -> dict:
    spec = industry_spec(industry, subindustry)
    rows, advice, sources = [], [], set()

    for key, metric in spec["metrics"].items():
        value = metrics.get(key)
        if value is None:
            continue
        status, message, score, cut = classify(float(value), metric)
        source_id = metric["source"]
        sources.add(source_id)
        rows.append({
            "metric": key,
            "title": metric["label"],
            "value": float(value),
            "unit": metric["unit"],
            "reference": float(metric["reference"]),
            "caution": cut["caution"],
            "danger": cut["danger"],
            "direction": metric["direction"],
            "status": status,
            "message": message,
            "score": score,
            "source": source_id,
        })
        if status in {"danger", "warning", "watch"}:
            advice.append((score, ADVICE[key]))

    overall_score = round(sum(row["score"] for row in rows) / len(rows)) if rows else 0
    advice.sort(key=lambda item: item[0])
    if not advice:
        advice = [(100, "入力された主要指標は現在の参考値から大きく外れていません。次は3〜5年の推移で悪化方向がないか確認してください。")]

    return {
        "industry": industry,
        "industry_label": spec["label"],
        "benchmark_version": DATA["benchmark_version"],
        "period": DATA["period"],
        "score": overall_score,
        "rows": rows,
        "advice": [text for _, text in advice[:3]],
        "sources": [DATA["sources"][source_id] for source_id in sorted(sources)],
        "threshold_note": DATA["threshold_policy"]["note"],
    }
