"""Japan-market industry benchmark advisor.

Commercial production benchmarks use public aggregate statistics and public
financial guidance. Company-level EDINET data is not used as a production
benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data" / "jp_benchmarks_2024.json").read_text(encoding="utf-8"))
BENCHMARKS = DATA["industries"]
ABSOLUTE_GUIDANCE = DATA["absolute_guidance"]
SOURCE_NOTES = [
    "日本市場向け。業種参考値は財務省「法人企業統計」を主な一次ソースとします。",
    "流動比率・自己資本比率など、一般的な目安を公的機関が明示している指標はその値を優先します。",
    "企業別EDINETデータは商用版の基準値には使用しません。",
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
    current_assets: Optional[float] = None,
    current_liabilities: Optional[float] = None,
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
    current_assets = _num("流動資産", current_assets)
    current_liabilities = _num("流動負債", current_liabilities)
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
        "current_ratio": None,
        "inventory_months": None,
        "receivable_months": None,
    }
    if avg_assets and avg_assets > 0:
        metrics["ordinary_roa"] = ordinary_profit / avg_assets * 100.0
    if current_total_assets and current_total_assets > 0 and equity is not None:
        metrics["equity_ratio"] = equity / current_total_assets * 100.0
    if current_assets is not None and current_liabilities and current_liabilities > 0:
        metrics["current_ratio"] = current_assets / current_liabilities * 100.0
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


def reference_score(value: float, metric: dict) -> tuple[str, str, int]:
    reference = float(metric["reference"])
    unit = metric["unit"]
    if metric["direction"] == "higher":
        if value >= reference:
            return "healthy", f"業界参考値 {reference:.2f}{unit} 以上です", 100
        ratio = value / reference if reference > 0 else 0
        return "watch", f"業界参考値 {reference:.2f}{unit} を下回っています", max(0, min(99, round(ratio * 100)))
    if value <= reference:
        return "healthy", f"業界参考値 {reference:.2f}{unit} 以下です", 100
    ratio = reference / value if value > 0 else 0
    return "watch", f"業界参考値 {reference:.2f}{unit} より長めです", max(0, min(99, round(ratio * 100)))


def assess_absolute(metric_key: str, value: Optional[float]) -> Optional[dict]:
    if value is None:
        return None
    if metric_key == "current_ratio":
        spec = ABSOLUTE_GUIDANCE["current_ratio"]
    elif metric_key == "equity_ratio":
        spec = ABSOLUTE_GUIDANCE["equity_ratio_30"]
    else:
        return None
    threshold = float(spec["threshold"])
    ok = float(value) >= threshold
    return {
        "metric": metric_key,
        "title": spec["label"],
        "value": float(value),
        "unit": spec["unit"],
        "threshold": threshold,
        "status": "healthy" if ok else "warning",
        "message": spec["display"],
        "source": spec["source"],
    }


ADVICE = {
    "operating_margin": "本業の利益率です。低い場合は価格、製品ミックス、原価、販管費を順に分解してください。",
    "ordinary_margin": "営業外損益を含む通常の収益力です。営業利益との差が大きい場合は支払利息などを確認してください。",
    "ordinary_roa": "総資産に対する通常の収益力です。利益率と資産回転のどちらが弱いか分解してください。",
    "equity_ratio": "財務の耐久力を見る指標です。低い場合は借入依存と利益蓄積の推移を確認してください。",
    "current_ratio": "短期の支払余力です。100%未満なら、流動負債の返済原資と資金繰りを優先して確認してください。",
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
        status, message, score = reference_score(float(value), metric)
        source_id = metric["source"]
        sources.add(source_id)
        rows.append({
            "metric": key,
            "title": metric["label"],
            "value": float(value),
            "unit": metric["unit"],
            "reference": float(metric["reference"]),
            "direction": metric["direction"],
            "status": status,
            "message": message,
            "score": score,
            "source": source_id,
        })
        if status != "healthy":
            advice.append((score, ADVICE[key]))

    guidance_rows = []
    for key in ("current_ratio", "equity_ratio"):
        row = assess_absolute(key, metrics.get(key))
        if row:
            guidance_rows.append(row)
            sources.add(row["source"])
            if row["status"] != "healthy":
                advice.append((40, ADVICE[key]))

    all_scores = [row["score"] for row in rows]
    for row in guidance_rows:
        all_scores.append(100 if row["status"] == "healthy" else 50)
    overall_score = round(sum(all_scores) / len(all_scores)) if all_scores else 0

    advice.sort(key=lambda item: item[0])
    if not advice:
        advice = [(100, "主要指標は現在の公的参考値・一般的な目安から大きく外れていません。次は3〜5年の推移を確認してください。")]

    return {
        "industry": industry,
        "industry_label": spec["label"],
        "benchmark_version": DATA["benchmark_version"],
        "period": DATA["period"],
        "score": overall_score,
        "rows": rows,
        "guidance_rows": guidance_rows,
        "advice": [text for _, text in advice[:3]],
        "sources": [DATA["sources"][source_id] for source_id in sorted(sources)],
        "threshold_note": DATA["threshold_policy"]["note"],
    }
