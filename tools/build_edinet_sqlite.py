"""Build/update the packaged EDINET SQLite dataset from normalized CSV.

No network access is performed. Acquisition and normalization happen outside the
production web process; this command only validates/imports prepared rows and
rebuilds peer-distribution statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from edinet_store import connect
from industry_benchmark import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data" / "edinet_schema.sql"
DEFAULT_DB = ROOT / "data" / "edinet_financials.sqlite3"

REQUIRED = {
    "doc_id", "edinet_code", "company_name", "fiscal_year_end",
    "accounting_standard", "industry_code", "industry_label",
    "source_url", "source_sha256",
    "revenue", "operating_profit", "ordinary_profit",
}
OPTIONAL_NUMERIC = {
    "sec_code", "net_profit", "total_assets", "prior_total_assets", "equity",
    "current_assets", "current_liabilities", "inventory", "prior_inventory",
    "receivables", "prior_receivables",
}
METRICS = (
    "operating_margin", "ordinary_margin", "ordinary_roa", "equity_ratio",
    "current_ratio", "inventory_months", "receivable_months",
)


def number(value: str):
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    return float(value)


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    weight = pos - low
    return values[low] * (1 - weight) + values[high] * weight


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))


def import_csv(path: Path, conn: sqlite3.Connection) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV必須列不足: " + ", ".join(sorted(missing)))

        count = 0
        for row in reader:
            if row["accounting_standard"].strip() != "J-GAAP":
                raise ValueError(f"{row['doc_id']}: J-GAAP以外はこのDBへ投入しません")
            sha = row["source_sha256"].strip().lower()
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"{row['doc_id']}: source_sha256が不正です")

            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO filings (
                    doc_id, edinet_code, sec_code, company_name, fiscal_year_end,
                    accounting_standard, industry_code, industry_label,
                    source_url, source_sha256, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    edinet_code=excluded.edinet_code,
                    sec_code=excluded.sec_code,
                    company_name=excluded.company_name,
                    fiscal_year_end=excluded.fiscal_year_end,
                    accounting_standard=excluded.accounting_standard,
                    industry_code=excluded.industry_code,
                    industry_label=excluded.industry_label,
                    source_url=excluded.source_url,
                    source_sha256=excluded.source_sha256,
                    imported_at=excluded.imported_at
                """,
                (
                    row["doc_id"].strip(), row["edinet_code"].strip(),
                    row.get("sec_code", "").strip(), row["company_name"].strip(),
                    row["fiscal_year_end"].strip(), "J-GAAP",
                    row["industry_code"].strip(), row["industry_label"].strip(),
                    row["source_url"].strip(), sha, now,
                ),
            )
            conn.execute(
                """
                INSERT INTO financials (
                    doc_id, revenue, operating_profit, ordinary_profit, net_profit,
                    total_assets, prior_total_assets, equity, current_assets,
                    current_liabilities, inventory, prior_inventory,
                    receivables, prior_receivables
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    revenue=excluded.revenue,
                    operating_profit=excluded.operating_profit,
                    ordinary_profit=excluded.ordinary_profit,
                    net_profit=excluded.net_profit,
                    total_assets=excluded.total_assets,
                    prior_total_assets=excluded.prior_total_assets,
                    equity=excluded.equity,
                    current_assets=excluded.current_assets,
                    current_liabilities=excluded.current_liabilities,
                    inventory=excluded.inventory,
                    prior_inventory=excluded.prior_inventory,
                    receivables=excluded.receivables,
                    prior_receivables=excluded.prior_receivables
                """,
                (
                    row["doc_id"].strip(),
                    number(row["revenue"]), number(row["operating_profit"]),
                    number(row["ordinary_profit"]), number(row.get("net_profit", "")),
                    number(row.get("total_assets", "")), number(row.get("prior_total_assets", "")),
                    number(row.get("equity", "")), number(row.get("current_assets", "")),
                    number(row.get("current_liabilities", "")), number(row.get("inventory", "")),
                    number(row.get("prior_inventory", "")), number(row.get("receivables", "")),
                    number(row.get("prior_receivables", "")),
                ),
            )
            count += 1
    return count


def rebuild_benchmarks(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM industry_benchmarks")
    groups: dict[tuple[str, str, int], dict[str, list[float]]] = {}

    rows = conn.execute(
        """
        SELECT f.industry_code, f.industry_label, f.fiscal_year_end,
               f.accounting_standard,
               n.revenue, n.operating_profit, n.ordinary_profit, n.net_profit,
               n.total_assets, n.prior_total_assets, n.equity,
               n.current_assets, n.current_liabilities,
               n.inventory, n.prior_inventory, n.receivables, n.prior_receivables
        FROM filings f JOIN financials n USING (doc_id)
        WHERE f.accounting_standard = 'J-GAAP'
        """
    )
    labels: dict[tuple[str, str, int], str] = {}
    for r in rows:
        fiscal_year = int(str(r["fiscal_year_end"])[:4])
        key = (r["industry_code"], r["accounting_standard"], fiscal_year)
        labels[key] = r["industry_label"]
        metrics = calculate_metrics(
            revenue=r["revenue"],
            operating_profit=r["operating_profit"],
            ordinary_profit=r["ordinary_profit"],
            net_profit=r["net_profit"] or 0.0,
            current_total_assets=r["total_assets"],
            prior_total_assets=r["prior_total_assets"],
            equity=r["equity"],
            current_assets=r["current_assets"],
            current_liabilities=r["current_liabilities"],
            current_inventory=r["inventory"],
            prior_inventory=r["prior_inventory"],
            current_receivables=r["receivables"],
            prior_receivables=r["prior_receivables"],
        )
        bucket = groups.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = metrics.get(metric)
            if value is not None and math.isfinite(value):
                bucket[metric].append(float(value))

    updated = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for (industry, standard, year), bucket in groups.items():
        for metric, values in bucket.items():
            if not values:
                continue
            conn.execute(
                """
                INSERT INTO industry_benchmarks (
                    industry_code, industry_label, fiscal_year, accounting_standard,
                    metric, sample_count, q10, q25, median, q75, q90, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    industry, labels[(industry, standard, year)], year, standard, metric,
                    len(values), percentile(values, .10), percentile(values, .25),
                    percentile(values, .50), percentile(values, .75),
                    percentile(values, .90), updated,
                ),
            )
            inserted += 1
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dataset-version", default="")
    args = parser.parse_args()

    conn = connect(args.db, read_only=False)
    try:
        init_db(conn)
        total = 0
        with conn:
            for path in args.csv:
                total += import_csv(path, conn)
            benchmarks = rebuild_benchmarks(conn)
            if args.dataset_version:
                conn.execute(
                    "INSERT OR REPLACE INTO dataset_metadata(key,value) VALUES('dataset_version',?)",
                    (args.dataset_version,),
                )
            conn.execute(
                "INSERT OR REPLACE INTO dataset_metadata(key,value) VALUES('updated_at',?)",
                (datetime.now(timezone.utc).isoformat(),),
            )
        print(json.dumps({
            "db": str(args.db),
            "imported_rows": total,
            "benchmark_rows": benchmarks,
        }, ensure_ascii=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
