"""Read-only/query helpers for the packaged EDINET SQLite dataset."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "edinet_financials.sqlite3"


class EdinetStoreError(RuntimeError):
    pass


def db_path() -> Path:
    return Path(os.environ.get("EDINET_DB_PATH", str(DEFAULT_DB))).expanduser()


def connect(path: Optional[Path] = None, *, read_only: bool = True) -> sqlite3.Connection:
    path = Path(path or db_path())
    if read_only:
        if not path.is_file():
            raise EdinetStoreError(f"EDINET SQLiteがありません: {path}")
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def dataset_summary(path: Optional[Path] = None) -> dict:
    with connect(path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS filings,
                   COUNT(DISTINCT edinet_code) AS companies,
                   MIN(fiscal_year_end) AS min_year,
                   MAX(fiscal_year_end) AS max_year
            FROM filings
            """
        ).fetchone()
        meta = {
            r["key"]: r["value"]
            for r in conn.execute("SELECT key, value FROM dataset_metadata ORDER BY key")
        }
    return {
        "filings": int(row["filings"] or 0),
        "companies": int(row["companies"] or 0),
        "min_year": row["min_year"],
        "max_year": row["max_year"],
        "metadata": meta,
    }


def search_companies(query: str = "", *, industry_code: str = "", limit: int = 100,
                     path: Optional[Path] = None) -> list[dict]:
    q = (query or "").strip()
    clauses = ["accounting_standard = 'J-GAAP'"]
    args: list[object] = []
    if q:
        clauses.append("(company_name LIKE ? OR edinet_code LIKE ? OR COALESCE(sec_code,'') LIKE ?)")
        like = f"%{q}%"
        args.extend([like, like, like])
    if industry_code:
        clauses.append("industry_code = ?")
        args.append(industry_code)
    args.append(max(1, min(int(limit), 500)))

    sql = f"""
        SELECT edinet_code,
               MAX(company_name) AS company_name,
               MAX(sec_code) AS sec_code,
               MAX(industry_code) AS industry_code,
               MAX(industry_label) AS industry_label,
               MAX(fiscal_year_end) AS latest_fiscal_year_end,
               COUNT(*) AS filing_count
        FROM filings
        WHERE {' AND '.join(clauses)}
        GROUP BY edinet_code
        ORDER BY company_name
        LIMIT ?
    """
    with connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def list_filings(edinet_code: str, *, path: Optional[Path] = None) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, edinet_code, sec_code, company_name, fiscal_year_end,
                   accounting_standard, industry_code, industry_label,
                   source_url, source_sha256, imported_at
            FROM filings
            WHERE edinet_code = ? AND accounting_standard = 'J-GAAP'
            ORDER BY fiscal_year_end DESC, imported_at DESC
            """,
            (edinet_code,),
        )
        return [dict(r) for r in rows]


def get_filing(doc_id: str, *, path: Optional[Path] = None) -> dict:
    with connect(path) as conn:
        row = conn.execute(
            """
            SELECT f.*, n.revenue, n.operating_profit, n.ordinary_profit, n.net_profit,
                   n.total_assets, n.prior_total_assets, n.equity,
                   n.current_assets, n.current_liabilities,
                   n.inventory, n.prior_inventory,
                   n.receivables, n.prior_receivables
            FROM filings f
            JOIN financials n USING (doc_id)
            WHERE f.doc_id = ?
            """,
            (doc_id,),
        ).fetchone()
        if row is None:
            raise EdinetStoreError(f"docIDが見つかりません: {doc_id}")
        return dict(row)


def benchmark_rows(industry_code: str, fiscal_year: int, *,
                   accounting_standard: str = "J-GAAP",
                   path: Optional[Path] = None) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT metric, sample_count, q10, q25, median, q75, q90, updated_at
            FROM industry_benchmarks
            WHERE industry_code = ? AND fiscal_year = ? AND accounting_standard = ?
            ORDER BY metric
            """,
            (industry_code, int(fiscal_year), accounting_standard),
        )
        return [dict(r) for r in rows]
