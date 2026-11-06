"""Small EDINET API v2 adapter for the Japan financial advisor.

The API key is passed by the caller and is never persisted here.
Only annual securities reports (docTypeCode=120) with CSV availability are
intended for the first production-facing MVP.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from typing import Iterable


BASE = "https://api.edinet-fsa.go.jp/api/v2"
TIMEOUT = 30

FIELD_CANDIDATES = {
    "revenue": ["jppfs_cor:NetSales", "jpcrp_cor:NetSalesSummaryOfBusinessResults"],
    "operating_profit": ["jppfs_cor:OperatingIncome", "jpcrp_cor:OperatingIncomeLossSummaryOfBusinessResults"],
    "ordinary_profit": ["jppfs_cor:OrdinaryIncome", "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults"],
    "net_profit": ["jppfs_cor:ProfitLoss", "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults"],
    "total_assets": ["jppfs_cor:Assets", "jpcrp_cor:TotalAssetsSummaryOfBusinessResults"],
    "current_assets": ["jppfs_cor:CurrentAssets"],
    "current_liabilities": ["jppfs_cor:CurrentLiabilities"],
    "equity": ["jppfs_cor:NetAssets", "jpcrp_cor:NetAssetsSummaryOfBusinessResults"],
    "inventory": ["jppfs_cor:Inventories"],
    "receivables": ["jppfs_cor:NotesAndAccountsReceivableTrade"],
    "notes_receivable": ["jppfs_cor:NotesReceivableTrade"],
    "accounts_receivable": ["jppfs_cor:AccountsReceivableTrade"],
}


class EdinetError(RuntimeError):
    pass


def _request(path: str, *, api_key: str, params: dict | None = None):
    import requests
    if not api_key.strip():
        raise EdinetError("EDINET APIキーが必要です")
    query = dict(params or {})
    query["Subscription-Key"] = api_key.strip()
    r = requests.get(BASE + path, params=query, timeout=TIMEOUT)
    if r.status_code != 200:
        raise EdinetError(f"EDINET API HTTP {r.status_code}")
    return r


def list_annual_reports(date: str, api_key: str) -> list[dict]:
    r = _request("/documents.json", api_key=api_key, params={"date": date, "type": 2})
    try:
        payload = r.json()
    except ValueError as exc:
        raise EdinetError("EDINET書類一覧がJSONではありません") from exc

    out = []
    for row in payload.get("results", []):
        if str(row.get("docTypeCode", "")) != "120":
            continue
        if str(row.get("csvFlag", "")) != "1":
            continue
        if str(row.get("withdrawalStatus", "0")) not in {"0", ""}:
            continue
        out.append({
            "doc_id": row.get("docID", ""),
            "edinet_code": row.get("edinetCode", ""),
            "sec_code": row.get("secCode", ""),
            "filer_name": row.get("filerName", ""),
            "period_start": row.get("periodStart", ""),
            "period_end": row.get("periodEnd", ""),
            "submitted_at": row.get("submitDateTime", ""),
            "description": row.get("docDescription", ""),
        })
    return out


def download_csv_zip(doc_id: str, api_key: str) -> bytes:
    if not re.fullmatch(r"S[0-9A-Z]{7,9}", doc_id or ""):
        raise EdinetError("docIDの形式が不正です")
    r = _request(f"/documents/{doc_id}", api_key=api_key, params={"type": 5})
    body = r.content
    if len(body) < 4 or body[:2] != b"PK":
        try:
            detail = r.json()
        except Exception:
            detail = {}
        raise EdinetError(f"CSV ZIPを取得できません: {detail or 'unexpected response'}")
    return body


def _decode_csv(raw: bytes) -> str:
    for enc in ("utf-16", "utf-16-le", "cp932", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise EdinetError("EDINET CSVの文字コードを判定できません")


def parse_csv_zip(blob: bytes) -> list[dict]:
    rows: list[dict] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise EdinetError("EDINET CSV ZIPが壊れています") from exc

    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        if "xbrl_to_csv" not in name.lower().replace("\\", "/"):
            continue
        text = _decode_csv(zf.read(name))
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        table = list(reader)
        if not table:
            continue
        header = [c.strip().lstrip("\ufeff") for c in table[0]]
        index = {name: i for i, name in enumerate(header)}
        required = ("要素ID", "コンテキストID", "値")
        if not all(k in index for k in required):
            continue
        for line in table[1:]:
            if len(line) < len(header):
                continue
            rows.append({
                "element_id": line[index["要素ID"]].strip(),
                "item_name": line[index.get("項目名", index["要素ID"])].strip(),
                "context_id": line[index["コンテキストID"]].strip(),
                "value": line[index["値"]].strip(),
                "unit_id": line[index["単位ID"]].strip() if "単位ID" in index else "",
                "source_csv": name,
            })
    if not rows:
        raise EdinetError("XBRL_TO_CSVの財務データを読み取れませんでした")
    return rows


def _number(text: str) -> float | None:
    s = (text or "").strip().replace(",", "").replace("−", "-")
    if not s:
        return None
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _context_ok(context: str, period: str) -> bool:
    if "NonConsolidated" in context:
        return False
    if "Member" in context:
        return False
    return context == period or context.startswith(period + "_")


def _pick(rows: Iterable[dict], candidates: list[str], period: str) -> dict | None:
    candidate_set = set(candidates)
    matches = [
        row for row in rows
        if row["element_id"] in candidate_set and _context_ok(row["context_id"], period)
        and _number(row["value"]) is not None
    ]
    if not matches:
        return None
    matches.sort(key=lambda r: (r["context_id"] != period, candidates.index(r["element_id"])))
    row = dict(matches[0])
    row["numeric_value"] = _number(row["value"])
    return row


def extract_jgaap_metrics(rows: list[dict]) -> dict:
    current_duration = "CurrentYearDuration"
    current_instant = "CurrentYearInstant"
    prior_duration = "Prior1YearDuration"
    prior_instant = "Prior1YearInstant"

    raw = {}
    for key in ("revenue", "operating_profit", "ordinary_profit", "net_profit"):
        raw[key] = _pick(rows, FIELD_CANDIDATES[key], current_duration)
    for key in ("total_assets", "current_assets", "current_liabilities", "equity", "inventory", "receivables"):
        raw[key] = _pick(rows, FIELD_CANDIDATES[key], current_instant)
    for key in ("total_assets", "inventory", "receivables"):
        raw["prior_" + key] = _pick(rows, FIELD_CANDIDATES[key], prior_instant)

    if raw["receivables"] is None:
        notes = _pick(rows, FIELD_CANDIDATES["notes_receivable"], current_instant)
        accounts = _pick(rows, FIELD_CANDIDATES["accounts_receivable"], current_instant)
        if notes and accounts:
            raw["receivables"] = {
                "element_id": notes["element_id"] + "+" + accounts["element_id"],
                "context_id": current_instant,
                "value": "",
                "numeric_value": notes["numeric_value"] + accounts["numeric_value"],
                "item_name": "受取手形+売掛金",
                "source_csv": notes["source_csv"],
            }

    if raw["prior_receivables"] is None:
        notes = _pick(rows, FIELD_CANDIDATES["notes_receivable"], prior_instant)
        accounts = _pick(rows, FIELD_CANDIDATES["accounts_receivable"], prior_instant)
        if notes and accounts:
            raw["prior_receivables"] = {
                "element_id": notes["element_id"] + "+" + accounts["element_id"],
                "context_id": prior_instant,
                "value": "",
                "numeric_value": notes["numeric_value"] + accounts["numeric_value"],
                "item_name": "前期 受取手形+売掛金",
                "source_csv": notes["source_csv"],
            }

    # First MVP is intentionally J-GAAP only. Ordinary income is a useful
    # fail-closed signal because IFRS statements generally do not report it.
    if raw["ordinary_profit"] is None:
        raise EdinetError("J-GAAP有報として必要な経常利益を特定できません。IFRS等はこのMVPでは対象外です")

    values = {key: (row["numeric_value"] if row else None) for key, row in raw.items()}
    return {
        "accounting_standard": "J-GAAP",
        "values": values,
        "evidence": {key: row for key, row in raw.items() if row},
    }


def provenance(blob: bytes, doc: dict) -> dict:
    return {
        "doc_id": doc.get("doc_id", ""),
        "edinet_code": doc.get("edinet_code", ""),
        "filer_name": doc.get("filer_name", ""),
        "period_end": doc.get("period_end", ""),
        "submitted_at": doc.get("submitted_at", ""),
        "csv_zip_sha256": hashlib.sha256(blob).hexdigest(),
    }
