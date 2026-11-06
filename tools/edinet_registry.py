"""Append one EDINET filing record to the local versioned registry.

This tool does not call EDINET. It records provenance for a downloaded source file
so the dataset can be accumulated safely before API automation is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "edinet_registry.json"
ALLOWED_STANDARDS = {"J-GAAP", "IFRS", "US-GAAP", "OTHER"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_record(data: dict, *, source_file: Path, edinet_code: str, doc_id: str,
               company_name: str, fiscal_year_end: str, accounting_standard: str,
               taxonomy_year: str, industry: str, source_url: str) -> dict:
    if accounting_standard not in ALLOWED_STANDARDS:
        raise ValueError("unknown accounting standard")
    if not source_file.is_file() or source_file.is_symlink():
        raise ValueError("source file missing or symlinked")
    if any(row.get("doc_id") == doc_id for row in data.get("filings", [])):
        raise ValueError(f"doc_id already registered: {doc_id}")

    record = {
        "edinet_code": edinet_code.strip(),
        "doc_id": doc_id.strip(),
        "company_name": company_name.strip(),
        "fiscal_year_end": fiscal_year_end.strip(),
        "accounting_standard": accounting_standard,
        "taxonomy_year": taxonomy_year.strip(),
        "industry": industry.strip(),
        "source_url": source_url.strip(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source_file),
        "source_filename": source_file.name,
    }
    if not all(record[key] for key in (
        "edinet_code", "doc_id", "company_name", "fiscal_year_end",
        "taxonomy_year", "industry", "source_url"
    )):
        raise ValueError("required metadata is blank")
    data.setdefault("filings", []).append(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--edinet-code", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--fiscal-year-end", required=True)
    parser.add_argument("--accounting-standard", choices=sorted(ALLOWED_STANDARDS), required=True)
    parser.add_argument("--taxonomy-year", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    record = add_record(
        data,
        source_file=args.file,
        edinet_code=args.edinet_code,
        doc_id=args.doc_id,
        company_name=args.company_name,
        fiscal_year_end=args.fiscal_year_end,
        accounting_standard=args.accounting_standard,
        taxonomy_year=args.taxonomy_year,
        industry=args.industry,
        source_url=args.source_url,
    )
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
