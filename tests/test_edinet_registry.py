import tempfile
import unittest
from pathlib import Path

from tools.edinet_registry import add_record


class EdinetRegistryTests(unittest.TestCase):
    def test_append_only_record_has_sha_and_standard(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "sample.xbrl"
            src.write_bytes(b"sample-edinet-xbrl")
            data = {"filings": []}
            row = add_record(
                data,
                source_file=src,
                edinet_code="E00001",
                doc_id="S100TEST",
                company_name="テスト株式会社",
                fiscal_year_end="2026-03-31",
                accounting_standard="J-GAAP",
                taxonomy_year="2026",
                industry="manufacturing",
                source_url="https://disclosure2.edinet-fsa.go.jp/",
            )
            self.assertEqual(row["accounting_standard"], "J-GAAP")
            self.assertEqual(len(row["source_sha256"]), 64)
            self.assertEqual(len(data["filings"]), 1)

    def test_duplicate_doc_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "sample.xbrl"
            src.write_bytes(b"x")
            data = {"filings": [{"doc_id": "S100TEST"}]}
            with self.assertRaisesRegex(ValueError, "already registered"):
                add_record(
                    data,
                    source_file=src,
                    edinet_code="E00001",
                    doc_id="S100TEST",
                    company_name="テスト株式会社",
                    fiscal_year_end="2026-03-31",
                    accounting_standard="J-GAAP",
                    taxonomy_year="2026",
                    industry="manufacturing",
                    source_url="https://disclosure2.edinet-fsa.go.jp/",
                )

    def test_unknown_standard_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "sample.xbrl"
            src.write_bytes(b"x")
            with self.assertRaises(ValueError):
                add_record(
                    {"filings": []},
                    source_file=src,
                    edinet_code="E00001",
                    doc_id="S100TEST",
                    company_name="テスト株式会社",
                    fiscal_year_end="2026-03-31",
                    accounting_standard="UNKNOWN",
                    taxonomy_year="2026",
                    industry="manufacturing",
                    source_url="https://disclosure2.edinet-fsa.go.jp/",
                )


if __name__ == "__main__":
    unittest.main()
