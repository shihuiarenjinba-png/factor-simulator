import csv
import tempfile
import unittest
from pathlib import Path

from edinet_store import benchmark_rows, dataset_summary, get_filing, list_filings, search_companies
from tools.build_edinet_sqlite import connect, import_csv, init_db, rebuild_benchmarks


HEADERS = [
    "doc_id","edinet_code","sec_code","company_name","fiscal_year_end",
    "accounting_standard","industry_code","industry_label","source_url","source_sha256",
    "revenue","operating_profit","ordinary_profit","net_profit","total_assets",
    "prior_total_assets","equity","current_assets","current_liabilities","inventory",
    "prior_inventory","receivables","prior_receivables"
]


class EdinetStoreTests(unittest.TestCase):
    def make_dataset(self, root: Path):
        csv_path = root / "normalized.csv"
        db_path = root / "edinet.sqlite3"
        rows = [
            ["S100AAA1","E00001","1111","甲製造株式会社","2026-03-31","J-GAAP","manufacturing","製造業",
             "https://example.invalid/S100AAA1","a"*64,
             "100000000","6000000","8000000","5000000","120000000","110000000",
             "60000000","50000000","40000000","14000000","13000000","18000000","17000000"],
            ["S100AAA2","E00002","2222","乙製造株式会社","2026-03-31","J-GAAP","manufacturing","製造業",
             "https://example.invalid/S100AAA2","b"*64,
             "200000000","8000000","12000000","6000000","250000000","240000000",
             "100000000","90000000","70000000","30000000","28000000","35000000","33000000"],
            ["S100AAA3","E00001","1111","甲製造株式会社","2025-03-31","J-GAAP","manufacturing","製造業",
             "https://example.invalid/S100AAA3","c"*64,
             "90000000","4500000","6500000","4000000","110000000","100000000",
             "52000000","45000000","38000000","12000000","11000000","16000000","15000000"],
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADERS)
            writer.writerows(rows)

        conn = connect(db_path, read_only=False)
        try:
            init_db(conn)
            with conn:
                self.assertEqual(import_csv(csv_path, conn), 3)
                self.assertGreater(rebuild_benchmarks(conn), 0)
                conn.execute("INSERT OR REPLACE INTO dataset_metadata VALUES('dataset_version','test-v1')")
        finally:
            conn.close()
        return db_path

    def test_query_company_and_filing(self):
        with tempfile.TemporaryDirectory() as td:
            db = self.make_dataset(Path(td))
            summary = dataset_summary(db)
            self.assertEqual(summary["companies"], 2)
            self.assertEqual(summary["filings"], 3)
            self.assertEqual(summary["metadata"]["dataset_version"], "test-v1")

            companies = search_companies("甲", path=db)
            self.assertEqual(len(companies), 1)
            self.assertEqual(companies[0]["edinet_code"], "E00001")

            filings = list_filings("E00001", path=db)
            self.assertEqual([r["fiscal_year_end"] for r in filings], ["2026-03-31", "2025-03-31"])

            row = get_filing("S100AAA1", path=db)
            self.assertEqual(row["revenue"], 100000000.0)
            self.assertEqual(row["source_sha256"], "a"*64)

    def test_peer_benchmarks_are_precomputed(self):
        with tempfile.TemporaryDirectory() as td:
            db = self.make_dataset(Path(td))
            rows = benchmark_rows("manufacturing", 2026, path=db)
            metrics = {row["metric"]: row for row in rows}
            self.assertIn("operating_margin", metrics)
            self.assertEqual(metrics["operating_margin"]["sample_count"], 2)
            self.assertAlmostEqual(metrics["operating_margin"]["median"], 5.0)
            self.assertIsNotNone(metrics["operating_margin"]["q25"])
            self.assertIsNotNone(metrics["operating_margin"]["q75"])

    def test_jgaap_only_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "ifrs.csv"
            db_path = root / "edinet.sqlite3"
            row = [
                "S100IFRS","E99999","9999","IFRS社","2026-03-31","IFRS","manufacturing","製造業",
                "https://example.invalid/ifrs","d"*64,
                "100","10","0","","","","","","","","","",""
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(HEADERS)
                writer.writerow(row)
            conn = connect(db_path, read_only=False)
            try:
                init_db(conn)
                with self.assertRaisesRegex(ValueError, "J-GAAP以外"):
                    import_csv(csv_path, conn)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
