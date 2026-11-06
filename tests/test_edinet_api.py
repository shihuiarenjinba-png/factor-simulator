import csv
import io
import unittest
import zipfile

from edinet_api import EdinetError, extract_jgaap_metrics, parse_csv_zip


def make_zip(rows):
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"])
    for row in rows:
        w.writerow(row)
    raw = out.getvalue().encode("utf-16")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("XBRL_TO_CSV/sample.csv", raw)
    return buf.getvalue()


class EdinetAdapterTests(unittest.TestCase):
    def test_extract_current_jgaap_values(self):
        blob = make_zip([
            ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration", "", "", "", "JPY", "円", "100000000"],
            ["jppfs_cor:OperatingIncome", "営業利益", "CurrentYearDuration", "", "", "", "JPY", "円", "6000000"],
            ["jppfs_cor:OrdinaryIncome", "経常利益", "CurrentYearDuration", "", "", "", "JPY", "円", "8000000"],
            ["jppfs_cor:Assets", "資産", "CurrentYearInstant", "", "", "", "JPY", "円", "120000000"],
            ["jppfs_cor:Assets", "資産", "Prior1YearInstant", "", "", "", "JPY", "円", "110000000"],
            ["jppfs_cor:CurrentAssets", "流動資産", "CurrentYearInstant", "", "", "", "JPY", "円", "50000000"],
            ["jppfs_cor:CurrentLiabilities", "流動負債", "CurrentYearInstant", "", "", "", "JPY", "円", "40000000"],
            ["jppfs_cor:NetAssets", "純資産", "CurrentYearInstant", "", "", "", "JPY", "円", "60000000"],
        ])
        rows = parse_csv_zip(blob)
        result = extract_jgaap_metrics(rows)
        self.assertEqual(result["accounting_standard"], "J-GAAP")
        self.assertEqual(result["values"]["revenue"], 100000000.0)
        self.assertEqual(result["values"]["ordinary_profit"], 8000000.0)
        self.assertEqual(result["values"]["prior_total_assets"], 110000000.0)

    def test_nonconsolidated_rows_are_ignored(self):
        blob = make_zip([
            ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration_NonConsolidatedMember", "", "", "", "JPY", "円", "999"],
            ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration", "", "", "", "JPY", "円", "100"],
            ["jppfs_cor:OperatingIncome", "営業利益", "CurrentYearDuration", "", "", "", "JPY", "円", "6"],
            ["jppfs_cor:OrdinaryIncome", "経常利益", "CurrentYearDuration", "", "", "", "JPY", "円", "8"],
        ])
        result = extract_jgaap_metrics(parse_csv_zip(blob))
        self.assertEqual(result["values"]["revenue"], 100.0)

    def test_ifrs_like_input_fails_closed(self):
        blob = make_zip([
            ["ifrs-full:Revenue", "売上収益", "CurrentYearDuration", "", "", "", "JPY", "円", "100"],
            ["ifrs-full:OperatingProfitLoss", "営業利益", "CurrentYearDuration", "", "", "", "JPY", "円", "10"],
        ])
        with self.assertRaisesRegex(EdinetError, "J-GAAP"):
            extract_jgaap_metrics(parse_csv_zip(blob))


if __name__ == "__main__":
    unittest.main()
