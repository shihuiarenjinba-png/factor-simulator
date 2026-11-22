import io
import unittest

import pandas as pd

from portfolio_input import load_portfolio_upload, parse_portfolio_frame


class NamedBytesIO(io.BytesIO):
    def __init__(self, data=b"", name="portfolio.csv"):
        super().__init__(data)
        self.name = name


class PortfolioInputDirtyTests(unittest.TestCase):
    def csv(self,text,encoding="utf-8-sig"):
        return NamedBytesIO(text.encode(encoding),name="portfolio.csv")

    def test_bom_semicolon_fullwidth_codes_percent_and_blank_rows(self):
        src=self.csv(
            "\ufeff コード ; ウェイト ; メモ\n"
            "７２０３ ; 40% ; a\n"
            "\n"
            "8306 ; 30 ; b\n"
            "9984 ; 30 ; c\n"
        )
        df=load_portfolio_upload(src)
        self.assertEqual(df["コード"].tolist(),["7203","8306","9984"])
        self.assertEqual(df["ウェイト"].tolist(),[40.0,30.0,30.0])

    def test_missing_weight_column_uses_equal_weights(self):
        df=parse_portfolio_frame(pd.DataFrame({"銘柄":["7203","8306","9984"]}))
        self.assertEqual(len(df),3)
        self.assertAlmostEqual(float(df["ウェイト"].sum()),100.0)

    def test_duplicate_code_is_rejected_not_silently_deduplicated(self):
        with self.assertRaisesRegex(ValueError,"重複"):
            parse_portfolio_frame(pd.DataFrame({
                "コード":["7203","7203"],
                "ウェイト":[60,40],
            }))

    def test_blank_bad_negative_inf_and_zero_total_weights_are_rejected(self):
        cases=[
            pd.DataFrame({"コード":["7203","8306"],"ウェイト":[50,""]}),
            pd.DataFrame({"コード":["7203","8306"],"ウェイト":[50,"abc"]}),
            pd.DataFrame({"コード":["7203","8306"],"ウェイト":[50,-1]}),
            pd.DataFrame({"コード":["7203","8306"],"ウェイト":[50,float("inf")]}),
            pd.DataFrame({"コード":["7203","8306"],"ウェイト":[0,0]}),
        ]
        for df in cases:
            with self.assertRaises(ValueError):
                parse_portfolio_frame(df)

    def test_invalid_ticker_does_not_shift_weight_alignment(self):
        with self.assertRaisesRegex(ValueError,"4桁"):
            parse_portfolio_frame(pd.DataFrame({
                "コード":["7203","BAD","8306"],
                "ウェイト":[20,30,50],
            }))

    def test_cp932_csv_is_supported(self):
        text="コード,ウェイト\n7203,60\n8306,40\n"
        src=NamedBytesIO(text.encode("cp932"),name="portfolio.csv")
        df=load_portfolio_upload(src)
        self.assertEqual(df["コード"].tolist(),["7203","8306"])

    def test_excel_uses_same_validation(self):
        bio=NamedBytesIO(name="portfolio.xlsx")
        raw=pd.DataFrame({"コード":["7203","8306"],"ウェイト":[55,45]})
        with pd.ExcelWriter(bio,engine="openpyxl") as writer:
            raw.to_excel(writer,index=False)
        bio.seek(0)
        df=load_portfolio_upload(bio)
        self.assertEqual(df["ウェイト"].tolist(),[55.0,45.0])


if __name__=="__main__":
    unittest.main()
