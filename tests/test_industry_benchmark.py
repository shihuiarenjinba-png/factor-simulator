import unittest

from industry_benchmark import calculate_metrics, diagnose, industry_spec, assess_absolute


class JapanIndustryBenchmarkTests(unittest.TestCase):
    def test_manufacturing_reference_values_are_versioned(self):
        spec = industry_spec("manufacturing")
        self.assertEqual(spec["metrics"]["operating_margin"]["reference"], 5.4)
        self.assertEqual(spec["metrics"]["equity_ratio"]["reference"], 50.3)
        self.assertEqual(spec["metrics"]["inventory_months"]["reference"], 1.64)

    def test_calculate_current_ratio_and_equity_ratio(self):
        m = calculate_metrics(
            revenue=100_000_000,
            operating_profit=6_000_000,
            ordinary_profit=8_000_000,
            current_total_assets=120_000_000,
            prior_total_assets=100_000_000,
            equity=36_000_000,
            current_assets=50_000_000,
            current_liabilities=50_000_000,
        )
        self.assertAlmostEqual(m["current_ratio"], 100.0)
        self.assertAlmostEqual(m["equity_ratio"], 30.0)

    def test_public_guideposts_are_exact(self):
        current = assess_absolute("current_ratio", 99.9)
        equity = assess_absolute("equity_ratio", 29.9)
        self.assertEqual(current["threshold"], 100.0)
        self.assertEqual(current["status"], "warning")
        self.assertEqual(equity["threshold"], 30.0)
        self.assertEqual(equity["status"], "warning")
        self.assertEqual(assess_absolute("current_ratio", 100.0)["status"], "healthy")
        self.assertEqual(assess_absolute("equity_ratio", 30.0)["status"], "healthy")

    def test_industry_reference_is_not_presented_as_danger_cutoff(self):
        m = calculate_metrics(
            revenue=100_000_000,
            operating_profit=2_000_000,
            ordinary_profit=3_000_000,
            current_total_assets=120_000_000,
            equity=24_000_000,
            current_assets=40_000_000,
            current_liabilities=50_000_000,
        )
        d = diagnose("manufacturing", m)
        self.assertTrue(d["guidance_rows"])
        self.assertTrue(all("reference" in row for row in d["rows"]))
        self.assertTrue(all("danger" not in row and "caution" not in row for row in d["rows"]))

    def test_zero_revenue_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics(revenue=0, operating_profit=0, ordinary_profit=0)


if __name__ == "__main__":
    unittest.main()
