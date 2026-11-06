import unittest

from industry_benchmark import BENCHMARKS, calculate_metrics, diagnose, industry_spec, thresholds


class JapanIndustryBenchmarkTests(unittest.TestCase):
    def test_manufacturing_reference_values_are_versioned(self):
        spec = industry_spec("manufacturing")
        self.assertEqual(spec["metrics"]["operating_margin"]["reference"], 5.4)
        self.assertEqual(spec["metrics"]["equity_ratio"]["reference"], 50.3)
        self.assertEqual(spec["metrics"]["inventory_months"]["reference"], 1.64)

    def test_food_and_chemical_override_only_known_profitability_metrics(self):
        food = industry_spec("manufacturing", "food")
        chem = industry_spec("manufacturing", "chemical")
        self.assertEqual(food["metrics"]["operating_margin"]["reference"], 3.6)
        self.assertEqual(food["metrics"]["ordinary_margin"]["reference"], 5.5)
        self.assertEqual(chem["metrics"]["operating_margin"]["reference"], 8.5)
        self.assertEqual(chem["metrics"]["ordinary_margin"]["reference"], 12.3)
        self.assertEqual(food["metrics"]["equity_ratio"]["reference"], 50.3)

    def test_calculate_japan_financial_metrics(self):
        m = calculate_metrics(
            revenue=100_000_000,
            operating_profit=6_000_000,
            ordinary_profit=8_000_000,
            net_profit=5_000_000,
            current_total_assets=120_000_000,
            prior_total_assets=100_000_000,
            equity=60_000_000,
            current_inventory=15_000_000,
            prior_inventory=13_000_000,
            current_receivables=20_000_000,
            prior_receivables=18_000_000,
        )
        self.assertAlmostEqual(m["operating_margin"], 6.0)
        self.assertAlmostEqual(m["ordinary_margin"], 8.0)
        self.assertAlmostEqual(m["ordinary_roa"], 8_000_000 / 110_000_000 * 100)
        self.assertAlmostEqual(m["equity_ratio"], 50.0)
        self.assertAlmostEqual(m["inventory_months"], 14_000_000 / (100_000_000 / 12))
        self.assertAlmostEqual(m["receivable_months"], 19_000_000 / (100_000_000 / 12))

    def test_higher_and_lower_magic_lines_are_transparent(self):
        higher = thresholds(50.0, "higher")
        lower = thresholds(2.0, "lower")
        self.assertEqual(higher, {"caution": 37.5, "danger": 25.0})
        self.assertEqual(lower, {"caution": 2.5, "danger": 3.0})

    def test_diagnosis_exposes_reference_caution_danger_and_sources(self):
        m = calculate_metrics(
            revenue=100_000_000,
            operating_profit=2_000_000,
            ordinary_profit=3_000_000,
            current_total_assets=120_000_000,
            prior_total_assets=120_000_000,
            equity=20_000_000,
            current_inventory=30_000_000,
            prior_inventory=30_000_000,
            current_receivables=30_000_000,
            prior_receivables=30_000_000,
        )
        d = diagnose("manufacturing", m)
        self.assertLess(d["score"], 80)
        self.assertTrue(d["sources"])
        self.assertTrue(all("reference" in row and "caution" in row and "danger" in row for row in d["rows"]))

    def test_zero_revenue_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics(revenue=0, operating_profit=0, ordinary_profit=0)


if __name__ == "__main__":
    unittest.main()
