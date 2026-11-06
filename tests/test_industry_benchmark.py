import unittest

from industry_benchmark import BENCHMARKS, calculate_metrics, classify, diagnose


class IndustryBenchmarkTests(unittest.TestCase):
    def test_basic_calculation(self):
        m = calculate_metrics(
            revenue=1_000_000,
            cogs=400_000,
            net_profit=80_000,
            labor_cost=300_000,
            fixed_costs=250_000,
        )
        self.assertAlmostEqual(m["gross_margin"], 60.0)
        self.assertAlmostEqual(m["net_margin"], 8.0)
        self.assertAlmostEqual(m["labor_ratio"], 30.0)
        self.assertAlmostEqual(m["break_even_revenue"], 416_666.6667, places=2)

    def test_restaurant_reference_can_score_healthy(self):
        m = calculate_metrics(
            revenue=1_000_000,
            cogs=400_000,
            net_profit=60_000,
            labor_cost=300_000,
            fixed_costs=250_000,
        )
        d = diagnose("restaurant", m)
        self.assertEqual(d["score"], 100)
        self.assertTrue(all(row["status"] == "healthy" for row in d["rows"]))

    def test_high_labor_ratio_warns(self):
        status, _, score = classify(45.0, BENCHMARKS["restaurant"]["labor_ratio"])
        self.assertEqual(status, "warning")
        self.assertLess(score, 100)

    def test_zero_revenue_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics(
                revenue=0,
                cogs=0,
                net_profit=0,
                labor_cost=0,
                fixed_costs=0,
            )

    def test_negative_gross_margin_has_no_break_even(self):
        m = calculate_metrics(
            revenue=100,
            cogs=120,
            net_profit=-40,
            labor_cost=10,
            fixed_costs=20,
        )
        self.assertIsNone(m["break_even_revenue"])
        d = diagnose("retail", m)
        self.assertIn("損益分岐売上", d["advice"][0])


if __name__ == "__main__":
    unittest.main()
