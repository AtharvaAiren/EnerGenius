import unittest

from evaluate import _energy_comparison


class ShadowEnergyEvaluationTests(unittest.TestCase):
    def test_energy_comparison_reports_fan_savings(self):
        warm_started = {
            "estimated_fan_energy_kwh": 15.0,
            "estimated_light_energy_kwh": 5.0,
        }
        baseline = {
            "estimated_fan_energy_kwh": 20.0,
            "estimated_light_energy_kwh": 5.0,
        }

        comparison = _energy_comparison(warm_started, baseline)

        self.assertAlmostEqual(
            comparison["estimated_fan_energy_savings_pct"], 25.0
        )
        self.assertAlmostEqual(
            comparison["estimated_total_energy_savings_pct"], 20.0
        )


if __name__ == "__main__":
    unittest.main()
