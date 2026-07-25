import csv
import json
import unittest
from pathlib import Path

from room_control.controllers import AdaptiveQLearningController
from room_control.dataset import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "real_room" / "observations.csv.gz"
POLICY = ROOT / "room_policy.json"
EVALUATION = ROOT / "reports" / "real_room" / "latest" / "evaluation.json"
DEMO_CSV = ROOT / "reports" / "real_room" / "demo" / "decision_audit.csv"


class ReleaseArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(POLICY.read_text(encoding="utf-8"))
        cls.evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))

    def test_policy_and_evaluation_match_the_current_dataset(self):
        dataset_hash = file_sha256(DATASET)
        self.assertEqual(
            self.policy["training_metadata"]["dataset_sha256"], dataset_hash
        )
        self.assertEqual(self.evaluation["dataset_sha256"], dataset_hash)
        self.assertEqual(
            self.policy["format_version"],
            AdaptiveQLearningController.FORMAT_VERSION,
        )

    def test_saved_whole_day_split_is_disjoint_and_complete(self):
        metadata = self.policy["training_metadata"]
        training = set(metadata["train_day_ids"])
        validation = set(metadata["validation_day_ids"])
        evaluation = set(metadata["evaluation_day_ids"])
        self.assertFalse(training & validation)
        self.assertFalse(training & evaluation)
        self.assertFalse(validation & evaluation)
        self.assertEqual(len(training), 28)
        self.assertEqual(len(validation), 7)
        self.assertEqual(len(evaluation), 9)
        self.assertEqual(training | validation | evaluation, set(range(44)))
        self.assertEqual(
            self.evaluation["validation_day_ids"], sorted(validation)
        )
        self.assertEqual(
            self.evaluation["evaluation_day_ids"], sorted(evaluation)
        )

    def test_energy_totals_and_savings_are_internally_consistent(self):
        warm = self.evaluation["overall"]
        baseline = self.evaluation["comfort_baseline"]
        comparison = self.evaluation["energy_comparison"]
        self.assertAlmostEqual(
            warm["estimated_total_energy_kwh"],
            warm["estimated_fan_energy_kwh"]
            + warm["estimated_light_energy_kwh"],
        )
        expected_savings = 100.0 * (
            baseline["estimated_total_energy_kwh"]
            - warm["estimated_total_energy_kwh"]
        ) / baseline["estimated_total_energy_kwh"]
        self.assertAlmostEqual(
            comparison["estimated_total_energy_savings_pct"], expected_savings
        )

    def test_offline_comfort_energy_gate_passes(self):
        gate = self.evaluation["offline_success_gate"]
        self.assertEqual(gate["status"], "PASS")
        self.assertGreaterEqual(gate["actual"]["average_comfort_delta"], 0.0)
        self.assertGreaterEqual(gate["actual"]["comfort_retained_pct"], 95.0)
        self.assertGreaterEqual(gate["actual"]["fan_energy_savings_pct"], 5.0)

    def test_demo_trace_contains_both_control_outputs(self):
        with DEMO_CSV.open("r", encoding="utf-8", newline="") as handle:
            fields = set(csv.DictReader(handle).fieldnames or ())
        self.assertIn("requested_fan_pwm", fields)
        self.assertIn("requested_light_pwm", fields)
        self.assertIn("behavior_prior_level", fields)
        self.assertIn("offline_q_level", fields)


if __name__ == "__main__":
    unittest.main()
