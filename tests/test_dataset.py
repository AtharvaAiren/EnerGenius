import csv
import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from room_control.dataset import (
    NORMALIZED_FIELDS,
    file_sha256,
    iter_real_transitions,
    load_observations,
    summarize,
)


class DatasetTests(unittest.TestCase):
    def test_normalized_dataset_loads_real_room_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
                writer.writeheader()
                for minute in (600, 601):
                    writer.writerow(
                        {
                            "room_id": "room_2",
                            "room_type": "study-zone",
                            "area_m2": 125,
                            "volume_m3": 418.75,
                            "seating_capacity": 32,
                            "year": 2019,
                            "month": 4,
                            "day_id": 8,
                            "minute_of_day": minute,
                            "is_workday": True,
                            "occupancy": 16,
                            "temperature_c": 24.2,
                            "humidity_pct": 48.0,
                            "co2_ppm": 850.0,
                            "illuminance_lux": 420.0,
                            "vav_damper_pct": 55.0,
                        }
                    )
            observations = load_observations(path)

        self.assertEqual(len(observations), 2)
        self.assertAlmostEqual(observations[0].occupancy_density, 16 / 125)
        self.assertEqual(len(list(iter_real_transitions(observations))), 1)
        self.assertEqual(summarize(observations).maximum_occupancy, 16)

    def test_wrong_schema_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write("wrong,column\n1,2\n")
            with self.assertRaises(ValueError):
                load_observations(path)

    def test_dataset_fingerprint_changes_with_file_content(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("first\n", encoding="utf-8")
            first_hash = file_sha256(path)
            path.write_text("second\n", encoding="utf-8")
            second_hash = file_sha256(path)
        self.assertEqual(len(first_hash), 64)
        self.assertNotEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
