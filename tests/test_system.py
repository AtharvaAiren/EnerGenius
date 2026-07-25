import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from room_control.comfort import assess_comfort, estimated_power_w
from room_control.controllers import (
    AdaptiveQLearningController,
    ComfortBaselineController,
    VentilationPrior,
    ventilation_level,
)
from room_control.dataset import split_day_ids, split_train_validation_test_day_ids
from room_control.models import (
    ActuatorCalibration,
    ControlAction,
    RoomObservation,
    UserFeedback,
    UserPreferences,
)
from room_control.offline_rl import ConservativeOfflineVentilationQ
from room_control.session import AdaptiveControlSession


def observation(**changes) -> RoomObservation:
    base = RoomObservation(
        room_id="room_1",
        room_type="lecture",
        area_m2=100.0,
        volume_m3=300.0,
        seating_capacity=50,
        year=2019,
        month=3,
        day_id=1,
        minute_of_day=9 * 60,
        is_workday=True,
        occupancy=10,
        temperature_c=25.0,
        humidity_pct=50.0,
        co2_ppm=700.0,
        illuminance_lux=300.0,
        logged_ventilation_pct=40.0,
    )
    return replace(base, **changes)


class RealRoomControlTests(unittest.TestCase):
    def test_actions_use_direct_pwm_range(self):
        action = ControlAction(fan_pwm=173, light_pwm=91)
        self.assertEqual(action.fan_pwm, 173)
        self.assertEqual(action.light_pwm, 91)
        with self.assertRaises(ValueError):
            ControlAction(fan_pwm=256)

    def test_density_uses_published_room_area(self):
        self.assertAlmostEqual(observation(occupancy=20).occupancy_density, 0.2)

    def test_empty_room_shuts_down_comfort_loads(self):
        action = ComfortBaselineController().act(
            observation(occupancy=0, co2_ppm=500.0, illuminance_lux=0.0)
        )
        self.assertEqual(action, ControlAction())

    def test_dim_occupied_room_requests_more_light(self):
        controller = ComfortBaselineController()
        dim = controller.act(observation(illuminance_lux=50.0)).light_pwm
        bright = controller.act(observation(illuminance_lux=700.0)).light_pwm
        self.assertGreater(dim, bright)

    def test_lighting_feedback_uses_previous_pwm(self):
        controller = ComfortBaselineController()
        previous = ControlAction(light_pwm=80)
        action = controller.act(
            observation(illuminance_lux=300.0, previous_action=previous)
        )
        self.assertGreater(action.light_pwm, previous.light_pwm)

    def test_high_co2_forces_maximum_ventilation(self):
        decision = ComfortBaselineController().decide(observation(co2_ppm=1300.0))
        self.assertEqual(decision.action.fan_pwm, 255)
        self.assertTrue(any(reason.startswith("CO2 demand") for reason in decision.reasons))

    def test_warm_room_increases_air_movement(self):
        controller = ComfortBaselineController(UserPreferences(target_temperature_c=24.0))
        warm = controller.act(observation(temperature_c=30.0)).fan_pwm
        cool = controller.act(observation(temperature_c=22.0)).fan_pwm
        self.assertGreater(warm, cool)

    def test_fan_power_estimate_changes_with_pwm(self):
        calibration = ActuatorCalibration()
        low = estimated_power_w(ControlAction(fan_pwm=100), calibration)
        high = estimated_power_w(ControlAction(fan_pwm=220), calibration)
        self.assertGreater(high, low)

    def test_light_power_estimate_changes_with_pwm(self):
        calibration = ActuatorCalibration()
        off = estimated_power_w(ControlAction(), calibration)
        on = estimated_power_w(ControlAction(light_pwm=255), calibration)
        self.assertEqual(off, 0.0)
        self.assertAlmostEqual(on, calibration.light_max_power_w)

    def test_ventilation_prior_uses_logged_vav_only(self):
        prior = VentilationPrior()
        records = [
            observation(logged_ventilation_pct=40.0),
            observation(logged_ventilation_pct=60.0),
        ]
        prior.fit(records)
        self.assertEqual(prior.samples, 2)
        self.assertAlmostEqual(prior.predict(records[0]), 128, delta=1)

    def test_ventilation_prior_transfers_to_unknown_room_type(self):
        prior = VentilationPrior()
        training = observation(room_type="lecture", logged_ventilation_pct=70.0)
        prior.fit([training])
        prototype = replace(training, room_type="prototype", logged_ventilation_pct=None)
        self.assertEqual(prior.predict_level(prototype), "HIGH")

    def test_vav_percentages_map_to_ventilation_levels(self):
        self.assertEqual(ventilation_level(0.0), "OFF")
        self.assertEqual(ventilation_level(20.0), "LOW")
        self.assertEqual(ventilation_level(50.0), "MEDIUM")
        self.assertEqual(ventilation_level(80.0), "HIGH")
        self.assertEqual(ventilation_level(100.0), "MAXIMUM")

    def test_split_keeps_whole_day_ids_disjoint(self):
        records = [observation(day_id=day) for day in range(10)]
        training, evaluation = split_day_ids(records, seed=7)
        self.assertFalse(training & evaluation)
        self.assertEqual(training | evaluation, set(range(10)))

    def test_three_way_split_keeps_whole_days_disjoint(self):
        records = [observation(day_id=day) for day in range(44)]
        training, validation, evaluation = split_train_validation_test_day_ids(
            records,
            seed=42,
        )
        self.assertEqual((len(training), len(validation), len(evaluation)), (28, 7, 9))
        self.assertFalse(training & validation)
        self.assertFalse(training & evaluation)
        self.assertFalse(validation & evaluation)
        self.assertEqual(training | validation | evaluation, set(range(44)))

    def test_feedback_updates_stored_preferences(self):
        controller = AdaptiveQLearningController(seed=1)
        before = controller.preferences
        after = controller.apply_feedback(
            UserFeedback(thermal="hot", air_quality="stale", lighting="dim")
        )
        self.assertLess(after.target_temperature_c, before.target_temperature_c)
        self.assertLess(after.max_co2_ppm, before.max_co2_ppm)
        self.assertGreater(
            after.target_illuminance_lux, before.target_illuminance_lux
        )

    def test_online_session_learns_only_after_real_next_state(self):
        controller = AdaptiveQLearningController(epsilon=0.0, seed=2)
        session = AdaptiveControlSession(controller, explore=False)
        _decision, reward = session.step(observation())
        self.assertIsNone(reward)
        _decision, reward = session.step(
            observation(minute_of_day=9 * 60 + 1, co2_ppm=680.0),
            feedback=UserFeedback(overall=1),
        )
        self.assertIsInstance(reward, float)
        self.assertEqual(controller.online_updates, 1)
        self.assertTrue(any(value != 0.0 for value in controller.q.values()))

    def test_policy_round_trip_preserves_prior_preferences_and_q(self):
        controller = AdaptiveQLearningController(seed=3)
        controller.training_metadata = {
            "dataset_sha256": "abc123",
            "train_day_ids": [1, 2],
            "validation_day_ids": [4],
            "evaluation_day_ids": [3],
            "rooms": ["room_1"],
        }
        controller.pretrain_ventilation([observation()])
        session = AdaptiveControlSession(controller, explore=False)
        session.step(observation())
        session.step(observation(minute_of_day=541, temperature_c=25.2))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            controller.save(path)
            loaded = AdaptiveQLearningController.load(path)
        self.assertEqual(loaded.ventilation_prior.samples, 1)
        self.assertEqual(loaded.preferences, controller.preferences)
        self.assertEqual(dict(loaded.q), dict(controller.q))
        self.assertEqual(
            loaded.offline_ventilation_q.transition_count,
            controller.offline_ventilation_q.transition_count,
        )
        self.assertEqual(loaded.training_metadata, controller.training_metadata)
        self.assertEqual(
            loaded.ventilation_prior.predict_level(
                observation(room_type="prototype", logged_ventilation_pct=None)
            ),
            "MEDIUM",
        )

    def test_comfort_is_preference_relative(self):
        comfortable = assess_comfort(observation(), UserPreferences(target_temperature_c=25.0))
        hot = assess_comfort(observation(), UserPreferences(target_temperature_c=20.0))
        self.assertGreater(comfortable.overall, hot.overall)

    def test_offline_q_uses_only_supported_logged_levels(self):
        records = []
        for day_id, vav in ((1, 10.0), (2, 40.0)):
            for minute in range(60):
                records.append(
                    observation(
                        day_id=day_id,
                        minute_of_day=9 * 60 + minute,
                        logged_ventilation_pct=vav,
                    )
                )
        model = ConservativeOfflineVentilationQ(
            horizon_minutes=5,
            stride_minutes=5,
            minimum_support=1,
            iterations=5,
            minimum_q_advantage=0.0,
        )
        preferences = UserPreferences()
        model.fit(records, preferences, ActuatorCalibration())
        recommendation = model.predict_level(records[0], preferences, "MEDIUM")
        self.assertIn(recommendation, {"LOW", "MEDIUM"})
        self.assertGreater(model.transition_count, 0)
        self.assertGreater(model.q_entries, 0)


if __name__ == "__main__":
    unittest.main()
