import unittest
from datetime import datetime, timezone

from hardware_demo import map_logical_fan_pwm
from room_control.models import ControlAction
from room_control.serial_hardware import LiveRoomConfig, SerialRoomBridge, Telemetry


VALID_LINE = "TEL,7,1234,25.5,51.0,321.5,3,1,1,70,177,HOST,OFF,500,400"
CO2_LINE = VALID_LINE + ",812.0,300"


class FakeSerial:
    def __init__(self, lines=None):
        self.lines = [line.encode("ascii") + b"\n" for line in (lines or [])]
        self.writes = []
        self.closed = False

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def write(self, payload):
        self.writes.append(payload.decode("ascii").strip())

    def flush(self):
        pass

    def close(self):
        self.closed = True


class SerialHardwareTests(unittest.TestCase):
    def test_telemetry_parser(self):
        telemetry = Telemetry.parse(VALID_LINE)
        self.assertEqual(telemetry.people_count, 3)
        self.assertEqual(telemetry.fan_pwm, 177)
        self.assertIsNone(telemetry.co2_ppm)

    def test_extended_telemetry_parser_reads_co2(self):
        telemetry = Telemetry.parse(CO2_LINE)
        self.assertEqual(telemetry.co2_ppm, 812.0)
        self.assertEqual(telemetry.co2_age_ms, 300)

    def test_telemetry_rejects_non_off_reserved_value(self):
        with self.assertRaises(ValueError):
            Telemetry.parse(VALID_LINE.replace(",HOST,OFF,", ",HOST,COOL,"))

    def test_bridge_builds_density_aware_observation(self):
        serial = FakeSerial([VALID_LINE])
        bridge = SerialRoomBridge(
            serial_port=serial,
            reset_delay_s=0,
            config=LiveRoomConfig(area_m2=30.0, volume_m3=90.0, seating_capacity=12),
            clock=lambda: datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc),
        )
        observation = bridge.read_observation()
        self.assertEqual(observation.occupancy, 3)
        self.assertAlmostEqual(observation.occupancy_density, 0.1)
        self.assertEqual(observation.illuminance_lux, 321.5)
        self.assertEqual(observation.co2_ppm, 600.0)
        self.assertFalse(observation.co2_is_measured)

    def test_bridge_prefers_fresh_measured_co2(self):
        serial = FakeSerial([CO2_LINE])
        bridge = SerialRoomBridge(serial_port=serial, reset_delay_s=0)
        observation = bridge.read_observation()
        self.assertEqual(observation.co2_ppm, 812.0)
        self.assertTrue(observation.co2_is_measured)

    def test_bridge_falls_back_when_co2_is_stale(self):
        stale_line = VALID_LINE + ",1200.0,15000"
        serial = FakeSerial([stale_line])
        bridge = SerialRoomBridge(
            serial_port=serial,
            reset_delay_s=0,
            config=LiveRoomConfig(assumed_co2_ppm=650.0, max_sensor_age_ms=10_000),
        )
        observation = bridge.read_observation()
        self.assertEqual(observation.co2_ppm, 650.0)
        self.assertFalse(observation.co2_is_measured)

    def test_bridge_sends_exact_pwm_values(self):
        serial = FakeSerial()
        bridge = SerialRoomBridge(serial_port=serial, reset_delay_s=0)
        bridge.apply_action(ControlAction(fan_pwm=193, light_pwm=87))
        self.assertEqual(serial.writes[-1], "CMD,1,87,193,OFF")

    def test_logical_fan_levels_remain_distinct_above_stall_speed(self):
        mapped = [
            map_logical_fan_pwm(pwm, minimum_effective_pwm=177)
            for pwm in (0, 64, 128, 204, 255)
        ]
        self.assertEqual(mapped[0], 0)
        self.assertEqual(mapped[-1], 255)
        self.assertTrue(all(left < right for left, right in zip(mapped, mapped[1:])))
        self.assertGreaterEqual(mapped[1], 177)

    def test_safe_close_sends_safe_mode(self):
        serial = FakeSerial()
        bridge = SerialRoomBridge(serial_port=serial, reset_delay_s=0)
        bridge.close(safe=True)
        self.assertIn("MODE,SAFE", serial.writes)
        self.assertTrue(serial.closed)


if __name__ == "__main__":
    unittest.main()
