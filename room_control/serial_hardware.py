from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from room_control.models import ControlAction, RoomObservation


@dataclass(frozen=True)
class Telemetry:
    sequence: int
    device_millis: int
    temperature: float
    humidity: float
    lux: float
    people_count: int
    occupied: bool
    motion: bool
    led_pwm: int
    fan_pwm: int
    control_mode: str
    reserved_request: str
    dht_age_ms: int
    lux_age_ms: int
    co2_ppm: float | None
    co2_age_ms: int | None

    @classmethod
    def parse(cls, line: str) -> "Telemetry":
        fields = line.strip().split(",")
        if len(fields) not in {15, 17} or fields[0] != "TEL":
            raise ValueError("not an EnerGenius telemetry line")
        try:
            raw_co2 = float(fields[15]) if len(fields) == 17 else -1.0
            telemetry = cls(
                sequence=int(fields[1]),
                device_millis=int(fields[2]),
                temperature=float(fields[3]),
                humidity=float(fields[4]),
                lux=float(fields[5]),
                people_count=int(fields[6]),
                occupied=_parse_flag(fields[7]),
                motion=_parse_flag(fields[8]),
                led_pwm=int(fields[9]),
                fan_pwm=int(fields[10]),
                control_mode=fields[11],
                reserved_request=fields[12],
                dht_age_ms=int(fields[13]),
                lux_age_ms=int(fields[14]),
                co2_ppm=raw_co2 if raw_co2 >= 0.0 else None,
                co2_age_ms=int(fields[16]) if len(fields) == 17 else None,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid telemetry values: {exc}") from exc
        telemetry.validate()
        return telemetry

    def validate(self, max_sensor_age_ms: int | None = None) -> None:
        if not all(math.isfinite(value) for value in (self.temperature, self.humidity, self.lux)):
            raise ValueError("telemetry contains a non-finite sensor value")
        if not -10.0 <= self.temperature <= 60.0:
            raise ValueError(f"temperature outside supported range: {self.temperature}")
        if not 0.0 <= self.humidity <= 100.0:
            raise ValueError(f"humidity outside supported range: {self.humidity}")
        if not 0.0 <= self.lux <= 200_000.0:
            raise ValueError(f"lux outside supported range: {self.lux}")
        if not 0 <= self.people_count <= 100:
            raise ValueError(f"people count outside supported range: {self.people_count}")
        if not 0 <= self.led_pwm <= 255 or not 0 <= self.fan_pwm <= 255:
            raise ValueError("actuator PWM outside 0..255")
        if self.control_mode not in {"LOCAL", "HOST", "SAFE"}:
            raise ValueError(f"unknown control mode: {self.control_mode}")
        if self.reserved_request != "OFF":
            raise ValueError(f"unknown reserved protocol value: {self.reserved_request}")
        if self.co2_ppm is not None:
            if not math.isfinite(self.co2_ppm) or not 350.0 <= self.co2_ppm <= 10_000.0:
                raise ValueError(f"CO2 outside supported range: {self.co2_ppm}")
            if self.co2_age_ms is None or self.co2_age_ms < 0:
                raise ValueError("measured CO2 requires a non-negative sensor age")
        if max_sensor_age_ms is not None:
            if self.dht_age_ms > max_sensor_age_ms or self.lux_age_ms > max_sensor_age_ms:
                raise ValueError("sensor telemetry is stale")


@dataclass(frozen=True)
class LiveRoomConfig:
    room_id: str = "energenius_room"
    room_type: str = "prototype"
    area_m2: float = 20.0
    volume_m3: float = 60.0
    seating_capacity: int = 10
    assumed_co2_ppm: float = 600.0
    max_sensor_age_ms: int = 10_000


class SerialRoomBridge:
    """Validated USB serial boundary for the EnerGenius ESP32 firmware."""

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        config: LiveRoomConfig | None = None,
        serial_port=None,
        read_timeout_s: float = 6.0,
        reset_delay_s: float = 2.0,
        clock: Callable[[], datetime] | None = None,
    ):
        if serial_port is None:
            if not port:
                raise ValueError("a serial port is required")
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is required; install requirements.txt") from exc
            serial_port = serial.Serial(port, baudrate=baudrate, timeout=0.5)
            if reset_delay_s > 0:
                time.sleep(reset_delay_s)

        self.serial = serial_port
        self.config = config or LiveRoomConfig()
        self.read_timeout_s = read_timeout_s
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.command_sequence = 0
        self.previous_action = ControlAction()
        self.last_telemetry: Telemetry | None = None
        self.last_co2_was_fallback = True
        self.last_device_message = ""

    def read_observation(self) -> RoomObservation:
        deadline = time.monotonic() + self.read_timeout_s
        last_problem = "no telemetry received"
        while time.monotonic() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            self.last_device_message = line
            if not line.startswith("TEL,"):
                continue
            try:
                telemetry = Telemetry.parse(line)
                telemetry.validate(self.config.max_sensor_age_ms)
            except ValueError as exc:
                last_problem = str(exc)
                continue

            self.last_telemetry = telemetry
            co2_is_fresh = (
                telemetry.co2_ppm is not None
                and telemetry.co2_age_ms is not None
                and telemetry.co2_age_ms <= self.config.max_sensor_age_ms
            )
            self.last_co2_was_fallback = not co2_is_fresh
            co2_ppm = (
                telemetry.co2_ppm
                if co2_is_fresh
                else self.config.assumed_co2_ppm
            )
            now = self.clock()
            occupancy = telemetry.people_count
            if occupancy == 0 and telemetry.occupied:
                occupancy = 1
            return RoomObservation(
                room_id=self.config.room_id,
                room_type=self.config.room_type,
                area_m2=self.config.area_m2,
                volume_m3=self.config.volume_m3,
                seating_capacity=self.config.seating_capacity,
                year=now.year,
                month=now.month,
                day_id=now.timetuple().tm_yday - 1,
                minute_of_day=now.hour * 60 + now.minute,
                is_workday=now.weekday() < 5,
                occupancy=occupancy,
                temperature_c=telemetry.temperature,
                humidity_pct=telemetry.humidity,
                co2_ppm=co2_ppm,
                co2_is_measured=co2_is_fresh,
                illuminance_lux=telemetry.lux,
                previous_action=self.previous_action,
                source="energenius_esp32",
            )
        raise TimeoutError(f"timed out waiting for valid ESP32 telemetry: {last_problem}")

    def apply_action(self, action: ControlAction) -> None:
        self.command_sequence += 1
        self._write_line(
            f"CMD,{self.command_sequence},{action.light_pwm},{action.fan_pwm},OFF"
        )
        self.previous_action = action

    def enter_local_mode(self) -> None:
        self._write_line("MODE,LOCAL")

    def enter_safe_mode(self) -> None:
        self._write_line("MODE,SAFE")
        self.previous_action = ControlAction()

    def close(self, safe: bool = True) -> None:
        if safe:
            try:
                self.enter_safe_mode()
            except Exception:
                pass
        close = getattr(self.serial, "close", None)
        if close:
            close()

    def _write_line(self, line: str) -> None:
        self.serial.write((line + "\n").encode("ascii"))
        flush = getattr(self.serial, "flush", None)
        if flush:
            flush()


def _parse_flag(value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"expected 0 or 1, got {value!r}")
    return value == "1"
