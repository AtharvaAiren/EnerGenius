from __future__ import annotations

import math
from dataclasses import dataclass, replace


def _validate_pwm(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{name} must be an integer from 0 to 255")


@dataclass(frozen=True)
class ControlAction:
    fan_pwm: int = 0
    light_pwm: int = 0

    def __post_init__(self) -> None:
        _validate_pwm("fan_pwm", self.fan_pwm)
        _validate_pwm("light_pwm", self.light_pwm)

    def label(self) -> str:
        return f"fan={self.fan_pwm},light={self.light_pwm}"


@dataclass(frozen=True)
class UserPreferences:
    target_temperature_c: float = 25.0
    humidity_min_pct: float = 40.0
    humidity_max_pct: float = 60.0
    max_co2_ppm: float = 900.0
    target_illuminance_lux: float = 500.0
    energy_sensitivity: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.target_temperature_c,
            self.humidity_min_pct,
            self.humidity_max_pct,
            self.max_co2_ppm,
            self.target_illuminance_lux,
            self.energy_sensitivity,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("preferences must contain finite values")
        if not 16.0 <= self.target_temperature_c <= 32.0:
            raise ValueError("target_temperature_c must be between 16 and 32")
        if not 0.0 <= self.humidity_min_pct < self.humidity_max_pct <= 100.0:
            raise ValueError("humidity bounds must be ordered within 0..100")
        if not 600.0 <= self.max_co2_ppm <= 1200.0:
            raise ValueError("max_co2_ppm must be between 600 and 1200")
        if not 100.0 <= self.target_illuminance_lux <= 1500.0:
            raise ValueError("target_illuminance_lux must be between 100 and 1500")
        if not 0.1 <= self.energy_sensitivity <= 3.0:
            raise ValueError("energy_sensitivity must be between 0.1 and 3.0")


@dataclass(frozen=True)
class ActuatorCalibration:
    fan_min_effective_pwm: int = 0
    fan_max_power_w: float = 40.0
    light_max_power_w: float = 24.0
    light_feedback_gain_pwm_per_lux: float = 0.25

    def __post_init__(self) -> None:
        _validate_pwm("fan_min_effective_pwm", self.fan_min_effective_pwm)
        powers = (self.fan_max_power_w, self.light_max_power_w)
        if not all(math.isfinite(value) and value >= 0.0 for value in powers):
            raise ValueError("calibration values must be finite and non-negative")
        if not math.isfinite(self.light_feedback_gain_pwm_per_lux):
            raise ValueError("light feedback gain must be finite")
        if not 0.0 < self.light_feedback_gain_pwm_per_lux <= 2.0:
            raise ValueError("light feedback gain must be in (0, 2]")


@dataclass(frozen=True)
class RoomObservation:
    room_id: str
    room_type: str
    area_m2: float
    volume_m3: float
    seating_capacity: int
    year: int
    month: int
    day_id: int
    minute_of_day: int
    is_workday: bool
    occupancy: int
    temperature_c: float
    humidity_pct: float
    co2_ppm: float
    illuminance_lux: float
    co2_is_measured: bool = True
    previous_action: ControlAction = ControlAction()
    source: str = "real_sensor"
    logged_ventilation_pct: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.area_m2,
            self.volume_m3,
            self.temperature_c,
            self.humidity_pct,
            self.co2_ppm,
            self.illuminance_lux,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("observation contains a non-finite value")
        if not self.room_id:
            raise ValueError("room_id is required")
        if self.area_m2 <= 0.0 or self.volume_m3 <= 0.0:
            raise ValueError("room area and volume must be positive")
        if self.seating_capacity <= 0:
            raise ValueError("seating_capacity must be positive")
        if not 1 <= self.month <= 12:
            raise ValueError("month must be in 1..12")
        if not 0 <= self.minute_of_day < 1440:
            raise ValueError("minute_of_day must be in 0..1439")
        if self.occupancy < 0:
            raise ValueError("occupancy must be non-negative")
        if not 0.0 <= self.humidity_pct <= 100.0:
            raise ValueError("humidity_pct must be in 0..100")
        if self.co2_ppm < 0.0 or self.illuminance_lux < 0.0:
            raise ValueError("CO2 and illuminance must be non-negative")
        if not isinstance(self.co2_is_measured, bool):
            raise ValueError("co2_is_measured must be a boolean")
        if self.logged_ventilation_pct is not None:
            if not math.isfinite(self.logged_ventilation_pct):
                raise ValueError("logged ventilation must be finite")
            if not 0.0 <= self.logged_ventilation_pct <= 100.0:
                raise ValueError("logged ventilation must be in 0..100")

    @property
    def occupancy_density(self) -> float:
        return self.occupancy / self.area_m2

    @property
    def capacity_fraction(self) -> float:
        return min(self.occupancy / self.seating_capacity, 1.5)

    @property
    def hour(self) -> int:
        return self.minute_of_day // 60

    def with_previous_action(self, action: ControlAction) -> "RoomObservation":
        return replace(self, previous_action=action)


@dataclass(frozen=True)
class UserFeedback:
    thermal: str | None = None
    humidity: str | None = None
    air_quality: str | None = None
    lighting: str | None = None
    overall: int = 0

    def __post_init__(self) -> None:
        allowed = {
            "thermal": {None, "cold", "comfortable", "hot"},
            "humidity": {None, "dry", "comfortable", "humid"},
            "air_quality": {None, "fresh", "comfortable", "stale"},
            "lighting": {None, "dim", "comfortable", "bright"},
        }
        for field, choices in allowed.items():
            if getattr(self, field) not in choices:
                raise ValueError(f"invalid {field} feedback: {getattr(self, field)!r}")
        if self.overall not in {-1, 0, 1}:
            raise ValueError("overall feedback must be -1, 0, or 1")


@dataclass(frozen=True)
class ControlDecision:
    action: ControlAction
    baseline_action: ControlAction
    policy_adjustment: str
    reasons: tuple[str, ...]
    safety_interventions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transition:
    observation: RoomObservation
    decision: ControlDecision
    next_observation: RoomObservation
    feedback: UserFeedback | None = None
    elapsed_minutes: float = 1.0
