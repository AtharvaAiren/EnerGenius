from __future__ import annotations

from dataclasses import dataclass

from room_control.models import (
    ActuatorCalibration,
    ControlAction,
    RoomObservation,
    UserFeedback,
    UserPreferences,
)


@dataclass(frozen=True)
class ComfortAssessment:
    overall: float
    thermal: float
    humidity: float
    air_quality: float
    visual: float
    violations: tuple[str, ...]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def assess_comfort(
    observation: RoomObservation,
    preferences: UserPreferences,
) -> ComfortAssessment:
    if observation.occupancy == 0:
        return ComfortAssessment(100.0, 100.0, 100.0, 100.0, 100.0, ())

    thermal_error = abs(observation.temperature_c - preferences.target_temperature_c)
    thermal = 100.0 * clamp(1.0 - thermal_error / 4.0)

    if observation.humidity_pct < preferences.humidity_min_pct:
        humidity_error = preferences.humidity_min_pct - observation.humidity_pct
        humidity_violation = "dry"
    elif observation.humidity_pct > preferences.humidity_max_pct:
        humidity_error = observation.humidity_pct - preferences.humidity_max_pct
        humidity_violation = "humid"
    else:
        humidity_error = 0.0
        humidity_violation = ""
    humidity = 100.0 * clamp(1.0 - humidity_error / 20.0)

    co2_excess = max(0.0, observation.co2_ppm - preferences.max_co2_ppm)
    air_quality = 100.0 * clamp(1.0 - co2_excess / 800.0)

    illuminance_error = abs(
        observation.illuminance_lux - preferences.target_illuminance_lux
    )
    visual = 100.0 * clamp(
        1.0 - illuminance_error / max(preferences.target_illuminance_lux, 100.0)
    )

    violations: list[str] = []
    if observation.temperature_c < preferences.target_temperature_c - 1.0:
        violations.append("cold")
    elif observation.temperature_c > preferences.target_temperature_c + 1.0:
        violations.append("hot")
    if humidity_violation:
        violations.append(humidity_violation)
    if co2_excess > 0.0:
        violations.append("high_co2")
    if observation.illuminance_lux < preferences.target_illuminance_lux - 100.0:
        violations.append("dim")
    elif observation.illuminance_lux > preferences.target_illuminance_lux + 250.0:
        violations.append("bright")
    overall = 0.30 * thermal + 0.10 * humidity + 0.35 * air_quality + 0.25 * visual
    return ComfortAssessment(
        overall=overall,
        thermal=thermal,
        humidity=humidity,
        air_quality=air_quality,
        visual=visual,
        violations=tuple(violations),
    )


def estimated_power_w(action: ControlAction, calibration: ActuatorCalibration) -> float:
    fan_fraction = action.fan_pwm / 255.0
    light_fraction = action.light_pwm / 255.0
    return (
        calibration.fan_max_power_w * fan_fraction**3
        + calibration.light_max_power_w * light_fraction
    )


def switching_magnitude(previous: ControlAction, current: ControlAction) -> float:
    return (
        abs(previous.fan_pwm - current.fan_pwm)
        + abs(previous.light_pwm - current.light_pwm)
    ) / 510.0


def transition_reward(
    next_observation: RoomObservation,
    action: ControlAction,
    preferences: UserPreferences,
    calibration: ActuatorCalibration,
    feedback: UserFeedback | None = None,
    elapsed_minutes: float = 1.0,
    previous_action: ControlAction | None = None,
) -> float:
    assessment = assess_comfort(next_observation, preferences)
    occupied = next_observation.occupancy > 0
    discomfort = (100.0 - assessment.overall) / 100.0 if occupied else 0.0
    energy_wh = estimated_power_w(action, calibration) * elapsed_minutes / 60.0
    switching = switching_magnitude(previous_action or next_observation.previous_action, action)

    value = -8.0 * discomfort
    value -= 0.012 * preferences.energy_sensitivity * energy_wh
    value -= 0.08 * switching
    if next_observation.co2_ppm > 1200.0:
        value -= 4.0 * (next_observation.co2_ppm - 1200.0) / 800.0
    if feedback is not None:
        value += 2.0 * feedback.overall
        disliked = sum(
            item in {"cold", "hot", "dry", "humid", "stale", "dim", "bright"}
            for item in (
                feedback.thermal,
                feedback.humidity,
                feedback.air_quality,
                feedback.lighting,
            )
        )
        liked = sum(
            item in {"comfortable", "fresh"}
            for item in (
                feedback.thermal,
                feedback.humidity,
                feedback.air_quality,
            )
        )
        value += 0.5 * liked - 1.25 * disliked
    return value
