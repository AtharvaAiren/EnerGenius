from simulator.models import ControlAction, FanSpeed, HVACMode, LightLevel, Observation, PreferenceProfile
from simulator.population import SyntheticOccupant


FAN_POWER_W = {
    FanSpeed.OFF: 0.0,
    FanSpeed.LOW: 35.0,
    FanSpeed.HIGH: 65.0,
}

LIGHT_POWER_W = {
    LightLevel.OFF: 0.0,
    LightLevel.LOW: 25.0,
    LightLevel.HIGH: 60.0,
}

HVAC_POWER_W = {
    HVACMode.OFF: 0.0,
    HVACMode.COOL: 950.0,
    HVACMode.HEAT: 1100.0,
}

LIGHT_OUTPUT_LUX = {
    LightLevel.OFF: 0.0,
    LightLevel.LOW: 220.0,
    LightLevel.HIGH: 520.0,
}


def energy_use_wh(action: ControlAction, minutes_per_step: int = 5) -> float:
    watts = FAN_POWER_W[action.fan] + LIGHT_POWER_W[action.light] + HVAC_POWER_W[action.hvac]
    return watts * (minutes_per_step / 60.0)


def comfort_score(observation: Observation, action: ControlAction, preferences: PreferenceProfile | None = None) -> float:
    if observation.occupancy == 0:
        return 0.0

    preferences = preferences or PreferenceProfile(
        target_temperature=observation.preferred_temperature,
        target_humidity=observation.preferred_humidity,
        target_light_when_occupied=observation.preferred_light,
        energy_sensitivity=observation.average_energy_sensitivity,
        discomfort_tolerance=observation.average_discomfort_tolerance,
        activity_level=observation.average_activity_level,
    )
    delivered_light = observation.ambient_light + LIGHT_OUTPUT_LUX[action.light]
    thermal_error = abs(observation.temperature - preferences.target_temperature)
    humidity_error = abs(observation.humidity - preferences.target_humidity)
    light_error = max(preferences.target_light_when_occupied - delivered_light, 0.0)
    crowding_penalty = max(observation.occupancy - 8, 0) * 0.25 * preferences.crowding_sensitivity
    air_quality_penalty = max(observation.co2 - 900.0, 0.0) / 180.0

    tolerance = max(preferences.discomfort_tolerance, 0.45)
    thermal = max(0.0, 10.0 - thermal_error * (1.35 / tolerance))
    humidity = max(0.0, 2.0 - humidity_error * (0.08 / tolerance))
    lighting = max(0.0, 2.0 - light_error / (260.0 * tolerance))
    airflow = 0.4 if action.fan == FanSpeed.LOW else 0.8 if action.fan == FanSpeed.HIGH else 0.0

    return max(0.0, thermal + humidity + lighting + airflow - crowding_penalty - air_quality_penalty)


def occupant_comfort_score(observation: Observation, action: ControlAction, occupant: SyntheticOccupant) -> float:
    return comfort_score(observation, action, occupant.preference_profile())


def reward(
    observation: Observation,
    action: ControlAction,
    preferences: PreferenceProfile,
    energy_weight: float = 0.012,
) -> tuple[float, float, float]:
    comfort = comfort_score(observation, action, preferences)
    energy = energy_use_wh(action)
    sensitivity = preferences.energy_sensitivity if observation.occupancy else 1.0
    return comfort - energy_weight * sensitivity * energy, comfort, energy
