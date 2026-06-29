import math
import random

from simulator.metrics import occupant_comfort_score, reward
from simulator.models import ControlAction, FanSpeed, HVACMode, LightLevel, Observation, PopulationSnapshot, StepResult
from simulator.population import SyntheticPopulation


class RoomEnvironment:
    def __init__(
        self,
        seed: int | None = None,
        population_size: int = 24,
        minutes_per_step: int = 5,
        population: SyntheticPopulation | None = None,
    ):
        self.seed = seed or 0
        self.rng = random.Random(seed)
        self.population = population or SyntheticPopulation.generate(size=population_size, seed=self.seed + 11)
        self.minutes_per_step = minutes_per_step
        self.step_count = 0
        self.temperature = 27.5
        self.humidity = 54.0
        self.co2 = 430.0
        self.previous_action = ControlAction()

    def reset(self) -> Observation:
        self.step_count = 0
        self.temperature = self.rng.uniform(24.5, 28.5)
        self.humidity = self.rng.uniform(45.0, 62.0)
        self.co2 = self.rng.uniform(420.0, 500.0)
        self.previous_action = ControlAction()
        return self.observe()

    def observe(self) -> Observation:
        day = self.step_count // self.steps_per_day
        minute = (self.step_count * self.minutes_per_step) % (24 * 60)
        hour = minute // 60
        snapshot = self.population.snapshot(day, minute)
        season = self._season(day)

        return Observation(
            step=self.step_count,
            day=day,
            hour=hour,
            minute=minute,
            is_weekend=snapshot.is_weekend,
            season=season,
            occupancy=snapshot.occupancy,
            temperature=self.temperature,
            humidity=self.humidity,
            ambient_light=self._ambient_light(day, minute, season),
            outdoor_temperature=self._outdoor_temperature(day, minute, season),
            co2=self.co2,
            preferred_temperature=snapshot.preferences.target_temperature,
            preferred_humidity=snapshot.preferences.target_humidity,
            preferred_light=snapshot.preferences.target_light_when_occupied,
            average_energy_sensitivity=snapshot.preferences.energy_sensitivity,
            average_discomfort_tolerance=snapshot.preferences.discomfort_tolerance,
            average_activity_level=snapshot.preferences.activity_level,
            previous_action=self.previous_action,
            archetype_counts=snapshot.active_archetypes,
        )

    def step(self, action: ControlAction) -> StepResult:
        observation = self.observe()
        snapshot = self.population.snapshot(observation.day, observation.minute)
        current_reward, comfort, energy = reward(observation, action, snapshot.preferences)
        active_occupants = self.population.active_occupants(snapshot)
        occupant_comfort = {
            occupant.occupant_id: occupant_comfort_score(observation, action, occupant)
            for occupant in active_occupants
        }

        self._advance_physics(observation, action, snapshot)
        self.previous_action = action
        self.step_count += 1
        return StepResult(
            observation=observation,
            action=action,
            comfort=comfort,
            energy_wh=energy,
            reward=current_reward,
            occupant_comfort=occupant_comfort,
        )

    @property
    def steps_per_day(self) -> int:
        return int(24 * 60 / self.minutes_per_step)

    def _season(self, day: int) -> str:
        season_index = (day // 30) % 4
        return ("spring", "summer", "autumn", "winter")[season_index]

    def _ambient_light(self, day: int, minute: int, season: str) -> float:
        sunrise, sunset = {
            "spring": (6 * 60 + 10, 18 * 60 + 35),
            "summer": (5 * 60 + 35, 19 * 60 + 20),
            "autumn": (6 * 60 + 20, 18 * 60),
            "winter": (7 * 60, 17 * 60 + 20),
        }[season]
        if minute < sunrise or minute > sunset:
            return 0.0

        cloud_factor = self._day_weather(day)["cloud_factor"]
        daylight_phase = (minute - sunrise) / (sunset - sunrise)
        solar_curve = math.sin(math.pi * daylight_phase)
        return max(0.0, 720.0 * solar_curve * cloud_factor)

    def _outdoor_temperature(self, day: int, minute: int, season: str) -> float:
        base, swing = {
            "spring": (24.0, 4.5),
            "summer": (30.5, 5.5),
            "autumn": (25.0, 4.0),
            "winter": (19.0, 3.2),
        }[season]
        weather = self._day_weather(day)
        hour = minute / 60.0
        daily_cycle = math.sin(2 * math.pi * (hour - 7) / 24)
        return base + swing * daily_cycle + weather["temperature_offset"]

    def _day_weather(self, day: int) -> dict[str, float]:
        rng = random.Random(self.seed + day * 7919 + 202)
        return {
            "cloud_factor": rng.uniform(0.55, 1.0),
            "temperature_offset": rng.gauss(0.0, 1.3),
        }

    def _advance_physics(
        self,
        observation: Observation,
        action: ControlAction,
        snapshot: PopulationSnapshot,
    ) -> None:
        activity_heat = 0.006 * observation.occupancy * snapshot.preferences.activity_level
        outside_pull = 0.030 * (observation.outdoor_temperature - self.temperature)
        solar_gain = 0.00012 * observation.ambient_light
        equipment_heat = 0.006 if action.light != LightLevel.OFF else 0.0

        fan_effect = {
            FanSpeed.OFF: 0.0,
            FanSpeed.LOW: -0.04,
            FanSpeed.HIGH: -0.08,
        }[action.fan]

        hvac_effect = {
            HVACMode.OFF: 0.0,
            HVACMode.COOL: -0.40,
            HVACMode.HEAT: 0.34,
        }[action.hvac]

        self.temperature += outside_pull + solar_gain + activity_heat + equipment_heat + fan_effect + hvac_effect
        self.temperature += self.rng.gauss(0.0, 0.025)

        occupancy_moisture = 0.012 * observation.occupancy * snapshot.preferences.activity_level
        outdoor_humidity_pull = 0.012 * (self._outdoor_humidity(observation.season) - self.humidity)
        dehumidification = -0.10 if action.hvac == HVACMode.COOL else 0.0
        self.humidity += outdoor_humidity_pull + occupancy_moisture + dehumidification
        self.humidity += self.rng.gauss(0.0, 0.10)
        self.humidity = max(25.0, min(85.0, self.humidity))

        co2_generation = 2.6 * observation.occupancy * snapshot.preferences.activity_level
        ventilation = 38.0 if action.fan == FanSpeed.HIGH else 22.0 if action.fan == FanSpeed.LOW else 8.0
        passive_decay = 0.026 * (self.co2 - 420.0)
        self.co2 += co2_generation - ventilation - passive_decay
        self.co2 += self.rng.gauss(0.0, 2.5)
        self.co2 = max(420.0, min(1800.0, self.co2))

    def _outdoor_humidity(self, season: str) -> float:
        return {
            "spring": 52.0,
            "summer": 62.0,
            "autumn": 55.0,
            "winter": 42.0,
        }[season]


Environment = RoomEnvironment
