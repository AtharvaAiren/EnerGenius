import ast
import json
import random
from collections import defaultdict
from pathlib import Path

from simulator.models import ControlAction, FanSpeed, HVACMode, LightLevel, Observation, default_action_space


class Controller:
    def act(self, observation: Observation) -> ControlAction:
        raise NotImplementedError


class BaselineController(Controller):
    def act(self, observation: Observation) -> ControlAction:
        if observation.occupancy == 0:
            return ControlAction()

        fan = FanSpeed.OFF
        hvac = HVACMode.OFF
        warm_threshold = observation.preferred_temperature + 2.4
        mild_threshold = observation.preferred_temperature + 1.0
        cold_threshold = observation.preferred_temperature - 3.2
        if observation.temperature > warm_threshold:
            hvac = HVACMode.COOL
            fan = FanSpeed.HIGH
        elif observation.temperature > mild_threshold:
            fan = FanSpeed.LOW
        elif observation.temperature < cold_threshold:
            hvac = HVACMode.HEAT

        delivered_light = observation.ambient_light
        if delivered_light >= observation.preferred_light:
            light = LightLevel.OFF
        elif delivered_light >= observation.preferred_light * 0.55:
            light = LightLevel.LOW
        else:
            light = LightLevel.HIGH

        return ControlAction(fan=fan, light=light, hvac=hvac)


class QLearningController(Controller):
    def __init__(
        self,
        action_space: list[ControlAction] | None = None,
        alpha: float = 0.12,
        gamma: float = 0.86,
        epsilon: float = 0.35,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ):
        self.action_space = action_space or default_action_space()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: dict[tuple[tuple[str, ...], str], float] = defaultdict(float)
        self.fallback = BaselineController()

    def act(self, observation: Observation, greedy: bool = False) -> ControlAction:
        state = discretize(observation)
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_space)

        if all(self.q[(state, action.label())] == 0.0 for action in self.action_space):
            return self.fallback.act(observation)

        candidate = max(self.action_space, key=lambda action: self.q[(state, action.label())])
        return comfort_safety_layer(observation, candidate, self.fallback.act(observation))

    def update(
        self,
        observation: Observation,
        action: ControlAction,
        reward: float,
        next_observation: Observation,
    ) -> None:
        state = discretize(observation)
        next_state = discretize(next_observation)
        action_label = action.label()
        current = self.q[(state, action_label)]
        future = max(self.q[(next_state, candidate.label())] for candidate in self.action_space)
        self.q[(state, action_label)] = current + self.alpha * (reward + self.gamma * future - current)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        serializable = {str(key): value for key, value in self.q.items()}
        Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.q = defaultdict(float, {ast.literal_eval(key): value for key, value in data.items()})


def discretize(observation: Observation) -> tuple[str, ...]:
    return (
        "weekend" if observation.is_weekend else "weekday",
        observation.season,
        _bucket_number(observation.hour, (7, 10, 14, 18, 22), ("night", "arrival", "morning", "midday", "afternoon", "evening")),
        _bucket_number(observation.occupancy, (1, 5, 10, 16), ("empty", "low", "medium", "high", "crowded")),
        _bucket_number(observation.temperature, (22, 25, 27, 29), ("cold", "cool", "comfortable", "warm", "hot")),
        _bucket_number(observation.humidity, (35, 45, 60, 70), ("dry", "low", "normal", "humid", "very_humid")),
        _bucket_number(observation.ambient_light, (120, 300, 550), ("dark", "dim", "lit", "bright")),
        _bucket_number(observation.co2, (650, 900, 1200), ("fresh", "okay", "stale", "poor")),
        _bucket_number(observation.preferred_temperature, (24, 25, 26), ("cool_pref", "neutral_pref", "warm_pref", "hot_pref")),
        _bucket_number(observation.preferred_light, (430, 520), ("low_light_pref", "normal_light_pref", "bright_light_pref")),
        _bucket_number(observation.average_energy_sensitivity, (0.8, 1.15), ("comfort_first", "balanced", "energy_sensitive")),
        observation.previous_action.fan.value,
        observation.previous_action.light.value,
        observation.previous_action.hvac.value,
    )


def _bucket_number(value: float, thresholds: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[index]
    return labels[-1]


def comfort_safety_layer(
    observation: Observation,
    candidate: ControlAction,
    fallback: ControlAction,
) -> ControlAction:
    if observation.occupancy == 0:
        return ControlAction()

    adjusted = candidate
    if observation.temperature < observation.preferred_temperature - 1.2 and candidate.hvac == HVACMode.COOL:
        adjusted = ControlAction(fan=candidate.fan, light=candidate.light, hvac=HVACMode.OFF)
    elif observation.temperature > observation.preferred_temperature + 1.2 and candidate.hvac == HVACMode.HEAT:
        adjusted = ControlAction(fan=candidate.fan, light=candidate.light, hvac=HVACMode.OFF)

    if observation.ambient_light > observation.preferred_light and adjusted.light != LightLevel.OFF:
        adjusted = ControlAction(fan=adjusted.fan, light=LightLevel.OFF, hvac=adjusted.hvac)

    if observation.temperature >= observation.preferred_temperature + 3.5 and adjusted.hvac != HVACMode.COOL:
        return fallback
    if observation.temperature <= observation.preferred_temperature - 4.2 and adjusted.hvac != HVACMode.HEAT:
        return fallback
    if observation.ambient_light < 150.0 and adjusted.light == LightLevel.OFF:
        return fallback
    if observation.co2 > 1100.0 and adjusted.fan == FanSpeed.OFF:
        return fallback
    return adjusted


def controller_from_q_table(path: str | Path, greedy: bool = True) -> Controller:
    learner = QLearningController(epsilon=0.0 if greedy else 0.05)
    learner.load(path)
    return learner
