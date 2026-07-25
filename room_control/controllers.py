from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from room_control.comfort import clamp, estimated_power_w, transition_reward
from room_control.models import (
    ActuatorCalibration,
    ControlAction,
    ControlDecision,
    RoomObservation,
    Transition,
    UserFeedback,
    UserPreferences,
)
from room_control.offline_rl import ConservativeOfflineVentilationQ


FAN_DELTAS = (-34, 0, 34)
LIGHT_DELTAS = (-32, 0, 32)
ADJUSTMENTS = tuple(
    (fan_delta, light_delta)
    for fan_delta in FAN_DELTAS
    for light_delta in LIGHT_DELTAS
)
NEUTRAL_ADJUSTMENT = (0, 0)
VENTILATION_LEVELS = ("OFF", "LOW", "MEDIUM", "HIGH", "MAXIMUM")
VENTILATION_LEVEL_PWM = {
    "OFF": 0,
    "LOW": 64,
    "MEDIUM": 128,
    "HIGH": 204,
    "MAXIMUM": 255,
}


def _pwm(value: float) -> int:
    return int(round(clamp(value, 0.0, 255.0)))


def _effective_fan_pwm(value: int, calibration: ActuatorCalibration) -> int:
    if value == 0:
        return 0
    return max(value, calibration.fan_min_effective_pwm)


def _adjustment_label(adjustment: tuple[int, int]) -> str:
    fan_delta, light_delta = adjustment
    return f"fan_delta={fan_delta},light_delta={light_delta}"


def _bucket(value: float, thresholds: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[index]
    return labels[-1]


class ComfortBaselineController:
    """Configurable startup policy used before the room has learned enough."""

    def __init__(
        self,
        preferences: UserPreferences | None = None,
        calibration: ActuatorCalibration | None = None,
    ):
        self.preferences = preferences or UserPreferences()
        self.calibration = calibration or ActuatorCalibration()

    def act(self, observation: RoomObservation) -> ControlAction:
        return self.decide(observation).action

    def decide(self, observation: RoomObservation) -> ControlDecision:
        preferences = self.preferences
        reasons: list[str] = []

        if observation.occupancy > 0:
            density_load = clamp(observation.occupancy_density / 0.20)
            capacity_load = clamp(observation.capacity_fraction)
            occupant_load = max(density_load, capacity_load)
            occupancy_demand = 0.18 + 0.62 * occupant_load
            reasons.append(
                f"occupied:{observation.occupancy} people, {observation.occupancy_density:.3f} people/m2"
            )
        else:
            occupancy_demand = 0.0
            reasons.append("unoccupied")

        co2_span = max(preferences.max_co2_ppm - 500.0, 100.0)
        co2_demand = clamp((observation.co2_ppm - 500.0) / co2_span)
        warm_error = observation.temperature_c - preferences.target_temperature_c
        warm_air_movement = clamp((warm_error - 0.3) / 3.0)

        if observation.occupancy > 0:
            fan_fraction = max(occupancy_demand, co2_demand, 0.25 + 0.55 * warm_air_movement)
        elif observation.co2_ppm > preferences.max_co2_ppm:
            fan_fraction = co2_demand
            reasons.append("unoccupied CO2 purge")
        else:
            fan_fraction = 0.0
        fan_pwm = _effective_fan_pwm(_pwm(255.0 * fan_fraction), self.calibration)
        if co2_demand > occupancy_demand:
            reasons.append(f"CO2 demand:{observation.co2_ppm:.0f} ppm")
        elif warm_air_movement > occupancy_demand:
            reasons.append(f"warm-room air movement:{warm_error:+.1f} C")

        if observation.occupancy > 0:
            light_error = preferences.target_illuminance_lux - observation.illuminance_lux
            if abs(light_error) <= 25.0:
                light_pwm = observation.previous_action.light_pwm
                reasons.append("lighting within deadband")
            else:
                light_pwm = _pwm(
                    observation.previous_action.light_pwm
                    + self.calibration.light_feedback_gain_pwm_per_lux * light_error
                )
                reasons.append(
                    f"lux feedback:{observation.illuminance_lux:.0f}/"
                    f"{preferences.target_illuminance_lux:.0f} lux"
                )
        else:
            light_pwm = 0

        action = ControlAction(fan_pwm=fan_pwm, light_pwm=light_pwm)
        safe_action, interventions = enforce_safety(
            observation,
            action,
            action,
            preferences,
            self.calibration,
        )
        return ControlDecision(
            action=safe_action,
            baseline_action=action,
            policy_adjustment=_adjustment_label(NEUTRAL_ADJUSTMENT),
            reasons=tuple(reasons),
            safety_interventions=interventions,
        )


class VentilationPrior:
    """Categorical behavior prior learned from logged VAV positions."""

    def __init__(self):
        self._level_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._coarse_level_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._transfer_level_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._transfer_coarse_level_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    @property
    def samples(self) -> int:
        return sum(sum(counts.values()) for counts in self._level_counts.values())

    @property
    def states(self) -> int:
        return len(self._level_counts)

    def observe(self, observation: RoomObservation) -> None:
        if observation.logged_ventilation_pct is None:
            return
        level = ventilation_level(observation.logged_ventilation_pct)
        state = ventilation_state(observation)
        coarse = state[:4]
        self._level_counts[state][level] += 1
        self._coarse_level_counts[coarse][level] += 1
        self._transfer_level_counts[state[1:]][level] += 1
        self._transfer_coarse_level_counts[state[1:4]][level] += 1

    def fit(self, observations: Iterable[RoomObservation]) -> None:
        for observation in observations:
            self.observe(observation)

    def predict(self, observation: RoomObservation) -> int | None:
        level = self.predict_level(observation)
        return VENTILATION_LEVEL_PWM[level] if level is not None else None

    def predict_level(self, observation: RoomObservation) -> str | None:
        state = ventilation_state(observation)
        if self._level_counts.get(state):
            return _most_common_level(self._level_counts[state])
        coarse = state[:4]
        if self._coarse_level_counts.get(coarse):
            return _most_common_level(self._coarse_level_counts[coarse])
        transferable = state[1:]
        if self._transfer_level_counts.get(transferable):
            return _most_common_level(self._transfer_level_counts[transferable])
        transferable_coarse = state[1:4]
        if self._transfer_coarse_level_counts.get(transferable_coarse):
            return _most_common_level(
                self._transfer_coarse_level_counts[transferable_coarse]
            )
        return None

    def to_dict(self) -> dict[str, object]:
        rows = [
            {"state": list(state), "level_counts": dict(sorted(counts.items()))}
            for state, counts in sorted(self._level_counts.items())
        ]
        coarse_rows = [
            {"state": list(state), "level_counts": dict(sorted(counts.items()))}
            for state, counts in sorted(self._coarse_level_counts.items())
        ]
        return {"rows": rows, "coarse_rows": coarse_rows}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VentilationPrior":
        prior = cls()
        for row in payload.get("rows", []):
            state = tuple(row["state"])
            level_counts = {
                level: int(count) for level, count in row["level_counts"].items()
            }
            prior._level_counts[state].update(level_counts)
            prior._transfer_level_counts[state[1:]].update(level_counts)
            prior._transfer_coarse_level_counts[state[1:4]].update(level_counts)
        for row in payload.get("coarse_rows", []):
            state = tuple(row["state"])
            prior._coarse_level_counts[state].update(
                {level: int(count) for level, count in row["level_counts"].items()}
            )
        return prior


def ventilation_level(vav_damper_pct: float) -> str:
    if vav_damper_pct <= 0.0:
        return "OFF"
    if vav_damper_pct <= 20.0:
        return "LOW"
    if vav_damper_pct <= 50.0:
        return "MEDIUM"
    if vav_damper_pct <= 80.0:
        return "HIGH"
    return "MAXIMUM"


def _most_common_level(counts: Counter[str]) -> str:
    return max(
        VENTILATION_LEVELS,
        key=lambda level: (counts[level], -VENTILATION_LEVELS.index(level)),
    )


def ventilation_state(observation: RoomObservation) -> tuple[str, ...]:
    return (
        observation.room_type,
        "workday" if observation.is_workday else "nonworkday",
        _bucket(
            observation.occupancy_density,
            (0.001, 0.025, 0.075, 0.15),
            ("empty", "sparse", "low", "busy", "dense"),
        ),
        _bucket(
            observation.co2_ppm,
            (550.0, 750.0, 900.0, 1100.0),
            ("fresh", "normal", "elevated", "high", "very_high"),
        ),
        _bucket(
            observation.hour,
            (7, 10, 13, 16, 19),
            ("night", "arrival", "morning", "midday", "afternoon", "evening"),
        ),
    )


class AdaptiveQLearningController:
    """Residual Q-learning around a safe, preference-aware PWM controller.

    The real-room dataset initializes a behavior prior and conservative absolute
    ventilation Q-values. The separate residual Q-table is updated only after a
    real action has actually affected a room.
    """

    FORMAT_VERSION = 7

    def __init__(
        self,
        preferences: UserPreferences | None = None,
        calibration: ActuatorCalibration | None = None,
        alpha: float = 0.12,
        gamma: float = 0.90,
        epsilon: float = 0.10,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        prior_weight: float = 0.55,
        offline_value_weight: float = 0.35,
        seed: int | None = None,
    ):
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("prior_weight must be between 0 and 1")
        if not 0.0 <= offline_value_weight <= 1.0:
            raise ValueError("offline_value_weight must be between 0 and 1")
        self.preferences = preferences or UserPreferences()
        self.calibration = calibration or ActuatorCalibration()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.prior_weight = prior_weight
        self.offline_value_weight = offline_value_weight
        self.rng = random.Random(seed)
        self.q: dict[tuple[tuple[str, ...], str], float] = defaultdict(float)
        self.ventilation_prior = VentilationPrior()
        self.offline_ventilation_q = ConservativeOfflineVentilationQ()
        self.feedback_count = 0
        self.online_updates = 0
        self.training_metadata: dict[str, object] = {}
        self.baseline = ComfortBaselineController(self.preferences, self.calibration)

    def pretrain_ventilation(self, observations: Iterable[RoomObservation]) -> None:
        self.ventilation_prior.fit(observations)

    def pretrain_offline_values(self, observations: Iterable[RoomObservation]) -> None:
        self.offline_ventilation_q.fit(
            observations,
            self.preferences,
            self.calibration,
        )

    def act(self, observation: RoomObservation, explore: bool = False) -> ControlAction:
        return self.decide(observation, explore=explore).action

    def decide(self, observation: RoomObservation, explore: bool = False) -> ControlDecision:
        baseline_decision = self.baseline.decide(observation)
        seed_action = self._seed_action(observation, baseline_decision.action)
        behavior_level = self.ventilation_prior.predict_level(observation)
        offline_level = None
        if behavior_level is not None and observation.occupancy > 0:
            offline_level = self.offline_ventilation_q.predict_level(
                observation,
                self.preferences,
                behavior_level,
            )
            if offline_level is not None:
                offline_pwm = VENTILATION_LEVEL_PWM[offline_level]
                seed_action = replace(
                    seed_action,
                    fan_pwm=_effective_fan_pwm(
                        _pwm(
                            (1.0 - self.offline_value_weight) * seed_action.fan_pwm
                            + self.offline_value_weight * offline_pwm
                        ),
                        self.calibration,
                    ),
                )
        state = policy_state(observation, self.preferences)
        candidates = self._candidate_adjustments(observation, seed_action)

        if explore and self.rng.random() < self.epsilon:
            adjustment = self.rng.choice(candidates)
        elif all(self.q.get((state, _adjustment_label(item)), 0.0) == 0.0 for item in candidates):
            adjustment = NEUTRAL_ADJUSTMENT
        else:
            adjustment = max(
                candidates,
                key=lambda item: self._action_value(state, seed_action, item),
            )

        candidate = apply_adjustment(seed_action, adjustment)
        action, interventions = enforce_safety(
            observation,
            candidate,
            baseline_decision.action,
            self.preferences,
            self.calibration,
        )
        reasons = list(baseline_decision.reasons)
        prior = self.ventilation_prior.predict(observation)
        if prior is not None:
            level = self.ventilation_prior.predict_level(observation)
            reasons.append(f"dataset ventilation prior:{level} ({prior} PWM target)")
        if offline_level is not None:
            reasons.append(
                f"comfort-energy offline Q:{offline_level} "
                f"({VENTILATION_LEVEL_PWM[offline_level]} PWM target)"
            )
        if adjustment != NEUTRAL_ADJUSTMENT:
            reasons.append(_adjustment_label(adjustment))
        return ControlDecision(
            action=action,
            baseline_action=baseline_decision.action,
            policy_adjustment=_adjustment_label(adjustment),
            reasons=tuple(reasons),
            safety_interventions=interventions,
        )

    def learn(self, transition: Transition) -> float:
        state = policy_state(transition.observation, self.preferences)
        action_label = transition.decision.policy_adjustment
        next_state = policy_state(transition.next_observation, self.preferences)
        next_baseline = self.baseline.act(transition.next_observation)
        next_seed = self._seed_action(transition.next_observation, next_baseline)
        next_candidates = self._candidate_adjustments(transition.next_observation, next_seed)
        future = max(
            self.q.get((next_state, _adjustment_label(adjustment)), 0.0)
            for adjustment in next_candidates
        )
        value = transition_reward(
            transition.next_observation,
            transition.decision.action,
            self.preferences,
            self.calibration,
            feedback=transition.feedback,
            elapsed_minutes=transition.elapsed_minutes,
            previous_action=transition.observation.previous_action,
        )
        current = self.q.get((state, action_label), 0.0)
        self.q[(state, action_label)] = current + self.alpha * (
            value + self.gamma * future - current
        )
        self.online_updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if transition.feedback is not None:
            self.apply_feedback(transition.feedback)
        return value

    def apply_feedback(self, feedback: UserFeedback) -> UserPreferences:
        preferences = self.preferences
        temperature = preferences.target_temperature_c
        humidity_min = preferences.humidity_min_pct
        humidity_max = preferences.humidity_max_pct
        max_co2 = preferences.max_co2_ppm
        target_lux = preferences.target_illuminance_lux

        if feedback.thermal == "cold":
            temperature += 0.25
        elif feedback.thermal == "hot":
            temperature -= 0.25
        if feedback.humidity == "dry":
            humidity_min += 1.0
            humidity_max += 1.0
        elif feedback.humidity == "humid":
            humidity_min -= 1.0
            humidity_max -= 1.0
        if feedback.air_quality == "stale":
            max_co2 -= 25.0
        if feedback.lighting == "dim":
            target_lux += 25.0
        elif feedback.lighting == "bright":
            target_lux -= 25.0

        self.preferences = UserPreferences(
            target_temperature_c=clamp(temperature, 18.0, 30.0),
            humidity_min_pct=clamp(humidity_min, 25.0, 70.0),
            humidity_max_pct=clamp(humidity_max, 30.0, 75.0),
            max_co2_ppm=clamp(max_co2, 650.0, 1000.0),
            target_illuminance_lux=clamp(target_lux, 150.0, 1000.0),
            energy_sensitivity=preferences.energy_sensitivity,
        )
        if self.preferences.humidity_max_pct <= self.preferences.humidity_min_pct:
            self.preferences = replace(
                self.preferences,
                humidity_max_pct=self.preferences.humidity_min_pct + 5.0,
            )
        self.feedback_count += 1
        self.baseline = ComfortBaselineController(self.preferences, self.calibration)
        return self.preferences

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "controller": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "prior_weight": self.prior_weight,
                "offline_value_weight": self.offline_value_weight,
                "feedback_count": self.feedback_count,
                "online_updates": self.online_updates,
            },
            "preferences": asdict(self.preferences),
            "calibration": asdict(self.calibration),
            "ventilation_prior": self.ventilation_prior.to_dict(),
            "offline_ventilation_q": self.offline_ventilation_q.to_dict(),
            "q_values": [
                {"state": list(state), "action": action, "value": value}
                for (state, action), value in sorted(self.q.items())
                if value != 0.0
            ],
            "training_metadata": self.training_metadata,
            "training_scope": {
                "offline": (
                    "logged VAV behavior prior plus conservative comfort-energy Q values "
                    "trained only from supported logged transitions; lighting starts from lux feedback"
                ),
                "online": "fan and lighting Q-learning from real post-action transitions and optional user feedback",
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "AdaptiveQLearningController":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported policy format")
        config = payload["controller"]
        controller = cls(
            preferences=UserPreferences(**payload["preferences"]),
            calibration=ActuatorCalibration(**payload["calibration"]),
            alpha=float(config["alpha"]),
            gamma=float(config["gamma"]),
            epsilon=float(config["epsilon"]),
            epsilon_min=float(config["epsilon_min"]),
            epsilon_decay=float(config["epsilon_decay"]),
            prior_weight=float(config["prior_weight"]),
            offline_value_weight=float(config["offline_value_weight"]),
            seed=seed,
        )
        controller.feedback_count = int(config.get("feedback_count", 0))
        controller.online_updates = int(config.get("online_updates", 0))
        controller.training_metadata = dict(payload.get("training_metadata", {}))
        controller.ventilation_prior = VentilationPrior.from_dict(payload["ventilation_prior"])
        controller.offline_ventilation_q = ConservativeOfflineVentilationQ.from_dict(
            payload["offline_ventilation_q"]
        )
        controller.q = defaultdict(
            float,
            {
                (tuple(row["state"]), row["action"]): float(row["value"])
                for row in payload.get("q_values", [])
            },
        )
        return controller

    def _seed_action(
        self,
        observation: RoomObservation,
        baseline: ControlAction,
    ) -> ControlAction:
        prior_pwm = self.ventilation_prior.predict(observation)
        if prior_pwm is None or observation.occupancy == 0:
            return baseline
        blended = _pwm(
            (1.0 - self.prior_weight) * baseline.fan_pwm
            + self.prior_weight * prior_pwm
        )
        return replace(
            baseline,
            fan_pwm=_effective_fan_pwm(blended, self.calibration),
        )

    def _candidate_adjustments(
        self,
        observation: RoomObservation,
        seed_action: ControlAction,
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for adjustment in ADJUSTMENTS:
            action = apply_adjustment(seed_action, adjustment)
            _fan_delta, light_delta = adjustment
            if observation.occupancy == 0 and light_delta != 0:
                continue
            if (
                observation.occupancy == 0
                and observation.co2_ppm <= self.preferences.max_co2_ppm
                and action.fan_pwm > 0
            ):
                continue
            candidates.append(adjustment)
        return candidates or [NEUTRAL_ADJUSTMENT]

    def _action_value(
        self,
        state: tuple[str, ...],
        seed_action: ControlAction,
        adjustment: tuple[int, int],
    ) -> float:
        action = apply_adjustment(seed_action, adjustment)
        power_penalty = 0.00005 * estimated_power_w(action, self.calibration)
        return self.q.get((state, _adjustment_label(adjustment)), 0.0) - power_penalty


def apply_adjustment(
    baseline: ControlAction,
    adjustment: tuple[int, int],
) -> ControlAction:
    fan_delta, light_delta = adjustment
    return ControlAction(
        fan_pwm=_pwm(baseline.fan_pwm + fan_delta),
        light_pwm=_pwm(baseline.light_pwm + light_delta),
    )


def enforce_safety(
    observation: RoomObservation,
    action: ControlAction,
    baseline: ControlAction,
    preferences: UserPreferences,
    calibration: ActuatorCalibration,
) -> tuple[ControlAction, tuple[str, ...]]:
    fan_pwm = action.fan_pwm
    light_pwm = action.light_pwm
    interventions: list[str] = []

    if (
        observation.occupancy == 0
        and observation.co2_ppm <= preferences.max_co2_ppm
        and fan_pwm > 0
    ):
        fan_pwm = 0
        interventions.append("empty-room fan shutoff")

    if observation.occupancy == 0 and light_pwm > 0:
        light_pwm = 0
        interventions.append("empty-room light shutoff")

    if observation.co2_ppm >= 1200.0:
        if fan_pwm < 255:
            fan_pwm = 255
            interventions.append("high-CO2 maximum ventilation")
    elif observation.co2_ppm > preferences.max_co2_ppm:
        required = max(baseline.fan_pwm, 180)
        if fan_pwm < required:
            fan_pwm = required
            interventions.append("CO2 ventilation floor")

    if observation.occupancy > 0 and observation.temperature_c >= 32.0:
        if fan_pwm < 220:
            fan_pwm = 220
            interventions.append("high-temperature fan floor")

    fan_pwm = _effective_fan_pwm(fan_pwm, calibration)
    return ControlAction(fan_pwm=fan_pwm, light_pwm=light_pwm), tuple(interventions)


def policy_state(
    observation: RoomObservation,
    preferences: UserPreferences,
) -> tuple[str, ...]:
    temperature_error = observation.temperature_c - preferences.target_temperature_c
    if observation.humidity_pct < preferences.humidity_min_pct:
        humidity_state = "dry"
    elif observation.humidity_pct > preferences.humidity_max_pct:
        humidity_state = "humid"
    else:
        humidity_state = "comfortable"
    return (
        observation.room_type,
        "workday" if observation.is_workday else "nonworkday",
        _bucket(
            observation.hour,
            (7, 10, 13, 16, 19),
            ("night", "arrival", "morning", "midday", "afternoon", "evening"),
        ),
        _bucket(
            observation.occupancy_density,
            (0.001, 0.025, 0.075, 0.15),
            ("empty", "sparse", "low", "busy", "dense"),
        ),
        _bucket(
            temperature_error,
            (-2.0, -0.75, 0.75, 2.0),
            ("very_cold", "cold", "comfortable", "warm", "hot"),
        ),
        humidity_state,
        _bucket(
            observation.co2_ppm - preferences.max_co2_ppm,
            (-250.0, 0.0, 200.0),
            ("fresh", "acceptable", "high", "critical"),
        ),
        _bucket(
            observation.illuminance_lux - preferences.target_illuminance_lux,
            (-200.0, -50.0, 50.0, 250.0),
            ("very_dim", "dim", "comfortable", "bright", "very_bright"),
        ),
        str(observation.previous_action.fan_pwm // 64),
        str(observation.previous_action.light_pwm // 64),
    )
