from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Iterable

from room_control.comfort import assess_comfort
from room_control.models import ActuatorCalibration, RoomObservation, UserPreferences


VENTILATION_LEVELS = ("OFF", "LOW", "MEDIUM", "HIGH", "MAXIMUM")
VENTILATION_LEVEL_PWM = {
    "OFF": 0,
    "LOW": 64,
    "MEDIUM": 128,
    "HIGH": 204,
    "MAXIMUM": 255,
}


def _bucket(value: float, thresholds: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[index]
    return labels[-1]


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


def offline_state(
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


def ventilation_comfort(
    observation: RoomObservation,
    preferences: UserPreferences,
) -> float:
    assessment = assess_comfort(observation, preferences)
    return (
        0.30 * assessment.thermal
        + 0.10 * assessment.humidity
        + 0.35 * assessment.air_quality
    ) / 0.75


@dataclass(frozen=True)
class LoggedVentilationTransition:
    state: tuple[str, ...]
    action_level: str
    reward: float
    next_state: tuple[str, ...]
    outcome_comfort: float
    co2_violation_rate: float
    estimated_energy_wh: float


@dataclass(frozen=True)
class OfflineValueEstimate:
    q_value: float
    support: int
    outcome_comfort: float
    co2_violation_rate: float
    estimated_energy_wh: float


class ConservativeOfflineVentilationQ:
    """Batch Q warm-start using only VAV actions that were actually logged.

    Action selection is constrained to well-supported levels no more than one
    category from the behavior prior. A candidate must retain observed comfort
    and CO2 outcomes relative to the behavior level before Q-value can favor it.
    """

    def __init__(
        self,
        horizon_minutes: int = 15,
        stride_minutes: int = 5,
        minimum_stability: float = 0.80,
        minimum_support: int = 4,
        gamma: float = 0.90,
        iterations: int = 30,
        comfort_margin: float = 2.0,
        co2_margin: float = 0.02,
        minimum_q_advantage: float = 0.02,
    ):
        self.horizon_minutes = horizon_minutes
        self.stride_minutes = stride_minutes
        self.minimum_stability = minimum_stability
        self.minimum_support = minimum_support
        self.gamma = gamma
        self.iterations = iterations
        self.comfort_margin = comfort_margin
        self.co2_margin = co2_margin
        self.minimum_q_advantage = minimum_q_advantage
        self.transition_count = 0
        self._exact: dict[tuple[tuple[str, ...], str], OfflineValueEstimate] = {}
        self._transfer: dict[tuple[tuple[str, ...], str], OfflineValueEstimate] = {}

    @property
    def states(self) -> int:
        return len({state for state, _level in self._exact})

    @property
    def q_entries(self) -> int:
        return len(self._exact)

    def fit(
        self,
        observations: Iterable[RoomObservation],
        preferences: UserPreferences,
        calibration: ActuatorCalibration,
    ) -> None:
        transitions = build_logged_transitions(
            observations,
            preferences,
            calibration,
            horizon_minutes=self.horizon_minutes,
            stride_minutes=self.stride_minutes,
            minimum_stability=self.minimum_stability,
        )
        self.transition_count = len(transitions)
        self._exact = self._fit_table(transitions, lambda state: state)
        self._transfer = self._fit_table(transitions, lambda state: state[1:])

    def predict_level(
        self,
        observation: RoomObservation,
        preferences: UserPreferences,
        behavior_level: str,
    ) -> str | None:
        state = offline_state(observation, preferences)
        exact = self._state_estimates(self._exact, state)
        selected = self._select(exact, behavior_level)
        if selected is not None:
            return selected
        transferable = self._state_estimates(self._transfer, state[1:])
        return self._select(transferable, behavior_level)

    def estimate(
        self,
        observation: RoomObservation,
        preferences: UserPreferences,
        level: str,
    ) -> OfflineValueEstimate | None:
        state = offline_state(observation, preferences)
        return self._exact.get((state, level)) or self._transfer.get((state[1:], level))

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "horizon_minutes": self.horizon_minutes,
                "stride_minutes": self.stride_minutes,
                "minimum_stability": self.minimum_stability,
                "minimum_support": self.minimum_support,
                "gamma": self.gamma,
                "iterations": self.iterations,
                "comfort_margin": self.comfort_margin,
                "co2_margin": self.co2_margin,
                "minimum_q_advantage": self.minimum_q_advantage,
            },
            "transition_count": self.transition_count,
            "exact_rows": self._serialize(self._exact),
            "transfer_rows": self._serialize(self._transfer),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConservativeOfflineVentilationQ":
        model = cls(**payload.get("config", {}))
        model.transition_count = int(payload.get("transition_count", 0))
        model._exact = model._deserialize(payload.get("exact_rows", []))
        model._transfer = model._deserialize(payload.get("transfer_rows", []))
        return model

    def _fit_table(
        self,
        transitions: list[LoggedVentilationTransition],
        state_transform: Callable[[tuple[str, ...]], tuple[str, ...]],
    ) -> dict[tuple[tuple[str, ...], str], OfflineValueEstimate]:
        transformed = [
            LoggedVentilationTransition(
                state=state_transform(item.state),
                action_level=item.action_level,
                reward=item.reward,
                next_state=state_transform(item.next_state),
                outcome_comfort=item.outcome_comfort,
                co2_violation_rate=item.co2_violation_rate,
                estimated_energy_wh=item.estimated_energy_wh,
            )
            for item in transitions
        ]
        support = Counter((item.state, item.action_level) for item in transformed)
        q_values: dict[tuple[tuple[str, ...], str], float] = defaultdict(float)
        for _iteration in range(self.iterations):
            targets: dict[tuple[tuple[str, ...], str], list[float]] = defaultdict(list)
            for item in transformed:
                next_values = [
                    q_values[(item.next_state, level)]
                    for level in VENTILATION_LEVELS
                    if support[(item.next_state, level)] >= self.minimum_support
                ]
                future = max(next_values) if next_values else 0.0
                targets[(item.state, item.action_level)].append(
                    item.reward + self.gamma * future
                )
            q_values = defaultdict(
                float,
                {key: fmean(values) for key, values in targets.items()},
            )

        grouped: dict[
            tuple[tuple[str, ...], str], list[LoggedVentilationTransition]
        ] = defaultdict(list)
        for item in transformed:
            grouped[(item.state, item.action_level)].append(item)
        return {
            key: OfflineValueEstimate(
                q_value=q_values[key],
                support=len(items),
                outcome_comfort=fmean(item.outcome_comfort for item in items),
                co2_violation_rate=fmean(item.co2_violation_rate for item in items),
                estimated_energy_wh=fmean(item.estimated_energy_wh for item in items),
            )
            for key, items in grouped.items()
        }

    def _select(
        self,
        estimates: dict[str, OfflineValueEstimate],
        behavior_level: str,
    ) -> str | None:
        supported = {
            level: estimate
            for level, estimate in estimates.items()
            if estimate.support >= self.minimum_support
        }
        if not supported:
            return None
        behavior_index = VENTILATION_LEVELS.index(behavior_level)
        behavior = supported.get(behavior_level)
        if behavior is None:
            nearest = min(
                supported,
                key=lambda level: abs(VENTILATION_LEVELS.index(level) - behavior_index),
            )
            behavior = supported[nearest]
        candidates = {
            level: estimate
            for level, estimate in supported.items()
            if abs(VENTILATION_LEVELS.index(level) - behavior_index) <= 1
            and estimate.outcome_comfort >= behavior.outcome_comfort - self.comfort_margin
            and estimate.co2_violation_rate
            <= behavior.co2_violation_rate + self.co2_margin
        }
        if not candidates:
            return behavior_level
        best_level = max(
            candidates,
            key=lambda level: (
                candidates[level].q_value,
                -VENTILATION_LEVEL_PWM[level],
            ),
        )
        behavior_q = behavior.q_value
        if candidates[best_level].q_value < behavior_q + self.minimum_q_advantage:
            return behavior_level
        return best_level

    @staticmethod
    def _state_estimates(
        table: dict[tuple[tuple[str, ...], str], OfflineValueEstimate],
        state: tuple[str, ...],
    ) -> dict[str, OfflineValueEstimate]:
        return {
            level: table[(state, level)]
            for level in VENTILATION_LEVELS
            if (state, level) in table
        }

    @staticmethod
    def _serialize(
        table: dict[tuple[tuple[str, ...], str], OfflineValueEstimate],
    ) -> list[dict[str, object]]:
        return [
            {
                "state": list(state),
                "level": level,
                "q_value": estimate.q_value,
                "support": estimate.support,
                "outcome_comfort": estimate.outcome_comfort,
                "co2_violation_rate": estimate.co2_violation_rate,
                "estimated_energy_wh": estimate.estimated_energy_wh,
            }
            for (state, level), estimate in sorted(table.items())
        ]

    @staticmethod
    def _deserialize(
        rows: Iterable[dict[str, object]],
    ) -> dict[tuple[tuple[str, ...], str], OfflineValueEstimate]:
        return {
            (tuple(row["state"]), str(row["level"])): OfflineValueEstimate(
                q_value=float(row["q_value"]),
                support=int(row["support"]),
                outcome_comfort=float(row["outcome_comfort"]),
                co2_violation_rate=float(row["co2_violation_rate"]),
                estimated_energy_wh=float(row["estimated_energy_wh"]),
            )
            for row in rows
        }


def build_logged_transitions(
    observations: Iterable[RoomObservation],
    preferences: UserPreferences,
    calibration: ActuatorCalibration,
    horizon_minutes: int = 15,
    stride_minutes: int = 5,
    minimum_stability: float = 0.80,
) -> list[LoggedVentilationTransition]:
    grouped: dict[tuple[str, int], list[RoomObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.room_id, observation.day_id)].append(observation)

    transitions: list[LoggedVentilationTransition] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item.minute_of_day)
        for start in range(0, len(rows) - horizon_minutes, stride_minutes):
            end = start + horizon_minutes
            if rows[end].minute_of_day != rows[start].minute_of_day + horizon_minutes:
                continue
            window = rows[start:end]
            if any(item.logged_ventilation_pct is None for item in window):
                continue
            levels = [ventilation_level(item.logged_ventilation_pct) for item in window]
            counts = Counter(levels)
            action_level, count = max(
                counts.items(),
                key=lambda item: (item[1], -VENTILATION_LEVELS.index(item[0])),
            )
            if count / horizon_minutes < minimum_stability:
                continue

            outcome_rows = rows[start + 1 : end + 1]
            occupied_outcomes = [item for item in outcome_rows if item.occupancy > 0]
            comfort_rows = occupied_outcomes or outcome_rows
            outcome_comfort = fmean(
                ventilation_comfort(item, preferences) for item in comfort_rows
            )
            co2_violation_rate = (
                fmean(item.co2_ppm > preferences.max_co2_ppm for item in occupied_outcomes)
                if occupied_outcomes
                else 0.0
            )
            pwm = VENTILATION_LEVEL_PWM[action_level]
            energy_wh = (
                calibration.fan_max_power_w
                * (pwm / 255.0) ** 3
                * horizon_minutes
                / 60.0
            )
            discomfort = (100.0 - outcome_comfort) / 100.0
            critical_co2_rate = (
                fmean(item.co2_ppm >= 1200.0 for item in occupied_outcomes)
                if occupied_outcomes
                else 0.0
            )
            previous_level = (
                ventilation_level(rows[start - 1].logged_ventilation_pct)
                if start > 0 and rows[start - 1].logged_ventilation_pct is not None
                else action_level
            )
            switching = abs(
                VENTILATION_LEVELS.index(action_level)
                - VENTILATION_LEVELS.index(previous_level)
            ) / (len(VENTILATION_LEVELS) - 1)
            reward = (
                -8.0 * discomfort
                -0.012 * preferences.energy_sensitivity * energy_wh
                -0.08 * switching
                -4.0 * critical_co2_rate
            )
            transitions.append(
                LoggedVentilationTransition(
                    state=offline_state(rows[start], preferences),
                    action_level=action_level,
                    reward=reward,
                    next_state=offline_state(rows[end], preferences),
                    outcome_comfort=outcome_comfort,
                    co2_violation_rate=co2_violation_rate,
                    estimated_energy_wh=energy_wh,
                )
            )
    return transitions
