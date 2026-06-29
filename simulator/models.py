from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class FanSpeed(str, Enum):
    OFF = "OFF"
    LOW = "LOW"
    HIGH = "HIGH"


class LightLevel(str, Enum):
    OFF = "OFF"
    LOW = "LOW"
    HIGH = "HIGH"


class HVACMode(str, Enum):
    OFF = "OFF"
    COOL = "COOL"
    HEAT = "HEAT"


@dataclass(frozen=True)
class ControlAction:
    fan: FanSpeed = FanSpeed.OFF
    light: LightLevel = LightLevel.OFF
    hvac: HVACMode = HVACMode.OFF

    def label(self) -> str:
        return f"fan={self.fan.value},light={self.light.value},hvac={self.hvac.value}"


@dataclass(frozen=True)
class Observation:
    step: int
    day: int
    hour: int
    minute: int
    is_weekend: bool
    season: str
    occupancy: int
    temperature: float
    humidity: float
    ambient_light: float
    outdoor_temperature: float
    co2: float
    preferred_temperature: float
    preferred_humidity: float
    preferred_light: float
    average_energy_sensitivity: float
    average_discomfort_tolerance: float
    average_activity_level: float
    previous_action: ControlAction
    archetype_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceProfile:
    target_temperature: float = 25.5
    target_humidity: float = 50.0
    target_light_when_occupied: float = 500.0
    energy_sensitivity: float = 1.0
    discomfort_tolerance: float = 1.0
    crowding_sensitivity: float = 1.0
    activity_level: float = 1.0


@dataclass(frozen=True)
class PopulationSnapshot:
    day: int
    minute: int
    is_weekend: bool
    active_occupant_ids: tuple[int, ...]
    active_archetypes: dict[str, int]
    preferences: PreferenceProfile

    @property
    def occupancy(self) -> int:
        return len(self.active_occupant_ids)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    action: ControlAction
    comfort: float
    energy_wh: float
    reward: float
    occupant_comfort: dict[int, float] = field(default_factory=dict)


def default_action_space() -> list[ControlAction]:
    return [
        ControlAction(fan=fan, light=light, hvac=hvac)
        for fan in (FanSpeed.OFF, FanSpeed.LOW, FanSpeed.HIGH)
        for light in (LightLevel.OFF, LightLevel.LOW, LightLevel.HIGH)
        for hvac in (HVACMode.OFF, HVACMode.COOL, HVACMode.HEAT)
    ]


def find_action(action_space: Iterable[ControlAction], label: str) -> ControlAction:
    for action in action_space:
        if action.label() == label:
            return action
    raise ValueError(f"Unknown action label: {label}")
