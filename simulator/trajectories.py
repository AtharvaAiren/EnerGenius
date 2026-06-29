from dataclasses import asdict, dataclass

from simulator.controllers import BaselineController, Controller, QLearningController
from simulator.environment import RoomEnvironment
from simulator.models import ControlAction


@dataclass(frozen=True)
class TrajectoryRow:
    day: int
    clock: str
    weekend: bool
    season: str
    occupancy: int
    archetypes: str
    temperature: float
    humidity: float
    ambient_light: float
    outdoor_temperature: float
    co2: float
    preferred_temperature: float
    preferred_humidity: float
    preferred_light: float
    energy_sensitivity: float
    action: str
    comfort: float
    energy_wh: float
    reward: float

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def generate_trajectory(
    controller: Controller | None = None,
    days: int = 1,
    seed: int = 123,
    stride_steps: int = 12,
) -> list[TrajectoryRow]:
    env = RoomEnvironment(seed=seed)
    controller = controller or BaselineController()
    env.reset()
    rows: list[TrajectoryRow] = []

    for step in range(days * env.steps_per_day):
        observation = env.observe()
        action = _controller_action(controller, observation)
        result = env.step(action)
        if step % stride_steps == 0:
            rows.append(
                TrajectoryRow(
                    day=observation.day,
                    clock=_clock(observation.minute),
                    weekend=observation.is_weekend,
                    season=observation.season,
                    occupancy=observation.occupancy,
                    archetypes=_format_archetypes(observation.archetype_counts),
                    temperature=round(observation.temperature, 2),
                    humidity=round(observation.humidity, 2),
                    ambient_light=round(observation.ambient_light, 1),
                    outdoor_temperature=round(observation.outdoor_temperature, 2),
                    co2=round(observation.co2, 1),
                    preferred_temperature=round(observation.preferred_temperature, 2),
                    preferred_humidity=round(observation.preferred_humidity, 1),
                    preferred_light=round(observation.preferred_light, 1),
                    energy_sensitivity=round(observation.average_energy_sensitivity, 2),
                    action=action.label(),
                    comfort=round(result.comfort, 3),
                    energy_wh=round(result.energy_wh, 3),
                    reward=round(result.reward, 3),
                )
            )

    return rows


def _controller_action(controller: Controller, observation) -> ControlAction:
    if isinstance(controller, QLearningController):
        return controller.act(observation, greedy=True)
    return controller.act(observation)


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _format_archetypes(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))
