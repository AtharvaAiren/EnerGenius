from dataclasses import dataclass
from statistics import mean

from simulator.controllers import BaselineController, Controller, QLearningController
from simulator.environment import RoomEnvironment
from simulator.models import StepResult


@dataclass(frozen=True)
class EvaluationSummary:
    steps: int
    occupied_steps: int
    total_reward: float
    total_energy_wh: float
    average_comfort_when_occupied: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "occupied_steps": self.occupied_steps,
            "total_reward": round(self.total_reward, 3),
            "total_energy_wh": round(self.total_energy_wh, 3),
            "average_comfort_when_occupied": round(self.average_comfort_when_occupied, 3),
        }


def train_q_controller(
    days: int = 45,
    seed: int = 7,
    save_path: str = "q_table.json",
) -> QLearningController:
    env = RoomEnvironment(seed=seed)
    controller = QLearningController(seed=seed)
    observation = env.reset()
    total_steps = days * env.steps_per_day

    for _ in range(total_steps):
        action = controller.act(observation)
        result = env.step(action)
        next_observation = env.observe()
        controller.update(observation, action, result.reward, next_observation)
        controller.decay_exploration()
        observation = next_observation

    controller.save(save_path)
    return controller


def evaluate_controller(controller: Controller, days: int = 7, seed: int = 123) -> EvaluationSummary:
    env = RoomEnvironment(seed=seed)
    env.reset()
    results: list[StepResult] = []

    for _ in range(days * env.steps_per_day):
        observation = env.observe()
        if isinstance(controller, QLearningController):
            action = controller.act(observation, greedy=True)
        else:
            action = controller.act(observation)
        results.append(env.step(action))

    occupied_comfort = [
        result.comfort
        for result in results
        if result.observation.occupancy > 0
    ]
    return EvaluationSummary(
        steps=len(results),
        occupied_steps=len(occupied_comfort),
        total_reward=sum(result.reward for result in results),
        total_energy_wh=sum(result.energy_wh for result in results),
        average_comfort_when_occupied=mean(occupied_comfort) if occupied_comfort else 0.0,
    )


def evaluate_baseline(days: int = 7, seed: int = 123) -> EvaluationSummary:
    return evaluate_controller(BaselineController(), days=days, seed=seed)
