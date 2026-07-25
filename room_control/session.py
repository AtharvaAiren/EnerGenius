from __future__ import annotations

from dataclasses import dataclass

from room_control.controllers import AdaptiveQLearningController
from room_control.models import ControlDecision, RoomObservation, Transition, UserFeedback


@dataclass
class AdaptiveControlSession:
    """Connect consecutive real sensor readings to online Q-learning updates.

    Feedback supplied to ``step`` describes the interval that ended with the new
    observation. The first call can decide but cannot learn because no action has
    produced a next state yet.
    """

    controller: AdaptiveQLearningController
    explore: bool = True
    previous_observation: RoomObservation | None = None
    previous_decision: ControlDecision | None = None

    def step(
        self,
        observation: RoomObservation,
        feedback: UserFeedback | None = None,
        elapsed_minutes: float = 1.0,
    ) -> tuple[ControlDecision, float | None]:
        learned_reward: float | None = None
        if self.previous_observation is not None and self.previous_decision is not None:
            learned_reward = self.controller.learn(
                Transition(
                    observation=self.previous_observation,
                    decision=self.previous_decision,
                    next_observation=observation,
                    feedback=feedback,
                    elapsed_minutes=elapsed_minutes,
                )
            )
        elif feedback is not None:
            self.controller.apply_feedback(feedback)

        decision = self.controller.decide(observation, explore=self.explore)
        self.previous_observation = observation
        self.previous_decision = decision
        return decision, learned_reward
