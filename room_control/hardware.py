from dataclasses import dataclass
from typing import Protocol

from room_control.models import ControlAction, RoomObservation


class SensorReader(Protocol):
    def read_observation(self) -> RoomObservation:
        """Read the latest validated room state."""


class ActuatorDriver(Protocol):
    def apply_action(self, action: ControlAction) -> None:
        """Apply direct fan and LED PWM outputs."""


@dataclass
class HardwareControlLoop:
    sensors: SensorReader
    actuators: ActuatorDriver

    def step(self, controller) -> ControlAction:
        observation = self.sensors.read_observation()
        action = controller.act(observation)
        self.actuators.apply_action(action)
        return action
