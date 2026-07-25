import argparse
import time
from dataclasses import replace
from pathlib import Path

from room_control.controllers import AdaptiveQLearningController, ComfortBaselineController
from room_control.models import ActuatorCalibration, ControlAction, UserPreferences
from room_control.serial_hardware import LiveRoomConfig, SerialRoomBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct PWM room control against an EnerGenius ESP32."
    )
    parser.add_argument("--port", help="Serial device, for example /dev/cu.usbserial-0001.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--controller", choices=("learned", "baseline"), default="learned")
    parser.add_argument("--policy", default="room_policy.json")
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--decision-interval", type=float, default=5.0)
    parser.add_argument("--room-area", type=float, default=20.0)
    parser.add_argument("--room-volume", type=float, default=60.0)
    parser.add_argument("--room-type", default="prototype")
    parser.add_argument("--seating-capacity", type=int, default=10)
    parser.add_argument("--target-temperature", type=float, default=25.0)
    parser.add_argument("--humidity-min", type=float, default=40.0)
    parser.add_argument("--humidity-max", type=float, default=60.0)
    parser.add_argument("--max-co2", type=float, default=900.0)
    parser.add_argument("--target-lux", type=float, default=500.0)
    parser.add_argument("--assumed-co2", type=float, default=600.0)
    parser.add_argument("--fan-min-effective-pwm", type=int, default=177)
    return parser.parse_args()


def preferences_from_args(args: argparse.Namespace) -> UserPreferences:
    return UserPreferences(
        target_temperature_c=args.target_temperature,
        humidity_min_pct=args.humidity_min,
        humidity_max_pct=args.humidity_max,
        max_co2_ppm=args.max_co2,
        target_illuminance_lux=args.target_lux,
    )


def load_controller(args: argparse.Namespace):
    preferences = preferences_from_args(args)
    calibration = ActuatorCalibration(fan_min_effective_pwm=0)
    if args.controller == "baseline":
        return ComfortBaselineController(preferences, calibration)
    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise SystemExit(f"policy not found: {policy_path}; run python main.py")
    controller = AdaptiveQLearningController.load(policy_path)
    controller.preferences = preferences
    controller.calibration = replace(
        controller.calibration,
        fan_min_effective_pwm=0,
    )
    controller.baseline = ComfortBaselineController(
        controller.preferences,
        controller.calibration,
    )
    controller.epsilon = 0.0
    return controller


def map_logical_fan_pwm(logical_pwm: int, minimum_effective_pwm: int) -> int:
    if not 0 <= minimum_effective_pwm <= 255:
        raise ValueError("minimum_effective_pwm must be between 0 and 255")
    if logical_pwm == 0:
        return 0
    usable_span = 255 - minimum_effective_pwm
    return int(round(minimum_effective_pwm + usable_span * logical_pwm / 255.0))


def adapt_to_prototype_hardware(
    action: ControlAction,
    minimum_effective_pwm: int,
) -> ControlAction:
    return replace(
        action,
        fan_pwm=map_logical_fan_pwm(action.fan_pwm, minimum_effective_pwm),
    )


def list_ports() -> None:
    try:
        from serial.tools import list_ports as serial_list_ports
    except ImportError as exc:
        raise SystemExit("pyserial is required; install requirements.txt") from exc
    ports = list(serial_list_ports.comports())
    if not ports:
        print("No serial devices found.")
        return
    for port in ports:
        print(f"{port.device}\t{port.description}")


def main() -> None:
    args = parse_args()
    if args.list_ports:
        list_ports()
        return
    if not args.port:
        raise SystemExit("--port is required; use --list-ports first")
    if args.decision_interval <= 0:
        raise SystemExit("--decision-interval must be positive")
    if not 0 <= args.fan_min_effective_pwm <= 255:
        raise SystemExit("--fan-min-effective-pwm must be between 0 and 255")

    bridge = SerialRoomBridge(
        port=args.port,
        baudrate=args.baud,
        config=LiveRoomConfig(
            area_m2=args.room_area,
            volume_m3=args.room_volume,
            room_type=args.room_type,
            seating_capacity=args.seating_capacity,
            assumed_co2_ppm=args.assumed_co2,
        ),
    )
    controller = None if args.observe_only else load_controller(args)
    print(f"Connected to {args.port} at {args.baud} baud")
    print(
        f"Room: {args.room_type}, {args.room_area:.1f} m2, {args.room_volume:.1f} m3, "
        f"capacity {args.seating_capacity}; CO2 fallback is {args.assumed_co2:.0f} ppm"
    )
    print("Fresh CO2 telemetry overrides the fallback automatically.")
    if args.observe_only:
        bridge.enter_local_mode()
        print("Observe-only mode; ESP32 local control remains active.")
    else:
        print("Host sends direct fan and LED PWM; the reserved field stays OFF.")
        print(
            "Nonzero logical fan demand is mapped to physical PWM "
            f"{args.fan_min_effective_pwm}-255."
        )

    started = time.monotonic()
    next_decision_at = 0.0
    logical_action = ControlAction()
    applied_action = ControlAction()
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            observation = bridge.read_observation()
            telemetry = bridge.last_telemetry
            if args.observe_only:
                print_observation(observation, telemetry)
                continue
            now = time.monotonic()
            if now >= next_decision_at:
                logical_action = controller.act(observation)
                applied_action = adapt_to_prototype_hardware(
                    logical_action,
                    args.fan_min_effective_pwm,
                )
                next_decision_at = now + args.decision_interval
            bridge.apply_action(applied_action)
            print_control_step(
                observation,
                telemetry,
                logical_action,
                applied_action,
            )
    except KeyboardInterrupt:
        print("\nStopping hardware control.")
    finally:
        if args.observe_only:
            bridge.enter_local_mode()
            bridge.close(safe=False)
        else:
            bridge.close(safe=True)


def print_observation(observation, telemetry) -> None:
    co2_source = "sensor" if observation.co2_is_measured else "fallback"
    print(
        f"mode={telemetry.control_mode:<5} people={observation.occupancy:<2} "
        f"density={observation.occupancy_density:.3f}/m2 "
        f"temp={observation.temperature_c:5.1f} C humidity={observation.humidity_pct:5.1f}% "
        f"CO2={observation.co2_ppm:6.0f}({co2_source}) "
        f"lux={observation.illuminance_lux:6.1f} "
        f"applied_fan={telemetry.fan_pwm:3d} applied_light={telemetry.led_pwm:3d}"
    )


def print_control_step(
    observation,
    telemetry,
    logical_action,
    applied_action,
) -> None:
    co2_source = "sensor" if observation.co2_is_measured else "fallback"
    print(
        f"people={observation.occupancy:<2} density={observation.occupancy_density:.3f}/m2 "
        f"temp={observation.temperature_c:5.1f} C RH={observation.humidity_pct:5.1f}% "
        f"CO2={observation.co2_ppm:6.0f}({co2_source}) "
        f"lux={observation.illuminance_lux:6.1f} | "
        f"logical_fan={logical_action.fan_pwm:3d} logical_light={logical_action.light_pwm:3d} "
        f"applied_fan={applied_action.fan_pwm:3d} "
        f"device_fan={telemetry.fan_pwm:3d} device_light={telemetry.led_pwm:3d}"
    )


if __name__ == "__main__":
    main()
