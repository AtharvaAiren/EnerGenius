import argparse
from pathlib import Path

from room_control.controllers import AdaptiveQLearningController
from room_control.models import UserFeedback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record explicit user preference feedback in a room policy."
    )
    parser.add_argument("--policy", default="room_policy.json")
    parser.add_argument("--output", help="Write a new policy instead of updating --policy.")
    parser.add_argument("--thermal", choices=("cold", "comfortable", "hot"))
    parser.add_argument("--humidity", choices=("dry", "comfortable", "humid"))
    parser.add_argument("--air-quality", choices=("fresh", "comfortable", "stale"))
    parser.add_argument("--lighting", choices=("dim", "comfortable", "bright"))
    parser.add_argument("--overall", type=int, choices=(-1, 0, 1), default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any(
        (args.thermal, args.humidity, args.air_quality, args.lighting, args.overall)
    ):
        raise SystemExit("provide at least one feedback field")
    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise SystemExit(f"policy not found: {policy_path}; run python main.py")

    controller = AdaptiveQLearningController.load(policy_path)
    before = controller.preferences
    after = controller.apply_feedback(
        UserFeedback(
            thermal=args.thermal,
            humidity=args.humidity,
            air_quality=args.air_quality,
            lighting=args.lighting,
            overall=args.overall,
        )
    )
    output = Path(args.output) if args.output else policy_path
    controller.save(output)

    print("Preference profile updated")
    print(
        f"Temperature: {before.target_temperature_c:.2f} -> "
        f"{after.target_temperature_c:.2f} C"
    )
    print(
        f"Humidity: {before.humidity_min_pct:.0f}-{before.humidity_max_pct:.0f}% -> "
        f"{after.humidity_min_pct:.0f}-{after.humidity_max_pct:.0f}%"
    )
    print(f"CO2 ceiling: {before.max_co2_ppm:.0f} -> {after.max_co2_ppm:.0f} ppm")
    print(
        f"Target light: {before.target_illuminance_lux:.0f} -> "
        f"{after.target_illuminance_lux:.0f} lux"
    )
    print(f"Feedback records: {controller.feedback_count}")
    print(f"Policy: {output.resolve()}")


if __name__ == "__main__":
    main()
