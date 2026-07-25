import argparse
from pathlib import Path

from room_control.controllers import AdaptiveQLearningController
from room_control.dataset import file_sha256, load_observations
from room_control.models import ActuatorCalibration, UserPreferences
from room_control.pretraining import pretrain_and_audit, write_pretraining_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-start the adaptive controller from real room observations."
    )
    parser.add_argument("--data", default="data/real_room/observations.csv.gz")
    parser.add_argument("--output", default="room_policy.json")
    parser.add_argument("--report-dir", default="reports/real_room/latest")
    parser.add_argument("--room", action="append", help="Limit training to a room such as room_1.")
    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.8,
        help="Fraction of whole days reserved for train plus validation.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of development days used for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-temperature", type=float, default=25.0)
    parser.add_argument("--humidity-min", type=float, default=40.0)
    parser.add_argument("--humidity-max", type=float, default=60.0)
    parser.add_argument("--max-co2", type=float, default=900.0)
    parser.add_argument("--target-lux", type=float, default=500.0)
    parser.add_argument("--energy-sensitivity", type=float, default=1.0)
    parser.add_argument("--fan-min-effective-pwm", type=int, default=0)
    parser.add_argument("--fan-max-power-w", type=float, default=40.0)
    parser.add_argument("--light-max-power-w", type=float, default=24.0)
    parser.add_argument("--light-feedback-gain", type=float, default=0.25)
    parser.add_argument("--offline-value-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"normalized dataset not found: {data_path}\n"
            "Run: python prepare_dataset.py"
        )

    preferences = UserPreferences(
        target_temperature_c=args.target_temperature,
        humidity_min_pct=args.humidity_min,
        humidity_max_pct=args.humidity_max,
        max_co2_ppm=args.max_co2,
        target_illuminance_lux=args.target_lux,
        energy_sensitivity=args.energy_sensitivity,
    )
    calibration = ActuatorCalibration(
        fan_min_effective_pwm=args.fan_min_effective_pwm,
        fan_max_power_w=args.fan_max_power_w,
        light_max_power_w=args.light_max_power_w,
        light_feedback_gain_pwm_per_lux=args.light_feedback_gain,
    )
    observations = load_observations(
        data_path,
        rooms=set(args.room) if args.room else None,
    )
    controller = AdaptiveQLearningController(
        preferences=preferences,
        calibration=calibration,
        epsilon=0.10,
        offline_value_weight=args.offline_value_weight,
        seed=args.seed,
    )
    report = pretrain_and_audit(
        controller,
        observations,
        development_fraction=args.development_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    controller.training_metadata = {
        "dataset_path": data_path.as_posix(),
        "dataset_sha256": file_sha256(data_path),
        "split_seed": args.seed,
        "development_fraction": args.development_fraction,
        "validation_fraction": args.validation_fraction,
        "train_day_ids": list(report.train_day_ids),
        "validation_day_ids": list(report.validation_day_ids),
        "evaluation_day_ids": list(report.evaluation_day_ids),
        "rooms": list(report.rooms),
    }
    controller.save(args.output)
    markdown_path, _json_path = write_pretraining_report(
        report,
        controller,
        args.report_dir,
    )

    print("Real-room warm start complete")
    print(f"Records: {report.dataset_records:,} across {len(report.rooms)} rooms")
    print(
        f"Whole-day split: {len(report.train_day_ids)} train / "
        f"{len(report.validation_day_ids)} validation / "
        f"{len(report.evaluation_day_ids)} final test"
    )
    print(
        f"Ventilation prior: {report.ventilation_prior_states:,} states, "
        f"{report.ventilation_prior_coverage_pct:.1f}% held-out coverage, "
        f"{report.ventilation_level_accuracy_pct:.1f}% exact level accuracy, "
        f"{report.ventilation_within_one_level_pct:.1f}% within one level"
    )
    print(
        f"Comfort-energy offline Q: {report.offline_transition_count:,} stable transitions, "
        f"{report.offline_q_entries:,} supported state-actions, "
        f"{report.offline_validation_coverage_pct:.1f}% validation coverage"
    )
    print(
        f"Held-out occupied requests: fan="
        f"{report.requested_occupied_average_fan_pwm:.1f} PWM, "
        f"light={report.requested_occupied_average_light_pwm:.1f} PWM"
    )
    print("Online Q-learning updates: 0 (the current hardware demo is inference-only)")
    print(f"Policy: {Path(args.output).resolve()}")
    print(f"Report: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
