import argparse
import csv
from dataclasses import replace
from pathlib import Path

from room_control.controllers import AdaptiveQLearningController
from room_control.dataset import file_sha256, load_observations
from room_control.models import ControlAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fan and lighting decisions on one held-out real-room day."
    )
    parser.add_argument("--data", default="data/real_room/observations.csv.gz")
    parser.add_argument("--policy", default="room_policy.json")
    parser.add_argument("--room", default="room_1")
    parser.add_argument("--day-id", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/real_room/demo")
    parser.add_argument("--max-lines", type=int, default=16)
    return parser.parse_args()


def choose_day(
    observations,
    room: str,
    requested: int | None,
    evaluation_days: set[int],
) -> int:
    room_observations = [item for item in observations if item.room_id == room]
    if not room_observations:
        raise SystemExit(f"room not found: {room}")
    if requested is not None:
        if requested not in evaluation_days:
            raise SystemExit(f"day {requested} is not in the policy's held-out split")
        if not any(item.day_id == requested for item in room_observations):
            raise SystemExit(f"day {requested} is not available for {room}")
        return requested
    candidates = [item for item in room_observations if item.day_id in evaluation_days]
    if not candidates:
        raise SystemExit(f"no held-out observations are available for {room}")
    occupancy_totals: dict[int, int] = {}
    for item in candidates:
        occupancy_totals[item.day_id] = occupancy_totals.get(item.day_id, 0) + item.occupancy
    return max(occupancy_totals, key=occupancy_totals.get)


def run_decisions(controller, day, interval: int):
    rows = []
    for observation in day:
        if observation.minute_of_day % interval:
            continue
        audit_observation = replace(observation, previous_action=ControlAction())
        rows.append((audit_observation, controller.decide(audit_observation)))
    return rows


def write_csv(rows, output: Path, controller: AdaptiveQLearningController) -> None:
    fields = (
        "minute_of_day",
        "occupancy",
        "occupancy_density_per_m2",
        "temperature_c",
        "humidity_pct",
        "co2_ppm",
        "illuminance_lux",
        "logged_vav_pct",
        "logged_vav_pwm_equivalent",
        "behavior_prior_level",
        "offline_q_level",
        "requested_fan_pwm",
        "requested_light_pwm",
        "safety_interventions",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for observation, decision in rows:
            behavior_level = controller.ventilation_prior.predict_level(observation)
            offline_level = (
                controller.offline_ventilation_q.predict_level(
                    observation,
                    controller.preferences,
                    behavior_level,
                )
                if behavior_level is not None and observation.occupancy > 0
                else None
            )
            writer.writerow(
                {
                    "minute_of_day": observation.minute_of_day,
                    "occupancy": observation.occupancy,
                    "occupancy_density_per_m2": observation.occupancy_density,
                    "temperature_c": observation.temperature_c,
                    "humidity_pct": observation.humidity_pct,
                    "co2_ppm": observation.co2_ppm,
                    "illuminance_lux": observation.illuminance_lux,
                    "logged_vav_pct": observation.logged_ventilation_pct,
                    "logged_vav_pwm_equivalent": (
                        observation.logged_ventilation_pct * 2.55
                        if observation.logged_ventilation_pct is not None
                        else None
                    ),
                    "behavior_prior_level": behavior_level,
                    "offline_q_level": offline_level,
                    "requested_fan_pwm": decision.action.fan_pwm,
                    "requested_light_pwm": decision.action.light_pwm,
                    "safety_interventions": "; ".join(decision.safety_interventions),
                }
            )


def write_plot(rows, output: Path, room: str, day_id: int, target_lux: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install requirements.txt") from exc

    hours = [observation.minute_of_day / 60.0 for observation, _decision in rows]
    occupancy = [observation.occupancy for observation, _decision in rows]
    temperature = [observation.temperature_c for observation, _decision in rows]
    humidity = [observation.humidity_pct for observation, _decision in rows]
    co2 = [observation.co2_ppm for observation, _decision in rows]
    lux = [observation.illuminance_lux for observation, _decision in rows]
    logged_vav_pwm = [
        observation.logged_ventilation_pct * 2.55
        for observation, _decision in rows
    ]
    requested_fan = [decision.action.fan_pwm for _observation, decision in rows]
    requested_light = [decision.action.light_pwm for _observation, decision in rows]

    figure, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    axes[0].step(hours, occupancy, where="post", color="#2a6f97")
    axes[0].set_ylabel("People")
    axes[0].set_title(f"Real state with fan and lighting decisions: {room}, DayId {day_id}")

    temperature_line = axes[1].plot(
        hours, temperature, label="Temperature C", color="#c44536"
    )
    humidity_axis = axes[1].twinx()
    humidity_line = humidity_axis.plot(
        hours, humidity, label="Humidity %", color="#386641"
    )
    axes[1].set_ylabel("Temperature C")
    humidity_axis.set_ylabel("Humidity %")
    axes[1].legend(
        temperature_line + humidity_line,
        ["Temperature C", "Humidity %"],
        loc="upper right",
    )

    axes[2].plot(hours, co2, color="#6a4c93")
    axes[2].axhline(900, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_ylabel("CO2 ppm")

    axes[3].plot(hours, lux, label="Observed lux", color="#7f5539")
    axes[3].axhline(
        target_lux,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label=f"Target {target_lux:.0f} lux",
    )
    axes[3].step(
        hours,
        requested_light,
        where="post",
        label="Requested LED PWM",
        color="#e09f3e",
    )
    axes[3].set_ylabel("Lux / PWM")
    axes[3].legend(loc="upper right")

    axes[4].step(
        hours,
        logged_vav_pwm,
        where="post",
        label="Logged VAV equivalent",
        color="#d4a017",
    )
    axes[4].step(
        hours,
        requested_fan,
        where="post",
        label="Requested fan PWM",
        color="#168aad",
    )
    axes[4].set_ylabel("0-255 demand")
    axes[4].set_ylim(0, 255)
    axes[4].set_xlabel("Hour")
    axes[4].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 24)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.interval <= 0 or 60 % args.interval:
        raise SystemExit("--interval must be a positive divisor of 60")
    data_path = Path(args.data)
    policy_path = Path(args.policy)
    if not data_path.exists() or not policy_path.exists():
        raise SystemExit(
            "prepare data and policy first:\n"
            "  python prepare_dataset.py\n"
            "  python main.py"
        )

    controller = AdaptiveQLearningController.load(policy_path, seed=args.seed)
    controller.epsilon = 0.0
    metadata = controller.training_metadata
    if file_sha256(data_path) != metadata.get("dataset_sha256"):
        raise SystemExit(
            "dataset checksum does not match the policy; regenerate it with: python main.py"
        )
    policy_rooms = {str(room) for room in metadata.get("rooms", [])}
    if args.room not in policy_rooms:
        raise SystemExit(
            f"{args.room} was not included when this policy was trained; "
            f"available rooms: {', '.join(sorted(policy_rooms))}"
        )
    evaluation_days = {int(day) for day in metadata.get("evaluation_day_ids", [])}
    if not evaluation_days:
        raise SystemExit("policy has no held-out split; regenerate it with: python main.py")

    observations = load_observations(data_path, rooms={args.room})
    day_id = choose_day(observations, args.room, args.day_id, evaluation_days)
    day = sorted(
        (
            item
            for item in observations
            if item.room_id == args.room and item.day_id == day_id
        ),
        key=lambda item: item.minute_of_day,
    )
    rows = run_decisions(controller, day, args.interval)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "decision_audit.csv"
    plot_path = output_directory / "decision_audit.png"
    write_csv(rows, csv_path, controller)
    write_plot(
        rows,
        plot_path,
        args.room,
        day_id,
        controller.preferences.target_illuminance_lux,
    )

    print("Time   People Density  Temp   RH    CO2    Lux    VAV  | Fan Light")
    print("-----  ------ -------  ----- ----- ------ ------ ------ | ---- -----")
    occupied_rows = [row for row in rows if row[0].occupancy > 0]
    display_rows = occupied_rows[: args.max_lines] or rows[: args.max_lines]
    for observation, decision in display_rows:
        hour, minute = divmod(observation.minute_of_day, 60)
        print(
            f"{hour:02d}:{minute:02d}  {observation.occupancy:6d} "
            f"{observation.occupancy_density:7.3f} "
            f"{observation.temperature_c:5.1f} {observation.humidity_pct:5.1f} "
            f"{observation.co2_ppm:6.0f} "
            f"{observation.illuminance_lux:6.0f} "
            f"{observation.logged_ventilation_pct:6.1f} | "
            f"{decision.action.fan_pwm:4d} {decision.action.light_pwm:5d}"
        )
    print()
    print("These are fan and lighting requests on logged observations, not a causal replay.")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
