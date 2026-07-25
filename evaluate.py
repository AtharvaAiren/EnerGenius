import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from room_control.comfort import assess_comfort
from room_control.controllers import (
    AdaptiveQLearningController,
    VENTILATION_LEVELS,
    ventilation_level,
)
from room_control.dataset import file_sha256, load_observations
from room_control.models import ActuatorCalibration


FAN_POWER_REFERENCE_URL = (
    "https://www.energystar.gov/productfinder/product/"
    "certified-ceiling-fans/details/2732012"
)


def _new_metrics() -> dict[str, object]:
    return {
        "records": 0,
        "occupied": 0,
        "comfort": [],
        "fan": [],
        "occupied_fan": [],
        "light": [],
        "occupied_light": [],
        "vav_error": [],
        "ventilation_predictions": 0,
        "ventilation_exact": 0,
        "ventilation_within_one": 0,
        "offline_value_predictions": 0,
        "offline_value_changes": 0,
        "offline_comfort_retained": 0,
        "offline_comfort_comparisons": 0,
        "offline_recommended_comfort": [],
        "offline_behavior_comfort": [],
        "requested_ventilation_exact": 0,
        "requested_ventilation_within_one": 0,
        "discomfort_minutes": 0,
        "fan_runtime_minutes": 0,
        "light_runtime_minutes": 0,
        "fan_energy_wh": 0.0,
        "light_energy_wh": 0.0,
        "occupied_fan_energy_wh": 0.0,
    }


def _record_observation(
    metrics,
    observation,
    decision,
    controller,
    power_calibration: ActuatorCalibration,
    comfort_threshold: float,
    include_ventilation_prior: bool,
) -> None:
    metrics["records"] += 1
    assessment = assess_comfort(observation, controller.preferences)
    if observation.occupancy > 0:
        metrics["occupied"] += 1
        metrics["comfort"].append(assessment.overall)
        metrics["discomfort_minutes"] += int(assessment.overall < comfort_threshold)
        metrics["occupied_fan"].append(decision.action.fan_pwm)
        metrics["occupied_light"].append(decision.action.light_pwm)
    metrics["fan"].append(decision.action.fan_pwm)
    metrics["light"].append(decision.action.light_pwm)
    metrics["fan_runtime_minutes"] += int(decision.action.fan_pwm > 0)
    metrics["light_runtime_minutes"] += int(decision.action.light_pwm > 0)

    if observation.logged_ventilation_pct is not None:
        requested_level = ventilation_level(decision.action.fan_pwm / 2.55)
        actual_level = ventilation_level(observation.logged_ventilation_pct)
        metrics["requested_ventilation_exact"] += int(requested_level == actual_level)
        metrics["requested_ventilation_within_one"] += int(
            abs(
                VENTILATION_LEVELS.index(requested_level)
                - VENTILATION_LEVELS.index(actual_level)
            )
            <= 1
        )

    fan_energy_wh = (
        power_calibration.fan_max_power_w
        * (decision.action.fan_pwm / 255.0) ** 3
        / 60.0
    )
    light_energy_wh = (
        power_calibration.light_max_power_w
        * (decision.action.light_pwm / 255.0)
        / 60.0
    )
    metrics["fan_energy_wh"] += fan_energy_wh
    metrics["light_energy_wh"] += light_energy_wh
    if observation.occupancy > 0:
        metrics["occupied_fan_energy_wh"] += fan_energy_wh

    if not include_ventilation_prior:
        return
    behavior_level = controller.ventilation_prior.predict_level(observation)
    if behavior_level is not None and observation.occupancy > 0:
        offline_level = controller.offline_ventilation_q.predict_level(
            observation,
            controller.preferences,
            behavior_level,
        )
        if offline_level is not None:
            metrics["offline_value_predictions"] += 1
            metrics["offline_value_changes"] += int(offline_level != behavior_level)
            behavior_estimate = controller.offline_ventilation_q.estimate(
                observation,
                controller.preferences,
                behavior_level,
            )
            offline_estimate = controller.offline_ventilation_q.estimate(
                observation,
                controller.preferences,
                offline_level,
            )
            if behavior_estimate is not None and offline_estimate is not None:
                metrics["offline_comfort_comparisons"] += 1
                metrics["offline_recommended_comfort"].append(
                    offline_estimate.outcome_comfort
                )
                metrics["offline_behavior_comfort"].append(
                    behavior_estimate.outcome_comfort
                )
                metrics["offline_comfort_retained"] += int(
                    offline_estimate.outcome_comfort
                    >= behavior_estimate.outcome_comfort
                    - controller.offline_ventilation_q.comfort_margin
                )
    predicted_pwm = controller.ventilation_prior.predict(observation)
    predicted_level = controller.ventilation_prior.predict_level(observation)
    if (
        predicted_pwm is None
        or predicted_level is None
        or observation.logged_ventilation_pct is None
    ):
        return
    actual_level = ventilation_level(observation.logged_ventilation_pct)
    metrics["ventilation_predictions"] += 1
    metrics["ventilation_exact"] += int(predicted_level == actual_level)
    metrics["ventilation_within_one"] += int(
        abs(
            VENTILATION_LEVELS.index(predicted_level)
            - VENTILATION_LEVELS.index(actual_level)
        )
        <= 1
    )
    metrics["vav_error"].append(
        abs(predicted_pwm - observation.logged_ventilation_pct * 2.55)
    )


def _summarize_metrics(metrics, include_ventilation_prior: bool) -> dict[str, object]:
    occupied_hours = metrics["occupied"] / 60.0
    summary = {
        "records": metrics["records"],
        "occupied_records": metrics["occupied"],
        "occupied_hours": occupied_hours,
        "observed_comfort_score": (
            fmean(metrics["comfort"]) if metrics["comfort"] else 100.0
        ),
        "discomfort_minutes_below_threshold": metrics["discomfort_minutes"],
        "requested_average_fan_pwm": (
            fmean(metrics["fan"]) if metrics["fan"] else 0.0
        ),
        "requested_occupied_average_fan_pwm": (
            fmean(metrics["occupied_fan"]) if metrics["occupied_fan"] else 0.0
        ),
        "requested_average_light_pwm": (
            fmean(metrics["light"]) if metrics["light"] else 0.0
        ),
        "requested_occupied_average_light_pwm": (
            fmean(metrics["occupied_light"]) if metrics["occupied_light"] else 0.0
        ),
        "fan_runtime_hours": metrics["fan_runtime_minutes"] / 60.0,
        "light_runtime_hours": metrics["light_runtime_minutes"] / 60.0,
        "estimated_fan_energy_kwh": metrics["fan_energy_wh"] / 1000.0,
        "estimated_light_energy_kwh": metrics["light_energy_wh"] / 1000.0,
        "estimated_total_energy_kwh": (
            metrics["fan_energy_wh"] + metrics["light_energy_wh"]
        ) / 1000.0,
        "estimated_occupied_fan_energy_kwh": (
            metrics["occupied_fan_energy_wh"] / 1000.0
        ),
        "estimated_fan_kwh_per_occupied_hour": (
            metrics["occupied_fan_energy_wh"] / 1000.0 / occupied_hours
            if occupied_hours
            else 0.0
        ),
        "requested_ventilation_level_accuracy_pct": (
            100.0 * metrics["requested_ventilation_exact"] / max(metrics["records"], 1)
        ),
        "requested_ventilation_within_one_level_pct": (
            100.0
            * metrics["requested_ventilation_within_one"]
            / max(metrics["records"], 1)
        ),
    }
    if include_ventilation_prior:
        summary.update(
            {
                "ventilation_level_accuracy_pct": (
                    100.0
                    * metrics["ventilation_exact"]
                    / max(metrics["ventilation_predictions"], 1)
                ),
                "ventilation_within_one_level_pct": (
                    100.0
                    * metrics["ventilation_within_one"]
                    / max(metrics["ventilation_predictions"], 1)
                ),
                "ventilation_pwm_equivalent_mae": (
                    fmean(metrics["vav_error"]) if metrics["vav_error"] else None
                ),
                "offline_value_coverage_pct": (
                    100.0
                    * metrics["offline_value_predictions"]
                    / max(metrics["occupied"], 1)
                ),
                "offline_value_changed_pct": (
                    100.0
                    * metrics["offline_value_changes"]
                    / max(metrics["offline_value_predictions"], 1)
                ),
                "offline_model_comfort_retained_pct": (
                    100.0
                    * metrics["offline_comfort_retained"]
                    / max(metrics["offline_comfort_comparisons"], 1)
                ),
                "offline_model_recommended_comfort_score": (
                    fmean(metrics["offline_recommended_comfort"])
                    if metrics["offline_recommended_comfort"]
                    else None
                ),
                "offline_model_behavior_comfort_score": (
                    fmean(metrics["offline_behavior_comfort"])
                    if metrics["offline_behavior_comfort"]
                    else None
                ),
            }
        )
    return summary


def _energy_comparison(
    warm_started: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float]:
    warm_energy = float(warm_started["estimated_fan_energy_kwh"])
    baseline_energy = float(baseline["estimated_fan_energy_kwh"])
    warm_light = float(warm_started["estimated_light_energy_kwh"])
    baseline_light = float(baseline["estimated_light_energy_kwh"])
    warm_total = warm_energy + warm_light
    baseline_total = baseline_energy + baseline_light
    return {
        "estimated_fan_energy_difference_kwh": warm_energy - baseline_energy,
        "estimated_fan_energy_savings_pct": (
            100.0 * (baseline_energy - warm_energy) / baseline_energy
            if baseline_energy
            else 0.0
        ),
        "estimated_light_energy_difference_kwh": warm_light - baseline_light,
        "estimated_total_energy_difference_kwh": warm_total - baseline_total,
        "estimated_total_energy_savings_pct": (
            100.0 * (baseline_total - warm_total) / baseline_total
            if baseline_total
            else 0.0
        ),
    }


def _write_markdown_report(payload: dict[str, object], output: Path) -> None:
    overall = payload["overall"]
    baseline = payload["comfort_baseline"]
    comparison = payload["energy_comparison"]
    power = payload["power_profile"]
    gate = payload["offline_success_gate"]
    room_rows = []
    for room, room_payload in payload["rooms"].items():
        warm_room = room_payload["warm_started"]
        room_comparison = room_payload["energy_comparison"]
        room_rows.append(
            f"| {room} | {warm_room['observed_comfort_score']:.1f} | "
            f"{warm_room['estimated_fan_energy_kwh']:.3f} | "
            f"{warm_room['estimated_light_energy_kwh']:.3f} | "
            f"{room_comparison['estimated_total_energy_savings_pct']:.1f}% | "
            f"{warm_room['ventilation_level_accuracy_pct']:.1f}% |"
        )

    markdown = f"""# Held-Out Fan and Lighting Evaluation

This report replays the saved room controller on {overall['records']:,} real
observations from held-out whole days. Requested fan and LED actions do not cause
the next dataset row, so energy is an engineering estimate rather than a measured outcome.

## Power Assumption

| Device | Rated power | Model |
|---|---:|---|
| Fan at maximum PWM | {power['fan_max_power_w']:.0f} W | Cubic with PWM fraction |
| LED strip at maximum PWM | {power['light_max_power_w']:.0f} W | Linear with PWM fraction |

## Overall Results

| Metric | Warm-started controller | Comfort baseline |
|---|---:|---:|
| Occupied average fan demand | {overall['requested_occupied_average_fan_pwm']:.1f} PWM | {baseline['requested_occupied_average_fan_pwm']:.1f} PWM |
| Occupied average LED demand | {overall['requested_occupied_average_light_pwm']:.1f} PWM | {baseline['requested_occupied_average_light_pwm']:.1f} PWM |
| Estimated fan energy | {overall['estimated_fan_energy_kwh']:.3f} kWh | {baseline['estimated_fan_energy_kwh']:.3f} kWh |
| Estimated lighting energy | {overall['estimated_light_energy_kwh']:.3f} kWh | {baseline['estimated_light_energy_kwh']:.3f} kWh |
| Estimated combined energy | {overall['estimated_total_energy_kwh']:.3f} kWh | {baseline['estimated_total_energy_kwh']:.3f} kWh |
| Fan energy per occupied hour | {overall['estimated_fan_kwh_per_occupied_hour']:.4f} kWh | {baseline['estimated_fan_kwh_per_occupied_hour']:.4f} kWh |
| Fan runtime | {overall['fan_runtime_hours']:.1f} h | {baseline['fan_runtime_hours']:.1f} h |
| Lighting runtime | {overall['light_runtime_hours']:.1f} h | {baseline['light_runtime_hours']:.1f} h |
| Logged occupied minutes below {payload['comfort_threshold']:.0f}/100 | {overall['discomfort_minutes_below_threshold']:,} | {baseline['discomfort_minutes_below_threshold']:,} |
| Requested-level exact agreement | {overall['requested_ventilation_level_accuracy_pct']:.1f}% | {baseline['requested_ventilation_level_accuracy_pct']:.1f}% |
| Requested-level within one | {overall['requested_ventilation_within_one_level_pct']:.1f}% | {baseline['requested_ventilation_within_one_level_pct']:.1f}% |

- Estimated fan-energy reduction: **{comparison['estimated_fan_energy_savings_pct']:.2f}%**
- Estimated combined-energy reduction: **{comparison['estimated_total_energy_savings_pct']:.2f}%**
- Learned VAV-prior exact accuracy: **{overall['ventilation_level_accuracy_pct']:.1f}%**
- Learned VAV prior within one level: **{overall['ventilation_within_one_level_pct']:.1f}%**
- Comfort-energy offline-Q coverage: **{overall['offline_value_coverage_pct']:.1f}%**
- Offline-Q recommendations changed: **{overall['offline_value_changed_pct']:.1f}%**
- Model-estimated comfort constraints retained: **{overall['offline_model_comfort_retained_pct']:.1f}%**
- Model-estimated recommended ventilation comfort: **{overall['offline_model_recommended_comfort_score']:.1f}/100**
- Logged-state comfort: **{overall['observed_comfort_score']:.1f}/100**

## Offline Comfort-Energy Gate: {gate['status']}

| Requirement | Threshold | Result |
|---|---:|---:|
| Average model-estimated comfort change | >= {gate['thresholds']['minimum_average_comfort_delta']:+.1f} points | {gate['actual']['average_comfort_delta']:+.2f} points |
| Model comfort constraints retained | >= {gate['thresholds']['minimum_comfort_retained_pct']:.0f}% | {gate['actual']['comfort_retained_pct']:.1f}% |
| Estimated fan-energy reduction | >= {gate['thresholds']['minimum_fan_energy_savings_pct']:.0f}% | {gate['actual']['fan_energy_savings_pct']:.2f}% |

## Per Room

| Room | Logged comfort /100 | Fan kWh | Light kWh | Combined savings | VAV prior exact |
|---|---:|---:|---:|---:|---:|
{chr(10).join(room_rows)}

## Interpretation

The learned prior estimates recorded VAV behavior. The final fan request blends
that prior with conservative logged-transition Q-values, occupancy, comfort, and
safety logic, so prior accuracy and final requested-level agreement are reported
separately. Offline comfort retention is a training-model constraint, not a causal
measurement of an unobserved action. VAV position is not physical
fan PWM, so airflow and energy claims require fan calibration or power-meter
measurements after deployment. Lux is observed, but LED PWM is not logged in the
dataset. Lighting therefore uses a feedback controller initially; its causal comfort
and energy performance can be validated only with deployed LED-to-lux transitions.
The logged comfort score and discomfort minutes describe conditions in the fixed
dataset before our actions; they are not controller outcomes.
"""
    output.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fan and lighting control on whole held-out real-room days."
    )
    parser.add_argument("--data", default="data/real_room/observations.csv.gz")
    parser.add_argument("--policy", default="room_policy.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/real_room/latest/evaluation.json")
    parser.add_argument("--fan-max-power-w", type=float, default=40.0)
    parser.add_argument("--light-max-power-w", type=float, default=24.0)
    parser.add_argument("--comfort-threshold", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.comfort_threshold <= 100.0:
        raise SystemExit("--comfort-threshold must be between 0 and 100")
    try:
        power_calibration = ActuatorCalibration(
            fan_max_power_w=args.fan_max_power_w,
            light_max_power_w=args.light_max_power_w,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid power profile: {exc}") from exc

    controller = AdaptiveQLearningController.load(args.policy, seed=args.seed)
    controller.epsilon = 0.0
    metadata = controller.training_metadata
    required_metadata = {
        "dataset_sha256",
        "train_day_ids",
        "evaluation_day_ids",
        "validation_day_ids",
        "rooms",
    }
    missing_metadata = sorted(required_metadata - metadata.keys())
    if missing_metadata:
        raise SystemExit(
            "policy is missing evaluation provenance "
            f"({', '.join(missing_metadata)}); regenerate it with: python main.py"
        )

    data_path = Path(args.data)
    actual_hash = file_sha256(data_path)
    if actual_hash != metadata["dataset_sha256"]:
        raise SystemExit(
            "dataset checksum does not match the policy's training dataset; "
            "run python main.py with this dataset before evaluating"
        )

    training_days = {int(day) for day in metadata["train_day_ids"]}
    validation_days = {int(day) for day in metadata["validation_day_ids"]}
    evaluation_days = {int(day) for day in metadata["evaluation_day_ids"]}
    if (
        not evaluation_days
        or training_days & validation_days
        or training_days & evaluation_days
        or validation_days & evaluation_days
    ):
        raise SystemExit("policy contains an invalid or overlapping whole-day split")
    policy_rooms = {str(room) for room in metadata["rooms"]}
    observations = load_observations(data_path, rooms=policy_rooms)
    available_days = {item.day_id for item in observations}
    missing_days = evaluation_days - available_days
    if missing_days:
        raise SystemExit(f"held-out days are missing from the dataset: {sorted(missing_days)}")
    evaluation = [item for item in observations if item.day_id in evaluation_days]
    if not evaluation:
        raise SystemExit("the policy's held-out split contains no observations")

    room_metrics = defaultdict(_new_metrics)
    baseline_room_metrics = defaultdict(_new_metrics)
    overall_metrics = _new_metrics()
    baseline_overall_metrics = _new_metrics()
    for observation in evaluation:
        decision = controller.decide(observation)
        baseline_decision = controller.baseline.decide(observation)
        for metrics, current_decision, include_prior in (
            (room_metrics[observation.room_id], decision, True),
            (baseline_room_metrics[observation.room_id], baseline_decision, False),
            (overall_metrics, decision, True),
            (baseline_overall_metrics, baseline_decision, False),
        ):
            _record_observation(
                metrics,
                observation,
                current_decision,
                controller,
                power_calibration,
                args.comfort_threshold,
                include_prior,
            )

    overall = _summarize_metrics(overall_metrics, True)
    baseline_overall = _summarize_metrics(baseline_overall_metrics, False)
    energy_comparison = _energy_comparison(overall, baseline_overall)
    gate_thresholds = {
        "minimum_average_comfort_delta": 0.0,
        "minimum_comfort_retained_pct": 95.0,
        "minimum_fan_energy_savings_pct": 5.0,
    }
    gate_actual = {
        "model_comfort": float(
            overall["offline_model_recommended_comfort_score"] or 0.0
        ),
        "behavior_comfort": float(
            overall["offline_model_behavior_comfort_score"] or 0.0
        ),
        "comfort_retained_pct": float(
            overall["offline_model_comfort_retained_pct"]
        ),
        "fan_energy_savings_pct": float(
            energy_comparison["estimated_fan_energy_savings_pct"]
        ),
    }
    gate_actual["average_comfort_delta"] = (
        gate_actual["model_comfort"] - gate_actual["behavior_comfort"]
    )
    gate_passed = (
        gate_actual["average_comfort_delta"]
        >= gate_thresholds["minimum_average_comfort_delta"]
        and gate_actual["comfort_retained_pct"]
        >= gate_thresholds["minimum_comfort_retained_pct"]
        and gate_actual["fan_energy_savings_pct"]
        >= gate_thresholds["minimum_fan_energy_savings_pct"]
    )
    payload = {
        "dataset_sha256": actual_hash,
        "policy_format_version": controller.FORMAT_VERSION,
        "split_seed": metadata.get("split_seed"),
        "training_day_ids": sorted(training_days),
        "validation_day_ids": sorted(validation_days),
        "evaluation_day_ids": sorted(evaluation_days),
        "causal_warning": (
            "Fan and lighting actions are audited against fixed logged observations. "
            "They do not cause the next row, so energy is estimated rather than measured."
        ),
        "power_profile": {
            "fan_max_power_w": power_calibration.fan_max_power_w,
            "fan_model": "cubic with PWM fraction",
            "light_max_power_w": power_calibration.light_max_power_w,
            "light_model": "linear with PWM fraction",
            "reference_url": FAN_POWER_REFERENCE_URL,
        },
        "comfort_threshold": args.comfort_threshold,
        "overall": overall,
        "comfort_baseline": baseline_overall,
        "energy_comparison": energy_comparison,
        "offline_success_gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "thresholds": gate_thresholds,
            "actual": gate_actual,
            "scope": "conservative offline model estimate; hardware validation still required",
        },
        "rooms": {},
    }
    for room, metrics in sorted(room_metrics.items()):
        warm_room = _summarize_metrics(metrics, True)
        baseline_room = _summarize_metrics(baseline_room_metrics[room], False)
        payload["rooms"][room] = {
            "warm_started": warm_room,
            "comfort_baseline": baseline_room,
            "energy_comparison": _energy_comparison(warm_room, baseline_room),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_output = output.with_suffix(".md")
    _write_markdown_report(payload, markdown_output)

    comparison = payload["energy_comparison"]
    print(f"Audited {len(evaluation):,} held-out real observations")
    print(
        f"VAV prior: exact={overall['ventilation_level_accuracy_pct']:.1f}%, "
        f"within one={overall['ventilation_within_one_level_pct']:.1f}%"
    )
    print(
        f"Comfort-energy offline Q: coverage="
        f"{overall['offline_value_coverage_pct']:.1f}%, "
        f"changed={overall['offline_value_changed_pct']:.1f}%, "
        f"model comfort retained={overall['offline_model_comfort_retained_pct']:.1f}%"
    )
    print(
        f"Offline comfort-energy gate: {'PASS' if gate_passed else 'FAIL'} "
        f"(model comfort delta={gate_actual['average_comfort_delta']:+.2f}, "
        f"retained={gate_actual['comfort_retained_pct']:.1f}%, "
        f"fan savings={gate_actual['fan_energy_savings_pct']:.2f}%)"
    )
    print(
        f"Final request agreement: exact="
        f"{overall['requested_ventilation_level_accuracy_pct']:.1f}%, "
        f"within one={overall['requested_ventilation_within_one_level_pct']:.1f}%"
    )
    print(
        f"Occupied fan demand: warm-started="
        f"{overall['requested_occupied_average_fan_pwm']:.1f} PWM, "
        f"baseline={baseline_overall['requested_occupied_average_fan_pwm']:.1f} PWM"
    )
    print(
        f"Estimated fan energy: warm-started="
        f"{overall['estimated_fan_energy_kwh']:.3f} kWh, "
        f"baseline={baseline_overall['estimated_fan_energy_kwh']:.3f} kWh, "
        f"savings={comparison['estimated_fan_energy_savings_pct']:.2f}%"
    )
    print(
        f"Estimated lighting energy: warm-started="
        f"{overall['estimated_light_energy_kwh']:.3f} kWh, "
        f"baseline={baseline_overall['estimated_light_energy_kwh']:.3f} kWh"
    )
    print(
        f"Estimated combined savings: "
        f"{comparison['estimated_total_energy_savings_pct']:.2f}%"
    )
    for room, room_payload in payload["rooms"].items():
        metrics = room_payload["warm_started"]
        room_comparison = room_payload["energy_comparison"]
        print(
            f"{room}: fan={metrics['estimated_fan_energy_kwh']:.3f} kWh, "
            f"light={metrics['estimated_light_energy_kwh']:.3f} kWh, "
            f"combined savings={room_comparison['estimated_total_energy_savings_pct']:.1f}%, "
            f"ventilation exact={metrics['ventilation_level_accuracy_pct']:.1f}%"
        )
    print("This is a held-out decision and energy audit, not a causal rollout.")
    print(f"JSON: {output.resolve()}")
    print(f"Report: {markdown_output.resolve()}")


if __name__ == "__main__":
    main()
