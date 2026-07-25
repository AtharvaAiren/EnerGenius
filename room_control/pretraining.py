from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from room_control.comfort import assess_comfort
from room_control.controllers import (
    AdaptiveQLearningController,
    VENTILATION_LEVELS,
    ventilation_level,
)
from room_control.dataset import (
    observations_for_days,
    split_train_validation_test_day_ids,
    summarize,
)
from room_control.models import RoomObservation


@dataclass(frozen=True)
class PretrainingReport:
    dataset_records: int
    train_records: int
    validation_records: int
    evaluation_records: int
    train_day_ids: tuple[int, ...]
    validation_day_ids: tuple[int, ...]
    evaluation_day_ids: tuple[int, ...]
    rooms: tuple[str, ...]
    ventilation_prior_states: int
    ventilation_prior_samples: int
    ventilation_prior_coverage_pct: float
    ventilation_level_accuracy_pct: float
    ventilation_within_one_level_pct: float
    ventilation_pwm_equivalent_mae: float
    offline_transition_count: int
    offline_q_states: int
    offline_q_entries: int
    offline_validation_coverage_pct: float
    offline_validation_changed_pct: float
    occupied_evaluation_records: int
    observed_comfort_score: float
    requested_average_fan_pwm: float
    requested_occupied_average_fan_pwm: float
    requested_average_light_pwm: float
    requested_occupied_average_light_pwm: float
    q_entries: int
    online_updates: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def pretrain_and_audit(
    controller: AdaptiveQLearningController,
    observations: list[RoomObservation],
    development_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    seed: int = 42,
    audit_stride: int = 5,
) -> PretrainingReport:
    train_days, validation_days, evaluation_days = split_train_validation_test_day_ids(
        observations,
        development_fraction=development_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    training = observations_for_days(observations, train_days)
    validation = observations_for_days(observations, validation_days)
    evaluation = observations_for_days(observations, evaluation_days)
    controller.pretrain_ventilation(training)
    controller.pretrain_offline_values(training)

    offline_validation_candidates = 0
    offline_validation_predictions = 0
    offline_validation_changes = 0
    for index, observation in enumerate(validation):
        if index % audit_stride or observation.occupancy == 0:
            continue
        behavior_level = controller.ventilation_prior.predict_level(observation)
        if behavior_level is None:
            continue
        offline_validation_candidates += 1
        recommendation = controller.offline_ventilation_q.predict_level(
            observation,
            controller.preferences,
            behavior_level,
        )
        if recommendation is None:
            continue
        offline_validation_predictions += 1
        offline_validation_changes += int(recommendation != behavior_level)

    absolute_errors: list[float] = []
    predicted = 0
    exact_level_matches = 0
    within_one_level = 0
    for observation in evaluation:
        predicted_level = controller.ventilation_prior.predict_level(observation)
        prediction_pwm = controller.ventilation_prior.predict(observation)
        if (
            predicted_level is None
            or prediction_pwm is None
            or observation.logged_ventilation_pct is None
        ):
            continue
        predicted += 1
        actual_level = ventilation_level(observation.logged_ventilation_pct)
        exact_level_matches += int(predicted_level == actual_level)
        within_one_level += int(
            abs(
                VENTILATION_LEVELS.index(predicted_level)
                - VENTILATION_LEVELS.index(actual_level)
            )
            <= 1
        )
        logged_pwm_equivalent = observation.logged_ventilation_pct * 2.55
        absolute_errors.append(abs(prediction_pwm - logged_pwm_equivalent))

    fan_pwms: list[int] = []
    occupied_fan_pwms: list[int] = []
    light_pwms: list[int] = []
    occupied_light_pwms: list[int] = []
    comfort_scores: list[float] = []
    occupied_evaluation = 0
    for index, observation in enumerate(evaluation):
        if observation.occupancy > 0:
            occupied_evaluation += 1
            comfort_scores.append(assess_comfort(observation, controller.preferences).overall)
        if index % audit_stride:
            continue
        decision = controller.decide(observation)
        fan_pwms.append(decision.action.fan_pwm)
        light_pwms.append(decision.action.light_pwm)
        if observation.occupancy > 0:
            occupied_fan_pwms.append(decision.action.fan_pwm)
            occupied_light_pwms.append(decision.action.light_pwm)

    summary = summarize(observations)
    return PretrainingReport(
        dataset_records=summary.records,
        train_records=len(training),
        validation_records=len(validation),
        evaluation_records=len(evaluation),
        train_day_ids=tuple(sorted(train_days)),
        validation_day_ids=tuple(sorted(validation_days)),
        evaluation_day_ids=tuple(sorted(evaluation_days)),
        rooms=summary.rooms,
        ventilation_prior_states=controller.ventilation_prior.states,
        ventilation_prior_samples=controller.ventilation_prior.samples,
        ventilation_prior_coverage_pct=100.0 * predicted / max(len(evaluation), 1),
        ventilation_level_accuracy_pct=100.0 * exact_level_matches / max(predicted, 1),
        ventilation_within_one_level_pct=100.0 * within_one_level / max(predicted, 1),
        ventilation_pwm_equivalent_mae=fmean(absolute_errors) if absolute_errors else 0.0,
        offline_transition_count=controller.offline_ventilation_q.transition_count,
        offline_q_states=controller.offline_ventilation_q.states,
        offline_q_entries=controller.offline_ventilation_q.q_entries,
        offline_validation_coverage_pct=(
            100.0
            * offline_validation_predictions
            / max(offline_validation_candidates, 1)
        ),
        offline_validation_changed_pct=(
            100.0
            * offline_validation_changes
            / max(offline_validation_predictions, 1)
        ),
        occupied_evaluation_records=occupied_evaluation,
        observed_comfort_score=fmean(comfort_scores) if comfort_scores else 100.0,
        requested_average_fan_pwm=fmean(fan_pwms) if fan_pwms else 0.0,
        requested_occupied_average_fan_pwm=(
            fmean(occupied_fan_pwms) if occupied_fan_pwms else 0.0
        ),
        requested_average_light_pwm=fmean(light_pwms) if light_pwms else 0.0,
        requested_occupied_average_light_pwm=(
            fmean(occupied_light_pwms) if occupied_light_pwms else 0.0
        ),
        q_entries=sum(value != 0.0 for value in controller.q.values()),
        online_updates=controller.online_updates,
    )


def write_pretraining_report(
    report: PretrainingReport,
    controller: AdaptiveQLearningController,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "pretraining_report.json"
    markdown_path = directory / "pretraining_report.md"
    payload = {
        "report": report.as_dict(),
        "preferences": asdict(controller.preferences),
        "training_metadata": controller.training_metadata,
        "interpretation": {
            "ventilation": (
                "VAV positions were grouped as OFF, LOW, MEDIUM, HIGH, or MAXIMUM. "
                "Each category has a representative 0..255 PWM target for initialization only; "
                "this is not a physical fan calibration."
            ),
            "comfort": "Observed comfort describes logged sensor conditions before our requested actions.",
            "actions": (
                "Requested fan and LED values are an audit on fixed logged observations. "
                "They do not change the next dataset row and therefore are not a causal evaluation."
            ),
            "rl": (
                "Conservative offline Q-values are initialized only from supported logged VAV "
                "transitions. The separate residual Q-table remains untrained until real actions "
                "produce real next states; user feedback can then update preferences and online values."
            ),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown = f"""# Real-Room Fan and Lighting Warm Start

## Scope

This run uses only the SDU real-room sensing dataset and whole-day
train/validation/final-test splits.

| Item | Result |
|---|---:|
| Records | {report.dataset_records:,} |
| Training records | {report.train_records:,} |
| Validation records | {report.validation_records:,} |
| Held-out records | {report.evaluation_records:,} |
| Training whole days | {len(report.train_day_ids)} |
| Validation whole days | {len(report.validation_day_ids)} |
| Held-out whole days | {len(report.evaluation_day_ids)} |
| Ventilation prior states | {report.ventilation_prior_states:,} |
| Ventilation prior coverage | {report.ventilation_prior_coverage_pct:.1f}% |
| Exact ventilation-level accuracy | {report.ventilation_level_accuracy_pct:.1f}% |
| Within one ventilation level | {report.ventilation_within_one_level_pct:.1f}% |
| Representative-PWM MAE | {report.ventilation_pwm_equivalent_mae:.1f} PWM |
| Stable logged transitions | {report.offline_transition_count:,} |
| Offline Q states | {report.offline_q_states:,} |
| Offline Q entries | {report.offline_q_entries:,} |
| Validation recommendation coverage | {report.offline_validation_coverage_pct:.1f}% |
| Validation recommendations changed | {report.offline_validation_changed_pct:.1f}% |
| Occupied held-out records | {report.occupied_evaluation_records:,} |
| Logged-state comfort score | {report.observed_comfort_score:.1f}/100 |
| Requested average fan | {report.requested_average_fan_pwm:.1f} PWM |
| Requested occupied fan | {report.requested_occupied_average_fan_pwm:.1f} PWM |
| Requested average light | {report.requested_average_light_pwm:.1f} PWM |
| Requested occupied light | {report.requested_occupied_average_light_pwm:.1f} PWM |
| Online RL updates | {report.online_updates} |

## Interpretation

- The only logged control signal is VAV damper openness. Exact zero is OFF, 1-20% is
  LOW, 21-50% is MEDIUM, 51-80% is HIGH, and 81-100% is MAXIMUM. Representative
  PWM targets initialize fan demand but are not a physical airflow calibration.
- Offline Q-values use only stable logged VAV windows. Candidate levels require
  sufficient support, remain within one level of recorded behavior, and must retain
  observed ventilation comfort and CO2 outcomes within conservative margins.
- The dataset does not contain physical fan PWM, LED PWM, device energy, or user
  comfort votes. It cannot establish counterfactual results for unobserved actions.
- Lux is an observation, not a logged LED action. Lighting starts with closed-loop
  lux feedback and can learn only after deployed LED actions produce new lux observations.
- The logged sensor rows are never treated as next states caused by arbitrary actions from
  our controller.
- The separate residual online Q-table requires real post-action transitions. The current
  hardware demo is inference-only; persistent online learning still needs a logged deployment
  workflow. Explicit user feedback can update the stored preference profile separately.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path, json_path
