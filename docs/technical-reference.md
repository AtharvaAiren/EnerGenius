# Technical reference and offline evaluation

[Project overview](../README.md) · [Portfolio](https://github.com/AtharvaAiren)

The following is the existing implementation reference and logged-data audit. Energy figures are engineering estimates, not measured closed-loop savings.

# EnerGenius Adaptive Fan and Lighting Control

An occupancy-aware room controller with two direct outputs:

- fan ventilation demand from `0-255` PWM;
- LED lighting demand from `0-255` PWM.

HVAC is not part of the project. The controller observes occupancy, CO2,
temperature, humidity, ambient lux, time, room context, and previous actions.

```text
occupancy + environment + previous actions
                    |
 behavior prior + offline Q + lux feedback
                    |
           fan PWM + LED PWM
                    |
      real next state + user feedback
                    |
             online Q-learning
```

## Control Strategy

### Fan and ventilation

The real dataset contains VAV damper openness. Logged positions are grouped into
five ventilation levels and used to warm-start fan demand:

| VAV position | Level | Representative demand |
|---|---|---:|
| 0% | OFF | 0 |
| 1-20% | LOW | 64 |
| 21-50% | MEDIUM | 128 |
| 51-80% | HIGH | 204 |
| 81-100% | MAXIMUM | 255 |

The learned VAV prior is blended with occupancy, CO2, temperature, conservative
offline Q-values, and safety logic. VAV percentage is not a physical fan calibration.

### Offline comfort-energy pretraining

Whole days are split into 28 training, 7 validation, and 9 untouched final-test
days. From training days, the system builds stable 15-minute logged transitions:

```text
state -> logged VAV level -> future logged state
```

The reward strongly penalizes ventilation discomfort and critical CO2 before
estimated fan energy and switching. Q-values are fitted only for actions actually
observed with sufficient support. At decision time, an offline recommendation may
move at most one VAV category from the behavior prior and must retain modeled
comfort and CO2 outcomes within conservative margins. This is an offline estimate,
not proof that an unobserved command caused a fixed dataset row.

### Lighting

Lux is observed but LED PWM is not logged in the dataset. Lighting therefore
starts as a closed-loop feedback controller:

1. Compare measured lux with the occupant's target, default `500 lux`.
2. Increase or decrease LED PWM using the previous LED command.
3. Hold the command inside a `25 lux` deadband.
4. Force lighting off when the room is empty.

This does not require a fixed lux-per-PWM conversion to operate. The feedback
loop corrects from new BH1750 readings. The default gain controls how quickly it
corrects and should be tuned on the real LED strip.

### Online learning

The residual Q-learning policy can adjust both fan and lighting. It learns only
after an issued action is followed by a genuine new observation. User feedback
can change thermal, humidity, air-quality, and lighting preferences. The fixed
dataset is never treated as though arbitrary fan or LED commands caused its next
row.

The learning session and persistence format are implemented, but
`hardware_demo.py` intentionally runs inference only during initial hardware
testing. Persistent deployment learning still needs action/transition logging,
policy versioning, review, and rollback before it should be enabled on the room.

Safety behavior remains outside RL: empty-room fan and lighting shutoff, high-CO2
ventilation floors, and maximum ventilation for critical CO2.

The serial adapter accepts optional measured CO2 telemetry. Fresh measurements
override the configured fallback automatically; missing or stale CO2 uses the
fallback and is marked as unmeasured in the observation and hardware log.

## Real Dataset

The project uses the CC0 dataset described in *Room-level occupant counts and
environmental quality from heterogeneous sensing modalities in a smart building*:

- [Dataset DOI](https://doi.org/10.6084/m9.figshare.9959417.v1)
- [Paper DOI](https://doi.org/10.1038/s41597-019-0274-4)

It contains 190,080 one-minute records from 44 complete DayIds and three rooms.
Complete days are split into 28 training, 7 validation, and 9 final-test days. The policy
stores the dataset SHA-256 and exact split so evaluation cannot silently use its
training rows.

## Run It

```bash
python -m pip install -r requirements.txt
python prepare_dataset.py
python main.py
python evaluate.py
python demo.py
python -m unittest discover -s tests -v
```

Useful overrides:

```bash
python main.py --fan-max-power-w 40 --target-lux 500 --light-feedback-gain 0.25
python evaluate.py --fan-max-power-w 40 --light-max-power-w 24
python feedback.py --lighting dim --overall -1
```

## Held-Out Results

The checked evaluation audits 38,880 observations from 9 untouched whole days:

| Result | Value |
|---|---:|
| VAV prior exact accuracy | 73.6% |
| VAV prior within one level | 90.4% |
| Stable logged transitions | 22,810 |
| Supported offline Q entries | 1,028 |
| Final-test offline-Q coverage | 91.0% |
| Model-estimated comfort change | +0.76 points |
| Model comfort constraints retained | 99.98% |
| Offline comfort-energy gate | PASS |
| Final fan request within one level | 89.3% |
| Baseline fan request within one level | 73.8% |
| Warm-started occupied fan demand | 90.7 PWM |
| Baseline occupied fan demand | 112.4 PWM |
| Warm-started estimated fan energy | 1.877 kWh |
| Baseline estimated fan energy | 2.440 kWh |
| Estimated fan-energy reduction | 23.08% |
| Estimated lighting energy | 2.525 kWh |
| Estimated combined-energy reduction | 11.34% |
| Online Q-learning updates | 0 |

Fan power assumes a representative `40 W` maximum with a cubic PWM curve.
Lighting assumes a `24 W` LED strip with a linear PWM curve. Replace both values
with nameplate or power-meter measurements before making hardware energy claims.
Lighting energy is identical for the warm-started and baseline controllers because
offline data provides no learned LED action advantage yet.

## Outputs

```text
room_policy.json

reports/real_room/latest/
  pretraining_report.md
  pretraining_report.json
  evaluation.md
  evaluation.json

reports/real_room/demo/
  decision_audit.csv
  decision_audit.png
```

## Architecture

```text
room_control/
  models.py           Observations, dual PWM actions, preferences, feedback
  dataset.py          Real-room loader and whole-day splitting
  controllers.py      Fan baseline, VAV prior, lux feedback, RL, safety
  offline_rl.py       Supported logged-transition comfort-energy Q pretraining
  comfort.py          Comfort, reward, switching, and power estimates
  pretraining.py      Logged-VAV warm start and fixed-data audit
  session.py          Genuine online transition learning
  hardware.py         Sensor and actuator protocols
  serial_hardware.py  Validated ESP32 serial adapter

main.py               Warm-start and save policy format 7
evaluate.py           Held-out action and engineering-energy audit
demo.py               Minute-level CSV and fan/lighting plot
feedback.py           Persist occupant preference feedback
```

## Honest Boundary

The held-out evaluation validates VAV behavior learning and shows what the
controller would request on real recorded conditions. It is not a causal rollout:
requested actions cannot change fixed future dataset rows.

The remaining work is to measure fan airflow and power by PWM, measure LED power,
tune the lux-feedback gain, record real post-action transitions, collect occupant
feedback, connect the online session to a versioned deployment workflow, and
evaluate closed-loop comfort and energy on unseen sessions. Those transitions will
train the implemented online Q-learning layer for the actual room.
