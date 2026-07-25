# Real-Room Fan and Lighting Warm Start

## Scope

This run uses only the SDU real-room sensing dataset and whole-day
train/validation/final-test splits.

| Item | Result |
|---|---:|
| Records | 190,080 |
| Training records | 120,960 |
| Validation records | 30,240 |
| Held-out records | 38,880 |
| Training whole days | 28 |
| Validation whole days | 7 |
| Held-out whole days | 9 |
| Ventilation prior states | 163 |
| Ventilation prior coverage | 100.0% |
| Exact ventilation-level accuracy | 73.6% |
| Within one ventilation level | 90.4% |
| Representative-PWM MAE | 25.6 PWM |
| Stable logged transitions | 22,810 |
| Offline Q states | 625 |
| Offline Q entries | 1,028 |
| Validation recommendation coverage | 86.8% |
| Validation recommendations changed | 24.9% |
| Occupied held-out records | 20,101 |
| Logged-state comfort score | 61.5/100 |
| Requested average fan | 49.2 PWM |
| Requested occupied fan | 90.8 PWM |
| Requested average light | 41.3 PWM |
| Requested occupied light | 80.0 PWM |
| Online RL updates | 0 |

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
