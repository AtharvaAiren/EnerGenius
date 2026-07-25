# Held-Out Fan and Lighting Evaluation

This report replays the saved room controller on 38,880 real
observations from held-out whole days. Requested fan and LED actions do not cause
the next dataset row, so energy is an engineering estimate rather than a measured outcome.

## Power Assumption

| Device | Rated power | Model |
|---|---:|---|
| Fan at maximum PWM | 40 W | Cubic with PWM fraction |
| LED strip at maximum PWM | 24 W | Linear with PWM fraction |

## Overall Results

| Metric | Warm-started controller | Comfort baseline |
|---|---:|---:|
| Occupied average fan demand | 90.7 PWM | 112.4 PWM |
| Occupied average LED demand | 80.1 PWM | 80.1 PWM |
| Estimated fan energy | 1.877 kWh | 2.440 kWh |
| Estimated lighting energy | 2.525 kWh | 2.525 kWh |
| Estimated combined energy | 4.402 kWh | 4.965 kWh |
| Fan energy per occupied hour | 0.0049 kWh | 0.0066 kWh |
| Fan runtime | 340.8 h | 340.8 h |
| Lighting runtime | 316.3 h | 316.3 h |
| Logged occupied minutes below 80/100 | 17,920 | 17,920 |
| Requested-level exact agreement | 58.0% | 56.6% |
| Requested-level within one | 89.3% | 73.8% |

- Estimated fan-energy reduction: **23.08%**
- Estimated combined-energy reduction: **11.34%**
- Learned VAV-prior exact accuracy: **73.6%**
- Learned VAV prior within one level: **90.4%**
- Comfort-energy offline-Q coverage: **91.0%**
- Offline-Q recommendations changed: **26.6%**
- Model-estimated comfort constraints retained: **100.0%**
- Model-estimated recommended ventilation comfort: **70.3/100**
- Logged-state comfort: **61.5/100**

## Offline Comfort-Energy Gate: PASS

| Requirement | Threshold | Result |
|---|---:|---:|
| Average model-estimated comfort change | >= +0.0 points | +0.76 points |
| Model comfort constraints retained | >= 95% | 100.0% |
| Estimated fan-energy reduction | >= 5% | 23.08% |

## Per Room

| Room | Logged comfort /100 | Fan kWh | Light kWh | Combined savings | VAV prior exact |
|---|---:|---:|---:|---:|---:|
| room_1 | 69.5 | 0.857 | 0.613 | 13.7% | 73.0% |
| room_2 | 61.2 | 0.315 | 0.992 | 13.1% | 75.2% |
| room_3 | 54.4 | 0.706 | 0.920 | 7.6% | 72.5% |

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
