# EnerGenius

**Adaptive fan and lighting control using occupancy, environmental sensing and occupant preferences.**

EnerGenius explores how a low-cost room controller can adapt its behaviour as conditions and user preferences change. It combines a logged-data ventilation prior, conservative offline Q-learning, lighting feedback and a residual online-learning layer.

**Status:** Ongoing development. Hardware integration and closed-loop evaluation remain in progress. The system controls fan and LED demand; HVAC is outside its scope.

[Portfolio](https://github.com/AtharvaAiren) · [Technical reference](docs/technical-reference.md) · [Hardware handoff](TEAMMATE_HANDOFF.md)

## What is implemented

- A room-observation model covering occupancy, temperature, humidity, CO2, ambient light and previous actions.
- A ventilation warm start from logged VAV positions, combined with supported-action offline Q-values and explicit safety logic.
- Closed-loop lighting feedback using measured lux and occupant targets.
- Preference feedback and an online Q-learning session that requires genuine post-action observations.
- An ESP32 serial adapter, policy persistence, evaluation scripts and unit checks.

The hardware demo currently runs inference only. Persistent deployment learning still needs action/transition logging and a versioned deployment workflow. Logged VAV positions provide a behaviour prior, not a measured calibration between fan PWM and airflow.

## Architecture

```text
Occupancy + environment + previous actions
                    |
    Ventilation prior + offline Q + lux feedback
                    |
            Fan PWM + LED PWM
                    |
       New observations + preference feedback
                    |
         Online-learning session
```

| Component | Entry point |
| --- | --- |
| Data preparation and whole-day splits | [prepare_dataset.py](prepare_dataset.py), [room_control/dataset.py](room_control/dataset.py) |
| Control policies and safety behaviour | [room_control/controllers.py](room_control/controllers.py) |
| Offline learning | [room_control/offline_rl.py](room_control/offline_rl.py) |
| Online transition learning | [room_control/session.py](room_control/session.py) |
| Hardware telemetry and commands | [room_control/serial_hardware.py](room_control/serial_hardware.py) |
| Held-out audit and plots | [evaluate.py](evaluate.py), [demo.py](demo.py) |

## Run locally

From a fresh checkout, create a Python environment and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_dataset.py
python main.py
python evaluate.py
python demo.py
```

Run the existing checks with:

```bash
python -m unittest discover -s tests -v
```

See the [technical reference](docs/technical-reference.md) for parameter overrides, dataset details, output files and hardware assumptions.

## Data and evaluation

The controller uses a public room-occupancy and environmental-quality dataset: [dataset DOI](https://doi.org/10.6084/m9.figshare.9959417.v1), [data paper](https://doi.org/10.1038/s41597-019-0274-4).

Offline reports are retained in [reports/real_room/latest/](reports/real_room/latest/). They describe logged-data checks and engineering estimates. Real-world comfort and energy performance remain under evaluation; provisional figures are documented in the technical reference rather than presented as deployment results.
