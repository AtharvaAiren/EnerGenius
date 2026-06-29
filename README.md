# Smart Home RL

An experimental intelligent room-control system for learning how to manage indoor comfort and energy use.

The project simulates a room with synthetic occupants, environmental dynamics, and controllable devices. A reinforcement-learning controller learns actions for fan, lighting, and HVAC control from generated human-centered trajectories.

## Project Goal

Build toward a miniature intelligent building operating system:

```text
synthetic users -> behavior -> environment -> observations -> decisions -> actions -> learning
```

The long-term vision is a system that can eventually connect to real sensors, real actuators, and real user feedback while preserving the same control loop.

## What It Models

The simulator includes:

- Synthetic occupants with recurring schedules and diverse preferences
- Office-like weekday occupancy patterns
- Evening and weekend usage
- Seasonal outdoor temperature and daylight changes
- Room temperature, humidity, ambient light, and CO2
- Fan, lighting, and HVAC actions
- Comfort and energy tradeoffs

Occupants can differ by:

- temperature preference
- humidity preference
- lighting preference
- energy sensitivity
- discomfort tolerance
- crowding sensitivity
- activity level
- schedule and attendance patterns

## Repository Structure

```text
main.py                  Train the RL controller
evaluate.py              Compare learned controller against baseline
q_table.json             Saved tabular Q-learning policy

simulator/
  models.py              Domain models: observations, actions, preferences, results
  population.py          Synthetic occupants, archetypes, schedules, preferences
  environment.py         Room dynamics driven by synthetic population snapshots
  metrics.py             Energy, comfort, and reward calculations
  controllers.py         Baseline controller and Q-learning controller
  experiment.py          Training and evaluation loops
  trajectories.py        Synthetic trajectory generation for inspection/logging
```

## Run It

From the project root:

```bash
python main.py
```

This trains the Q-learning controller and writes the learned policy to:

```text
q_table.json
```

Then evaluate it:

```bash
python evaluate.py
```

Useful options:

```bash
python main.py --days 45 --seed 7 --output q_table.json
python evaluate.py --days 7 --seed 123 --policy q_table.json
```

## Generate Sample Trajectories

You can inspect generated synthetic room behavior from Python:

```bash
python -c "from simulator.trajectories import generate_trajectory; [print(row.as_dict()) for row in generate_trajectory(days=1, seed=42, stride_steps=36)]"
```

This prints sampled rows containing occupancy, archetype mix, environmental state, aggregate preferences, selected action, comfort, energy, and reward.

## Current Learning Approach

The current RL layer uses tabular Q-learning. It observes discretized room state including:

- weekday vs weekend
- season
- time of day
- occupancy
- temperature
- humidity
- ambient light
- CO2
- aggregate active-user preferences
- previous action

It chooses among combinations of:

- fan: off, low, high
- light: off, low, high
- HVAC: off, cool, heat

A baseline controller remains available for comparison and safety fallback.

## Important Metrics

`total_energy_wh` is total simulated energy usage in watt-hours.

Example:

```text
60 W for 1 hour = 60 Wh
1000 Wh = 1 kWh
```

The reward currently balances comfort against energy cost, weighted by the active occupants' energy sensitivity.

## Current Limitations

This is still a research prototype.

Remaining assumptions include:

- occupant archetypes are handcrafted
- room physics are simplified
- comfort is modeled, not learned from real feedback
- weather and seasonal effects are synthetic
- there are no hardware sensor or actuator adapters yet
- tabular Q-learning will not scale well to richer real-world control

## Next Improvements

Good next steps:

- Add richer evaluation metrics such as discomfort minutes, energy per occupied hour, HVAC runtime, and per-archetype comfort
- Use a comfort-first constrained reward
- Train and evaluate across many seeds
- Add action-switching penalties for realistic device behavior
- Add occupancy and thermal prediction
- Replace tabular Q-learning with a stronger method such as DQN, fitted Q-iteration, or model-predictive control
- Add sensor and actuator interfaces for hardware deployment

## Design Direction

The simulator should keep moving away from arbitrary random states and toward believable synthetic human environments. Simulation is a temporary stand-in for reality, but the architecture should make that swap natural:

```text
synthetic occupants today
real occupancy and feedback tomorrow
```
