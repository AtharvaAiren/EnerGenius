# Real-Room Dataset

Source: [Room-level occupant counts and environmental quality from heterogeneous
sensing modalities in a smart building](https://doi.org/10.6084/m9.figshare.9959417.v1)

- License: CC0
- Paper: https://doi.org/10.1038/s41597-019-0274-4
- Rooms: three
- Days: 44 nonconsecutive full days
- Normalized interval: one minute
- Expected normalized rows: 190,080

Create `observations.csv.gz` and its provenance metadata from the verified public
archive:

```bash
python prepare_dataset.py
```

The normalized schema is:

```text
room_id, room_type, area_m2, volume_m3, seating_capacity,
year, month, day_id, minute_of_day, is_workday, occupancy,
temperature_c, humidity_pct, co2_ppm, illuminance_lux, vav_damper_pct
```

The source dates were privacy-suppressed. `day_id` values are randomized and days
must not be treated as one continuous calendar trajectory. Training and evaluation
split complete DayIds to prevent minute-level leakage.

The normalized file uses the source archive's `filleddata` variant, which was
forward/backward-filled by the dataset authors. Treat repeated or imputed values as
real dataset limitations, not new independent sensor evidence.

The source VAV damper position is a logged ventilation action proxy. It is not fan
PWM. Illuminance is a measured state, not a logged LED action. The dataset contains
no physical fan PWM, LED PWM, device energy, or user comfort votes.

The active warm start groups VAV position into `OFF` (0%), `LOW` (1-20%),
`MEDIUM` (21-50%), `HIGH` (51-80%), and `MAXIMUM` (81-100%).
