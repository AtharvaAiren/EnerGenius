# EnerGenius Teammate Handoff

Copy the project to the computer connected to the ESP32, but exclude
`ino files/energenius_bh1750/secrets.h` from email, ZIP files, cloud drives, and
Git. The teammate should create that local file from `secrets.example.h`. The
active controller produces two outputs: logical fan ventilation demand and LED
lighting demand, both from `0-255`. HVAC is not used. Because the prototype fan
reportedly stalls below PWM `177`, nonzero
demand is proportionally mapped across the usable physical range `177-255`,
preserving distinct levels. The existing firmware command format is retained for
compatibility, and host mode sends both requested PWM channels.

## Hardware Specification

### Controller

| Item | Specification |
|---|---|
| Board | ESP32 DevKit V1 |
| MCU | Dual-core ESP32 with integrated 2.4 GHz Wi-Fi and Bluetooth |
| Logic level | 3.3 V; never apply a 5 V signal directly to a GPIO |
| Programming | Arduino IDE, board `ESP32 Dev Module`, or `hardware_setup.py` |
| Board power | USB 5 V |
| Serial | USB serial at 115200 baud |

### Sensor Inputs

| Sensor | Measurement | ESP32 connection | Interface | Power |
|---|---|---|---|---|
| BH1750 / GY-302 | Ambient light in lux | SDA `GPIO21`, SCL `GPIO22` | I2C | 3V3 |
| DHT22 / AM2302 | Temperature and humidity | Data `GPIO4` | One-wire digital | 3V3 |
| IR obstacle sensor A | First doorway beam | OUT `GPIO18` | Digital | 3V3 |
| IR obstacle sensor B | Second doorway beam | OUT `GPIO19` | Digital | 3V3 |
| PIR HC-SR501 | Motion and occupancy backup | OUT `GPIO23` | Digital | VIN / 5 V |
| Future CO2 sensor | Carbon dioxide in ppm | Choose after sensor model is known | Sensor-specific | Sensor-specific |

The current firmware has no physical CO2 driver. Python uses fresh CO2 telemetry
when available and otherwise supplies an explicit fallback, `600 ppm` by default.
Terminal output labels every value as `sensor` or `fallback`; never present a
fallback value as a measurement.

### Actuator Outputs

| Actuator | Driver | ESP32 pin | Control | External power |
|---|---|---|---|---|
| 12 V LED strip | IRLZ44N N-channel MOSFET | `GPIO25` | 25 kHz PWM, `0-255` | 12 V |
| 12 V DC fan | IRLZ44N N-channel MOSFET | `GPIO26` | 25 kHz PWM, `0-255` | 12 V |

Each MOSFET gate uses a `220 ohm` series resistor and `10 kohm` pulldown. The
reported fan stall threshold is PWM `177`; firmware gives the fan a startup kick
and Python maps nonzero logical demand into the usable physical range.

IRLZ44N is described as logic-level, but its low-resistance performance is not
guaranteed at every 3.3 V ESP32 gate drive. Check MOSFET temperature under load;
prefer a device explicitly characterized at 2.5 V or 3.3 V for the final build.

### Component List

- ESP32 DevKit V1 x1
- BH1750 light sensor x1
- DHT22 temperature/humidity sensor x1
- IR obstacle sensors x2
- PIR HC-SR501 x1
- IRLZ44N MOSFETs x2
- 220 ohm gate resistors x2
- 10 kohm gate pulldown resistors x2
- 12 V LED strip x1
- 12 V DC fan x1
- 12 V 2 A adapter x1
- Perfboard, female headers, three screw terminals, and USB cable
- Fan flyback diode x1; a fast or Schottky diode is preferred for 25 kHz PWM

The retired relay module and DO-type LDR are not part of the active build. The
BH1750 replaces the LDR, and the fan MOSFET replaces the relay.

### Power Architecture

```text
USB 5 V -> ESP32 -> onboard 3.3 V regulator -> BH1750, DHT22, IR-A, IR-B
ESP32 VIN 5 V -> PIR
12 V adapter positive -> LED strip positive and fan positive
LED/fan negative -> MOSFET drains; MOSFET sources -> common ground
12 V adapter negative + ESP32 GND + USB GND -> one common ground
```

Do not feed 12 V into an ESP32 GPIO or 3V3 pin. The flyback diode goes across the
fan, with its stripe toward fan `+12 V`. A 1N4007 may be present on the perfboard,
but a fast or Schottky flyback diode is the better final choice at this PWM rate.

### Communication and Firmware Modes

- Primary dashboard: ESP32 2.4 GHz Wi-Fi to ThingSpeak using HTTP every 20 seconds.
- ThingSpeak fields: temperature, humidity, people count, occupied, motion,
  LED PWM, applied fan PWM, and lux.
- Local control and host telemetry: USB serial at 115200 baud.
- Bluetooth is available on the ESP32 but unused. MQTT is not implemented.
- `LOCAL` mode runs the firmware fallback rules and is the boot default.
- `HOST` mode accepts Python commands in the form
  `CMD,<sequence>,<led_pwm>,<fan_pwm>,OFF`.
- Legacy telemetry has 15 fields. CO2-enabled firmware appends
  `,<co2_ppm>,<co2_age_ms>`. Python accepts both formats and uses measured CO2 only
  while its age is at most 10 seconds; missing, negative, or stale data falls back.
- `SAFE` mode sets both PWM outputs to zero. Sensor staleness or a five-second host
  command timeout also enters `SAFE` mode.
- The final `OFF` command field is reserved only for wire-format compatibility;
  no third actuator is controlled.

## Before Connecting 12 V

1. Rotate the Wi-Fi password and ThingSpeak write key that were shared previously.
2. Put them in `ino files/energenius_bh1750/secrets.h`. Do not share that file.
3. Confirm USB ground, ESP32 ground, MOSFET sources, and 12 V negative are common.
4. Place the fan flyback diode across the fan, stripe toward `+12 V`.
5. Prefer a fast/Schottky flyback diode and a MOSFET specified for a 3.3 V gate.
6. Keep 12 V disconnected during the initial sensor test.

## Software Setup

From the repository root, first verify the offline artifacts:

```bash
python main.py
python evaluate.py
python demo.py
python -m unittest discover -s tests -v
```

The current reference run uses 28 training, 7 validation, and 9 untouched test
days. It audits 38,880 final-test observations, reports 73.6% exact VAV-level
accuracy, 90.4% within one level, +0.76 model-estimated comfort points, 23.08%
estimated fan-energy reduction, and 11.34% estimated combined reduction. The
offline comfort-energy gate passes, but these remain fixed-data engineering
estimates rather than hardware measurements.

Then install hardware dependencies:

```bash
python -m pip install -r requirements.txt
python hardware_setup.py
python hardware_setup.py --prepare
```

Upload after selecting the detected serial port:

```bash
python hardware_setup.py --upload --port /dev/cu.usbserial-0001
```

On Windows, the port will look like `COM4`. Arduino IDE is also supported: select
**ESP32 Dev Module** and flash only
`ino files/energenius_bh1750/energenius_bh1750.ino`.

## Sensor Verification

Keep 12 V disconnected, close Arduino Serial Monitor, and run:

```bash
python hardware_demo.py --port /dev/cu.usbserial-0001 --observe-only
```

Verify:

- DHT22 temperature and humidity are plausible.
- BH1750 lux should rise when the LED PWM rises and fall when it is reduced.
- PIR motion changes between `0` and `1`.
- Crossing IR-A then IR-B increments people count.
- Crossing IR-B then IR-A decrements people count.

## Actuator Verification

After checking wiring, disconnect power, connect 12 V, and power up again.

Start with the safe baseline:

```bash
python hardware_demo.py \
  --port /dev/cu.usbserial-0001 \
  --controller baseline \
  --room-area 20 \
  --room-volume 60 \
  --seating-capacity 10
```

Then run the warm-started policy:

```bash
python hardware_demo.py \
  --port /dev/cu.usbserial-0001 \
  --controller learned \
  --policy room_policy.json \
  --room-area 20 \
  --room-volume 60 \
  --seating-capacity 10
```

Replace the example room metadata with measurements for the actual test space.
Press `Ctrl+C` after each run; Python sends `MODE,SAFE` before closing.

The default lighting target is `500 lux`. Change it with `--target-lux`. The
controller adjusts LED PWM from repeated BH1750 readings, so verify that lux moves
in the expected direction before leaving host control unattended.

## Current Hardware Gaps

- The CO2 sensor model, wiring, and firmware driver still need to be selected.
  Until then, the explicit fallback is `600 ppm`; override it with `--assumed-co2`
  only for controlled demonstrations.
- The software defaults to the reported fan stall threshold of PWM `177`. Confirm
  that value on this exact fan with `--fan-min-effective-pwm`; the terminal prints
  both logical and applied fan PWM for verification.
- Fan PWM-to-airflow and fan PWM-to-power are not calibrated yet.
- LED PWM-to-power is not calibrated, and the default lux feedback gain still
  needs tuning on the physical strip and room.
- Online RL should remain conservative until real post-action readings and user
  feedback can be recorded safely.
- `hardware_demo.py` currently performs inference and actuator control only; it
  does not persist online Q-learning updates. This avoids modifying the reference
  policy during first hardware tests. Persistent deployment learning still needs
  an explicit logging, review, and rollback workflow.

Do not claim hardware energy savings from a short demonstration. The current
dataset run is a real-data warm start and decision audit, not a closed-loop energy
experiment.
