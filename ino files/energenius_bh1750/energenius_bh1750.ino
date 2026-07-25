/*
  EnerGenius - LOCAL/HOST CONTROL + THINGSPEAK + USB TELEMETRY

  Pins: BH1750 SDA->21 SCL->22 | DHT22->4  IR-A->18  IR-B->19  PIR->23  LED->25  Fan->26
  Needs: "DHT sensor library" by Adafruit  AND  "BH1750" by Christopher Laws.
         (WiFi is built into the ESP32; ThingSpeak is plain HTTP - no library.)

  Control modes:
    LOCAL - standalone rules (boot default)
    HOST  - Python sends CMD lines over USB serial
    SAFE  - both PWM outputs off after MODE,SAFE or a host/sensor timeout

  ThingSpeak field map (8 fields):
    1 Temperature (C)   2 Humidity (%)     3 People count    4 Occupied (1/0)
    5 Motion (1/0)      6 LED PWM (0-255)  7 Fan PWM (0-255) 8 Lux

  Optional CO2 extension for a future sensor:
    Append ,<co2_ppm>,<co2_age_ms> to each TEL line. The current 15-field TEL
    format remains valid; Python uses its configured fallback when these fields
    are absent, negative, or stale.
*/

#include <WiFi.h>
#include <Wire.h>
#include <BH1750.h>
#include "DHT.h"
#include "secrets.h"
#include <stdio.h>
#include <string.h>

// Copy secrets.example.h to secrets.h and put local credentials there.

// ---------------- network timing ----------------
const unsigned long TS_INTERVAL_MS   = 20000UL;
const unsigned long WIFI_ATTEMPT_MS  = 10000UL;
const unsigned long WIFI_RETRY_MS    = 5000UL;
const unsigned long HTTP_TIMEOUT_MS  = 5000UL;

// ---------------- pins ----------------
#define SDA_PIN 21
#define SCL_PIN 22
#define DHT_PIN 4
#define IRA_PIN 18
#define IRB_PIN 19
#define PIR_PIN 23
#define LED_PIN 25
#define FAN_PIN 26
#define DHTTYPE DHT22
DHT dht(DHT_PIN, DHTTYPE);
BH1750 lightMeter;

// ---------------- behaviour switch ----------------
#define ONLY_RUN_WITH_PEOPLE  false

// ---------------- tunable knobs ----------------
// light -> LED (real lux now)
const float LUX_DARK   = 50.0;    // at/below this the room is "dark"  -> LED brightest
const float LUX_BRIGHT = 400.0;   // at/above this the room is "bright" -> LED dimmest
const int LED_MAX  = 180;         // brightness when dark
const int LED_MEAN = 40;          // brightness when bright (set 0 to switch off in daylight)
const int LED_PER_PERSON = 20;

const float TEMP_COMFORT = 28.0;
const float TEMP_HOT     = 35.0;
const int   FAN_MIN      = 40;
const int   FAN_KICK     = 255;
const int   FAN_KICK_MS  = 500;
const int   FAN_PER_PERSON = 15;
const int   FAN_STALL_BELOW = 177;
const unsigned long FAN_REKICK_EVERY_MS = 3000;
// Actuator updates run every 200 ms, so this produces one short full-power pulse.
const unsigned long FAN_REKICK_MS = 200;

const unsigned long MOTION_TIMEOUT = 30000UL;

#define BEAM_BLOCKED LOW

const unsigned long DHT_INTERVAL  = 2000;
const unsigned long CTRL_INTERVAL = 200;
const unsigned long PRINT_INTERVAL = 1000;
const unsigned long TELEMETRY_INTERVAL = 1000;
const unsigned long HOST_TIMEOUT_MS = 5000;
const unsigned long SENSOR_STALE_MS = 10000;

// ---------------- state ----------------
volatile int   peopleCount = 0;
volatile bool  occupied    = false;
volatile float lastLux     = 200.0;    // last good lux (survives a bad read)
volatile float lastTemp    = TEMP_COMFORT;
volatile float lastHum     = 50.0;
volatile int   ledTarget   = 0;
volatile int   fanTarget   = 0;
volatile int   fanAppliedPwm = 0;
volatile unsigned long lastDhtGoodMs = 0;
volatile unsigned long lastLuxGoodMs = 0;

volatile bool iraTripped = false;
volatile bool irbTripped = false;

unsigned long lastCtrlMs = 0, lastDhtMs = 0, lastPrintMs = 0, lastMotionMs = 0;
unsigned long lastTelemetryMs = 0, lastHostCommandMs = 0;
bool  motionEverSeen = false;

enum ControlMode { LOCAL_CONTROL, HOST_CONTROL, SAFE_CONTROL };
ControlMode controlMode = LOCAL_CONTROL;
char reservedRequest[8] = "OFF";
char serialBuffer[96];
size_t serialLength = 0;
unsigned long telemetrySequence = 0;

enum BeamFirst { NONE_FIRST, A_FIRST, B_FIRST };
BeamFirst firstBeam = NONE_FIRST;
bool      secondSeen = false;

int           lastFanTarget = 0;
unsigned long kickUntil = 0;
unsigned long lastRekickMs = 0;

// ---------------- helpers ----------------
int tempToFan(float t) {
  if (t <= TEMP_COMFORT) return 0;
  if (t >= TEMP_HOT)     return 255;
  float frac = (t - TEMP_COMFORT) / (TEMP_HOT - TEMP_COMFORT);
  return FAN_MIN + (int)(frac * (255 - FAN_MIN));
}

// lux -> LED base brightness: bright room = dim, dark room = bright (smooth)
int luxToLed(float lux) {
  if (lux <= LUX_DARK)   return LED_MAX;
  if (lux >= LUX_BRIGHT) return LED_MEAN;
  float frac = (lux - LUX_DARK) / (LUX_BRIGHT - LUX_DARK);   // 0 at dark, 1 at bright
  return LED_MAX - (int)(frac * (LED_MAX - LED_MEAN));
}

void updatePeopleCounter() {
  bool aBlk = (digitalRead(IRA_PIN) == BEAM_BLOCKED);
  bool bBlk = (digitalRead(IRB_PIN) == BEAM_BLOCKED);

  if (aBlk) iraTripped = true;
  if (bBlk) irbTripped = true;

  if (firstBeam == NONE_FIRST) {
    if      (aBlk && !bBlk) firstBeam = A_FIRST;
    else if (bBlk && !aBlk) firstBeam = B_FIRST;
  } else {
    if (aBlk && bBlk) secondSeen = true;
    if (!aBlk && !bBlk) {
      if (secondSeen) {
        if (firstBeam == A_FIRST) {
          peopleCount++;
          Serial.print(">>> ENTER (A then B), count="); Serial.println(peopleCount);
        } else {
          if (peopleCount > 0) peopleCount--;
          Serial.print(">>> EXIT  (B then A), count="); Serial.println(peopleCount);
        }
      }
      firstBeam = NONE_FIRST;
      secondSeen = false;
    }
  }
}

void updateOccupancy() {
  unsigned long now = millis();
  if (digitalRead(PIR_PIN) == HIGH) { lastMotionMs = now; motionEverSeen = true; }
  bool motionRecent = motionEverSeen && (now - lastMotionMs < MOTION_TIMEOUT);
  occupied = (peopleCount > 0) || motionRecent;
}

void readDht() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t) && !isnan(h)) {
    lastTemp = t;
    lastHum = h;
    lastDhtGoodMs = millis();
  }
}

void readLux() {
  float l = lightMeter.readLightLevel();
  if (l >= 0) {
    lastLux = l;
    lastLuxGoodMs = millis();
  }
}

const char* controlModeName() {
  if (controlMode == HOST_CONTROL) return "HOST";
  if (controlMode == SAFE_CONTROL) return "SAFE";
  return "LOCAL";
}

void setSafeOutputs() {
  ledTarget = 0;
  fanTarget = 0;
  strcpy(reservedRequest, "OFF");
}

bool validReservedRequest(const char* value) {
  return strcmp(value, "OFF") == 0;
}

void processSerialCommand(char* line) {
  if (strcmp(line, "PING") == 0) {
    Serial.println("PONG,ENERGENIUS,1");
    return;
  }
  if (strcmp(line, "MODE,LOCAL") == 0) {
    controlMode = LOCAL_CONTROL;
    strcpy(reservedRequest, "OFF");
    Serial.println("ACK,MODE,LOCAL");
    return;
  }
  if (strcmp(line, "MODE,SAFE") == 0) {
    controlMode = SAFE_CONTROL;
    setSafeOutputs();
    Serial.println("ACK,MODE,SAFE");
    return;
  }

  unsigned long sequence = 0;
  int requestedLed = 0;
  int requestedFan = 0;
  char requestedReserved[8] = "OFF";
  int parsed = sscanf(line, "CMD,%lu,%d,%d,%7s", &sequence, &requestedLed, &requestedFan, requestedReserved);
  if (parsed != 4 || requestedLed < 0 || requestedLed > 255 ||
      requestedFan < 0 || requestedFan > 255 || !validReservedRequest(requestedReserved)) {
    Serial.println("ERR,BAD_COMMAND");
    return;
  }

  ledTarget = requestedLed;
  fanTarget = requestedFan;
  strcpy(reservedRequest, requestedReserved);
  lastHostCommandMs = millis();
  controlMode = HOST_CONTROL;
  Serial.print("ACK,"); Serial.print(sequence);
  Serial.print(","); Serial.print(ledTarget);
  Serial.print(","); Serial.print(fanTarget);
  Serial.print(","); Serial.println(reservedRequest);
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char value = (char)Serial.read();
    if (value == '\r') continue;
    if (value == '\n') {
      if (serialLength > 0) {
        serialBuffer[serialLength] = '\0';
        processSerialCommand(serialBuffer);
        serialLength = 0;
      }
      continue;
    }
    if (serialLength < sizeof(serialBuffer) - 1) {
      serialBuffer[serialLength++] = value;
    } else {
      serialLength = 0;
      Serial.println("ERR,COMMAND_TOO_LONG");
    }
  }
}

// ============================================================
//  DECISION LOGIC  <-- ML MODEL PLUGS IN HERE.
// ============================================================
void computeActuators() {
  bool active = ONLY_RUN_WITH_PEOPLE ? (peopleCount > 0) : occupied;

  if (!active) { ledTarget = 0; fanTarget = 0; return; }

  int ledBase = luxToLed(lastLux);
  ledTarget = min(ledBase + peopleCount * LED_PER_PERSON, 255);

  int fanBase = tempToFan(lastTemp);
  int f = min(fanBase + peopleCount * FAN_PER_PERSON, 255);
  if (f > 0 && f < FAN_MIN) f = FAN_MIN;
  fanTarget = f;
}

void applyActuators() {
  analogWrite(LED_PIN, ledTarget);

  unsigned long now = millis();
  if (fanTarget == 0) {
    analogWrite(FAN_PIN, 0);
    fanAppliedPwm = 0;
    lastFanTarget = 0;
    kickUntil = 0;
    return;
  }
  if (lastFanTarget == 0) {
    kickUntil = now + FAN_KICK_MS;
    lastRekickMs = now;
  } else if (fanTarget < FAN_STALL_BELOW && now - lastRekickMs >= FAN_REKICK_EVERY_MS) {
    kickUntil = now + FAN_REKICK_MS;
    lastRekickMs = now;
  }

  fanAppliedPwm = now < kickUntil ? FAN_KICK : fanTarget;
  analogWrite(FAN_PIN, fanAppliedPwm);
  lastFanTarget = fanTarget;
}

void publishTelemetry() {
  unsigned long now = millis();
  unsigned long dhtAge = lastDhtGoodMs == 0 ? 0xFFFFFFFFUL : now - lastDhtGoodMs;
  unsigned long luxAge = lastLuxGoodMs == 0 ? 0xFFFFFFFFUL : now - lastLuxGoodMs;
  String line;
  line.reserve(180);
  line = "TEL,";
  line += String(++telemetrySequence);
  line += ','; line += String(now);
  line += ','; line += String((float)lastTemp, 2);
  line += ','; line += String((float)lastHum, 2);
  line += ','; line += String((float)lastLux, 1);
  line += ','; line += String((int)peopleCount);
  line += ','; line += String(occupied ? 1 : 0);
  line += ','; line += String(digitalRead(PIR_PIN) == HIGH ? 1 : 0);
  line += ','; line += String((int)ledTarget);
  line += ','; line += String((int)fanAppliedPwm);
  line += ','; line += controlModeName();
  line += ','; line += reservedRequest;
  line += ','; line += String(dhtAge);
  line += ','; line += String(luxAge);
  Serial.println(line);
}

void printStatus() {
  Serial.print("A:");  Serial.print(digitalRead(IRA_PIN) == BEAM_BLOCKED ? "BLK" : "clr");
  Serial.print(" B:"); Serial.print(digitalRead(IRB_PIN) == BEAM_BLOCKED ? "BLK" : "clr");
  Serial.print(" Motion:"); Serial.print(digitalRead(PIR_PIN) == HIGH ? "YES" : "no ");
  Serial.print(" | Lux:"); Serial.print(lastLux, 0);
  Serial.print(" Temp:"); Serial.print(lastTemp, 1); Serial.print("C");
  Serial.print(" Hum:");  Serial.print(lastHum, 1);  Serial.print("%");
  Serial.print(" | People:"); Serial.print(peopleCount);
  Serial.print(" Occ:"); Serial.print(occupied ? "yes" : "no");
  Serial.print(" Mode:"); Serial.print(controlModeName());
  Serial.print("  =>  LED:"); Serial.print(ledTarget);
  Serial.print(" FAN target:"); Serial.print(fanTarget);
  Serial.print(" applied:"); Serial.print(fanAppliedPwm);
  Serial.print(" Reserved:"); Serial.println(reservedRequest);
}

// ============================================================
//  THINGSPEAK - plain HTTP, on core 0, never blocks the control loop
// ============================================================
int tsSend(const String& query) {
  WiFiClient c;
  if (!c.connect("api.thingspeak.com", 80)) return -1;

  c.print("GET /update?" + query + " HTTP/1.1\r\n"
          "Host: api.thingspeak.com\r\n"
          "Connection: close\r\n\r\n");

  unsigned long t0 = millis();
  while (!c.available()) {
    if (millis() - t0 > HTTP_TIMEOUT_MS) { c.stop(); return -2; }
    vTaskDelay(20 / portTICK_PERIOD_MS);
  }

  String resp;
  while (c.available() && resp.length() < 600) resp += (char)c.read();
  c.stop();

  int split = resp.indexOf("\r\n\r\n");
  if (split < 0) return -3;
  String body = resp.substring(split + 4);
  body.trim();
  return body.toInt();
}

void thingSpeakTask(void* pv) {
  WiFi.mode(WIFI_STA);

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[TS] WiFi: connecting...");
      WiFi.begin(WIFI_SSID, WIFI_PASS);
      unsigned long t0 = millis();
      while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_ATTEMPT_MS)
        vTaskDelay(250 / portTICK_PERIOD_MS);

      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[TS] WiFi: failed - retrying (control loop unaffected)");
        WiFi.disconnect();
        vTaskDelay(WIFI_RETRY_MS / portTICK_PERIOD_MS);
        continue;
      }
      Serial.print("[TS] WiFi: connected, IP "); Serial.println(WiFi.localIP());
    }

    String q = "api_key=" + String(TS_WRITE_KEY);
    q += "&field1=" + String((float)lastTemp, 1);
    q += "&field2=" + String((float)lastHum, 1);
    q += "&field3=" + String((int)peopleCount);
    q += "&field4=" + String(occupied ? 1 : 0);
    q += "&field5=" + String(digitalRead(PIR_PIN) == HIGH ? 1 : 0);
    q += "&field6=" + String((int)ledTarget);
    q += "&field7=" + String((int)fanAppliedPwm);
    q += "&field8=" + String((float)lastLux, 0);

    int r = tsSend(q);
    if (r > 0) {
      Serial.print("[TS] uploaded ok, entry #"); Serial.println(r);
      iraTripped = false;
      irbTripped = false;
    } else if (r == 0) {
      Serial.println("[TS] rejected - wrong API key, or sending faster than 15s");
    } else {
      Serial.print("[TS] network problem, code "); Serial.println(r);
    }

    vTaskDelay(TS_INTERVAL_MS / portTICK_PERIOD_MS);
  }
}

// ---------------- setup / loop ----------------
void setup() {
  Serial.begin(115200);
  delay(400);

  pinMode(IRA_PIN, INPUT);
  pinMode(IRB_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  analogWriteFrequency(LED_PIN, 25000);
  analogWriteFrequency(FAN_PIN, 25000);
  analogWrite(LED_PIN, 0);
  analogWrite(FAN_PIN, 0);
  dht.begin();

  Wire.begin(SDA_PIN, SCL_PIN);
  if (lightMeter.begin()) Serial.println("BH1750 OK");
  else                    Serial.println("BH1750 NOT found - check SDA/SCL/power");

  xTaskCreatePinnedToCore(thingSpeakTask, "thingspeak", 8192, NULL, 1, NULL, 0);

  Serial.println();
  Serial.println("=== EnerGenius control loop running (LOCAL mode) ===");
}

void loop() {
  unsigned long now = millis();
  readSerialCommands();
  updatePeopleCounter();
  if (now - lastDhtMs  >= DHT_INTERVAL)  { lastDhtMs  = now; readDht(); readLux(); }
  if (now - lastCtrlMs >= CTRL_INTERVAL) {
    lastCtrlMs = now;
    updateOccupancy();
    if (controlMode == LOCAL_CONTROL) {
      computeActuators();
    } else if (controlMode == HOST_CONTROL) {
      bool commandExpired = now - lastHostCommandMs > HOST_TIMEOUT_MS;
      bool sensorsReady = lastDhtGoodMs != 0 && lastLuxGoodMs != 0;
      bool sensorsFresh = sensorsReady && now - lastDhtGoodMs <= SENSOR_STALE_MS &&
                          now - lastLuxGoodMs <= SENSOR_STALE_MS;
      if (commandExpired || !sensorsFresh) {
        controlMode = SAFE_CONTROL;
        setSafeOutputs();
        Serial.println(commandExpired ? "EVENT,HOST_TIMEOUT,SAFE" : "EVENT,SENSOR_STALE,SAFE");
      } else if (!occupied) {
        setSafeOutputs();
      }
    } else {
      setSafeOutputs();
    }
    applyActuators();
  }
  if (now - lastPrintMs >= PRINT_INTERVAL) { lastPrintMs = now; printStatus(); }
  if (now - lastTelemetryMs >= TELEMETRY_INTERVAL) {
    lastTelemetryMs = now;
    publishTelemetry();
  }
  delay(5);
}
