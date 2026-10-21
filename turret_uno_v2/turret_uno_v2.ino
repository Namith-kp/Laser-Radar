// Laser turret - Arduino Uno + CNC Shield V3, 2 stepper motors (A4988)   -- VERSION 2 (laser safety)
// Library: AccelStepper (Library Manager -> "AccelStepper")
//
// Serial commands (115200 baud, newline-terminated):
//   P<deg>,T<deg>,F<0|1>   e.g. P30,T-10,F1   pan angle, tilt angle, laser request
//   H                      go home (0,0) and laser off
//   X                      emergency stop (drivers disabled, laser + buzzer off)
// Reply: "OK" after each command.
//
// LASER SAFETY BUILT INTO THIS FIRMWARE (independent of the PC program):
//   1. The laser is OFF at power-up and after any reset.
//   2. Failsafe: if no command arrives for 200 ms the laser turns off.  (the PC must send at least every 100 ms while firing)
//   3. The buzzer sounds whenever the laser is requested, and the laser only switches on after a
//      WARN_MS warning beep, so anybody nearby hears it first.
//   4. Maximum continuous burn MAX_ON_MS, then a forced COOLDOWN_MS pause.
//   5. HARDWARE (do this too): a pull-down resistor (10k) from the MOSFET gate to GND so the laser stays off
//      while the Uno is resetting, and a physical kill switch in the laser's power line.

#include <AccelStepper.h>

// ---------- PINS (change to match your wiring) ----------
const uint8_t PAN_STEP = 2, PAN_DIR = 5;     // CNC shield X slot
const uint8_t TILT_STEP = 3, TILT_DIR = 6;   // CNC shield Y slot
const uint8_t EN_PIN = 8;       // CNC shield enable. REMOVE the EN/GND jumper on the shield first!
const uint8_t LASER_PIN = 12;   // shield 'SpnEn' pin -> 220 ohm -> gate of a logic-level MOSFET (never drive the laser from the pin)
const uint8_t BUZZER_PIN = A3;  // shield 'CoolEn' pin -> active buzzer (+). Sounds whenever the laser is requested/on

// ---------- MECHANICS (calibrate these!) ----------
const float PAN_STEPS_PER_DEG  = 8.889;  // 200 steps x 16 microsteps / 360 (A4988, all 3 MS jumpers)
const float TILT_STEPS_PER_DEG = 8.889;

const float PAN_MIN = -90,  PAN_MAX = 90;    // soft limits (degrees)
const float TILT_MIN = -30, TILT_MAX = 45;

const float MAX_SPEED = 3000;   // steps/sec cap
const float ACCEL     = 5000;   // steps/sec^2

// ---------- LASER SAFETY ----------
const unsigned long LASER_TIMEOUT_MS = 200;   // no command for this long -> laser off
const unsigned long WARN_MS          = 250;   // buzzer-only warning before the laser really switches on
const unsigned long MAX_ON_MS        = 3000;  // longest continuous burn
const unsigned long COOLDOWN_MS      = 1500;  // forced pause after MAX_ON_MS

AccelStepper pan(AccelStepper::DRIVER, PAN_STEP, PAN_DIR);
AccelStepper tilt(AccelStepper::DRIVER, TILT_STEP, TILT_DIR);

unsigned long lastCmd = 0;
bool laserWanted = false;       // PC is asking for the laser (buzzer sounds)
bool laserOn = false;           // laser really on
unsigned long wantSince = 0, onSince = 0, cooldownUntil = 0;

void requestLaser(bool want) {
  unsigned long now = millis();
  if (want && (long)(cooldownUntil - now) > 0) want = false;   // still cooling down
  if (want && !laserWanted) wantSince = now;
  laserWanted = want;
}

void updateLaser() {
  unsigned long now = millis();
  if (laserWanted && now - lastCmd > LASER_TIMEOUT_MS) laserWanted = false;   // failsafe
  bool on = laserWanted && (now - wantSince >= WARN_MS);
  if (on && !laserOn) onSince = now;
  if (on && now - onSince > MAX_ON_MS) {                                        // burn limit
    laserWanted = false; on = false; cooldownUntil = now + COOLDOWN_MS;
  }
  laserOn = on;
  digitalWrite(LASER_PIN, on ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, laserWanted ? HIGH : LOW);
}

void laserOffNow() {
  laserWanted = false; laserOn = false;
  digitalWrite(LASER_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

void goTo(float panDeg, float tiltDeg) {
  panDeg = constrain(panDeg, PAN_MIN, PAN_MAX);
  tiltDeg = constrain(tiltDeg, TILT_MIN, TILT_MAX);
  pan.moveTo((long)(panDeg * PAN_STEPS_PER_DEG));
  tilt.moveTo((long)(tiltDeg * TILT_STEPS_PER_DEG));
}

void handleLine(String s) {
  s.trim();
  if (s.length() == 0) return;
  lastCmd = millis();

  if (s[0] == 'X') {
    laserOffNow();
    digitalWrite(EN_PIN, HIGH);          // disable drivers
    Serial.println("OK");
    return;
  }
  digitalWrite(EN_PIN, LOW);             // (re)enable drivers

  if (s[0] == 'H') {
    laserOffNow();
    goTo(0, 0);
    Serial.println("OK");
    return;
  }

  // Parse P..,T..,F.. (missing fields keep their current value)
  float p = pan.targetPosition() / PAN_STEPS_PER_DEG;
  float t = tilt.targetPosition() / TILT_STEPS_PER_DEG;
  int f = laserWanted ? 1 : 0;

  int start = 0;
  while (start < (int)s.length()) {
    int comma = s.indexOf(',', start);
    if (comma < 0) comma = s.length();
    String tok = s.substring(start, comma);
    tok.trim();
    if (tok.length() > 1) {
      float v = tok.substring(1).toFloat();
      if (tok[0] == 'P') p = v;
      else if (tok[0] == 'T') t = v;
      else if (tok[0] == 'F') f = (int)v;
    }
    start = comma + 1;
  }
  goTo(p, t);
  requestLaser(f == 1);
  Serial.println("OK");
}

void setup() {
  pinMode(LASER_PIN, OUTPUT);  digitalWrite(LASER_PIN, LOW);     // laser OFF first of all
  pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);

  pan.setMaxSpeed(MAX_SPEED);   pan.setAcceleration(ACCEL);
  tilt.setMaxSpeed(MAX_SPEED);  tilt.setAcceleration(ACCEL);
  pan.setCurrentPosition(0);    tilt.setCurrentPosition(0);

  Serial.begin(115200);
  Serial.println("READY");
}

String buf;

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (buf.length() < 40) buf += c;
  }

  pan.run();
  tilt.run();
  updateLaser();
}
