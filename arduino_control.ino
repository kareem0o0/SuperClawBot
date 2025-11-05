#include <Servo.h>

Servo motorLeft;   // pin 3
Servo motorRight;  // pin 5
Servo armMotor1;   // pin 6
Servo armMotor2;   // pin 9
Servo armMotor3;   // **NEW** pin 10

const int ledPin = 13;

// state flags (so we can stop only what is moving)
bool driving   = false;
bool arm1      = false;
bool arm2      = false;
bool arm3      = false;

void setup() {
  Serial.begin(9600);

  motorLeft .attach(3);
  motorRight.attach(5);
  armMotor1 .attach(6);
  armMotor2 .attach(9);
  armMotor3 .attach(10);
  pinMode(ledPin, OUTPUT);

  stopAll();
  delay(3000);
  while (Serial.available()) Serial.read();
  Serial.println("Robot Ready");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    switch (c) {
      /* ---------- DRIVE (reversed) ---------- */
      case 'F': drive(-1, -1); driving = true; break;   // was forward to now backward
      case 'B': drive( 1,  1); driving = true; break;   // was backward to now forward
      case 'L': drive( 1, -1); driving = true; break;   // left turn (right motor forward, left backward)
      case 'R': drive(-1,  1); driving = true; break;   // right turn
      case '0': stopDrive();   driving = false; break;

      /* ---------- ARM 1 (reversed) ---------- */
      case 'A': moveArm1(-1); arm1 = true; break;   // was forward to now backward
      case 'Z': moveArm1( 1); arm1 = true; break;
      case 'a': stopArm1();   arm1 = false; break;

      /* ---------- ARM 2 (reversed) ---------- */
      case 'S': moveArm2(-1); arm2 = true; break;
      case 'X': moveArm2( 1); arm2 = true; break;
      case 's': stopArm2();   arm2 = false; break;

      /* ---------- ARM 3 – NEW (reversed) ---------- */
      case 'C': moveArm3(-1); arm3 = true; break;   // forward (2)
      case 'V': moveArm3( 1); arm3 = true; break;   // backward (5)
      case 'c': stopArm3();   arm3 = false; break;

      /* ---------- LED TOGGLE ---------- */
      case 'Q': digitalWrite(ledPin, !digitalRead(ledPin)); break;

      /* ---------- FULL STOP ---------- */
      case '!': stopAll(); break;
    }
  }
}

/* ----------------- HELPERS ----------------- */
void drive(int left, int right) {               // +/-1  to  60-120 range
  motorLeft .write(90 + left  * 30);
  motorRight.write(90 + right * 30);
}
void moveArm1(int dir) { armMotor1.write(90 + dir * 30); }
void moveArm2(int dir) { armMotor2.write(90 + dir * 30); }
void moveArm3(int dir) { armMotor3.write(90 + dir * 30); }

void stopDrive() { motorLeft.write(90); motorRight.write(90); }
void stopArm1()  { armMotor1.write(90); }
void stopArm2()  { armMotor2.write(90); }
void stopArm3()  { armMotor3.write(90); }

void stopAll() {
  stopDrive(); stopArm1(); stopArm2(); stopArm3();
  driving = arm1 = arm2 = arm3 = false;
}