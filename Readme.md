# SuperClawBot

SuperClawBot is a robotic project that can be controlled using multiple modes, including keyboard input and hand gesture recognition.

---

## Project Structure

* **`SuperClawBot_keyboard_control.py`**
  Control the robot's movement and arm functions using the keyboard.

* **`SuperClawBot_gesture_key_control.py`**
  Extends keyboard control with gesture-based commands. Hold **SPACE** to activate gesture mode.

* **`UART_arduino_com.py`**
  Utility script for direct UART communication with the Arduino, useful for testing and debugging servo motors.

* **`arduino_control.ino`**
  Arduino sketch running on the robot to interpret commands received via Bluetooth or UART.

* **`gesture_classifier/`**
  TensorFlow Lite model and labels for hand gesture recognition.

* **`sound_classifier/`**
  TensorFlow Lite model and labels for sound classification.

* **`up_down vision model/`**
  TensorFlow Lite model for up/down vision classification.

* **`requirements.txt`**
  Python dependencies required for the project.

---

## Setup

### 1. Python Dependencies

Ensure Python 3 is installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Bluetooth Setup

Pair and connect your computer to the robot's Bluetooth module (e.g., HC-05 or HC-06).
Refer to **`Bluetooth_Setup.txt`** for detailed instructions. The default port is `/dev/rfcomm0`.

---

## Running the Scripts

### Keyboard Control

Run the robot using only the keyboard:

```bash
python SuperClawBot_keyboard_control.py
```

**Controls:**

* **Arrow Up/Down**: Move forward/backward
* **Arrow Left/Right**: Turn left/right
* **1 / 4**: Control Arm 1
* **3 / 6**: Control Arm 2
* **0 / 2**: Control Arm 3
* **Q**: Toggle LED
* **ESC**: Quit program

---

### Gesture + Keyboard Control

Adds gesture commands to keyboard control:

```bash
python SuperClawBot_gesture_key_control.py
```

**Controls:**

* All keyboard controls from the previous script are available
* **Hold SPACE**: Activate gesture mode

  * **"start" gesture**: Begin movement
  * **"stop" gesture**: Stop movement and toggle next start direction

---

### UART Communication

Directly control servo motors through Arduino:

```bash
python UART_arduino_com.py
```

* Enter an angle (0-180°) to move the servo motor.

---

This README is now concise, structured, and easy to copy/paste.

I can also make a **“minimal setup instructions” version** for sharing with others, which focuses just on installing the venv and running scripts — would you like me to do that?
