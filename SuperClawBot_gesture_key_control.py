import serial
import cv2
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
from pynput import keyboard
import threading
import time

# ---------------- CONFIG ----------------
PORT = "/dev/rfcomm0"
BAUD = 9600

# ---------- KEY MAP ----------
DRIVE = {
    keyboard.Key.up:    'F',
    keyboard.Key.down:  'B',
    keyboard.Key.left:  'L',
    keyboard.Key.right: 'R',
}

ARM1 = {'1': 'A', '4': 'Z'}
ARM2 = {'3': 'S', '6': 'X'}
ARM3 = {'0': 'C', '2': 'V'}

STOP_DRIVE = '0'
STOP_ARM1  = 'a'
STOP_ARM2  = 's'
STOP_ARM3  = 'c'
TOGGLE_LED = 'Q'

# ---------- MODEL SETUP ----------
interpreter = tflite.Interpreter(model_path="gesture_classifier/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("gesture_classifier/labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# ---------- AUTO FIND CAMERA ----------
def find_camera(max_tries=5):
    for i in range(max_tries):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index: {i}")
            return cap
        cap.release()
    return None

# Try to get camera
camera = find_camera()
if not camera:
    print("WARNING: No camera found! Gesture mode will be DISABLED.")
else:
    print(f"Camera ready: {camera.get(3)}x{camera.get(4)}")

# ---------- ROBOT CONTROLLER ----------
class RobotController:
    def __init__(self, camera):
        self.bt = None
        self.active = {'drive': None, 'arm1': None, 'arm2': None, 'arm3': None}
        self.lock = threading.Lock()

        # Vision state
        self.camera = camera  # <-- NOW SAVED IN INSTANCE
        self.space_held = False
        self.gesture_mode = False
        self.direction_toggle = True  # True = start → forward
        self.last_gesture = None
        self.last_gesture_time = 0
        self.vision_thread = None
        self.running = True
        self.drive_from_gesture = False

    def connect(self):
        try:
            self.bt = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            print("\n=== ROBOT READY ===")
            print("Arrows: move | 1/4: arm1 | 3/6: arm2 | 0/2: arm3 | Q: LED")
            if self.camera:
                print("Hold SPACE → use hand gestures: 'start' = move, 'stop' = stop + toggle")
            else:
                print("No camera → gesture mode DISABLED")
            return True
        except Exception as e:
            print("BT error:", e)
            return False

    def send(self, cmd):
        with self.lock:
            if self.bt and self.bt.is_open:
                self.bt.write(cmd.encode())
                print(f"Sent: {cmd}")

    # ---------------- VISION THREAD ----------------
    def vision_loop(self):
        if not self.camera:
            print("No camera → vision thread skipped")
            return

        print("Vision thread started...")
        while self.running:
            if not self.space_held:
                time.sleep(0.05)
                continue

            ret, frame = self.camera.read()
            if not ret or frame is None:
                print("Camera read failed! Retrying...")
                time.sleep(0.1)
                continue

            # Preprocess
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                image_array = np.asarray(image, dtype=np.float32)
                normalized_image = (image_array / 127.5) - 1
                input_data = np.expand_dims(normalized_image, axis=0)
            except Exception as e:
                print(f"Preprocess error: {e}")
                continue

            # Inference
            try:
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = prediction[index]
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            # Display
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mode = "BACKWARD" if not self.direction_toggle else "FORWARD"
            cv2.putText(frame, f"Next: {mode}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Hold SPACE + Gesture", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("Gesture Control (Hold SPACE)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            # === GESTURE LOGIC ===
            if confidence < 0.7:
                continue

            current_time = time.time()
            if current_time - self.last_gesture_time < 1.0:
                continue

            if class_name == "start" and self.last_gesture != "start":
                cmd = 'B' if self.direction_toggle else 'F'
                self.send(cmd)
                print(f"GESTURE: start → {'FORWARD' if self.direction_toggle else 'BACKWARD'}")
                self.active['drive'] = cmd
                self.drive_from_gesture = True
                self.last_gesture = "start"
                self.last_gesture_time = current_time

            elif class_name == "stop" and self.last_gesture != "stop":
                self.send(STOP_DRIVE)
                print("GESTURE: stop → STOP + TOGGLE")
                self.direction_toggle = not self.direction_toggle
                self.active['drive'] = None
                self.drive_from_gesture = False
                self.last_gesture = "stop"
                self.last_gesture_time = current_time

        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Vision thread ended.")

    # ---------------- KEYBOARD ----------------
    def on_press(self, key):
        try:
            char = key.char if hasattr(key, 'char') else None
        except:
            char = None

        if key == keyboard.Key.space:
            if not self.space_held and self.camera:
                self.space_held = True
                self.gesture_mode = True
                self.drive_from_gesture = False
                self.last_gesture = None
                self.direction_toggle = True  # Reset on new hold
                self.last_gesture_time = 0
                print("SPACE HELD → GESTURE MODE ON")
            return

        if not self.space_held:
            if key in DRIVE and self.active['drive'] != DRIVE[key]:
                self.send(DRIVE[key])
                self.active['drive'] = DRIVE[key]
                self.drive_from_gesture = False
            elif char in ARM1 and self.active['arm1'] != ARM1[char]:
                self.send(ARM1[char]); self.active['arm1'] = ARM1[char]
            elif char in ARM2 and self.active['arm2'] != ARM2[char]:
                self.send(ARM2[char]); self.active['arm2'] = ARM2[char]
            elif char in ARM3 and self.active['arm3'] != ARM3[char]:
                self.send(ARM3[char]); self.active['arm3'] = ARM3[char]
            elif char and char.lower() == 'q':
                self.send(TOGGLE_LED)

        if key == keyboard.Key.esc:
            self.stop_all()
            return False

    def on_release(self, key):
        try:
            char = key.char if hasattr(key, 'char') else None
        except:
            char = None

        if key == keyboard.Key.space:
            if self.space_held:
                self.space_held = False
                self.gesture_mode = False
                print("SPACE RELEASED → GESTURE MODE OFF")
                if self.drive_from_gesture and self.active['drive'] in ['F', 'B']:
                    self.send(STOP_DRIVE)
                    print("STOPPING GESTURE MOVEMENT")
                    self.active['drive'] = None
                    self.drive_from_gesture = False
                self.last_gesture = None
            return

        if not self.space_held:
            if key in DRIVE and self.active['drive'] == DRIVE[key]:
                self.send(STOP_DRIVE); self.active['drive'] = None
            if char in ARM1 and self.active['arm1'] == ARM1[char]:
                self.send(STOP_ARM1); self.active['arm1'] = None
            if char in ARM2 and self.active['arm2'] == ARM2[char]:
                self.send(STOP_ARM2); self.active['arm2'] = None
            if char in ARM3 and self.active['arm3'] == ARM3[char]:
                self.send(STOP_ARM3); self.active['arm3'] = None

    def stop_all(self):
        self.send('!')
        self.active = {k: None for k in self.active}
        self.space_held = False
        self.gesture_mode = False
        self.drive_from_gesture = False

    def run(self):
        if not self.connect():
            return

        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.vision_thread.start()

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

        self.running = False
        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1)
        if self.bt and self.bt.is_open:
            self.bt.close()
        print("Disconnected.")


# =============== MAIN ===============
if __name__ == "__main__":
    controller = RobotController(camera)  # <-- PASS CAMERA HERE
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.stop_all()