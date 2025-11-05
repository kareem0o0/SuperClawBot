# robot_voice_control.py
# FINAL: NO OVERLAP + STOP WORKS + EXIT STOPS + READY AFTER STOP

import serial
from pynput import keyboard
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import threading
import os

# ---------------- CONFIG ----------------
PORT = "/dev/rfcomm0"
BAUD = 9600
MODEL = "sound_classifier/soundclassifier_with_metadata.tflite"
LABELS = "sound_classifier/labels.txt"
RATE = 44100
OVERLAP = 0.5
ACTION_DURATION = 3.0   # CHANGE THIS TO CONTROL DURATION

# ---------- KEY MAP ----------
DRIVE = {keyboard.Key.up: 'F', keyboard.Key.down: 'B', keyboard.Key.left: 'L', keyboard.Key.right: 'R'}
ARM1 = {'1': 'A', '4': 'Z'}
ARM2 = {'3': 'S', '6': 'X'}
ARM3 = {'0': 'C', '2': 'V'}
STOP_DRIVE = '0'
STOP_ARM1  = 'a'
STOP_ARM2  = 's'
STOP_ARM3  = 'c'
TOGGLE_LED = 'Q'

# ---------- VOICE COMMANDS ----------
VOICE_CMDS = {
    "forward":    ('F', STOP_DRIVE),
    "backward":   ('B', STOP_DRIVE),
    "left":       ('L', STOP_DRIVE),
    "right":      ('R', STOP_DRIVE),
    "up":         ('Z', STOP_ARM1),
    "down":       ('A', STOP_ARM1),
    "2up":        ('S', STOP_ARM2),
    "2down":      ('X', STOP_ARM2),
    "clockwise":  ('C', STOP_ARM3),
    "anti":       ('V', STOP_ARM3),
    "stop":       ('!', None),
}

# ---------------------------------------
class RobotController:
    def __init__(self):
        self.bt = None
        self.active = {'drive': None, 'arm1': None, 'arm2': None, 'arm3': None}
        self.lock = threading.Lock()
        self.voice_mode = False
        self.current_action = None
        self.stop_event = threading.Event()
        self.action_lock = threading.Lock()

        # Load model
        self.interp = tflite.Interpreter(MODEL)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.samples = self.inp['shape'][1]

        with open(LABELS) as f:
            self.labels = [line.strip().split()[-1] for line in f]

        self.audio_buf = np.zeros(self.samples, dtype=np.float32)
        self.pos = 0

    def connect(self):
        try:
            self.bt = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            print("\n=== ROBOT CONNECTED ===")
            print("Arrows: move | 1/4: arm1 | 3/6: arm2 | 0/2: arm3 | Q: LED | 'a': Voice Mode | ESC: quit")
            return True
        except Exception as e:
            print("BT error:", e)
            return False

    def send(self, cmd):
        with self.lock:
            if self.bt and self.bt.is_open:
                self.bt.write(cmd.encode())
                print(f"→ {cmd}")

    def stop_all(self):
        self.send('!')
        self.active = {k: None for k in self.active}
        self.stop_event.set()

    # =============== KEYBOARD ===============
    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.stop_all()
            return False
        try: char = key.char.lower() if hasattr(key, 'char') else None
        except: char = None

        if char == 'a':
            self.toggle_voice_mode()
            return True

        if self.voice_mode: return True

        if key in DRIVE:
            cmd = DRIVE[key]
            if self.active['drive'] != cmd:
                self.send(cmd)
                self.active['drive'] = cmd
        elif char in ARM1:
            cmd = ARM1[char]
            if self.active['arm1'] != cmd:
                self.send(cmd)
                self.active['arm1'] = cmd
        elif char in ARM2:
            cmd = ARM2[char]
            if self.active['arm2'] != cmd:
                self.send(cmd)
                self.active['arm2'] = cmd
        elif char in ARM3:
            cmd = ARM3[char]
            if self.active['arm3'] != cmd:
                self.send(cmd)
                self.active['arm3'] = cmd
        elif char == 'q':
            self.send(TOGGLE_LED)
        return True

    def on_release(self, key):
        if self.voice_mode: return
        try: char = key.char if hasattr(key, 'char') else None
        except: char = None
        if key in DRIVE and self.active['drive'] == DRIVE[key]:
            self.send(STOP_DRIVE)
            self.active['drive'] = None
        if char in ARM1 and self.active['arm1'] == ARM1[char]:
            self.send(STOP_ARM1)
            self.active['arm1'] = None
        if char in ARM2 and self.active['arm2'] == ARM2[char]:
            self.send(STOP_ARM2)
            self.active['arm2'] = None
        if char in ARM3 and self.active['arm3'] == ARM3[char]:
            self.send(STOP_ARM3)
            self.active['arm3'] = None

    # =============== VOICE MODE ===============
    def toggle_voice_mode(self):
        self.voice_mode = not self.voice_mode
        self.stop_event.set()
        
        if self.voice_mode:
            # ENTER VOICE MODE
            print("\n🔊 VOICE MODE ON")
            self.start_voice_stream()
        else:
            # EXIT VOICE MODE → STOP EVERYTHING
            print("\n🔇 VOICE MODE OFF")
            self.stop_all()  # ← SENDS '!' + releases lock
            self.stop_voice_stream()

    def start_voice_stream(self):
        self.audio_buf = np.zeros(self.samples, dtype=np.float32)
        self.pos = 0
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype='float32',
                                    blocksize=int(RATE*0.1), callback=self.voice_callback)
        self.stream.start()

    def stop_voice_stream(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def voice_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        n = min(len(chunk), self.samples - self.pos)
        self.audio_buf[self.pos:self.pos+n] = chunk[:n]
        self.pos += n
        if self.pos >= self.samples:
            audio = self.audio_buf.copy()
            maxv = np.max(np.abs(audio))
            if maxv > 0: audio /= maxv
            x = audio.reshape(1, self.samples).astype(np.float32)
            self.interp.set_tensor(self.inp['index'], x)
            self.interp.invoke()
            scores = self.interp.get_tensor(self.out['index'])[0]
            i = np.argmax(scores)
            label = self.labels[i]
            conf = scores[i]

            if conf > 0.7 and label in VOICE_CMDS:
                print(f"VOICE: {label} ({conf:.1%})")

                # ========== STOP COMMAND ==========
                if label == "stop":
                    self.stop_event.set()
                    self.send('!')
                    # FORCE RELEASE LOCK
                    if self.action_lock.locked():
                        self.action_lock.release()
                    self.stop_event = threading.Event()
                    print("🛑 STOPPED — READY FOR COMMANDS")
                    return

                # ========== OTHER COMMANDS ==========
                # Check if robot is busy
                if self.action_lock.locked():
                    print(f"⏳ IGNORED: '{label}' — robot busy")
                    return

                # Start new action
                self.action_lock.acquire()
                self.stop_event.set()
                self.stop_event = threading.Event()
                cmd, stop_cmd = VOICE_CMDS[label]
                self.current_action = threading.Thread(
                    target=self.run_action, args=(cmd, stop_cmd)
                )
                self.current_action.start()

            # Slide buffer
            shift = int(self.samples * (1 - OVERLAP))
            self.audio_buf[:shift] = self.audio_buf[-shift:]
            self.pos = shift

    def run_action(self, cmd, stop_cmd):
        self.send(cmd)
        if self.stop_event.wait(ACTION_DURATION):
            pass  # Interrupted
        else:
            if stop_cmd:
                self.send(stop_cmd)
        # RELEASE LOCK WHEN FINISHED
        if self.action_lock.locked():
            self.action_lock.release()

    # =============== RUN ===============
    def run(self):
        if not self.connect(): return
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all()
            if self.voice_mode: self.stop_voice_stream()
            self.listener.stop()
            self.bt.close()
            print("\nDisconnected.")

# =============== MAIN ===============
if __name__ == "__main__":
    if not os.path.exists(MODEL): print(f"Model not found: {MODEL}"); exit(1)
    if not os.path.exists(LABELS): print(f"Labels not found: {LABELS}"); exit(1)
    RobotController().run()