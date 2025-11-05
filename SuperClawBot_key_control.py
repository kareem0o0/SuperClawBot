import serial
from pynput import keyboard
import time
import threading

# ---------------- CONFIG ----------------
PORT = "/dev/rfcomm0"
BAUD = 9600

# ---------- KEY MAP ----------
DRIVE = {
    keyboard.Key.up:    'F',   # now backward
    keyboard.Key.down:  'B',   # now forward
    keyboard.Key.left:  'L',   # left turn
    keyboard.Key.right: 'R',   # right turn
}

ARM1 = {'1': 'A', '4': 'Z'}          # reversed
ARM2 = {'3': 'S', '6': 'X'}          # reversed
ARM3 = {'0': 'C', '2': 'V'}          # NEW

STOP_DRIVE = '0'
STOP_ARM1  = 'a'
STOP_ARM2  = 's'
STOP_ARM3  = 'c'
TOGGLE_LED = 'Q'

# ---------------------------------------
class RobotController:
    def __init__(self):
        self.bt = None
        self.active = {
            'drive': None,
            'arm1' : None,
            'arm2' : None,
            'arm3' : None,
        }
        self.lock = threading.Lock()

    # ---- connection ----
    def connect(self):
        try:
            self.bt = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            print("\n=== ROBOT READY ===")
            print("Arrows: move | 1/4: arm1 | 3/6: arm2 | 0/2: arm3 | Q: LED | ESC: quit")
            return True
        except Exception as e:
            print("BT error:", e)
            return False

    # ---- safe send ----
    def send(self, cmd):
        with self.lock:
            if self.bt and self.bt.is_open:
                self.bt.write(cmd.encode())
                print(f"→ {cmd}")

    # ---- press ----
    def on_press(self, key):
        try:
            char = key.char if hasattr(key, 'char') else None
        except:
            char = None

        updated = False

        # ---- DRIVE ----
        if key in DRIVE:
            cmd = DRIVE[key]
            if self.active['drive'] != cmd:
                self.send(cmd)
                self.active['drive'] = cmd
                updated = True

        # ---- ARM 1 ----
        elif char in ARM1:
            cmd = ARM1[char]
            if self.active['arm1'] != cmd:
                self.send(cmd)
                self.active['arm1'] = cmd
                updated = True

        # ---- ARM 2 ----
        elif char in ARM2:
            cmd = ARM2[char]
            if self.active['arm2'] != cmd:
                self.send(cmd)
                self.active['arm2'] = cmd
                updated = True

        # ---- ARM 3 (NEW) ----
        elif char in ARM3:
            cmd = ARM3[char]
            if self.active['arm3'] != cmd:
                self.send(cmd)
                self.active['arm3'] = cmd
                updated = True

        # ---- LED TOGGLE ----
        elif key == keyboard.Key.space:          # optional: space to toggle
            self.send(TOGGLE_LED)

        # ---- Q = LED ----
        elif char and char.lower() == 'q':
            self.send(TOGGLE_LED)

        if key == keyboard.Key.esc:
            self.stop_all()
            return False

        return True

    # ---- release ----
    def on_release(self, key):
        try:
            char = key.char if hasattr(key, 'char') else None
        except:
            char = None

        # ---- DRIVE ----
        if key in DRIVE and self.active['drive'] == DRIVE[key]:
            self.send(STOP_DRIVE)
            self.active['drive'] = None

        # ---- ARM 1 ----
        if char in ARM1 and self.active['arm1'] == ARM1[char]:
            self.send(STOP_ARM1)
            self.active['arm1'] = None

        # ---- ARM 2 ----
        if char in ARM2 and self.active['arm2'] == ARM2[char]:
            self.send(STOP_ARM2)
            self.active['arm2'] = None

        # ---- ARM 3 ----
        if char in ARM3 and self.active['arm3'] == ARM3[char]:
            self.send(STOP_ARM3)
            self.active['arm3'] = None

    # ---- full stop ----
    def stop_all(self):
        self.send('!')
        for k in self.active: self.active[k] = None

    # ---- run ----
    def run(self):
        if not self.connect(): return
        with keyboard.Listener(on_press=self.on_press,
                               on_release=self.on_release) as listener:
            listener.join()
        self.bt.close()
        print("Disconnected.")

# =============== MAIN ===============
if __name__ == "__main__":
    RobotController().run()