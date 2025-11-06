"""
Unified Robot Controller with Three Modes:
1. Keyboard Control (default)
2. Voice Control (toggle with 'a' key)
3. Gesture Control (toggle with spacebar)

Author: Unified Control System
Version: 1.0
"""

import serial
import cv2
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
from pynput import keyboard
import sounddevice as sd
import threading
import time
import os

# ============================================================
#                    CONFIGURATION SECTION
# ============================================================

# Bluetooth Configuration
PORT = "/dev/rfcomm0"
BAUD = 9600

# Voice Control Configuration
VOICE_MODEL = "sound_classifier/soundclassifier_with_metadata.tflite"
VOICE_LABELS = "sound_classifier/labels.txt"
VOICE_SAMPLE_RATE = 44100
VOICE_OVERLAP = 0.5
VOICE_CONFIDENCE_THRESHOLD = 0.7

# Gesture Control Configuration
GESTURE_MODEL = "gesture_classifier/model.tflite"
GESTURE_LABELS = "gesture_classifier/labels.txt"
GESTURE_CONFIDENCE_THRESHOLD = 0.7
GESTURE_COOLDOWN = 1.0  # Seconds between gesture commands

# Action Timing Configuration
COMMAND_DURATION = 3.0  # Duration for voice commands (seconds)
COOLDOWN_TIME = 1.0     # Cooldown after stop command (seconds)

# ============================================================
#                    COMMAND MAPPINGS
# ============================================================

# Keyboard mappings
DRIVE_KEYS = {
    keyboard.Key.up: 'F',
    keyboard.Key.down: 'B',
    keyboard.Key.left: 'L',
    keyboard.Key.right: 'R',
}

ARM1_KEYS = {'1': 'A', '4': 'Z'}
ARM2_KEYS = {'3': 'S', '6': 'X'}
ARM3_KEYS = {'0': 'C', '2': 'V'}

# Stop commands
STOP_DRIVE = '0'
STOP_ARM1 = 'a'
STOP_ARM2 = 's'
STOP_ARM3 = 'c'
STOP_ALL = '!'
TOGGLE_LED = 'Q'

# TODO: Update this dictionary when new voice commands are added
# Format: "command_name": (start_command, stop_command)
VOICE_COMMANDS = {
    "forward": ('F', STOP_DRIVE),
    "backward": ('B', STOP_DRIVE),
    "left": ('L', STOP_DRIVE),
    "right": ('R', STOP_DRIVE),
    "up": ('Z', STOP_ARM1),
    "down": ('A', STOP_ARM1),
    "2up": ('S', STOP_ARM2),
    "2down": ('X', STOP_ARM2),
    "clockwise": ('C', STOP_ARM3),
    "anti": ('V', STOP_ARM3),
    "stop": (STOP_ALL, None),
}

# TODO: Update this dictionary when new gesture commands are added
# Format: "gesture_name": command_character
# Note: "start" gesture uses forward/backward toggle logic
# Note: "stop" gesture stops movement and toggles direction
GESTURE_COMMANDS = {
    "start": None,  # Special handling with direction toggle
    "stop": STOP_DRIVE,
}


# ============================================================
#                    UTILITY FUNCTIONS
# ============================================================

def find_camera(max_tries=5):
    """Auto-detect available camera."""
    for i in range(max_tries):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✓ Camera found at index: {i}")
            return cap
        cap.release()
    return None


# ============================================================
#                    MAIN ROBOT CONTROLLER
# ============================================================

class UnifiedRobotController:
    """
    Unified controller supporting keyboard, voice, and gesture control modes.
    """
    
    def __init__(self):
        # Bluetooth connection
        self.bt = None
        self.bt_lock = threading.Lock()
        
        # Active command tracking (for keyboard mode)
        self.active_cmds = {
            'drive': None,
            'arm1': None,
            'arm2': None,
            'arm3': None
        }
        
        # Mode control
        self.current_mode = "KEYBOARD"  # KEYBOARD, VOICE, or GESTURE
        self.mode_lock = threading.Lock()
        self.running = True
        
        # Voice mode components
        self.voice_active = False
        self.voice_stream = None
        self.voice_interpreter = None
        self.voice_labels = []
        self.voice_buffer = None
        self.voice_position = 0
        self.voice_action_lock = threading.Lock()
        self.voice_stop_event = threading.Event()
        self.voice_cooldown_active = False
        
        # Gesture mode components
        self.gesture_active = False
        self.gesture_thread = None
        self.camera = None
        self.gesture_interpreter = None
        self.gesture_labels = []
        self.gesture_direction_forward = True  # True = forward, False = backward
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_driving = False
        
        # Initialize models
        self._init_voice_model()
        self._init_gesture_model()
        self._init_camera()
    
    # ========================================================
    #                  INITIALIZATION
    # ========================================================
    
    def _init_voice_model(self):
        """Initialize voice recognition model."""
        if not os.path.exists(VOICE_MODEL):
            print(f"⚠ Voice model not found: {VOICE_MODEL}")
            print("  Voice control will be DISABLED")
            return
        
        try:
            self.voice_interpreter = tflite.Interpreter(VOICE_MODEL)
            self.voice_interpreter.allocate_tensors()
            
            inp = self.voice_interpreter.get_input_details()[0]
            samples = inp['shape'][1]
            self.voice_buffer = np.zeros(samples, dtype=np.float32)
            
            with open(VOICE_LABELS) as f:
                self.voice_labels = [line.strip().split()[-1] for line in f]
            
            print("✓ Voice model loaded successfully")
        except Exception as e:
            print(f"⚠ Voice model error: {e}")
            self.voice_interpreter = None
    
    def _init_gesture_model(self):
        """Initialize gesture recognition model."""
        if not os.path.exists(GESTURE_MODEL):
            print(f"⚠ Gesture model not found: {GESTURE_MODEL}")
            print("  Gesture control will be DISABLED")
            return
        
        try:
            self.gesture_interpreter = tflite.Interpreter(GESTURE_MODEL)
            self.gesture_interpreter.allocate_tensors()
            
            with open(GESTURE_LABELS, "r") as f:
                self.gesture_labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]
            
            print("✓ Gesture model loaded successfully")
        except Exception as e:
            print(f"⚠ Gesture model error: {e}")
            self.gesture_interpreter = None
    
    def _init_camera(self):
        """Initialize camera for gesture control."""
        self.camera = find_camera()
        if not self.camera:
            print("⚠ No camera found - Gesture control will be DISABLED")
        else:
            print(f"✓ Camera ready: {int(self.camera.get(3))}x{int(self.camera.get(4))}")
    
    # ========================================================
    #                  BLUETOOTH CONTROL
    # ========================================================
    
    def connect_bluetooth(self):
        """Establish Bluetooth connection to robot."""
        try:
            self.bt = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            print("\n" + "="*50)
            print("       🤖 ROBOT CONNECTED SUCCESSFULLY 🤖")
            print("="*50)
            self._print_controls()
            return True
        except Exception as e:
            print(f"❌ Bluetooth connection error: {e}")
            return False
    
    def _print_controls(self):
        """Print available controls."""
        print("\n📋 CONTROL MODES:")
        print("  ├─ KEYBOARD (default):")
        print("  │   ├─ Arrows: Drive")
        print("  │   ├─ 1/4: Arm1 up/down")
        print("  │   ├─ 3/6: Arm2 up/down")
        print("  │   ├─ 0/2: Arm3 clockwise/counter")
        print("  │   └─ Q: Toggle LED")
        print("  ├─ VOICE: Press 'a' to toggle")
        print("  └─ GESTURE: Press SPACE to toggle")
        print("\n  ESC: Emergency stop & exit")
        print("="*50 + "\n")
    
    def send_command(self, cmd):
        """Send command to robot via Bluetooth."""
        with self.bt_lock:
            if self.bt and self.bt.is_open:
                self.bt.write(cmd.encode())
                print(f"→ Sent: {cmd}")
    
    def stop_all_motors(self):
        """Stop all robot motors."""
        self.send_command(STOP_ALL)
        self.active_cmds = {k: None for k in self.active_cmds}
        print("🛑 All motors stopped")
    
    # ========================================================
    #                  MODE SWITCHING
    # ========================================================
    
    def switch_mode(self, new_mode):
        """
        Switch between control modes with proper cleanup.
        
        Args:
            new_mode: "KEYBOARD", "VOICE", or "GESTURE"
        """
        with self.mode_lock:
            if self.current_mode == new_mode:
                return
            
            print(f"\n🔄 Switching: {self.current_mode} → {new_mode}")
            
            # Stop all motors before switching
            self.stop_all_motors()
            
            # Cleanup old mode
            if self.current_mode == "VOICE":
                self._stop_voice_mode()
            elif self.current_mode == "GESTURE":
                self._stop_gesture_mode()
            
            # Start new mode
            self.current_mode = new_mode
            
            if new_mode == "VOICE":
                self._start_voice_mode()
            elif new_mode == "GESTURE":
                self._start_gesture_mode()
            
            print(f"✓ Now in {new_mode} mode\n")
    
    # ========================================================
    #                  KEYBOARD CONTROL
    # ========================================================
    
    def on_key_press(self, key):
        """Handle keyboard press events."""
        # Emergency stop
        if key == keyboard.Key.esc:
            print("\n⚠ EMERGENCY STOP REQUESTED")
            self.stop_all_motors()
            self.running = False
            return False
        
        try:
            char = key.char.lower() if hasattr(key, 'char') else None
        except:
            char = None
        
        # Mode toggles (work in any mode)
        if char == 'a':
            if self.voice_interpreter:
                new_mode = "KEYBOARD" if self.current_mode == "VOICE" else "VOICE"
                self.switch_mode(new_mode)
            else:
                print("⚠ Voice mode not available (model not loaded)")
            return
        
        if key == keyboard.Key.space:
            if self.gesture_interpreter and self.camera:
                new_mode = "KEYBOARD" if self.current_mode == "GESTURE" else "GESTURE"
                self.switch_mode(new_mode)
            else:
                print("⚠ Gesture mode not available (model/camera not loaded)")
            return
        
        # Keyboard commands (only in KEYBOARD mode)
        if self.current_mode != "KEYBOARD":
            return
        
        # Drive controls
        if key in DRIVE_KEYS:
            cmd = DRIVE_KEYS[key]
            if self.active_cmds['drive'] != cmd:
                self.send_command(cmd)
                self.active_cmds['drive'] = cmd
        
        # Arm controls
        elif char in ARM1_KEYS:
            cmd = ARM1_KEYS[char]
            if self.active_cmds['arm1'] != cmd:
                self.send_command(cmd)
                self.active_cmds['arm1'] = cmd
        
        elif char in ARM2_KEYS:
            cmd = ARM2_KEYS[char]
            if self.active_cmds['arm2'] != cmd:
                self.send_command(cmd)
                self.active_cmds['arm2'] = cmd
        
        elif char in ARM3_KEYS:
            cmd = ARM3_KEYS[char]
            if self.active_cmds['arm3'] != cmd:
                self.send_command(cmd)
                self.active_cmds['arm3'] = cmd
        
        # LED toggle
        elif char == 'q':
            self.send_command(TOGGLE_LED)
    
    def on_key_release(self, key):
        """Handle keyboard release events."""
        if self.current_mode != "KEYBOARD":
            return
        
        try:
            char = key.char if hasattr(key, 'char') else None
        except:
            char = None
        
        # Stop drive
        if key in DRIVE_KEYS and self.active_cmds['drive'] == DRIVE_KEYS[key]:
            self.send_command(STOP_DRIVE)
            self.active_cmds['drive'] = None
        
        # Stop arms
        if char in ARM1_KEYS and self.active_cmds['arm1'] == ARM1_KEYS[char]:
            self.send_command(STOP_ARM1)
            self.active_cmds['arm1'] = None
        
        if char in ARM2_KEYS and self.active_cmds['arm2'] == ARM2_KEYS[char]:
            self.send_command(STOP_ARM2)
            self.active_cmds['arm2'] = None
        
        if char in ARM3_KEYS and self.active_cmds['arm3'] == ARM3_KEYS[char]:
            self.send_command(STOP_ARM3)
            self.active_cmds['arm3'] = None
    
    # ========================================================
    #                  VOICE CONTROL
    # ========================================================
    
    def _start_voice_mode(self):
        """Start voice recognition stream."""
        if not self.voice_interpreter:
            print("❌ Cannot start voice mode: model not loaded")
            return
        
        self.voice_active = True
        self.voice_position = 0
        self.voice_buffer.fill(0)
        self.voice_cooldown_active = False
        
        inp = self.voice_interpreter.get_input_details()[0]
        
        try:
            self.voice_stream = sd.InputStream(
                samplerate=VOICE_SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=int(VOICE_SAMPLE_RATE * 0.1),
                callback=self._voice_callback
            )
            self.voice_stream.start()
            print("🎤 Voice recognition active - speak commands!")
        except Exception as e:
            print(f"❌ Voice stream error: {e}")
            self.voice_active = False
    
    def _stop_voice_mode(self):
        """Stop voice recognition stream."""
        self.voice_active = False
        if self.voice_stream:
            self.voice_stream.stop()
            self.voice_stream.close()
            self.voice_stream = None
        self.voice_stop_event.set()
        print("🔇 Voice recognition stopped")
    
    def _voice_callback(self, indata, frames, time_info, status):
        """Process audio input for voice recognition."""
        if not self.voice_active:
            return
        
        chunk = indata[:, 0].copy()
        inp = self.voice_interpreter.get_input_details()[0]
        samples = inp['shape'][1]
        
        # Fill buffer
        n = min(len(chunk), samples - self.voice_position)
        self.voice_buffer[self.voice_position:self.voice_position + n] = chunk[:n]
        self.voice_position += n
        
        # Process when buffer is full
        if self.voice_position >= samples:
            audio = self.voice_buffer.copy()
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio /= max_val
            
            # Run inference
            x = audio.reshape(1, samples).astype(np.float32)
            self.voice_interpreter.set_tensor(inp['index'], x)
            self.voice_interpreter.invoke()
            
            out = self.voice_interpreter.get_output_details()[0]
            scores = self.voice_interpreter.get_tensor(out['index'])[0]
            
            idx = np.argmax(scores)
            label = self.voice_labels[idx]
            confidence = scores[idx]
            
            # Process command if confident
            if confidence > VOICE_CONFIDENCE_THRESHOLD and label in VOICE_COMMANDS:
                self._handle_voice_command(label, confidence)
            
            # Slide buffer for overlap
            shift = int(samples * (1 - VOICE_OVERLAP))
            self.voice_buffer[:shift] = self.voice_buffer[-shift:]
            self.voice_position = shift
    
    def _handle_voice_command(self, label, confidence):
        """Execute voice command."""
        print(f"🎤 Voice: '{label}' ({confidence:.1%})")
        
        # STOP command - always immediate
        if label == "stop":
            self.voice_stop_event.set()
            self.send_command(STOP_ALL)
            
            # Release action lock if held
            if self.voice_action_lock.locked():
                self.voice_action_lock.release()
            
            # Reset and start cooldown
            self.voice_stop_event = threading.Event()
            self._start_voice_cooldown()
            print("🛑 STOPPED - Ready after cooldown")
            return
        
        # Check if in cooldown
        if self.voice_cooldown_active:
            print("⏳ Cooldown active - command ignored")
            return
        
        # Check if robot is busy
        if self.voice_action_lock.locked():
            print("⏳ Wait until current command finishes")
            return
        
        # Execute new command
        self.voice_action_lock.acquire()
        self.voice_stop_event.set()
        self.voice_stop_event = threading.Event()
        
        cmd, stop_cmd = VOICE_COMMANDS[label]
        threading.Thread(
            target=self._execute_voice_action,
            args=(cmd, stop_cmd),
            daemon=True
        ).start()
    
    def _execute_voice_action(self, cmd, stop_cmd):
        """Execute a timed voice action."""
        self.send_command(cmd)
        
        # Wait for duration or interruption
        if self.voice_stop_event.wait(COMMAND_DURATION):
            pass  # Interrupted
        else:
            # Duration completed, send stop command
            if stop_cmd:
                self.send_command(stop_cmd)
        
        # Release lock
        if self.voice_action_lock.locked():
            self.voice_action_lock.release()
    
    def _start_voice_cooldown(self):
        """Start cooldown period after stop command."""
        self.voice_cooldown_active = True
        threading.Thread(target=self._voice_cooldown_timer, daemon=True).start()
    
    def _voice_cooldown_timer(self):
        """Timer for voice command cooldown."""
        time.sleep(COOLDOWN_TIME)
        self.voice_cooldown_active = False
        print("✓ Cooldown complete - Ready for commands")
    
    # ========================================================
    #                  GESTURE CONTROL
    # ========================================================
    
    def _start_gesture_mode(self):
        """Start gesture recognition."""
        if not self.gesture_interpreter or not self.camera:
            print("❌ Cannot start gesture mode: model/camera not available")
            return
        
        self.gesture_active = True
        self.gesture_direction_forward = True
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_driving = False
        
        self.gesture_thread = threading.Thread(
            target=self._gesture_loop,
            daemon=True
        )
        self.gesture_thread.start()
        print("👋 Gesture recognition active - show hand gestures!")
    
    def _stop_gesture_mode(self):
        """Stop gesture recognition."""
        self.gesture_active = False
        
        if self.gesture_driving:
            self.send_command(STOP_DRIVE)
            self.gesture_driving = False
        
        if self.gesture_thread and self.gesture_thread.is_alive():
            self.gesture_thread.join(timeout=1)
        
        cv2.destroyAllWindows()
        print("👋 Gesture recognition stopped")
    
    def _gesture_loop(self):
        """Main gesture recognition loop."""
        print("📷 Gesture camera opened")
        
        while self.gesture_active and self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                print("⚠ Camera read failed")
                time.sleep(0.1)
                continue
            
            try:
                # Preprocess frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                image_array = np.asarray(image, dtype=np.float32)
                normalized = (image_array / 127.5) - 1
                input_data = np.expand_dims(normalized, axis=0)
                
                # Run inference
                inp = self.gesture_interpreter.get_input_details()[0]
                out = self.gesture_interpreter.get_output_details()[0]
                
                self.gesture_interpreter.set_tensor(inp['index'], input_data)
                self.gesture_interpreter.invoke()
                prediction = self.gesture_interpreter.get_tensor(out['index'])[0]
                
                idx = np.argmax(prediction)
                gesture = self.gesture_labels[idx]
                confidence = prediction[idx]
                
                # Display info on frame
                direction = "BACKWARD" if self.gesture_direction_forward else "FORWARD"
                cv2.putText(frame, f"{gesture}: {confidence:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Next: {direction}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to exit",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                cv2.imshow("Gesture Control", frame)
                cv2.waitKey(1)
                
                # Process gesture command
                if confidence > GESTURE_CONFIDENCE_THRESHOLD:
                    self._handle_gesture_command(gesture)
                
            except Exception as e:
                print(f"⚠ Gesture processing error: {e}")
                continue
        
        cv2.destroyAllWindows()
    
    def _handle_gesture_command(self, gesture):
        """Execute gesture command."""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_gesture_time < GESTURE_COOLDOWN:
            return
        
        # Ignore repeated gestures
        if gesture == self.last_gesture:
            return
        
        # START gesture - move in current direction
        if gesture == "start":
            cmd = 'B' if self.gesture_direction_forward else 'F'
            self.send_command(cmd)
            direction = "FORWARD" if self.gesture_direction_forward else "BACKWARD"
            print(f"👋 Gesture: START → {direction}")
            self.gesture_driving = True
            self.last_gesture = "start"
            self.last_gesture_time = current_time
        
        # STOP gesture - stop and toggle direction
        elif gesture == "stop":
            self.send_command(STOP_DRIVE)
            self.gesture_direction_forward = not self.gesture_direction_forward
            print("👋 Gesture: STOP → Direction toggled")
            self.gesture_driving = False
            self.last_gesture = "stop"
            self.last_gesture_time = current_time
    
    # ========================================================
    #                  MAIN RUN LOOP
    # ========================================================
    
    def run(self):
        """Main execution loop."""
        if not self.connect_bluetooth():
            return
        
        print("✓ Starting in KEYBOARD mode\n")
        
        # Start keyboard listener
        listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        listener.start()
        
        try:
            # Main loop - just keep running
            while self.running:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n⚠ Keyboard interrupt detected")
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            self.running = False
            self.stop_all_motors()
            
            if self.current_mode == "VOICE":
                self._stop_voice_mode()
            elif self.current_mode == "GESTURE":
                self._stop_gesture_mode()
            
            listener.stop()
            
            if self.camera:
                self.camera.release()
            
            if self.bt and self.bt.is_open:
                self.bt.close()
            
            cv2.destroyAllWindows()
            print("✓ Disconnected. Goodbye! 👋\n")


# ============================================================
#                    PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("    UNIFIED ROBOT CONTROL SYSTEM v1.0")
    print("="*50)
    
    controller = UnifiedRobotController()
    controller.run()