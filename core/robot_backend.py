"""
Main robot controller backend.
"""

import threading

from config import MODE_KEYBOARD, MODE_VOICE, MODE_GESTURE
from .bluetooth_manager import BluetoothManager
from .command_executor import CommandExecutor
from .profile_manager import ProfileManager
from controllers import VoiceController, GestureController
from ui.theme_manager import ThemeManager


class RobotControllerBackend:
    """Main backend controller managing robot communication and control modes."""
    
    def __init__(self, signal_emitter):
        self.signals = signal_emitter
        
        # Core components
        self.bluetooth = BluetoothManager(signal_emitter)
        self.executor = CommandExecutor(self.bluetooth, signal_emitter)
        self.profile_manager = ProfileManager()
        self.theme_manager = ThemeManager()
        
        # Control modes
        self.voice_controller = VoiceController(self.executor, signal_emitter)
        self.gesture_controller = GestureController(self.executor, signal_emitter)
        
        # State
        self.current_mode = MODE_KEYBOARD
        self.mode_lock = threading.Lock()
        self.running = True
        
        # Active commands for keyboard mode
        self.active_cmds = {
            'drive': None,
            'arm1': None,
            'arm2': None,
            'arm3': None
        }
        
        self._log_availability()
        self._load_last_profiles()
    
    def _log_availability(self):
        """Log availability of control modes."""
        if self.voice_controller.is_available():
            self.signals.log_signal.emit("Voice control available", "success")
        else:
            self.signals.log_signal.emit("Voice control DISABLED", "warning")
        
        if self.gesture_controller.is_available():
            self.signals.log_signal.emit("Gesture control available", "success")
        else:
            self.signals.log_signal.emit("Gesture control DISABLED", "warning")
    
    def _load_last_profiles(self):
        """Load last used profiles on startup or use defaults."""
        from config import DEFAULT_VOICE_MAPPING, DEFAULT_GESTURE_MAPPING
        
        # Load last voice profile or use default
        if self.profile_manager.last_used_voice:
            try:
                profile = self.profile_manager.get_profile(self.profile_manager.last_used_voice)
                if profile:
                    model_name = profile.model_path.split('/')[-1].replace('.tflite', '')
                    self.voice_controller.load_new_model(model_name)
                    self.voice_controller.model.set_mapping(profile.class_to_letter)
                    self.signals.log_signal.emit(f"Auto-loaded voice profile: {profile.name}", "info")
            except Exception as e:
                self.signals.log_signal.emit(f"Failed to auto-load voice profile: {e}", "warning")
        else:
            # Apply default mapping if model exists
            if self.voice_controller.model and self.voice_controller.model.is_loaded():
                self.voice_controller.model.set_mapping(DEFAULT_VOICE_MAPPING)
                self.signals.log_signal.emit("Using default voice mappings", "info")
        
        # Load last gesture profile or use default
        if self.profile_manager.last_used_gesture:
            try:
                profile = self.profile_manager.get_profile(self.profile_manager.last_used_gesture)
                if profile:
                    model_name = profile.model_path.split('/')[-1].replace('.tflite', '')
                    self.gesture_controller.load_new_model(model_name)
                    self.gesture_controller.model.set_mapping(profile.class_to_letter)
                    if profile.custom_gestures:
                        self.gesture_controller.custom_gesture_manager.from_dict(profile.custom_gestures)
                    self.signals.log_signal.emit(f"Auto-loaded gesture profile: {profile.name}", "info")
            except Exception as e:
                self.signals.log_signal.emit(f"Failed to auto-load gesture profile: {e}", "warning")
        else:
            # Apply default mapping if model exists
            if self.gesture_controller.model and self.gesture_controller.model.is_loaded():
                self.gesture_controller.model.set_mapping(DEFAULT_GESTURE_MAPPING)
                self.signals.log_signal.emit("Using default gesture mappings", "info")
    def send_command(self, command):
        """Send command to robot."""
        self.executor.send_command(command)
    
    def stop_all_motors(self):
        """Stop all motors."""
        self.executor.stop_all_motors()
        self.active_cmds = {k: None for k in self.active_cmds}
        if self.gesture_controller.active:
            self.gesture_controller.current_cmd = None
    
    def switch_mode(self, new_mode):
        """
        Switch control mode.
        
        Args:
            new_mode: New control mode (KEYBOARD, VOICE, or GESTURE)
        """
        with self.mode_lock:
            if self.current_mode == new_mode:
                return
            
            self.signals.log_signal.emit(f"Switching: {self.current_mode} → {new_mode}", "info")
            
            # Stop all motors
            self.stop_all_motors()
            
            # Cleanup old mode
            if self.current_mode == MODE_VOICE:
                self.voice_controller.stop()
            elif self.current_mode == MODE_GESTURE:
                self.gesture_controller.stop()
            
            # Start new mode
            self.current_mode = new_mode
            self.signals.mode_signal.emit(new_mode)
            
            if new_mode == MODE_VOICE:
                self.voice_controller.start()
            elif new_mode == MODE_GESTURE:
                self.gesture_controller.start()
            
            self.signals.log_signal.emit(f"Now in {new_mode} mode", "success")
    
    def cleanup(self):
        """Cleanup all resources."""
        self.running = False
        self.stop_all_motors()
        
        # Save current profiles before cleanup
        if self.voice_controller.model:
            self._save_current_profile('voice')
        if self.gesture_controller.model:
            self._save_current_profile('gesture')
        
        # Stop controllers
        if self.current_mode == MODE_VOICE:
            self.voice_controller.stop()
        elif self.current_mode == MODE_GESTURE:
            self.gesture_controller.stop()
        
        # Release camera
        if self.gesture_controller.camera:
            self.gesture_controller.camera.release()
        
        # Disconnect Bluetooth
        self.bluetooth.disconnect()
        
        self.signals.log_signal.emit("Cleanup complete", "info")
    
    def _save_current_profile(self, model_type):
        """Save current model state to profile."""
        try:
            if model_type == 'voice':
                controller = self.voice_controller
                if not controller.model:
                    return
                
                profile_name = controller.current_model_name
                profile = self.profile_manager.get_profile(profile_name)
                
                if not profile:
                    profile = self.profile_manager.create_profile(profile_name, 'voice')
                
                profile.model_path = controller.model.model_dir + f"/{controller.current_model_name}.tflite"
                profile.labels_path = controller.model.model_dir + f"/{controller.current_model_name}_labels.txt"
                profile.classes = controller.model.get_labels()
                profile.class_to_letter = controller.get_current_mapping()
                
            else:  # gesture
                controller = self.gesture_controller
                if not controller.model:
                    return
                
                profile_name = controller.current_model_name
                profile = self.profile_manager.get_profile(profile_name)
                
                if not profile:
                    profile = self.profile_manager.create_profile(profile_name, 'gesture')
                
                profile.model_path = controller.model.model_dir + f"/{controller.current_model_name}.tflite"
                profile.labels_path = controller.model.model_dir + f"/{controller.current_model_name}_labels.txt"
                profile.classes = controller.model.get_labels()
                profile.class_to_letter = controller.model.class_to_letter
                profile.custom_gestures = controller.custom_gesture_manager.to_dict()
            
            self.profile_manager.save_profiles()
        
        except Exception as e:
            self.signals.log_signal.emit(f"Error saving profile: {e}", "warning")