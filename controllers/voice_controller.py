"""
Voice control implementation.
"""

import threading
import numpy as np
import sounddevice as sd

from .base_controller import BaseController
from models import VoiceModel
from config import VOICE_SAMPLE_RATE, VOICE_OVERLAP, VOICE_CONFIDENCE_THRESHOLD, COOLDOWN_TIME
from core.model_manager import ModelManager


class VoiceController(BaseController):
    """Voice recognition control mode with dynamic model support."""
    
    def __init__(self, command_executor, signal_emitter):
        super().__init__(command_executor, signal_emitter)
        
        self.model_manager = ModelManager(signal_emitter)
        self.model = None
        self.buffer = None
        self.current_model_name = None
        
        # Try to load default model
        try:
            self._load_model("soundclassifier_with_metadata")
        except Exception as e:
            self.signals.log_signal.emit(f"Default voice model not found - load one via Models menu", "warning")
        
        self.stream = None
        self.position = 0
        self.action_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cooldown_active = False
    
    def _load_model(self, model_name):
        """Load a voice model by name."""
        try:
            self.model = VoiceModel(model_name)
            
            # Check if model actually loaded
            if not self.model.is_loaded():
                self.model = None
                return False
            
            self.current_model_name = model_name
            self.buffer = np.zeros(self.model.buffer_size, dtype=np.float32)
            
            # Load or create mapping
            mapping = self.model_manager.load_mapping(model_name, "voice")
            if mapping is None:
                # Create default mapping
                labels = self.model.get_labels()
                mapping = self.model_manager.create_default_mapping(labels)
                self.model_manager.save_mapping(model_name, "voice", mapping)
            
            self.model.set_mapping(mapping)
            self.signals.log_signal.emit(f"Voice model loaded: {model_name}", "success")
            
            # Check if mapping is complete
            if not self.model.is_mapping_complete():
                self.signals.log_signal.emit("Warning: Model mapping incomplete - configure letters first", "warning")
            
            return True
        except Exception as e:
            self.signals.log_signal.emit(f"Failed to load voice model: {e}", "error")
            self.model = None
            return False
    
    def load_new_model(self, model_name):
        """Load a different voice model."""
        # Stop if active
        was_active = self.active
        if was_active:
            self.stop()
        
        success = self._load_model(model_name)
        
        # Restart if was active
        if was_active and success:
            self.start()
        
        return success
    
    def get_current_mapping(self):
        """Get current class-to-letter mapping."""
        if self.model:
            return self.model.class_to_letter.copy()
        return {}
    
    def update_mapping(self, mapping):
        """Update class-to-letter mapping."""
        if self.model:
            self.model.set_mapping(mapping)
            self.model_manager.save_mapping(self.current_model_name, "voice", mapping)
            self.signals.log_signal.emit("Voice mapping updated", "success")
            return True
        return False
    
    def is_available(self):
        """Check if voice control is available."""
        return (self.model is not None and 
                self.model.is_loaded() and 
                self.model.is_mapping_complete())
    
    def start(self):
        """Start voice recognition."""
        if not self.model or not self.model.is_loaded():
            self.signals.log_signal.emit("Voice model not loaded", "error")
            return
        
        if not self.model.is_mapping_complete():
            self.signals.log_signal.emit("Voice mapping incomplete - configure letters first", "error")
            return
        
        self.active = True
        self.position = 0
        self.buffer.fill(0)
        self.cooldown_active = False
        
        try:
            self.stream = sd.InputStream(
                samplerate=VOICE_SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=int(VOICE_SAMPLE_RATE * 0.1),
                callback=self._audio_callback
            )
            self.stream.start()
            self.signals.log_signal.emit("Voice recognition active", "success")
        except Exception as e:
            self.signals.log_signal.emit(f"Voice stream error: {e}", "error")
            self.active = False
    
    def stop(self):
        """Stop voice recognition."""
        self.active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.stop_event.set()
        self.signals.log_signal.emit("Voice recognition stopped", "info")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio stream."""
        if not self.active or not self.model:
            return
        
        chunk = indata[:, 0].copy()
        samples = self.model.buffer_size
        
        n = min(len(chunk), samples - self.position)
        self.buffer[self.position:self.position + n] = chunk[:n]
        self.position += n
        
        if self.position >= samples:
            audio = self.buffer.copy()
            
            # Run prediction
            class_name, letter, confidence = self.model.predict(audio)
            
            if confidence > VOICE_CONFIDENCE_THRESHOLD and letter:
                self._handle_command(class_name, letter, confidence)
            
            # Slide buffer
            shift = int(samples * (1 - VOICE_OVERLAP))
            self.buffer[:shift] = self.buffer[-shift:]
            self.position = shift
    
    def _handle_command(self, class_name, letter, confidence):
        """Execute voice command."""
        self.signals.voice_command_signal.emit(f"{class_name}→{letter}", confidence)
        
        if self.cooldown_active or self.action_lock.locked():
            return
        
        # Send the letter as command
        self.executor.send_command(letter)
        
        self.signals.log_signal.emit(f"Voice: {class_name} → {letter}", "info")
    
    def _start_cooldown(self):
        """Start cooldown after command."""
        self.cooldown_active = True
        threading.Thread(target=self._cooldown_timer, daemon=True).start()
    
    def _cooldown_timer(self):
        """Cooldown timer."""
        import time
        time.sleep(COOLDOWN_TIME)
        self.cooldown_active = False