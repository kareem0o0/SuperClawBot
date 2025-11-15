"""
Gesture control implementation with custom gesture support.
"""

import time
import threading
import cv2
import os

from .base_controller import BaseController
from models import GestureModel
from config import GESTURE_CONFIDENCE_THRESHOLD, GESTURE_COOLDOWN
from utils.camera import find_camera
from core.model_manager import ModelManager
from core.embedding_extractor import EmbeddingExtractor, CustomGestureManager
from utils.resource_loader import resource_path


class GestureController(BaseController):
    """Gesture recognition control mode with dynamic model and custom gesture support."""
    
    def __init__(self, command_executor, signal_emitter):
        super().__init__(command_executor, signal_emitter)
        
        self.model_manager = ModelManager(signal_emitter)
        self.model = None
        self.current_model_name = None
        self.custom_gesture_manager = CustomGestureManager()
        self.embedding_extractor = None
        
        # Try to load default model
        try:
            self._load_model("model")
        except Exception as e:
            self.signals.log_signal.emit(f"Default gesture model not found - load one via Models menu", "warning")
        
        self.camera = find_camera()
        if not self.camera:
            self.signals.log_signal.emit("No camera found", "warning")
        else:
            w, h = int(self.camera.get(3)), int(self.camera.get(4))
            self.signals.log_signal.emit(f"Camera ready: {w}x{h}", "success")
        
        self.thread = None
        self.last_gesture = None
        self.last_gesture_time = 0
        self.current_cmd = None
        self.custom_gesture_threshold = 0.75
    
    def _load_model(self, model_name):
        """Load a gesture model by name."""
        try:
            self.model = GestureModel(model_name)
            
            # Check if model actually loaded
            if not self.model.is_loaded():
                self.model = None
                return False
            
            self.current_model_name = model_name
            
            # Initialize embedding extractor for custom gestures
            # Build the correct path
            model_path = resource_path(os.path.join(self.model.model_dir, f"{model_name}.tflite"))
            
            print(f"Initializing EmbeddingExtractor with path: {model_path}")
            
            try:
                self.embedding_extractor = EmbeddingExtractor(model_path)
                print("EmbeddingExtractor initialized successfully")
            except Exception as e:
                print(f"Failed to initialize EmbeddingExtractor: {e}")
                self.embedding_extractor = None
            
            # Load or create mapping
            mapping = self.model_manager.load_mapping(model_name, "gesture")
            if mapping is None:
                # Create default mapping
                labels = self.model.get_labels()
                mapping = self.model_manager.create_default_mapping(labels)
                self.model_manager.save_mapping(model_name, "gesture", mapping)
            
            self.model.set_mapping(mapping)
            self.signals.log_signal.emit(f"Gesture model loaded: {model_name}", "success")
            
            # Check if mapping is complete
            if not self.model.is_mapping_complete():
                self.signals.log_signal.emit("Warning: Model mapping incomplete - configure letters first", "warning")
            
            return True
        except Exception as e:
            self.signals.log_signal.emit(f"Failed to load gesture model: {e}", "error")
            print(f"Error in _load_model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False
    
    def load_new_model(self, model_name):
        """Load a different gesture model."""
        # Stop if active
        was_active = self.active
        if was_active:
            self.stop()
        
        success = self._load_model(model_name)
        
        # Restart if was active
        if was_active and success:
            self.start()
        
        return success
    
    def add_custom_gesture(self, name, embeddings, letter):
        """Add a new custom gesture."""
        self.custom_gesture_manager.add_gesture(name, embeddings, letter)
        self.signals.log_signal.emit(f"Custom gesture added: {name} → {letter}", "success")
    
    def remove_custom_gesture(self, name):
        """Remove a custom gesture."""
        self.custom_gesture_manager.remove_gesture(name)
        self.signals.log_signal.emit(f"Custom gesture removed: {name}", "info")
    
    def get_custom_gestures(self):
        """Get all custom gesture names."""
        return self.custom_gesture_manager.get_all_gestures()
    
    def get_current_mapping(self):
        """Get current class-to-letter mapping."""
        if self.model:
            mapping = self.model.class_to_letter.copy()
            # Add custom gestures
            for name in self.custom_gesture_manager.get_all_gestures():
                letter = self.custom_gesture_manager.get_gesture_letter(name)
                mapping[f"[CUSTOM] {name}"] = letter
            return mapping
        return {}
    
    def update_mapping(self, mapping):
        """Update class-to-letter mapping."""
        if self.model:
            # Separate custom gestures from regular classes
            regular_mapping = {}
            for key, value in mapping.items():
                if key.startswith("[CUSTOM] "):
                    # Update custom gesture letter
                    custom_name = key.replace("[CUSTOM] ", "")
                    self.custom_gesture_manager.update_letter(custom_name, value)
                else:
                    regular_mapping[key] = value
            
            self.model.set_mapping(regular_mapping)
            self.model_manager.save_mapping(self.current_model_name, "gesture", regular_mapping)
            self.signals.log_signal.emit("Gesture mapping updated", "success")
            return True
        return False
    
    def is_available(self):
        """Check if gesture control is available."""
        return (self.model is not None and 
                self.model.is_loaded() and 
                self.model.is_mapping_complete() and 
                self.camera is not None)
    
    def start(self):
        """Start gesture recognition."""
        if not self.model or not self.model.is_loaded():
            self.signals.log_signal.emit("Gesture model not loaded", "error")
            return
        
        if not self.model.is_mapping_complete():
            self.signals.log_signal.emit("Gesture mapping incomplete - configure letters first", "error")
            return
        
        if not self.camera:
            self.signals.log_signal.emit("No camera available", "error")
            return
        
        self.active = True
        self.last_gesture = None
        self.last_gesture_time = 0
        self.current_cmd = None
        
        self.thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.thread.start()
        self.signals.log_signal.emit("Gesture recognition active", "success")
    
    def stop(self):
        """Stop gesture recognition."""
        self.active = False
        
        if self.current_cmd:
            self.executor.send_command('!')  # Stop all
            self.current_cmd = None
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        
        self.signals.log_signal.emit("Gesture recognition stopped", "info")
    
    def _recognition_loop(self):
        """Main gesture recognition loop with custom gesture support."""
        while self.active:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            try:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check custom gestures first (higher priority)
                detected_class = None
                detected_letter = None
                confidence = 0
                is_custom = False
                
                if self.embedding_extractor:
                    try:
                        embedding = self.embedding_extractor.extract_from_frame(rgb_frame)
                        if embedding is not None:
                            custom_name, custom_letter, custom_conf = self.custom_gesture_manager.predict(
                                embedding, self.custom_gesture_threshold
                            )
                            
                            if custom_name:
                                detected_class = f"[CUSTOM] {custom_name}"
                                detected_letter = custom_letter
                                confidence = custom_conf
                                is_custom = True
                    except Exception as e:
                        print(f"Custom gesture error: {e}")
                
                # If no custom gesture, use regular model
                if not is_custom:
                    input_data = self.model.preprocess_frame(rgb_frame)
                    class_name, letter, conf = self.model.predict(input_data)
                    detected_class = class_name
                    detected_letter = letter
                    confidence = conf
                
                # Annotate frame
                if detected_class:
                    cv2.putText(frame, f"Class: {detected_class}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if detected_letter:
                    cv2.putText(frame, f"Letter: {detected_letter}", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if is_custom:
                    cv2.putText(frame, "CUSTOM GESTURE", (10, 135),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                if self.current_cmd:
                    cv2.putText(frame, f"Active: {self.current_cmd}", (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Send frame to UI
                self.signals.frame_signal.emit(frame)
                
                # Process command
                threshold = self.custom_gesture_threshold if is_custom else GESTURE_CONFIDENCE_THRESHOLD
                if confidence > threshold and detected_letter:
                    self._handle_gesture(detected_class, detected_letter, confidence)
            
            except Exception as e:
                self.signals.log_signal.emit(f"Gesture error: {e}", "error")
                print(f"Recognition loop error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _handle_gesture(self, gesture, letter, confidence):
        """Execute gesture command."""
        current_time = time.time()
        
        # Apply cooldown
        if current_time - self.last_gesture_time < GESTURE_COOLDOWN:
            return
        
        # Ignore repeated gestures
        if gesture == self.last_gesture:
            return
        
        self.signals.gesture_command_signal.emit(f"{gesture}→{letter}", confidence)
        
        # Send the letter as command
        self.executor.send_command(letter)
        self.current_cmd = gesture
        
        self.signals.log_signal.emit(f"Gesture: {gesture} → {letter}", "info")
        
        self.last_gesture = gesture
        self.last_gesture_time = current_time