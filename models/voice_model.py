"""
Voice recognition model handler.
"""

import os
import numpy as np
import tflite_runtime.interpreter as tflite

from utils.resource_loader import resource_path


class VoiceModel:
    """Voice recognition model wrapper with dynamic loading."""
    
    def __init__(self, model_name="soundclassifier_with_metadata", model_dir="resources/sound_classifier"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.interpreter = None
        self.labels = []
        self.buffer_size = 0
        self.class_to_letter = {}  # Mapping from class name to letter
        
        # Try to load model
        try:
            self._load_model()
        except FileNotFoundError as e:
            # Model files not found - this is OK, user can load later
            print(f"Voice model not found: {e}")
        except Exception as e:
            # Other errors
            print(f"Error loading voice model: {e}")
    
    def _load_model(self):
        """Load TFLite model and labels."""
        # Build paths
        if os.path.isabs(self.model_name):
            # Absolute path provided
            model_path = self.model_name
            labels_path = self.model_name.replace('.tflite', '_labels.txt')
        else:
            # Relative path - use resource_path
            model_path = resource_path(os.path.join(self.model_dir, f"{self.model_name}.tflite"))
            labels_path = resource_path(os.path.join(self.model_dir, f"{self.model_name}_labels.txt"))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Voice labels not found: {labels_path}")
        
        try:
            self.interpreter = tflite.Interpreter(model_path)
            self.interpreter.allocate_tensors()
            
            # Get buffer size from model input shape
            inp = self.interpreter.get_input_details()[0]
            self.buffer_size = inp['shape'][1]
            
            # Load labels
            with open(labels_path) as f:
                lines = f.readlines()
                self.labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.labels.append(parts[-1])
                    elif len(parts) == 1:
                        self.labels.append(parts[0])
        
        except Exception as e:
            raise RuntimeError(f"Failed to load voice model: {e}")
    
    def set_mapping(self, mapping):
        """
        Set class-to-letter mapping.
        
        Args:
            mapping: Dictionary mapping class names to letters
        """
        self.class_to_letter = mapping
    
    def get_labels(self):
        """Get list of all class labels."""
        return self.labels.copy()
    
    def predict(self, audio_buffer):
        """
        Run inference on audio buffer.
        
        Args:
            audio_buffer: Numpy array of audio samples
            
        Returns:
            Tuple of (class_name, letter, confidence)
        """
        if self.interpreter is None:
            return None, None, 0.0
        
        # Normalize audio
        max_val = np.max(np.abs(audio_buffer))
        if max_val > 0:
            audio_buffer = audio_buffer / max_val
        
        # Prepare input
        input_data = audio_buffer.reshape(1, -1).astype(np.float32)
        
        # Run inference
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        
        self.interpreter.set_tensor(inp['index'], input_data)
        self.interpreter.invoke()
        
        # Get results
        scores = self.interpreter.get_tensor(out['index'])[0]
        idx = np.argmax(scores)
        class_name = self.labels[idx]
        confidence = scores[idx]
        
        # Get letter from mapping
        letter = self.class_to_letter.get(class_name, class_name)
        
        return class_name, letter, confidence
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.interpreter is not None
    
    def is_mapping_complete(self):
        """Check if all classes have valid single-letter mappings."""
        if not self.class_to_letter:
            return False
        
        for class_name in self.labels:
            letter = self.class_to_letter.get(class_name)
            if not letter or len(letter) != 1:
                return False
        
        return True