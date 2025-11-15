"""
Gesture recognition model handler.
"""

import os
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite

from config import GESTURE_IMAGE_SIZE
from utils.resource_loader import resource_path


class GestureModel:
    """Gesture recognition model wrapper with dynamic loading."""
    
    def __init__(self, model_name="model", model_dir="resources/gesture_classifier"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.interpreter = None
        self.labels = []
        self.class_to_letter = {}  # Mapping from class name to letter
        
        # Try to load model
        try:
            self._load_model()
        except FileNotFoundError as e:
            # Model files not found - this is OK, user can load later
            print(f"Gesture model not found: {e}")
        except Exception as e:
            # Other errors
            print(f"Error loading gesture model: {e}")
    
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
            raise FileNotFoundError(f"Gesture model not found: {model_path}")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Gesture labels not found: {labels_path}")
        
        try:
            self.interpreter = tflite.Interpreter(model_path)
            self.interpreter.allocate_tensors()
            
            # Load labels
            with open(labels_path, "r") as f:
                lines = f.readlines()
                self.labels = []
                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        self.labels.append(parts[1])
                    else:
                        self.labels.append(parts[0])
        
        except Exception as e:
            raise RuntimeError(f"Failed to load gesture model: {e}")
    
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
    
    def preprocess_frame(self, frame):
        """
        Preprocess camera frame for model input.
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            Preprocessed numpy array
        """
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Resize and fit
        image = ImageOps.fit(image, GESTURE_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        image_array = np.asarray(image, dtype=np.float32)
        normalized = (image_array / 127.5) - 1
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def predict(self, input_data):
        """
        Run inference on preprocessed frame.
        
        Args:
            input_data: Preprocessed image array
            
        Returns:
            Tuple of (class_name, letter, confidence)
        """
        if self.interpreter is None:
            return None, None, 0.0
        
        # Run inference
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        
        self.interpreter.set_tensor(inp['index'], input_data)
        self.interpreter.invoke()
        
        # Get results
        prediction = self.interpreter.get_tensor(out['index'])[0]
        idx = np.argmax(prediction)
        class_name = self.labels[idx]
        confidence = prediction[idx]
        
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