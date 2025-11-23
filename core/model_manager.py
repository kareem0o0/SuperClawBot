"""
Model manager for dynamic loading and class-to-letter mapping.
"""

import os
import json
import shutil
from pathlib import Path

from utils.resource_loader import resource_path


class ModelManager:
    """Manages dynamic model loading and class-to-letter mappings."""
    
    def __init__(self, signal_emitter):
        self.signals = signal_emitter
        self.mappings_dir = "model_mappings"
        self._ensure_mappings_dir()
    
    def _ensure_mappings_dir(self):
        """Ensure mappings directory exists."""
        os.makedirs(self.mappings_dir, exist_ok=True)
    
    def _get_mapping_file(self, model_name, model_type):
        """Get path to mapping file for a model."""
        filename = f"{model_type}_{model_name}_mapping.json"
        return os.path.join(self.mappings_dir, filename)
    
    def load_labels_from_file(self, labels_path):
        """
        Load class labels from labels.txt file.
        
        Args:
            labels_path: Path to labels.txt file
            
        Returns:
            List of class names
        """
        try:
            with open(labels_path, "r") as f:
                lines = f.readlines()
                labels = []
                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        labels.append(parts[1])
                    elif len(parts) == 1:
                        labels.append(parts[0])
                return labels
        except Exception as e:
            self.signals.log_signal.emit(f"Error loading labels: {e}", "error")
            return []
    
    def create_default_mapping(self, labels):
        """
        Create default mapping (class name -> class name).
        
        Args:
            labels: List of class names
            
        Returns:
            Dictionary mapping class to letter
        """
        return {label: label for label in labels}
    
    def save_mapping(self, model_name, model_type, mapping):
        """
        Save class-to-letter mapping to JSON file.
        
        Args:
            model_name: Name of the model
            model_type: Type ('voice' or 'gesture')
            mapping: Dictionary of class->letter mappings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mapping_file = self._get_mapping_file(model_name, model_type)
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            
            self.signals.log_signal.emit(f"Mapping saved: {mapping_file}", "success")
            return True
        except Exception as e:
            self.signals.log_signal.emit(f"Error saving mapping: {e}", "error")
            return False
    
    def load_mapping(self, model_name, model_type):
        """
        Load class-to-letter mapping from JSON file.
        
        Args:
            model_name: Name of the model
            model_type: Type ('voice' or 'gesture')
            
        Returns:
            Dictionary of class->letter mappings, or None if not found
        """
        try:
            mapping_file = self._get_mapping_file(model_name, model_type)
            if not os.path.exists(mapping_file):
                return None
            
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            
            return mapping
        except Exception as e:
            self.signals.log_signal.emit(f"Error loading mapping: {e}", "error")
            return None
    
    def validate_mapping(self, mapping):
        """
        Validate that all letters are unique.
        
        Args:
            mapping: Dictionary of class->letter mappings
            
        Returns:
            Tuple (is_valid, duplicate_letter, duplicate_classes)
        """
        letter_to_classes = {}
        
        for class_name, letter in mapping.items():
            if letter in letter_to_classes:
                # Duplicate found
                letter_to_classes[letter].append(class_name)
            else:
                letter_to_classes[letter] = [class_name]
        
        # Check for duplicates
        for letter, classes in letter_to_classes.items():
            if len(classes) > 1:
                return False, letter, classes
        
        return True, None, None
    
    def install_model(self, tflite_path, labels_path, model_type, destination_dir):
        """
        Copy model files to the resources directory.
        
        Args:
            tflite_path: Path to .tflite file
            labels_path: Path to labels.txt file
            model_type: Type ('voice' or 'gesture')
            destination_dir: Destination directory in resources
            
        Returns:
            Tuple (model_name, success)
        """
        try:
            # Ensure destination exists
            os.makedirs(destination_dir, exist_ok=True)
            
            # Get model name from tflite filename
            model_name = Path(tflite_path).stem
            
            # Check against default model names to prevent overwriting
            from config import DEFAULT_GESTURE_MODEL, DEFAULT_VOICE_MODEL
            
            is_default_name = False
            if model_type == 'gesture' and model_name == DEFAULT_GESTURE_MODEL:
                is_default_name = True
            elif model_type == 'voice' and model_name == DEFAULT_VOICE_MODEL:
                is_default_name = True
            
            # If it matches default or file already exists, append timestamp
            dest_tflite = os.path.join(destination_dir, f"{model_name}.tflite")
            if is_default_name or os.path.exists(dest_tflite):
                import time
                timestamp = int(time.time())
                model_name = f"{model_name}_{timestamp}"
                self.signals.log_signal.emit(f"Model renamed to {model_name} to prevent overwriting", "info")
            
            # Copy files
            dest_tflite = os.path.join(destination_dir, f"{model_name}.tflite")
            dest_labels = os.path.join(destination_dir, f"{model_name}_labels.txt")
            
            shutil.copy(tflite_path, dest_tflite)
            shutil.copy(labels_path, dest_labels)
            
            self.signals.log_signal.emit(f"Model installed: {model_name}", "success")
            return model_name, True
        except Exception as e:
            self.signals.log_signal.emit(f"Error installing model: {e}", "error")
            return None, False
    
    def get_available_models(self, model_type):
        """
        Get list of available models of a given type.
        
        Args:
            model_type: Type ('voice' or 'gesture')
            
        Returns:
            List of model names
        """
        if model_type == "voice":
            search_dir = "resources/sound_classifier"
        else:
            search_dir = "resources/gesture_classifier"
        
        if not os.path.exists(search_dir):
            return []
        
        models = []
        for file in os.listdir(search_dir):
            if file.endswith(".tflite"):
                model_name = file.replace(".tflite", "")
                models.append(model_name)
        
        return models