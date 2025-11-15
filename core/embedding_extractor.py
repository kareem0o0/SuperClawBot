"""
Embedding extractor for gesture auto-learning.
Extracts feature vectors from TFLite models without classification.
"""

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps

from config import GESTURE_IMAGE_SIZE


class EmbeddingExtractor:
    """Extracts embeddings from gesture models for custom gesture learning."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.embedding_layer_index = None
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model and find embedding layer."""
        try:
            self.interpreter = tflite.Interpreter(self.model_path)
            self.interpreter.allocate_tensors()  # CRITICAL: Allocate first
            
            # Get output details - use the main output layer
            output_details = self.interpreter.get_output_details()
            
            # For most models, just use the output layer itself as "embeddings"
            # This works fine for similarity comparison
            self.embedding_layer_index = output_details[0]['index']
            
            print(f"EmbeddingExtractor loaded successfully, using output index: {self.embedding_layer_index}")
        
        except Exception as e:
            print(f"Failed to load embedding extractor: {e}")
            raise RuntimeError(f"Failed to load embedding extractor: {e}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess camera frame for model input.
        
        Args:
            frame: OpenCV BGR or RGB frame
            
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
    
    def extract_embedding(self, input_data):
        """
        Extract embedding vector from preprocessed frame.
        
        Args:
            input_data: Preprocessed image array
            
        Returns:
            Embedding vector as numpy array
        """
        if self.interpreter is None:
            print("Interpreter is None!")
            return None
        
        try:
            # Run inference
            inp = self.interpreter.get_input_details()[0]
            
            self.interpreter.set_tensor(inp['index'], input_data)
            self.interpreter.invoke()
            
            # Get embedding (output layer)
            embedding = self.interpreter.get_tensor(self.embedding_layer_index)
            
            # Flatten to 1D vector
            embedding_flat = embedding.flatten()
            
            return embedding_flat
        
        except Exception as e:
            print(f"Error in extract_embedding: {e}")
            return None
    
    def extract_from_frame(self, frame):
        """
        Extract embedding directly from a frame.
        
        Args:
            frame: OpenCV BGR or RGB frame
            
        Returns:
            Embedding vector
        """
        input_data = self.preprocess_frame(frame)
        return self.extract_embedding(input_data)


class CustomGestureManager:
    """Manages custom learned gestures."""
    
    def __init__(self):
        self.custom_gestures = {}  # name -> {'embeddings': [], 'letter': str}
    
    def add_gesture(self, name, embeddings, letter):
        """
        Add a new custom gesture.
        
        Args:
            name: Gesture name
            embeddings: List of embedding vectors
            letter: Assigned letter
        """
        self.custom_gestures[name] = {
            'embeddings': [emb.tolist() for emb in embeddings],
            'letter': letter
        }
    
    def remove_gesture(self, name):
        """Remove a custom gesture."""
        if name in self.custom_gestures:
            del self.custom_gestures[name]
    
    def get_gesture_letter(self, name):
        """Get letter assigned to a gesture."""
        if name in self.custom_gestures:
            return self.custom_gestures[name]['letter']
        return None
    
    def update_letter(self, name, new_letter):
        """Update letter for a custom gesture."""
        if name in self.custom_gestures:
            self.custom_gestures[name]['letter'] = new_letter
    
    def predict(self, embedding, threshold=0.8):
        """
        Find matching custom gesture.
        
        Args:
            embedding: Current frame embedding
            threshold: Similarity threshold (0-1)
            
        Returns:
            Tuple of (gesture_name, letter, similarity) or (None, None, 0)
        """
        if embedding is None:
            return None, None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, data in self.custom_gestures.items():
            # Compare with all stored embeddings for this gesture
            similarities = []
            for stored_emb in data['embeddings']:
                stored_emb_np = np.array(stored_emb)
                # Cosine similarity
                similarity = self._cosine_similarity(embedding, stored_emb_np)
                similarities.append(similarity)
            
            # Use average similarity
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = name
        
        if best_similarity > threshold:
            return best_match, self.custom_gestures[best_match]['letter'], best_similarity
        
        return None, None, 0
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return self.custom_gestures.copy()
    
    def from_dict(self, data):
        """Load from dictionary."""
        self.custom_gestures = data.copy()
    
    def get_all_gestures(self):
        """Get list of all custom gesture names."""
        return list(self.custom_gestures.keys())