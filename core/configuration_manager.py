"""
Configuration manager for saving/loading custom gestures and voices.
"""

import json
import os
from datetime import datetime


class ConfigurationManager:
    """Manages saving and loading of custom configurations."""
    
    def __init__(self, config_dir="saved_configurations"):
        self.config_dir = config_dir
        self.recent_file = os.path.join(config_dir, "recent.json")
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists."""
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_configuration(self, name, gesture_controller, voice_controller, profile_manager=None):
        """
        Save current custom gestures and voices.
        
        Args:
            name: Configuration name
            gesture_controller: Gesture controller instance
            voice_controller: Voice controller instance
            profile_manager: ProfileManager instance (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_data = {
                'name': name,
                'created': datetime.now().isoformat(),
                'custom_gestures': gesture_controller.custom_gesture_manager.to_dict(),
                'custom_voices': voice_controller.custom_voice_manager.to_dict(),
                'gesture_model': gesture_controller.current_model_name,
                'voice_model': voice_controller.current_model_name,
                'gesture_mapping': gesture_controller.model.class_to_letter if gesture_controller.model else {},
                'voice_mapping': voice_controller.model.class_to_letter if voice_controller.model else {}
            }
            
            # Save active profile names if profile_manager is provided
            if profile_manager:
                config_data['active_voice_profile'] = profile_manager.active_voice_profile
                config_data['active_gesture_profile'] = profile_manager.active_gesture_profile
            
            # Save configuration file
            filename = f"{name.replace(' ', '_')}.json"
            filepath = os.path.join(self.config_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Update recent list
            self._add_to_recent(name, filepath)
            
            return True
        
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_configuration(self, filepath, gesture_controller, voice_controller, profile_manager=None):
        """
        Load a saved configuration.
        
        Args:
            filepath: Path to configuration file
            gesture_controller: Gesture controller instance
            voice_controller: Voice controller instance
            profile_manager: ProfileManager instance (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Always load the saved models and mappings first (configuration takes priority)
            
            # Load voice model and mapping
            saved_voice_model = config_data.get('voice_model')
            if saved_voice_model and saved_voice_model != voice_controller.current_model_name:
                print(f"Switching voice model to: {saved_voice_model}")
                voice_controller.load_new_model(saved_voice_model)
            
            # Apply saved voice mapping
            saved_voice_mapping = config_data.get('voice_mapping')
            if saved_voice_mapping and voice_controller.model:
                voice_controller.model.set_mapping(saved_voice_mapping)
                print("Applied saved voice mapping")
            
            # Load gesture model and mapping
            saved_gesture_model = config_data.get('gesture_model')
            if saved_gesture_model and saved_gesture_model != gesture_controller.current_model_name:
                print(f"Switching gesture model to: {saved_gesture_model}")
                gesture_controller.load_new_model(saved_gesture_model)
            
            # Apply saved gesture mapping
            saved_gesture_mapping = config_data.get('gesture_mapping')
            if saved_gesture_mapping and gesture_controller.model:
                gesture_controller.model.set_mapping(saved_gesture_mapping)
                print("Applied saved gesture mapping")
            
            # If profile_manager is provided, set the active profiles (if they match the loaded models)
            if profile_manager:
                active_voice_profile = config_data.get('active_voice_profile')
                if active_voice_profile:
                    voice_profile = profile_manager.get_profile(active_voice_profile)
                    if voice_profile:
                        profile_model_name = voice_profile.model_path.split('/')[-1].replace('.tflite', '')
                        # Only set as active if the profile's model matches what we loaded
                        if profile_model_name == saved_voice_model:
                            profile_manager.set_active_profile(active_voice_profile)
                            print(f"Set active voice profile: {active_voice_profile}")
                        else:
                            print(f"Note: Profile '{active_voice_profile}' uses different model, not setting as active")
                
                active_gesture_profile = config_data.get('active_gesture_profile')
                if active_gesture_profile:
                    gesture_profile = profile_manager.get_profile(active_gesture_profile)
                    if gesture_profile:
                        profile_model_name = gesture_profile.model_path.split('/')[-1].replace('.tflite', '')
                        # Only set as active if the profile's model matches what we loaded
                        if profile_model_name == saved_gesture_model:
                            profile_manager.set_active_profile(active_gesture_profile)
                            print(f"Set active gesture profile: {active_gesture_profile}")
                        else:
                            print(f"Note: Profile '{active_gesture_profile}' uses different model, not setting as active")
            
            # Load custom gestures (always load these on top)
            gesture_controller.custom_gesture_manager.from_dict(
                config_data.get('custom_gestures', {})
            )
            
            # Load custom voices (always load these on top)
            voice_controller.custom_voice_manager.from_dict(
                config_data.get('custom_voices', {})
            )
            
            # Update recent list
            self._add_to_recent(config_data['name'], filepath)
            
            return True
        
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def _add_to_recent(self, name, filepath):
        """Add configuration to recent list."""
        try:
            # Load existing recent list
            recent = []
            if os.path.exists(self.recent_file):
                with open(self.recent_file, 'r') as f:
                    recent = json.load(f)
            
            # Add new entry (remove if already exists)
            recent = [r for r in recent if r['filepath'] != filepath]
            recent.insert(0, {
                'name': name,
                'filepath': filepath,
                'accessed': datetime.now().isoformat()
            })
            
            # Keep only last 10
            recent = recent[:10]
            
            # Save
            with open(self.recent_file, 'w') as f:
                json.dump(recent, f, indent=2)
        
        except Exception as e:
            print(f"Error updating recent list: {e}")
    
    def get_recent_configurations(self):
        """
        Get list of recent configurations.
        
        Returns:
            List of recent configuration dictionaries
        """
        if not os.path.exists(self.recent_file):
            return []
        
        try:
            with open(self.recent_file, 'r') as f:
                recent = json.load(f)
            
            # Filter out non-existent files
            recent = [r for r in recent if os.path.exists(r['filepath'])]
            
            return recent
        
        except Exception as e:
            print(f"Error loading recent configurations: {e}")
            return []
    
    def get_all_configurations(self):
        """
        Get list of all saved configurations.
        
        Returns:
            List of configuration file paths
        """
        if not os.path.exists(self.config_dir):
            return []
        
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json') and file != 'recent.json':
                filepath = os.path.join(self.config_dir, file)
                configs.append(filepath)
        
        return configs