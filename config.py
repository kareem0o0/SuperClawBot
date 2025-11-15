"""
Configuration and Constants
All global configuration values for the robot controller.
"""

import os

# ============================================================
#                    BLUETOOTH CONFIGURATION
# ============================================================
BLUETOOTH_PORT = "/dev/rfcomm0"
BLUETOOTH_BAUD = 9600
DEFAULT_RFCOMM_CHANNEL = 1

# ============================================================
#                    VOICE CONTROL CONFIGURATION
# ============================================================
VOICE_MODEL_PATH = "resources/sound_classifier/soundclassifier_with_metadata.tflite"
VOICE_LABELS_PATH = "resources/sound_classifier/labels.txt"
VOICE_SAMPLE_RATE = 44100
VOICE_OVERLAP = 0.5
VOICE_CONFIDENCE_THRESHOLD = 0.7

# ============================================================
#                    GESTURE CONTROL CONFIGURATION
# ============================================================
GESTURE_MODEL_PATH = "resources/gesture_classifier/model.tflite"
GESTURE_LABELS_PATH = "resources/gesture_classifier/labels.txt"
GESTURE_CONFIDENCE_THRESHOLD = 0.7
GESTURE_COOLDOWN = 1.0
GESTURE_IMAGE_SIZE = (224, 224)

# ============================================================
#                    TIMING CONFIGURATION
# ============================================================
COMMAND_DURATION = 2.0
COOLDOWN_TIME = 1.0

# ============================================================
#                    ROBOT COMMANDS
# ============================================================

# Stop commands
STOP_DRIVE = '0'
STOP_ARM1 = 'a'
STOP_ARM2 = 's'
STOP_ARM3 = 'c'
STOP_ALL = '!'
TOGGLE_LED = 'Q'

# ============================================================
#                    UI CONFIGURATION
# ============================================================
WINDOW_TITLE = "🤖 Unified Robot Controller v2.0"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600

# Control modes
MODE_KEYBOARD = "KEYBOARD"
MODE_VOICE = "VOICE"
MODE_GESTURE = "GESTURE"

# ============================================================
#                    CAMERA CONFIGURATION
# ============================================================
MAX_CAMERA_TRIES = 5
# ============================================================
#                    DEFAULT MAPPINGS
# ============================================================

# Default voice command mappings
DEFAULT_VOICE_MAPPING = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
    "up": "Z",
    "down": "A",
    "2up": "S",
    "2down": "X",
    "clockwise": "C",
    "anti": "V",
    "clap": "Q",
    "stop": "!",
    "Noise": ".",
}

# Default gesture command mappings
DEFAULT_GESTURE_MAPPING = {
    "start": "F",
    "stop": "!",
}
# ============================================================
#                    MODEL MANAGEMENT
# ============================================================
# Default model names (without extensions)
DEFAULT_VOICE_MODEL = "soundclassifier_with_metadata"
DEFAULT_GESTURE_MODEL = "model"

# Model directories
VOICE_MODEL_DIR = "resources/sound_classifier"
GESTURE_MODEL_DIR = "resources/gesture_classifier"

# Mapping storage
MAPPINGS_DIR = "model_mappings"