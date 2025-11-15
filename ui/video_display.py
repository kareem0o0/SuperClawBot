"""
Video display widget for camera feed.
"""

import cv2
from PySide6.QtWidgets import QLabel, QGroupBox, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from config import VIDEO_WIDTH, VIDEO_HEIGHT


class VideoDisplay(QGroupBox):
    """Video display widget for live camera feed."""
    
    def __init__(self, parent=None):
            super().__init__("📹 Live Camera Feed (Gesture Mode)", parent)
            self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        self.video_label = QLabel("Waiting for camera...")
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setStyleSheet("""
            background: #000; 
            border: 3px solid #00ff88; 
            color: white;
            border-radius: 10px;
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.video_label)
        self.setLayout(layout)
    
    def update_frame(self, frame):
            """
            Update video display with new frame.
            
            Args:
                frame: OpenCV BGR frame or None to clear
            """
            if frame is None:
                # Clear the display
                self.video_label.clear()
                self.video_label.setText("Waiting for camera...")
                return
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame_rgb.shape
            bytes_per_line = 3 * width
            
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.video_label.setPixmap(scaled)