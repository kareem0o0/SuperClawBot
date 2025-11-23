"""
Manual control panel UI component.
"""

from PySide6.QtWidgets import QGroupBox, QGridLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from config import STOP_DRIVE, STOP_ARM1, STOP_ARM2, STOP_ARM3, STOP_ALL, TOGGLE_LED
from .editable_label import EditableLabel


class ControlPanel(QGroupBox):
    """Manual control buttons for robot."""
    
    def __init__(self, backend, parent=None):
        super().__init__("", parent)  # Empty title, we'll add custom widget
        self.backend = backend
        self.all_buttons = []  # Store all button references
        
        # Create title widget with edit button
        title_widget = QWidget()
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(5, 5, 5, 5)
        title_layout.setSpacing(5)
        
        # Title label (editable)
        self.title_label = EditableLabel("🕹️ Manual Controls", "manual_controls_title", bold=True, font_size=14)
        title_layout.addWidget(self.title_label)
        
        # Push button to the right
        title_layout.addStretch()
        
        # Small edit/apply button (pen icon without background)
        self.edit_mode_btn = QPushButton("✏️")
        self.edit_mode_btn.setMaximumWidth(30)
        self.edit_mode_btn.setMaximumHeight(25)
        self.edit_mode_btn.setToolTip("Edit labels")
        self.edit_mode_active = False  # Track edit mode state manually
        self.edit_mode_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                font-size: 16px;
                padding: 0.5px;
            }
            QPushButton:hover {
                background: rgba(0, 0, 0, 0.1);
                border-radius: 3px;
            }
        """)
        self.edit_mode_btn.clicked.connect(self._toggle_edit_mode)
        title_layout.addWidget(self.edit_mode_btn)
        
        title_widget.setLayout(title_layout)
        
        # Set the title widget
        self.setTitle("")  # Clear default title
        
        self._init_ui()
        
        # Add title widget at the top of the layout
        layout = self.layout()
        layout.insertWidget(0, title_widget)
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Store all editable labels
        self.editable_labels = []
        
        # Drive controls section with editable title
        drive_title_widget = QWidget()
        drive_title_layout = QHBoxLayout()
        drive_title_layout.setContentsMargins(0, 0, 0, 0)
        self.drive_label = EditableLabel("🚗 Drive Controls", "drive_controls_title", bold=True, font_size=12)
        self.editable_labels.append(self.drive_label)
        drive_title_layout.addWidget(self.drive_label)
        drive_title_widget.setLayout(drive_title_layout)
        
        drive_group = QGroupBox()
        drive_main_layout = QVBoxLayout()
        drive_main_layout.addWidget(drive_title_widget)
        
        drive_layout = QGridLayout()
        
        btn_forward = QPushButton("⬆️ Forward")
        btn_forward.pressed.connect(lambda: self.backend.send_command('F'))
        btn_forward.released.connect(lambda: self.backend.send_command(STOP_DRIVE))
        drive_layout.addWidget(btn_forward, 0, 1)
        self.all_buttons.append(btn_forward)
        
        btn_left = QPushButton("⬅️ Left")
        btn_left.pressed.connect(lambda: self.backend.send_command('L'))
        btn_left.released.connect(lambda: self.backend.send_command(STOP_DRIVE))
        drive_layout.addWidget(btn_left, 1, 0)
        self.all_buttons.append(btn_left)
        
        btn_stop = QPushButton("⏹️ STOP")
        btn_stop.setStyleSheet("background: #f11444; font-weight: bold; color: white;")
        btn_stop.clicked.connect(lambda: self.backend.send_command(STOP_ALL))
        drive_layout.addWidget(btn_stop, 1, 1)
        self.all_buttons.append(btn_stop)
        
        btn_right = QPushButton("➡️ Right")
        btn_right.pressed.connect(lambda: self.backend.send_command('R'))
        btn_right.released.connect(lambda: self.backend.send_command(STOP_DRIVE))
        drive_layout.addWidget(btn_right, 1, 2)
        self.all_buttons.append(btn_right)
        
        btn_backward = QPushButton("⬇️ Backward")
        btn_backward.pressed.connect(lambda: self.backend.send_command('B'))
        btn_backward.released.connect(lambda: self.backend.send_command(STOP_DRIVE))
        drive_layout.addWidget(btn_backward, 2, 1)
        self.all_buttons.append(btn_backward)
        
        drive_main_layout.addLayout(drive_layout)
        drive_group.setLayout(drive_main_layout)
        layout.addWidget(drive_group)
        
        # Arm controls section with editable title
        arm_title_widget = QWidget()
        arm_title_layout = QHBoxLayout()
        arm_title_layout.setContentsMargins(0, 0, 0, 0)
        self.arm_label = EditableLabel("🦾 Arm Controls", "arm_controls_title", bold=True, font_size=12)
        self.editable_labels.append(self.arm_label)
        arm_title_layout.addWidget(self.arm_label)
        arm_title_widget.setLayout(arm_title_layout)
        
        arm_group = QGroupBox()
        arm_main_layout = QVBoxLayout()
        arm_main_layout.addWidget(arm_title_widget)
        
        arm_layout = QGridLayout()
        
        # Arm 1 - Column 0 with editable label
        self.arm1_label = EditableLabel("Arm 1", "arm1_label", bold=True, font_size=12)
        self.editable_labels.append(self.arm1_label)
        arm_layout.addWidget(self.arm1_label, 0, 0)
        
        btn_arm1_up = QPushButton("Up")
        btn_arm1_up.pressed.connect(lambda: self.backend.send_command('Z'))
        btn_arm1_up.released.connect(lambda: self.backend.send_command(STOP_ARM1))
        arm_layout.addWidget(btn_arm1_up, 1, 0)
        self.all_buttons.append(btn_arm1_up)
        
        btn_arm1_down = QPushButton("Down")
        btn_arm1_down.pressed.connect(lambda: self.backend.send_command('A'))
        btn_arm1_down.released.connect(lambda: self.backend.send_command(STOP_ARM1))
        arm_layout.addWidget(btn_arm1_down, 2, 0)
        self.all_buttons.append(btn_arm1_down)
        
        # Arm 2 - Column 1 with editable label
        self.arm2_label = EditableLabel("Arm 2", "arm2_label", bold=True, font_size=12)
        self.editable_labels.append(self.arm2_label)
        arm_layout.addWidget(self.arm2_label, 0, 1)
        
        btn_arm2_up = QPushButton("Up")
        btn_arm2_up.pressed.connect(lambda: self.backend.send_command('S'))
        btn_arm2_up.released.connect(lambda: self.backend.send_command(STOP_ARM2))
        arm_layout.addWidget(btn_arm2_up, 1, 1)
        self.all_buttons.append(btn_arm2_up)
        
        btn_arm2_down = QPushButton("Down")
        btn_arm2_down.pressed.connect(lambda: self.backend.send_command('X'))
        btn_arm2_down.released.connect(lambda: self.backend.send_command(STOP_ARM2))
        arm_layout.addWidget(btn_arm2_down, 2, 1)
        self.all_buttons.append(btn_arm2_down)
        
        # Arm 3 - Column 2 with editable label
        self.arm3_label = EditableLabel("Arm 3", "arm3_label", bold=True, font_size=12)
        self.editable_labels.append(self.arm3_label)
        arm_layout.addWidget(self.arm3_label, 0, 2)
        
        btn_arm3_cw = QPushButton("↻ CW")
        btn_arm3_cw.pressed.connect(lambda: self.backend.send_command('C'))
        btn_arm3_cw.released.connect(lambda: self.backend.send_command(STOP_ARM3))
        arm_layout.addWidget(btn_arm3_cw, 1, 2)
        self.all_buttons.append(btn_arm3_cw)
        
        btn_arm3_ccw = QPushButton("↺ CCW")
        btn_arm3_ccw.pressed.connect(lambda: self.backend.send_command('V'))
        btn_arm3_ccw.released.connect(lambda: self.backend.send_command(STOP_ARM3))
        arm_layout.addWidget(btn_arm3_ccw, 2, 2)
        self.all_buttons.append(btn_arm3_ccw)
        
        arm_main_layout.addLayout(arm_layout)
        arm_group.setLayout(arm_main_layout)
        layout.addWidget(arm_group)
        
        # LED toggle
        btn_led = QPushButton("💡 Toggle LED")
        btn_led.clicked.connect(lambda: self.backend.send_command(TOGGLE_LED))
        layout.addWidget(btn_led)
        self.all_buttons.append(btn_led)
        
        self.setLayout(layout)
    
    def _toggle_edit_mode(self):
        """Toggle edit mode for all labels."""
        if not self.edit_mode_active:
            # Enable edit mode
            self.edit_mode_active = True
            self.edit_mode_btn.setText("✓")
            self.edit_mode_btn.setToolTip("Apply changes")
            self.edit_mode_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: #4CAF50;
                    color: white;
                    font-size: 16px;
                    padding: 2px;
                    border-radius: 3px;
                }
            """)
            # Add title to editable labels
            self.title_label.enable_edit_mode()
            for label in self.editable_labels:
                label.enable_edit_mode()
        else:
            # Disable edit mode and save
            self.edit_mode_active = False
            self.edit_mode_btn.setText("✏️")
            self.edit_mode_btn.setToolTip("Edit labels")
            self.edit_mode_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: transparent;
                    font-size: 16px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background: rgba(0, 0, 0, 0.1);
                    border-radius: 3px;
                }
            """)
            # Save title and all labels
            self.title_label.disable_edit_mode(save=True)
            for label in self.editable_labels:
                label.disable_edit_mode(save=True)
    
    def refresh_theme(self):
        """Refresh button styles after theme change."""
        # Force style update for all buttons except STOP
        for button in self.all_buttons:
            if "STOP" not in button.text():
                # Clear inline styles to use palette
                button.setStyleSheet("")