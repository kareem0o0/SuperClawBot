"""
Main UI window for robot controller.
"""

import time
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QGroupBox, QPushButton, QLabel, QTextEdit, QMessageBox,
                               QMenuBar, QMenu)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QAction

from config import (WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, 
                   MODE_KEYBOARD, MODE_VOICE, MODE_GESTURE,
                   STOP_DRIVE, STOP_ARM1, STOP_ARM2, STOP_ARM3, TOGGLE_LED, STOP_ALL)
from .video_display import VideoDisplay
from .bluetooth_panel import BluetoothPanel
from .control_panel import ControlPanel
from .model_config_dialog import ModelConfigDialog
from utils.logger import get_log_color, LogLevel


class RobotControlUI(QMainWindow):
    """Main UI window for robot control."""
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.signals = backend.signals
        
        # Connect signals
        self.signals.log_signal.connect(self.add_log)
        self.signals.frame_signal.connect(self.update_video)
        self.signals.mode_signal.connect(self.update_mode_display)
        self.signals.status_signal.connect(self.update_status)
        self.signals.voice_command_signal.connect(self.show_voice_command)
        self.signals.gesture_command_signal.connect(self.show_gesture_command)
        
        self.init_ui()
        self.add_log("UI initialized", "success")
    
    def init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Main widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left panel: Video + Status
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_display = VideoDisplay()
        left_panel.addWidget(self.video_display)
        
        # Status bar
        status_group = self._create_status_bar()
        left_panel.addWidget(status_group)
        
        main_layout.addLayout(left_panel, 2)
        
        # Right panel: Controls
        right_panel = QVBoxLayout()
        
        # Mode selection
        mode_group = self._create_mode_selector()
        right_panel.addWidget(mode_group)
        
        # Bluetooth panel
        self.bt_panel = BluetoothPanel(self.backend, self.signals)
        right_panel.addWidget(self.bt_panel)
        
        # Manual controls
        self.control_panel = ControlPanel(self.backend)
        right_panel.addWidget(self.control_panel)
        
        # Command info
        info_group = self._create_info_panel()
        right_panel.addWidget(info_group)
        
        # Log display
        log_group = self._create_log_panel()
        right_panel.addWidget(log_group)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)
    
    def _create_menu_bar(self):
        """Create menu bar with model configuration option."""
        menubar = self.menuBar()
        
        # Models menu
        models_menu = menubar.addMenu("Models")
        
        config_action = QAction("Configure Models", self)
        config_action.triggered.connect(self._open_model_config)
        models_menu.addAction(config_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _open_model_config(self):
        """Open model configuration dialog."""
        dialog = ModelConfigDialog(self.backend, self)
        dialog.exec()
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Robot Controller",
            "Unified Robot Controller v2.0\n\n"
            "Features:\n"
            "• Keyboard, Voice, and Gesture Control\n"
            "• Dynamic Model Loading\n"
            "• Customizable Class-to-Letter Mapping\n"
            "• Bluetooth Communication\n\n"
            "Configure models via Models → Configure Models menu."
        )
    
    def _create_status_bar(self):
            """Create status bar widget."""
            status_group = QGroupBox("📊 Connection & Mode Status")
            status_group.setMinimumHeight(80)  # Ensure enough space
            layout = QHBoxLayout()
            
            self.connection_status = QLabel("🔴 Disconnected")
            self.connection_status.setFont(QFont("Arial", 11, QFont.Bold))
            self.connection_status.setMinimumWidth(150)
            layout.addWidget(self.connection_status)
            
            self.mode_display = QLabel(f"Mode: {MODE_KEYBOARD}")
            self.mode_display.setFont(QFont("Arial", 11, QFont.Bold))
            self.mode_display.setStyleSheet("color: #00ff88;")
            self.mode_display.setMinimumWidth(150)
            layout.addWidget(self.mode_display)
            
            self.voice_indicator = QLabel("")
            self.voice_indicator.setFont(QFont("Arial", 10))
            self.voice_indicator.setMinimumWidth(120)
            layout.addWidget(self.voice_indicator)
            
            self.gesture_indicator = QLabel("")
            self.gesture_indicator.setFont(QFont("Arial", 10))
            self.gesture_indicator.setMinimumWidth(120)
            layout.addWidget(self.gesture_indicator)
            
            layout.addStretch()
            status_group.setLayout(layout)
            return status_group
    
    def _create_mode_selector(self):
        """Create mode selection widget."""
        mode_group = QGroupBox("🎮 Control Mode")
        layout = QVBoxLayout()
        
        self.keyboard_btn = QPushButton("⌨️ Keyboard Control")
        self.keyboard_btn.setCheckable(True)
        self.keyboard_btn.setChecked(True)
        self.keyboard_btn.clicked.connect(lambda: self.backend.switch_mode(MODE_KEYBOARD))
        layout.addWidget(self.keyboard_btn)
        
        self.voice_btn = QPushButton("🎤 Voice Control")
        self.voice_btn.setCheckable(True)
        self.voice_btn.clicked.connect(lambda: self.backend.switch_mode(MODE_VOICE))
        layout.addWidget(self.voice_btn)
        
        self.gesture_btn = QPushButton("👋 Gesture Control")
        self.gesture_btn.setCheckable(True)
        self.gesture_btn.clicked.connect(lambda: self.backend.switch_mode(MODE_GESTURE))
        layout.addWidget(self.gesture_btn)
        
        mode_group.setLayout(layout)
        return mode_group
    
    def _create_info_panel(self):
        """Create command info panel."""
        info_group = QGroupBox("ℹ️ Quick Reference")
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(150)
        info_text.setHtml("""
        <b>Dynamic Models:</b><br>
        Use <b>Models → Configure Models</b> to load new models and assign letters to classes.<br><br>
        <b>Keyboard (WASD + Number Pad):</b><br>
        W: Forward | S: Backward | A: Left | D: Right | 1/4: Arm1 | 3/6: Arm2 | 0/2: Arm3 | Q: LED
        """)
        layout.addWidget(info_text)
        
        info_group.setLayout(layout)
        return info_group
    
    def _create_log_panel(self):
        """Create activity log panel."""
        log_group = QGroupBox("📝 Activity Log")
        layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        layout.addWidget(self.log_display)
        
        log_group.setLayout(layout)
        return log_group
    
    # ========================================================
    #                  UI UPDATE METHODS
    # ========================================================
    
    def add_log(self, message, level="info"):
        """Add log message with color coding."""
        try:
            log_level = LogLevel(level)
        except ValueError:
            log_level = LogLevel.INFO
        
        color = get_log_color(log_level)
        timestamp = time.strftime("%H:%M:%S")
        self.log_display.append(
            f'<span style="color:{color};">[{timestamp}] {message}</span>'
        )
        self.log_display.ensureCursorVisible()
    
    def update_video(self, frame):
        """Update live video feed."""
        self.video_display.update_frame(frame)
    
    def update_mode_display(self, mode):
        """Update current control mode display."""
        self.mode_display.setText(f"Mode: <b>{mode}</b>")
        self.keyboard_btn.setChecked(mode == MODE_KEYBOARD)
        self.voice_btn.setChecked(mode == MODE_VOICE)
        self.gesture_btn.setChecked(mode == MODE_GESTURE)
    
    def update_status(self, status):
        """Update connection status indicator."""
        if status == "Connected":
            self.connection_status.setText("🟢 Connected")
            self.connection_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.connection_status.setText("🔴 Disconnected")
            self.connection_status.setStyleSheet("color: #ff4444; font-weight: bold;")
    
    def show_voice_command(self, command, confidence):
        """Show recognized voice command."""
        self.voice_indicator.setText(f"🎤 {command} ({confidence:.2f})")
        QTimer.singleShot(2000, lambda: self.voice_indicator.setText(""))
    
    def show_gesture_command(self, gesture, confidence):
        """Show recognized gesture command."""
        self.gesture_indicator.setText(f"👋 {gesture} ({confidence:.2f})")
        QTimer.singleShot(2000, lambda: self.gesture_indicator.setText(""))
    
    # ========================================================
    #                  KEYBOARD CONTROL
    # ========================================================
    
    def keyPressEvent(self, event):
        """Handle keyboard press for robot control."""
        if self.backend.current_mode != MODE_KEYBOARD:
            return super().keyPressEvent(event)
        
        key = event.key()
        cmd = None
        cmd_type = None
        
        # Drive controls (WASD)
        if key == Qt.Key_W:
            cmd, cmd_type = 'F', 'drive'
        elif key == Qt.Key_S:
            cmd, cmd_type = 'B', 'drive'
        elif key == Qt.Key_A:
            cmd, cmd_type = 'L', 'drive'
        elif key == Qt.Key_D:
            cmd, cmd_type = 'R', 'drive'
        
        # Arm 1 (1 / 4)
        elif key == Qt.Key_1:
            cmd, cmd_type = 'Z', 'arm1'
        elif key == Qt.Key_4:
            cmd, cmd_type = 'A', 'arm1'
        
        # Arm 2 (3 / 6)
        elif key == Qt.Key_3:
            cmd, cmd_type = 'S', 'arm2'
        elif key == Qt.Key_6:
            cmd, cmd_type = 'X', 'arm2'
        
        # Arm 3 (0 / 2)
        elif key == Qt.Key_0:
            cmd, cmd_type = 'C', 'arm3'
        elif key == Qt.Key_2:
            cmd, cmd_type = 'V', 'arm3'
        
        # LED toggle
        elif key == Qt.Key_Q:
            self.backend.send_command(TOGGLE_LED)
            return
        
        # Emergency stop
        elif key == Qt.Key_Escape:
            self.backend.stop_all_motors()
            return
        
        if cmd and cmd_type:
            self.backend.active_cmds[cmd_type] = cmd
            self.backend.send_command(cmd)
    
    def keyReleaseEvent(self, event):
        """Handle key release to stop motors."""
        if self.backend.current_mode != MODE_KEYBOARD:
            return super().keyReleaseEvent(event)
        
        key = event.key()
        stop_cmd = None
        cmd_type = None
        
        if key in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D):
            stop_cmd, cmd_type = STOP_DRIVE, 'drive'
        elif key in (Qt.Key_1, Qt.Key_4):
            stop_cmd, cmd_type = STOP_ARM1, 'arm1'
        elif key in (Qt.Key_3, Qt.Key_6):
            stop_cmd, cmd_type = STOP_ARM2, 'arm2'
        elif key in (Qt.Key_0, Qt.Key_2):
            stop_cmd, cmd_type = STOP_ARM3, 'arm3'
        
        if stop_cmd and cmd_type:
            # Only stop if this was the active command
            if self.backend.active_cmds.get(cmd_type):
                self.backend.send_command(stop_cmd)
                self.backend.active_cmds[cmd_type] = None
    
    # ========================================================
    #                  CLEANUP ON CLOSE
    # ========================================================
    
    def closeEvent(self, event):
        """Handle window close - clean shutdown."""
        reply = QMessageBox.question(
            self, "Exit", "Are you sure you want to quit?\nAll motors will be stopped.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.backend.cleanup()
            event.accept()
        else:
            event.ignore()
    def _create_menu_bar(self):
        """Create menu bar with model configuration and settings."""
        menubar = self.menuBar()
        
        # Models menu
        models_menu = menubar.addMenu("Models")
        
        config_action = QAction("Configure Models", self)
        config_action.triggered.connect(self._open_model_config)
        models_menu.addAction(config_action)
        
        profile_action = QAction("Manage Profiles", self)
        profile_action.triggered.connect(self._open_profile_manager)
        models_menu.addAction(profile_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        settings_menu.addAction(theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _open_profile_manager(self):
        """Open profile manager dialog."""
        from .profile_manager_dialog import ProfileManagerDialog
        dialog = ProfileManagerDialog(self.backend.profile_manager, self.backend, self)
        dialog.exec()
    
    def _toggle_theme(self):
        """Toggle application theme."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        new_theme = self.backend.theme_manager.toggle_theme(app)
        self.add_log(f"Theme changed to: {new_theme}", "success")
    def _create_menu_bar(self):
        """Create menu bar with model configuration and settings."""
        menubar = self.menuBar()
        
        # Models menu
        models_menu = menubar.addMenu("Models")
        
        config_action = QAction("Configure Models", self)
        config_action.triggered.connect(self._open_model_config)
        models_menu.addAction(config_action)
        
        profile_action = QAction("Manage Profiles", self)
        profile_action.triggered.connect(self._open_profile_manager)
        models_menu.addAction(profile_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        monitor_action = QAction("📡 Virtual BT Monitor", self)
        monitor_action.triggered.connect(self._open_virtual_monitor)
        tools_menu.addAction(monitor_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        settings_menu.addAction(theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _open_virtual_monitor(self):
        """Open virtual Bluetooth monitor window."""
        if not self.backend.bluetooth.is_virtual():
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Virtual Mode Required",
                "Virtual Bluetooth Monitor requires Virtual Connection mode.\n\n"
                "Would you like to switch to virtual mode now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.backend.bluetooth.connect_virtual()
            else:
                return
        
        from .virtual_bt_monitor import VirtualBluetoothMonitor
        monitor = VirtualBluetoothMonitor(self.backend, self)
        monitor.show()