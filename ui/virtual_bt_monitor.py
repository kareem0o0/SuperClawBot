"""
Virtual Bluetooth data visualization window.
Real-time display of commands sent through virtual connection.
"""

import time
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                               QTableWidgetItem, QPushButton, QLabel, QHeaderView,
                               QGroupBox, QTextEdit)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont


class VirtualBluetoothMonitor(QDialog):
    """Real-time monitor for virtual Bluetooth commands."""
    
    # Command descriptions
    COMMAND_DESCRIPTIONS = {
        'F': 'Forward',
        'B': 'Backward',
        'L': 'Left',
        'R': 'Right',
        'Z': 'Arm1 Up',
        'A': 'Arm1 Down',
        'S': 'Arm2 Up',
        'X': 'Arm2 Down',
        'C': 'Arm3 Clockwise',
        'V': 'Arm3 Counter-CW',
        'Q': 'Toggle LED',
        '0': 'Stop Drive',
        'a': 'Stop Arm1',
        's': 'Stop Arm2',
        'c': 'Stop Arm3',
        '!': 'EMERGENCY STOP'
    }
    
    # Mode colors
    MODE_COLORS = {
        'KEYBOARD': QColor(100, 150, 255),    # Blue
        'VOICE': QColor(100, 255, 150),       # Green
        'GESTURE': QColor(255, 150, 100),     # Orange
        'UNKNOWN': QColor(200, 200, 200)      # Gray
    }
    
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.setWindowTitle("🔍 Virtual Bluetooth Monitor")
        self.setMinimumSize(900, 600)
        
        # Track statistics
        self.total_commands = 0
        self.mode_counts = {'KEYBOARD': 0, 'VOICE': 0, 'GESTURE': 0}
        
        self._init_ui()
        self._connect_signals()
        self._start_updates()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("📡 Virtual Bluetooth Data Stream Monitor")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Status info
        status_group = QGroupBox("Connection Status")
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("🟢 VIRTUAL MODE - Simulation Active")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #00ff88;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.stats_label = QLabel("Commands: 0 | KB: 0 | Voice: 0 | Gesture: 0")
        self.stats_label.setFont(QFont("Arial", 9))
        status_layout.addWidget(self.stats_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Command table
        table_group = QGroupBox("Command History (Real-Time)")
        table_layout = QVBoxLayout()
        
        self.command_table = QTableWidget()
        self.command_table.setColumnCount(5)
        self.command_table.setHorizontalHeaderLabels([
            "Time", "Command", "Description", "Mode", "Raw Data"
        ])
        
        # Set column widths
        header = self.command_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Time
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Command
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Description
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Mode
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Raw
        
        self.command_table.setAlternatingRowColors(True)
        self.command_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        table_layout.addWidget(self.command_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Command legend
        legend_group = QGroupBox("Command Reference")
        legend_layout = QVBoxLayout()
        
        legend_text = QTextEdit()
        legend_text.setReadOnly(True)
        legend_text.setMaximumHeight(120)
        legend_text.setHtml("""
        <b>Drive:</b> F=Forward, B=Backward, L=Left, R=Right, 0=Stop<br>
        <b>Arms:</b> Z/A=Arm1, S/X=Arm2, C/V=Arm3, a/s/c=Stop Arms<br>
        <b>Special:</b> Q=LED Toggle, !=Emergency Stop<br>
        <b>Mode Colors:</b> 
        <span style="color:#6496FF;">■ Keyboard</span> | 
        <span style="color:#64FF96;">■ Voice</span> | 
        <span style="color:#FF9664;">■ Gesture</span>
        """)
        legend_layout.addWidget(legend_text)
        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🗑️ Clear History")
        clear_btn.clicked.connect(self._clear_history)
        btn_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("💾 Export to File")
        export_btn.clicked.connect(self._export_history)
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect to backend signals."""
        # Connect mode signal
        self.backend.signals.mode_signal.connect(self._on_mode_changed)
    
    def _start_updates(self):
        """Start periodic UI updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # Update every 100ms
    
    def _update_display(self):
        """Update display with latest commands."""
        if not self.backend.bluetooth or not hasattr(self.backend.bluetooth, 'connection'):
            return
        
        connection = self.backend.bluetooth.connection
        if not connection or not hasattr(connection, 'get_history'):
            return
        
        history = connection.get_history()
        
        # Update table if new commands
        if len(history) != self.command_table.rowCount():
            self._refresh_table(history)
            self._update_statistics(history)
    
    def _refresh_table(self, history):
        """Refresh command table with history."""
        self.command_table.setRowCount(len(history))
        
        for i, cmd_data in enumerate(history):
            row = len(history) - 1 - i  # Newest first
            
            # Time
            time_item = QTableWidgetItem(cmd_data['timestamp_str'])
            time_item.setFont(QFont("Courier", 9))
            self.command_table.setItem(row, 0, time_item)
            
            # Command
            command = cmd_data['command']
            cmd_item = QTableWidgetItem(command)
            cmd_item.setFont(QFont("Courier", 11, QFont.Bold))
            cmd_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code emergency stop
            if command == '!':
                cmd_item.setForeground(QColor(255, 0, 0))
            
            self.command_table.setItem(row, 1, cmd_item)
            
            # Description
            description = self.COMMAND_DESCRIPTIONS.get(command, 'Unknown')
            desc_item = QTableWidgetItem(description)
            self.command_table.setItem(row, 2, desc_item)
            
            # Mode
            mode = self.backend.current_mode
            mode_item = QTableWidgetItem(mode)
            mode_item.setFont(QFont("Arial", 9, QFont.Bold))
            mode_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code by mode
            mode_color = self.MODE_COLORS.get(mode, self.MODE_COLORS['UNKNOWN'])
            mode_item.setForeground(mode_color)
            
            self.command_table.setItem(row, 3, mode_item)
            
            # Raw data (hex + ascii)
            raw_hex = ' '.join(f'{ord(c):02X}' for c in command)
            raw_item = QTableWidgetItem(f"{raw_hex} ({repr(command)})")
            raw_item.setFont(QFont("Courier", 8))
            self.command_table.setItem(row, 4, raw_item)
        
        # Scroll to top (newest)
        self.command_table.scrollToTop()
    
    def _update_statistics(self, history):
        """Update statistics display."""
        self.total_commands = len(history)
        
        # Count by mode (approximate - use current mode for recent commands)
        # This is simplified; for exact tracking, mode should be stored with each command
        self.stats_label.setText(
            f"Commands: {self.total_commands} | "
            f"Total Sent: {self.total_commands}"
        )
    
    def _on_mode_changed(self, mode):
        """Handle mode change."""
        # Mode is tracked in backend, no action needed here
        pass
    
    def _clear_history(self):
        """Clear command history."""
        if self.backend.bluetooth and hasattr(self.backend.bluetooth, 'connection'):
            connection = self.backend.bluetooth.connection
            if connection:
                connection.clear_history()
        
        self.command_table.setRowCount(0)
        self.total_commands = 0
        self._update_statistics([])
    
    def _export_history(self):
        """Export command history to file."""
        from PySide6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Command History",
            f"bt_monitor_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        
        if not filename:
            return
        
        try:
            if not self.backend.bluetooth or not hasattr(self.backend.bluetooth, 'connection'):
                return
            
            connection = self.backend.bluetooth.connection
            if not connection:
                return
            
            history = connection.get_history()
            
            with open(filename, 'w') as f:
                f.write("Virtual Bluetooth Command History\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                if filename.endswith('.csv'):
                    f.write("Timestamp,Command,Description,Raw\n")
                    for cmd_data in history:
                        command = cmd_data['command']
                        desc = self.COMMAND_DESCRIPTIONS.get(command, 'Unknown')
                        raw = ' '.join(f'{ord(c):02X}' for c in command)
                        f.write(f"{cmd_data['timestamp_str']},{command},{desc},{raw}\n")
                else:
                    for cmd_data in history:
                        command = cmd_data['command']
                        desc = self.COMMAND_DESCRIPTIONS.get(command, 'Unknown')
                        raw = ' '.join(f'{ord(c):02X}' for c in command)
                        f.write(f"[{cmd_data['timestamp_str']}] {command} - {desc} (Raw: {raw})\n")
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Success", f"Exported {len(history)} commands to:\n{filename}")
        
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def closeEvent(self, event):
        """Stop updates on close."""
        self.update_timer.stop()
        event.accept()