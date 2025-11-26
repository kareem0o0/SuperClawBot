"""
Bluetooth connection panel UI component.
"""

import platform
import re
import threading

from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QListWidget, QMessageBox, QListWidgetItem)
from PySide6.QtCore import Signal, Slot


class BluetoothPanel(QGroupBox):
    """Bluetooth device discovery and connection panel."""

    devices_found = Signal(list)
    scan_error_signal = Signal(str)
    connection_failed_signal = Signal(str)

    def __init__(self, backend, signal_emitter, parent=None):
        super().__init__("Bluetooth Setup", parent)
        self.backend = backend
        self.signals = signal_emitter
        self.selected_port = None

        self._init_ui()

        self.devices_found.connect(self._update_scan_result)
        self.scan_error_signal.connect(self._scan_error)
        self.connection_failed_signal.connect(self._connection_failed)

        # Load serial ports on startup
        self.refresh_serial_ports()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Status label
        self.bt_status = QLabel("Status: Not connected")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        layout.addWidget(self.bt_status)

        # Virtual connection button (toggle)
        self.virtual_btn = QPushButton("🔧 Connect Virtual (Testing Mode)")
        self.virtual_btn.setStyleSheet("background-color: #6495ED; font-weight: bold;")
        self.virtual_btn.setCheckable(True)
        self.virtual_btn.clicked.connect(self.toggle_virtual)
        layout.addWidget(self.virtual_btn)

        # Scan buttons
        btn_layout = QHBoxLayout()

        scan_btn = QPushButton("Scan for Bluetooth Devices")
        scan_btn.clicked.connect(self.scan_devices)
        btn_layout.addWidget(scan_btn)

        refresh_serial_btn = QPushButton("Refresh Serial Ports")
        refresh_serial_btn.clicked.connect(self.refresh_serial_ports)
        btn_layout.addWidget(refresh_serial_btn)

        layout.addLayout(btn_layout)

        # Device list
        self.bt_list = QListWidget()
        self.bt_list.itemClicked.connect(self.select_device)
        layout.addWidget(self.bt_list)

        # Connection button
        self.connect_btn = QPushButton("Connect to Selected Port")
        self.connect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_device)
        layout.addWidget(self.connect_btn)

        self.setLayout(layout)

    def toggle_virtual(self):
        """Toggle virtual Bluetooth connection."""
        if self.virtual_btn.isChecked():
            if self.backend.bluetooth.connect_virtual():
                self.bt_status.setText("VIRTUAL MODE - Simulation Active")
                self.bt_status.setStyleSheet("color: #6495ED; font-weight: bold;")
                self.signals.log_signal.emit("Virtual Bluetooth ready for testing", "success")
                self.virtual_btn.setText("🔌 Disconnect Virtual")
            else:
                self.bt_status.setText("Virtual connection failed")
                self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
                self.virtual_btn.setChecked(False)
        else:
            self.backend.bluetooth.disconnect()
            self.bt_status.setText("Status: Not connected")
            self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
            self.signals.log_signal.emit("Virtual Bluetooth disconnected", "info")
            self.virtual_btn.setText("🔧 Connect Virtual (Testing Mode)")

    def scan_devices(self):
        """Start Bluetooth device discovery."""
        self.bt_list.clear()
        self.bt_status.setText("Scanning for devices...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.signals.log_signal.emit("Starting Bluetooth discovery...", "info")

        thread = threading.Thread(target=self._scan_devices_thread, daemon=True)
        thread.start()

    def _scan_devices_thread(self):
        """Background thread for device discovery."""
        try:
            devices = self.backend.bluetooth.scan_devices()
            self.devices_found.emit(devices)
        except Exception as e:
            self.scan_error_signal.emit(str(e))

    def refresh_serial_ports(self):
        """Refresh the list of available serial ports."""
        self.bt_list.clear()
        self.bt_status.setText("Refreshing serial ports...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")

        ports = self.backend.bluetooth.list_serial_ports()
        self._update_scan_result(ports, is_serial=True)

    @Slot(list)
    def _update_scan_result(self, devices, is_serial=False):
        """Update UI with scan results."""
        self.bt_list.clear()

        if not devices:
            self.bt_status.setText("No devices found")
            self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
            self.signals.log_signal.emit("No devices found.", "warning")
            return

        if is_serial:
            for port in devices:
                is_bluetooth = ("Bluetooth" in port.description or "BTHENUM" in str(port.hwid).upper())
                prefix = "🔵 [BT-SERIAL]" if is_bluetooth else "[SERIAL]"
                display_text = f"{prefix} {port.device} - {port.description}"
                item = QListWidgetItem(display_text)
                item.setData(1, ("serial", port.device))
                self.bt_list.addItem(item)
        else:
            for device_type, name, address in devices:
                prefix = "[BLE]" if device_type == "ble" else "[CLASSIC]"
                display_text = f"{prefix} {name} ({address})"
                item = QListWidgetItem(display_text)
                item.setData(1, (device_type, address))
                self.bt_list.addItem(item)

        self.bt_status.setText(f"Found {len(devices)} device(s)")
        self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.signals.log_signal.emit(f"Found {len(devices)} device(s)", "success")

    @Slot(str)
    def _scan_error(self, msg):
        """Handle scan error."""
        self.bt_status.setText("Scan failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Scan error: {msg}", "error")

    def select_device(self, item):
        """Handle device selection."""
        device_data = item.data(1)
        if not device_data:
            return

        device_type, device_id = device_data
        if device_type == "serial":
            self.selected_port = device_id
            self.connect_btn.setEnabled(True)
            self.bt_status.setText(f"Selected: {self.selected_port}")
            self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.connect_btn.setEnabled(False)
            QMessageBox.information(self, "Connection Info",
                                      "To connect, please pair the device in your OS settings, "
                                      "then refresh the serial ports list and select the COM port.")

    def connect_device(self):
        """Connect to the selected serial port."""
        if not self.selected_port:
            self.signals.log_signal.emit("No serial port selected!", "error")
            return

        self.bt_status.setText(f"Connecting to {self.selected_port}...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")

        thread = threading.Thread(target=self._connect_serial_thread, daemon=True)
        thread.start()

    def _connect_serial_thread(self):
        """Background thread for serial connection."""
        success = self.backend.bluetooth.connect_serial(self.selected_port)
        if success:
            self.bt_status.setText(f"Connected to {self.selected_port}")
            self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.connection_failed_signal.emit("Serial connection failed")

    @Slot(str)
    def _connection_failed(self, msg):
        """Handle connection failure."""
        self.bt_status.setText("Connection failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Connection failed: {msg}", "error")
