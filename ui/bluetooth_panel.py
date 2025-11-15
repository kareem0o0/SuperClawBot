"""
Bluetooth connection panel UI component.
"""

import re
import subprocess
import time
import threading
import bluetooth

from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QListWidget, QSpinBox)
from PySide6.QtCore import QTimer


class BluetoothPanel(QGroupBox):
    """Bluetooth device discovery and connection panel."""
    
    def __init__(self, backend, signal_emitter, parent=None):
        super().__init__("Bluetooth Setup", parent)
        self.backend = backend
        self.signals = signal_emitter
        self.discovered_devices = []
        self.selected_mac = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Status label
        self.bt_status = QLabel("Status: Not connected")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        layout.addWidget(self.bt_status)
        
        # Scan buttons
        btn_layout = QHBoxLayout()
        
        scan_new_btn = QPushButton("Discover New Devices")
        scan_new_btn.clicked.connect(self.scan_bluetooth_devices)
        btn_layout.addWidget(scan_new_btn)
        
        scan_paired_btn = QPushButton("Show Paired Devices")
        scan_paired_btn.clicked.connect(self._get_paired_devices_thread)
        btn_layout.addWidget(scan_paired_btn)
        
        layout.addLayout(btn_layout)
        
        # Device list
        self.bt_list = QListWidget()
        self.bt_list.itemClicked.connect(self.select_bt_device)
        layout.addWidget(self.bt_list)
        
        # Connection buttons
        connect_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect (rfcomm)")
        self.connect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_via_rfcomm)
        connect_layout.addWidget(self.connect_btn)
        
        self.connect_direct_btn = QPushButton("Direct Connect")
        self.connect_direct_btn.setEnabled(False)
        self.connect_direct_btn.clicked.connect(self.connect_via_socket)
        connect_layout.addWidget(self.connect_direct_btn)
        
        layout.addLayout(connect_layout)
        
        # Channel selector
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("RFCOMM Channel:"))
        
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 30)
        self.channel_spin.setValue(1)
        channel_layout.addWidget(self.channel_spin)
        
        layout.addLayout(channel_layout)
        self.setLayout(layout)
    
    def scan_bluetooth_devices(self):
        """Start Bluetooth device discovery."""
        self.bt_list.clear()
        self.bt_status.setText("Scanning for devices...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.signals.log_signal.emit("Starting Bluetooth discovery...", "info")
        threading.Thread(target=self._discover_devices_thread, daemon=True).start()
    
    def _discover_devices_thread(self):
        """Background thread for device discovery."""
        try:
            self.signals.log_signal.emit("Discovering (≈10 s)...", "info")
            devices = bluetooth.discover_devices(
                duration=8, lookup_names=True, flush_cache=True, lookup_class=False
            )
            self.discovered_devices = []
            
            if not devices:
                QTimer.singleShot(0, lambda: self._update_scan_result([]))
                return
            
            for addr, name in devices:
                try:
                    services = bluetooth.find_service(address=addr)
                    channels = [svc["port"] for svc in services if "port" in svc]
                except Exception:
                    channels = []
                
                self.discovered_devices.append({
                    "name": name or "Unknown Device",
                    "mac": addr,
                    "channels": channels or [1],
                })
            
            QTimer.singleShot(0, lambda: self._update_scan_result(self.discovered_devices))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._scan_error(str(e)))
    
    def _get_paired_devices_thread(self):
        """Get paired devices using bluetoothctl."""
        threading.Thread(target=self._fetch_paired_devices, daemon=True).start()
    
    def _fetch_paired_devices(self):
        """Fetch paired devices from bluetoothctl."""
        try:
            result = subprocess.run(
                ["bluetoothctl", "paired-devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            devices = []
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("Device "):
                        parts = line.split(" ", 2)
                        mac = parts[1]
                        name = parts[2] if len(parts) > 2 else "Unknown"
                        devices.append({
                            "name": name,
                            "mac": mac,
                            "channels": [1],
                            "paired": True
                        })
            else:
                self.signals.log_signal.emit(f"bluetoothctl error: {result.stderr}", "error")
            
            self.discovered_devices = devices
            QTimer.singleShot(0, lambda: self._update_scan_result(devices))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._scan_error(str(e)))
    
    def _update_scan_result(self, devices):
        """Update UI with scan results."""
        self.bt_list.clear()
        if not devices:
            self.bt_status.setText("No devices found")
            self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
            self.signals.log_signal.emit("No devices. Pair via bluetoothctl first.", "warning")
            return
        
        for dev in devices:
            ch = ",".join(map(str, dev["channels"]))
            paired = " [PAIRED]" if dev.get("paired") else ""
            self.bt_list.addItem(f"{dev['name']} ({dev['mac']}) [Ch: {ch}]{paired}")
        
        self.bt_status.setText(f"Found {len(devices)} device(s)")
        self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.signals.log_signal.emit(f"Found {len(devices)} device(s)", "success")
    
    def _scan_error(self, msg):
        """Handle scan error."""
        self.bt_status.setText("Scan failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Scan error: {msg}", "error")
        self.signals.log_signal.emit("Check: sudo systemctl start bluetooth", "warning")
    
    def select_bt_device(self, item):
        """Handle device selection."""
        text = item.text()
        mac_match = re.search(r'\(([0-9A-F:]+)\)', text)
        if not mac_match:
            self.signals.log_signal.emit("Could not parse MAC", "error")
            return
        
        self.selected_mac = mac_match.group(1)
        
        ch_match = re.search(r'\[Ch: ([0-9,]+)\]', text)
        if ch_match:
            first_ch = ch_match.group(1).split(',')[0]
            self.channel_spin.setValue(int(first_ch))
        
        self.connect_btn.setEnabled(True)
        self.connect_direct_btn.setEnabled(True)
        self.bt_status.setText(f"Selected: {self.selected_mac}")
        self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.signals.log_signal.emit(f"Selected: {text}", "info")
    
    def connect_via_rfcomm(self):
        """Connect via rfcomm bind."""
        if not self.selected_mac:
            self.signals.log_signal.emit("No device selected!", "error")
            return
        
        self.bt_status.setText("Connecting via rfcomm...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        threading.Thread(target=self._connect_rfcomm_thread, daemon=True).start()
    
    def _connect_rfcomm_thread(self):
        """Background thread for rfcomm connection."""
        mac = self.selected_mac
        channel = self.channel_spin.value()
        try:
            subprocess.run(["sudo", "rfcomm", "release", "0"], capture_output=True, timeout=3)
            time.sleep(0.5)
            
            res = subprocess.run(
                ["sudo", "rfcomm", "bind", "0", mac, str(channel)],
                capture_output=True, text=True, timeout=10
            )
            
            if res.returncode == 0:
                QTimer.singleShot(0, lambda: self.bt_status.setText(f"Bound to {mac}"))
                QTimer.singleShot(0, lambda: self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;"))
                QTimer.singleShot(500, lambda: self.backend.bluetooth.connect_serial())
            else:
                QTimer.singleShot(0, lambda: self._connection_failed(res.stderr or "unknown"))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._connection_failed(str(e)))
    
    def connect_via_socket(self):
        """Connect via direct socket."""
        if not self.selected_mac:
            self.signals.log_signal.emit("No device selected!", "error")
            return
        
        self.bt_status.setText("Connecting via socket...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        channel = self.channel_spin.value()
        threading.Thread(
            target=self._connect_socket_thread,
            args=(channel,),
            daemon=True
        ).start()
    
    def _connect_socket_thread(self, channel):
        """Background thread for socket connection."""
        success = self.backend.bluetooth.connect_direct(self.selected_mac, channel)
        if success:
            QTimer.singleShot(0, lambda: self.bt_status.setText(f"Connected to {self.selected_mac}"))
            QTimer.singleShot(0, lambda: self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;"))
        else:
            QTimer.singleShot(0, lambda: self._connection_failed("socket failed"))
    
    def _connection_failed(self, msg):
        """Handle connection failure."""
        self.bt_status.setText("Connection failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Connection failed: {msg}", "error")
    def create_bluetooth_panel(self):
        """Create the Bluetooth setup panel."""
        bt_group = QGroupBox("Bluetooth Setup")
        bt_layout = QVBoxLayout()

        # ---- status ------------------------------------------------
        self.bt_status = QLabel("Status: Not connected")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        bt_layout.addWidget(self.bt_status)

        # ---- virtual connection button -----------------------------
        virtual_btn = QPushButton("🔧 Connect Virtual (Testing Mode)")
        virtual_btn.setStyleSheet("background-color: #6495ED; font-weight: bold;")
        virtual_btn.clicked.connect(self.connect_virtual)
        bt_layout.addWidget(virtual_btn)

        # ---- scan buttons -----------------------------------------
        btn_layout = QHBoxLayout()
        scan_new_btn = QPushButton("Discover New Devices")
        scan_new_btn.clicked.connect(self.scan_bluetooth_devices)
        btn_layout.addWidget(scan_new_btn)

        scan_paired_btn = QPushButton("Show Paired Devices")
        scan_paired_btn.clicked.connect(self._get_paired_devices_thread)
        btn_layout.addWidget(scan_paired_btn)
        bt_layout.addLayout(btn_layout)

        # ---- device list -------------------------------------------
        self.bt_list = QListWidget()
        self.bt_list.itemClicked.connect(self.select_bt_device)
        bt_layout.addWidget(self.bt_list)

        # ---- connection buttons ------------------------------------
        connect_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect (rfcomm)")
        self.connect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_via_rfcomm)
        connect_layout.addWidget(self.connect_btn)

        self.connect_direct_btn = QPushButton("Direct Connect")
        self.connect_direct_btn.setEnabled(False)
        self.connect_direct_btn.clicked.connect(self.connect_via_socket)
        connect_layout.addWidget(self.connect_direct_btn)
        bt_layout.addLayout(connect_layout)

        # ---- channel selector --------------------------------------
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("RFCOMM Channel:"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 30)
        self.channel_spin.setValue(1)
        channel_layout.addWidget(self.channel_spin)
        bt_layout.addLayout(channel_layout)

        bt_group.setLayout(bt_layout)
        return bt_group
    
    def connect_virtual(self):
        """Connect virtual Bluetooth for testing."""
        self.bt_status.setText("Connecting virtual...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        success = self.backend.bluetooth.connect_virtual()
        
        if success:
            self.bt_status.setText("VIRTUAL MODE - Simulation Active")
            self.bt_status.setStyleSheet("color: #6495ED; font-weight: bold;")
            self.signals.log_signal.emit("Virtual Bluetooth ready for testing", "success")
        else:
            self.bt_status.setText("Virtual connection failed")
            self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")