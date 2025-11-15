"""
Bluetooth connection panel UI component.
"""

import re
import subprocess
import time
import threading
import sys

from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QListWidget, QSpinBox, QMessageBox)
from PySide6.QtCore import QTimer, Signal, QObject

# Try importing bluetooth, handle if not available
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    print("WARNING: PyBluez not available. Bluetooth scanning will not work.")


class BluetoothPanel(QGroupBox):
    """Bluetooth device discovery and connection panel."""
    
    def __init__(self, backend, signal_emitter, parent=None):
        super().__init__("Bluetooth Setup", parent)
        self.backend = backend
        self.signals = signal_emitter
        self.discovered_devices = []
        self.selected_mac = None
        self._init_ui()
        
        print("BluetoothPanel initialized")  # DEBUG
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Status label
        self.bt_status = QLabel("Status: Not connected")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        layout.addWidget(self.bt_status)
        
        # Virtual connection button
        virtual_btn = QPushButton("🔧 Connect Virtual (Testing Mode)")
        virtual_btn.setStyleSheet("background-color: #6495ED; font-weight: bold;")
        virtual_btn.clicked.connect(self.connect_virtual)
        layout.addWidget(virtual_btn)
        
        # Scan buttons
        btn_layout = QHBoxLayout()
        
        scan_new_btn = QPushButton("Discover New Devices")
        scan_new_btn.clicked.connect(self.scan_bluetooth_devices)
        btn_layout.addWidget(scan_new_btn)
        
        scan_paired_btn = QPushButton("Show Paired Devices")
        scan_paired_btn.clicked.connect(self.show_paired_devices)
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
    
    def connect_virtual(self):
        """Connect virtual Bluetooth for testing."""
        print("connect_virtual called")  # DEBUG
        self.bt_status.setText("Connecting virtual...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        
        try:
            success = self.backend.bluetooth.connect_virtual()
            print(f"Virtual connection result: {success}")  # DEBUG
            
            if success:
                self.bt_status.setText("VIRTUAL MODE - Simulation Active")
                self.bt_status.setStyleSheet("color: #6495ED; font-weight: bold;")
                self.signals.log_signal.emit("Virtual Bluetooth ready for testing", "success")
            else:
                self.bt_status.setText("Virtual connection failed")
                self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        except Exception as e:
            print(f"Error in connect_virtual: {e}")  # DEBUG
            self.signals.log_signal.emit(f"Virtual connection error: {e}", "error")
    
    def scan_bluetooth_devices(self):
        """Start Bluetooth device discovery."""
        print("scan_bluetooth_devices called")  # DEBUG
        
        if not BLUETOOTH_AVAILABLE:
            QMessageBox.warning(
                self, 
                "Bluetooth Not Available",
                "PyBluez library is not installed.\n\n"
                "Install it with: pip install pybluez\n\n"
                "Use 'Show Paired Devices' or 'Virtual Connection' instead."
            )
            return
        
        self.bt_list.clear()
        self.bt_status.setText("Scanning for devices...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.signals.log_signal.emit("Starting Bluetooth discovery...", "info")
        
        # Start discovery in thread
        thread = threading.Thread(target=self._discover_devices_thread, daemon=True)
        thread.start()
        print("Discovery thread started")  # DEBUG
    
    def _discover_devices_thread(self):
        """Background thread for device discovery."""
        print("_discover_devices_thread started")  # DEBUG
        try:
            self.signals.log_signal.emit("Discovering (≈10 s)...", "info")
            devices = bluetooth.discover_devices(
                duration=8, lookup_names=True, flush_cache=True, lookup_class=False
            )
            print(f"Found {len(devices)} devices")  # DEBUG
            
            self.discovered_devices = []
            
            if not devices:
                QTimer.singleShot(0, lambda: self._update_scan_result([]))
                return
            
            for addr, name in devices:
                try:
                    services = bluetooth.find_service(address=addr)
                    channels = [svc["port"] for svc in services if "port" in svc]
                except Exception as e:
                    print(f"Error getting services for {addr}: {e}")  # DEBUG
                    channels = []
                
                self.discovered_devices.append({
                    "name": name or "Unknown Device",
                    "mac": addr,
                    "channels": channels or [1],
                })
            
            print(f"Processed {len(self.discovered_devices)} devices")  # DEBUG
            QTimer.singleShot(0, lambda: self._update_scan_result(self.discovered_devices))
        
        except Exception as e:
            print(f"Error in discovery thread: {e}")  # DEBUG
            import traceback
            traceback.print_exc()
            QTimer.singleShot(0, lambda: self._scan_error(str(e)))
    
    def show_paired_devices(self):
        """Get paired devices using bluetoothctl."""
        print("show_paired_devices called")  # DEBUG
        
        self.bt_list.clear()
        self.bt_status.setText("Loading paired devices...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.signals.log_signal.emit("Fetching paired devices...", "info")
        
        # Start in thread
        thread = threading.Thread(target=self._fetch_paired_devices, daemon=True)
        thread.start()
        print("Paired devices thread started")  # DEBUG
    
    def _fetch_paired_devices(self):
        """Fetch paired devices from bluetoothctl."""
        print("_fetch_paired_devices started")  # DEBUG
        try:
            result = subprocess.run(
                ["bluetoothctl", "paired-devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(f"bluetoothctl return code: {result.returncode}")  # DEBUG
            print(f"bluetoothctl stdout: {result.stdout}")  # DEBUG
            print(f"bluetoothctl stderr: {result.stderr}")  # DEBUG
            
            devices = []
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    print(f"Processing line: {line}")  # DEBUG
                    
                    if line.startswith("Device "):
                        parts = line.split(" ", 2)
                        if len(parts) >= 2:
                            mac = parts[1]
                            name = parts[2] if len(parts) > 2 else "Unknown"
                            devices.append({
                                "name": name,
                                "mac": mac,
                                "channels": [1],
                                "paired": True
                            })
                            print(f"Added device: {name} ({mac})")  # DEBUG
            else:
                error_msg = f"bluetoothctl error: {result.stderr}"
                print(error_msg)  # DEBUG
                self.signals.log_signal.emit(error_msg, "error")
            
            print(f"Total devices found: {len(devices)}")  # DEBUG
            self.discovered_devices = devices
            QTimer.singleShot(0, lambda devs=devices: self._update_scan_result(devs))
        
        except FileNotFoundError:
            error_msg = "bluetoothctl not found. Install bluez-utils."
            print(error_msg)  # DEBUG
            QTimer.singleShot(0, lambda: self._scan_error(error_msg))
        
        except Exception as e:
            print(f"Error in _fetch_paired_devices: {e}")  # DEBUG
            import traceback
            traceback.print_exc()
            QTimer.singleShot(0, lambda: self._scan_error(str(e)))
    
    def _update_scan_result(self, devices):
        """Update UI with scan results."""
        print(f"_update_scan_result called with {len(devices)} devices")  # DEBUG
        
        self.bt_list.clear()
        
        if not devices:
            self.bt_status.setText("No devices found")
            self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
            self.signals.log_signal.emit("No devices found. Try pairing via system settings first.", "warning")
            return
        
        for dev in devices:
            ch = ",".join(map(str, dev["channels"]))
            paired = " [PAIRED]" if dev.get("paired") else ""
            item_text = f"{dev['name']} ({dev['mac']}) [Ch: {ch}]{paired}"
            print(f"Adding item to list: {item_text}")  # DEBUG
            self.bt_list.addItem(item_text)
        
        self.bt_status.setText(f"Found {len(devices)} device(s)")
        self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.signals.log_signal.emit(f"Found {len(devices)} device(s)", "success")
        
        print("Device list updated")  # DEBUG
    
    def _scan_error(self, msg):
        """Handle scan error."""
        print(f"_scan_error called: {msg}")  # DEBUG
        self.bt_status.setText("Scan failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Scan error: {msg}", "error")
        self.signals.log_signal.emit("Check: sudo systemctl start bluetooth", "warning")
    
    def select_bt_device(self, item):
        """Handle device selection."""
        text = item.text()
        print(f"Device selected: {text}")  # DEBUG
        
        mac_match = re.search(r'\(([0-9A-Fa-f:]+)\)', text)
        if not mac_match:
            self.signals.log_signal.emit("Could not parse MAC address", "error")
            return
        
        self.selected_mac = mac_match.group(1)
        print(f"Selected MAC: {self.selected_mac}")  # DEBUG
        
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