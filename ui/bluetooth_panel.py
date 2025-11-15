"""
Bluetooth connection panel UI component.
"""

import re
import subprocess
import time
import threading

from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QListWidget, QSpinBox, QMessageBox)
from PySide6.QtCore import Signal, Slot

# Try importing bluetooth, handle if not available
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    print("WARNING: PyBluez not available. Bluetooth scanning will not work.")


class BluetoothPanel(QGroupBox):
    """Bluetooth device discovery and connection panel."""
    
    # Custom signals for thread-safe UI updates
    devices_found = Signal(list)
    scan_error_signal = Signal(str)
    connection_failed_signal = Signal(str)
    
    def __init__(self, backend, signal_emitter, parent=None):
        super().__init__("Bluetooth Setup", parent)
        self.backend = backend
        self.signals = signal_emitter
        self.discovered_devices = []
        self.selected_mac = None
        
        self._init_ui()
        
        # Connect internal signals to slots
        self.devices_found.connect(self._update_scan_result)
        self.scan_error_signal.connect(self._scan_error)
        self.connection_failed_signal.connect(self._connection_failed)
        
        print("BluetoothPanel initialized")
    
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
            
            # Connection button (Direct only)
            self.connect_btn = QPushButton("Connect to Device")
            self.connect_btn.setEnabled(False)
            self.connect_btn.clicked.connect(self.connect_via_socket)
            layout.addWidget(self.connect_btn)
            
            self.setLayout(layout)
    
    def toggle_virtual(self):
            """Toggle virtual Bluetooth connection."""
            if self.virtual_btn.isChecked():
                # Connect
                print("Connecting virtual...")
                self.bt_status.setText("Connecting virtual...")
                self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
                
                try:
                    success = self.backend.bluetooth.connect_virtual()
                    print(f"Virtual connection result: {success}")
                    
                    if success:
                        self.bt_status.setText("VIRTUAL MODE - Simulation Active")
                        self.bt_status.setStyleSheet("color: #6495ED; font-weight: bold;")
                        self.signals.log_signal.emit("Virtual Bluetooth ready for testing", "success")
                        self.virtual_btn.setText("🔌 Disconnect Virtual")
                    else:
                        self.bt_status.setText("Virtual connection failed")
                        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
                        self.virtual_btn.setChecked(False)
                except Exception as e:
                    print(f"Error in toggle_virtual: {e}")
                    self.signals.log_signal.emit(f"Virtual connection error: {e}", "error")
                    self.virtual_btn.setChecked(False)
            else:
                # Disconnect
                print("Disconnecting virtual...")
                self.backend.bluetooth.disconnect()
                self.bt_status.setText("Status: Not connected")
                self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
                self.signals.log_signal.emit("Virtual Bluetooth disconnected", "info")
                self.virtual_btn.setText("🔧 Connect Virtual (Testing Mode)")
    
    def scan_bluetooth_devices(self):
        """Start Bluetooth device discovery."""
        print("scan_bluetooth_devices called")
        
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
        print("Discovery thread started")
    
    def _discover_devices_thread(self):
        """Background thread for device discovery."""
        print("_discover_devices_thread started")
        try:
            self.signals.log_signal.emit("Discovering (≈10 s)...", "info")
            devices = bluetooth.discover_devices(
                duration=8, lookup_names=True, flush_cache=True, lookup_class=False
            )
            print(f"Found {len(devices)} devices")
            
            self.discovered_devices = []
            
            if not devices:
                self.devices_found.emit([])
                return
            
            for addr, name in devices:
                try:
                    services = bluetooth.find_service(address=addr)
                    channels = [svc["port"] for svc in services if "port" in svc]
                except Exception as e:
                    print(f"Error getting services for {addr}: {e}")
                    channels = []
                
                self.discovered_devices.append({
                    "name": name or "Unknown Device",
                    "mac": addr,
                    "channels": channels or [1],
                })
            
            print(f"Processed {len(self.discovered_devices)} devices")
            self.devices_found.emit(self.discovered_devices)
        
        except Exception as e:
            print(f"Error in discovery thread: {e}")
            import traceback
            traceback.print_exc()
            self.scan_error_signal.emit(str(e))
    
    def show_paired_devices(self):
        """Get paired devices using bluetoothctl."""
        print("show_paired_devices called")
        
        self.bt_list.clear()
        self.bt_status.setText("Loading paired devices...")
        self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.signals.log_signal.emit("Fetching paired devices...", "info")
        
        # Start in thread
        thread = threading.Thread(target=self._fetch_paired_devices, daemon=True)
        thread.start()
        print("Paired devices thread started")
    
    def _fetch_paired_devices(self):
        """Fetch paired devices from bluetoothctl."""
        print("_fetch_paired_devices started")
        try:
            result = subprocess.run(
                ["bluetoothctl", "paired-devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(f"bluetoothctl return code: {result.returncode}")
            print(f"bluetoothctl stdout: {result.stdout}")
            print(f"bluetoothctl stderr: {result.stderr}")
            
            devices = []
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    print(f"Processing line: {line}")
                    
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
                            print(f"Added device: {name} ({mac})")
            else:
                error_msg = f"bluetoothctl error: {result.stderr}"
                print(error_msg)
                self.signals.log_signal.emit(error_msg, "error")
            
            print(f"Total devices found: {len(devices)}")
            self.discovered_devices = devices
            
            # Emit signal to update UI
            self.devices_found.emit(devices)
        
        except FileNotFoundError:
            error_msg = "bluetoothctl not found. Install bluez-utils."
            print(error_msg)
            self.scan_error_signal.emit(error_msg)
        
        except Exception as e:
            print(f"Error in _fetch_paired_devices: {e}")
            import traceback
            traceback.print_exc()
            self.scan_error_signal.emit(str(e))
    
    @Slot(list)
    def _update_scan_result(self, devices):
        """Update UI with scan results. Runs on main thread."""
        print(f"_update_scan_result called with {len(devices)} devices (on main thread)")
        
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
            print(f"Adding item to list: {item_text}")
            self.bt_list.addItem(item_text)
        
        self.bt_status.setText(f"Found {len(devices)} device(s)")
        self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        self.signals.log_signal.emit(f"Found {len(devices)} device(s)", "success")
        
        print(f"Device list updated - list now has {self.bt_list.count()} items")
    
    @Slot(str)
    def _scan_error(self, msg):
        """Handle scan error. Runs on main thread."""
        print(f"_scan_error called: {msg} (on main thread)")
        self.bt_status.setText("Scan failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Scan error: {msg}", "error")
        self.signals.log_signal.emit("Check: sudo systemctl start bluetooth", "warning")
    
    def select_bt_device(self, item):
            """Handle device selection."""
            text = item.text()
            print(f"Device selected: {text}")
            
            mac_match = re.search(r'\(([0-9A-Fa-f:]+)\)', text)
            if not mac_match:
                self.signals.log_signal.emit("Could not parse MAC address", "error")
                return
            
            self.selected_mac = mac_match.group(1)
            print(f"Selected MAC: {self.selected_mac}")
            
            self.connect_btn.setEnabled(True)
            self.bt_status.setText(f"Selected: {self.selected_mac}")
            self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
            self.signals.log_signal.emit(f"Selected: {text}", "info")
    
    def connect_via_socket(self):
            """Connect via direct socket."""
            if not self.selected_mac:
                self.signals.log_signal.emit("No device selected!", "error")
                return
            
            self.bt_status.setText("Connecting via socket...")
            self.bt_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
            
            # Use default channel 1
            threading.Thread(
                target=self._connect_socket_thread,
                args=(1,),  # Default channel
                daemon=True
            ).start()
    
    def _connect_socket_thread(self, channel):
        """Background thread for socket connection."""
        success = self.backend.bluetooth.connect_direct(self.selected_mac, channel)
        if success:
            # Update on main thread
            self.devices_found.emit([])  # Dummy signal
            self.bt_status.setText(f"Connected to {self.selected_mac}")
            self.bt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.connection_failed_signal.emit("socket failed")
    
    @Slot(str)
    def _connection_failed(self, msg):
        """Handle connection failure. Runs on main thread."""
        self.bt_status.setText("Connection failed")
        self.bt_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.signals.log_signal.emit(f"Connection failed: {msg}", "error")