"""
Bluetooth connection manager with virtual connection support.
"""

import asyncio
import platform
import re
import subprocess
import threading
import time
import serial
import serial.tools.list_ports
from bleak import BleakScanner
from collections import deque
from datetime import datetime

from config import BLUETOOTH_BAUD
from .virtual_bluetooth import VirtualBluetoothConnection


class BluetoothManager:
    """Manages Bluetooth connections via serial, socket, or virtual."""

    def __init__(self, signal_emitter):
        self.signals = signal_emitter
        self.connection = None
        self.lock = threading.Lock()
        self.connection_type = None  # 'serial', 'socket', or 'virtual'
        self.command_history = deque(maxlen=1000)  # Store last 1000 commands for all connection types

    def scan_devices(self):
        """Scan for both BLE and Classic Bluetooth devices."""
        try:
            all_devices = []

            # Scan for BLE devices
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ble_devices = loop.run_until_complete(BleakScanner.discover(timeout=5.0))
            for d in ble_devices:
                all_devices.append(("ble", d.name or "Unknown", d.address))

            # Scan for Classic Bluetooth devices (Windows only)
            if platform.system() == "Windows":
                try:
                    classic_devices = self._scan_classic_bluetooth_windows()
                    all_devices.extend(classic_devices)
                except Exception as e:
                    print(f"Classic BT scan error: {e}")

            return all_devices
        except Exception as e:
            print(f"Scan error: {e}")
            return []

    def _scan_classic_bluetooth_windows(self):
        """Scan for classic Bluetooth devices using Windows PowerShell."""
        devices = []
        try:
            # Use PowerShell to get Bluetooth devices
            ps_command = '''
            Get-PnpDevice -Class Bluetooth | Where-Object {$_.Status -eq "OK" -or $_.Status -eq "Unknown"} |
            Select-Object FriendlyName, InstanceId | ConvertTo-Json
            '''
            result = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                import json
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        data = [data]

                    for item in data:
                        name = item.get('FriendlyName', 'Unknown')
                        instance = item.get('InstanceId', '')

                        # Extract MAC address if present
                        mac_match = re.search(r'([0-9A-Fa-f]{12})', instance)
                        address = mac_match.group(1) if mac_match else instance

                        if name and 'Bluetooth' in name or 'HC-' in name.upper():
                            devices.append(("classic", name, address))
                except:
                    pass
        except Exception as e:
            print(f"PowerShell scan error: {e}")

        return devices

    def list_serial_ports(self):
        """List available serial ports."""
        return serial.tools.list_ports.comports()

    def connect_serial(self, port, baud=BLUETOOTH_BAUD):
        """
        Connect via serial port.

        Args:
            port: Serial port path
            baud: Baud rate

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection:
                self.disconnect()

            self.connection = serial.Serial(port, baud, timeout=1)
            self.connection_type = 'serial'
            self.command_history.clear()  # Clear history on new connection
            time.sleep(2)

            self.signals.log_signal.emit(f"Connected to {port}", "success")
            self.signals.status_signal.emit("Connected")
            return True

        except Exception as e:
            self.signals.log_signal.emit(f"Connection error: {e}", "error")
            self.signals.status_signal.emit("Disconnected")
            return False

    def connect_virtual(self):
        """
        Connect via virtual Bluetooth (simulation mode).

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection:
                self.disconnect()

            self.connection = VirtualBluetoothConnection(self.signals)
            self.connection_type = 'virtual'

            success = self.connection.connect()
            if success:
                self.signals.log_signal.emit("Virtual Bluetooth connected (SIMULATION)", "success")
                self.signals.status_signal.emit("Connected")

            return success

        except Exception as e:
            self.signals.log_signal.emit(f"Virtual connection failed: {e}", "error")
            self.signals.status_signal.emit("Disconnected")
            return False

    def send(self, command):
        """
        Send command to robot.

        Args:
            command: Command string to send
        """
        with self.lock:
            if not self.connection:
                return

            try:
                # Record command in history for all connection types
                timestamp = datetime.now()
                command_data = {
                    'command': command,
                    'timestamp': timestamp,
                    'timestamp_str': timestamp.strftime("%H:%M:%S.%f")[:-3],
                }
                self.command_history.append(command_data)
                
                if self.connection_type == 'serial':
                    self.connection.write(command.encode())
                    self.signals.log_signal.emit(f"Sent: {command}", "info")
                elif self.connection_type == 'virtual':
                    self.connection.send(command)

            except Exception as e:
                self.signals.log_signal.emit(f"Send error: {e}", "error")

    def disconnect(self):
        """Disconnect from robot."""
        with self.lock:
            if self.connection:
                try:
                    if self.connection_type == 'serial':
                        self.connection.close()
                    elif self.connection_type == 'virtual':
                        self.connection.disconnect()
                except Exception:
                    pass

                self.connection = None
                self.connection_type = None
                self.signals.status_signal.emit("Disconnected")

    def is_connected(self):
        """Check if connected."""
        return self.connection is not None

    def is_virtual(self):
        """Check if using virtual connection."""
        return self.connection_type == 'virtual'
    
    def get_history(self):
        """Get command history for all connection types."""
        with self.lock:
            if self.connection_type == 'virtual' and hasattr(self.connection, 'get_history'):
                return self.connection.get_history()
            return list(self.command_history)
    
    def clear_history(self):
        """Clear command history."""
        with self.lock:
            if self.connection_type == 'virtual' and hasattr(self.connection, 'clear_history'):
                self.connection.clear_history()
            self.command_history.clear()
