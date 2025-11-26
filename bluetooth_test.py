import sys
import asyncio
import serial
import serial.tools.list_ports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QListWidget, QTextEdit, 
                               QLineEdit, QLabel, QMessageBox, QListWidgetItem)
from PySide6.QtCore import QThread, Signal
from bleak import BleakScanner
import subprocess
import re

class BluetoothScanner(QThread):
    """Thread for scanning both BLE and Classic Bluetooth devices"""
    devices_found = Signal(list)
    
    def run(self):
        try:
            all_devices = []
            
            # Scan for BLE devices
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ble_devices = loop.run_until_complete(BleakScanner.discover(timeout=5.0))
            for d in ble_devices:
                all_devices.append(("ble", d.name or "Unknown", d.address))
            
            # Scan for Classic Bluetooth devices (Windows only)
            try:
                classic_devices = self.scan_classic_bluetooth()
                all_devices.extend(classic_devices)
            except Exception as e:
                print(f"Classic BT scan error: {e}")
            
            self.devices_found.emit(all_devices)
        except Exception as e:
            print(f"Scan error: {e}")
            self.devices_found.emit([])
    
    def scan_classic_bluetooth(self):
        """Scan for classic Bluetooth devices using Windows PowerShell"""
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

class SerialReader(QThread):
    """Thread for reading serial data"""
    data_received = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.running = False
    
    def set_port(self, port):
        self.serial_port = port
    
    def run(self):
        self.running = True
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    try:
                        decoded = data.decode('utf-8', errors='replace')
                        self.data_received.emit(decoded)
                    except:
                        self.data_received.emit(str(data))
                self.msleep(50)
            except Exception as e:
                print(f"Read error: {e}")
                break
    
    def stop(self):
        self.running = False

class BluetoothUARTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bluetooth UART Communication Tool")
        self.setGeometry(100, 100, 800, 600)
        
        self.serial_port = None
        self.serial_reader = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Bluetooth UART Communication")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "📌 For HC-05: Pair in Windows Settings first, then click 'Refresh Serial Ports'"
        )
        instructions.setStyleSheet("padding: 5px; background-color: #e3f2fd; border-radius: 3px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Scan section
        scan_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Scan All Bluetooth")
        self.scan_btn.clicked.connect(self.scan_devices)
        scan_layout.addWidget(self.scan_btn)
        
        self.refresh_serial_btn = QPushButton("Refresh Serial Ports")
        self.refresh_serial_btn.clicked.connect(self.refresh_serial_ports)
        scan_layout.addWidget(self.refresh_serial_btn)
        
        layout.addLayout(scan_layout)
        
        # Device list section
        device_label = QLabel("Available Devices & Serial Ports:")
        layout.addWidget(device_label)
        
        self.device_list = QListWidget()
        self.device_list.setMaximumHeight(150)
        layout.addWidget(self.device_list)
        
        # Connect buttons
        connect_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect to Selected")
        self.connect_btn.clicked.connect(self.connect_device)
        connect_layout.addWidget(self.connect_btn)
        
        self.test_btn = QPushButton("Test Selected Port")
        self.test_btn.clicked.connect(self.test_port)
        connect_layout.addWidget(self.test_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_device)
        self.disconnect_btn.setEnabled(False)
        connect_layout.addWidget(self.disconnect_btn)
        
        layout.addLayout(connect_layout)
        
        # Status label
        self.status_label = QLabel("Status: Not connected")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.status_label)
        
        # Received data
        received_label = QLabel("Received Data:")
        layout.addWidget(received_label)
        
        self.received_text = QTextEdit()
        self.received_text.setReadOnly(True)
        self.received_text.setMaximumHeight(250)
        layout.addWidget(self.received_text)
        
        # Send data section
        send_label = QLabel("Send Data:")
        layout.addWidget(send_label)
        
        send_layout = QHBoxLayout()
        self.send_input = QLineEdit()
        self.send_input.setPlaceholderText("Type message to send...")
        self.send_input.returnPressed.connect(self.send_data)
        send_layout.addWidget(self.send_input)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_data)
        self.send_btn.setEnabled(False)
        send_layout.addWidget(self.send_btn)
        
        layout.addLayout(send_layout)
        
        # Clear button
        self.clear_btn = QPushButton("Clear Received Data")
        self.clear_btn.clicked.connect(lambda: self.received_text.clear())
        layout.addWidget(self.clear_btn)
        
        # Load serial ports on startup
        self.refresh_serial_ports()
    
    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        self.device_list.clear()
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            item = QListWidgetItem("No serial ports found - pair your HC-05 in Windows Settings")
            item.setData(1, None)
            self.device_list.addItem(item)
        else:
            for port in ports:
                # Build detailed description
                details = []
                
                # Check if it's a Bluetooth device
                is_bluetooth = ("Bluetooth" in port.description or 
                              "BT" in port.description or
                              "BTHENUM" in str(port.hwid).upper())
                
                # Highlight Bluetooth serial ports
                prefix = "🔵 [BT-SERIAL]" if is_bluetooth else "[SERIAL]"
                
                # Add manufacturer if available
                if port.manufacturer and port.manufacturer != "n/a":
                    details.append(f"Mfr: {port.manufacturer}")
                
                # Add serial number or partial HWID
                if port.serial_number and port.serial_number != "n/a":
                    details.append(f"S/N: {port.serial_number}")
                elif port.hwid:
                    # Extract MAC address from hwid if present
                    hwid_upper = str(port.hwid).upper()
                    if "BTHENUM" in hwid_upper:
                        # Extract Bluetooth address
                        import re
                        mac_match = re.search(r'([0-9A-F]{12})', hwid_upper)
                        if mac_match:
                            mac = mac_match.group(1)
                            formatted_mac = ':'.join([mac[i:i+2] for i in range(0, 12, 2)])
                            details.append(f"BT-MAC: {formatted_mac}")
                
                # Build display text
                detail_str = " | ".join(details) if details else ""
                if detail_str:
                    display_text = f"{prefix} {port.device} - {port.description}\n    {detail_str}"
                else:
                    display_text = f"{prefix} {port.device} - {port.description}"
                
                item = QListWidgetItem(display_text)
                item.setData(1, ("serial", port.device, port.description, is_bluetooth))
                self.device_list.addItem(item)
        
        bt_count = sum(1 for p in ports if "Bluetooth" in p.description or "BTHENUM" in str(p.hwid).upper())
        self.status_label.setText(f"Status: Found {len(ports)} serial port(s) ({bt_count} Bluetooth)")
    
    def scan_devices(self):
        """Scan for all Bluetooth devices"""
        self.scan_btn.setEnabled(False)
        self.status_label.setText("Status: Scanning for Bluetooth devices...")
        self.device_list.clear()
        
        item = QListWidgetItem("Scanning...")
        self.device_list.addItem(item)
        
        self.scanner = BluetoothScanner()
        self.scanner.devices_found.connect(self.on_devices_found)
        self.scanner.start()
    
    def on_devices_found(self, devices):
        """Handle discovered Bluetooth devices"""
        self.device_list.clear()
        
        # Add serial ports first
        ports = serial.tools.list_ports.comports()
        for port in ports:
            is_bluetooth = ("Bluetooth" in port.description or 
                          "BT" in port.description or
                          "BTHENUM" in str(port.hwid).upper())
            
            prefix = "🔵 [BT-SERIAL]" if is_bluetooth else "[SERIAL]"
            
            # Extract details
            details = []
            if port.manufacturer and port.manufacturer != "n/a":
                details.append(f"Mfr: {port.manufacturer}")
            
            if port.hwid:
                hwid_upper = str(port.hwid).upper()
                if "BTHENUM" in hwid_upper:
                    import re
                    mac_match = re.search(r'([0-9A-F]{12})', hwid_upper)
                    if mac_match:
                        mac = mac_match.group(1)
                        formatted_mac = ':'.join([mac[i:i+2] for i in range(0, 12, 2)])
                        details.append(f"BT-MAC: {formatted_mac}")
            
            detail_str = " | ".join(details) if details else ""
            if detail_str:
                display_text = f"{prefix} {port.device} - {port.description}\n    {detail_str}"
            else:
                display_text = f"{prefix} {port.device} - {port.description}"
            
            item = QListWidgetItem(display_text)
            item.setData(1, ("serial", port.device, port.description, is_bluetooth))
            self.device_list.addItem(item)
        
        # Add discovered Bluetooth devices
        if devices:
            for device_type, name, address in devices:
                if device_type == "ble":
                    display_text = f"[BLE] {name} ({address})"
                else:  # classic
                    display_text = f"[CLASSIC] {name}"
                
                item = QListWidgetItem(display_text)
                item.setData(1, (device_type, address, name))
                self.device_list.addItem(item)
            
            self.status_label.setText(f"Status: Found {len(devices)} Bluetooth device(s)")
        else:
            if not ports:
                item = QListWidgetItem("No devices found")
                self.device_list.addItem(item)
            self.status_label.setText("Status: No Bluetooth devices found")
        
        self.scan_btn.setEnabled(True)
    
    def connect_device(self):
        """Connect to the selected device"""
        current_item = self.device_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a device to connect")
            return
        
        device_data = current_item.data(1)
        if not device_data:
            return
        
        if device_data[0] == "serial":
            self.connect_serial(device_data[1])
        else:
            QMessageBox.information(self, "Connection Info", 
                "To connect to HC-05:\n"
                "1. Pair it in Windows Bluetooth Settings\n"
                "2. Click 'Refresh Serial Ports'\n"
                "3. Connect to the COM port that appears")
    
    def test_port(self):
        """Test the selected port to see if it responds"""
        current_item = self.device_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a port to test")
            return
        
        device_data = current_item.data(1)
        if not device_data or device_data[0] != "serial":
            return
        
        port = device_data[1]
        
        # Try to open the port briefly
        test_serial = None
        try:
            test_serial = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=0.5
            )
            
            # Try common baud rates
            baud_rates = [9600, 38400, 115200]
            success = False
            
            for baud in baud_rates:
                try:
                    test_serial.baudrate = baud
                    test_serial.write(b'AT\r\n')  # HC-05 AT command
                    test_serial.flush()
                    
                    import time
                    time.sleep(0.2)
                    
                    if test_serial.in_waiting > 0:
                        response = test_serial.read(test_serial.in_waiting)
                        test_serial.close()
                        
                        msg = f"✅ Port {port} responded!\n\n"
                        msg += f"Baud rate: {baud}\n"
                        msg += f"Response: {response}\n\n"
                        msg += "This is likely your HC-05!"
                        
                        QMessageBox.information(self, "Port Test Result", msg)
                        success = True
                        return
                except:
                    continue
            
            test_serial.close()
            
            if not success:
                QMessageBox.information(self, "Port Test Result", 
                    f"Port {port} opened successfully but no response received.\n\n"
                    f"This could still be your HC-05 if:\n"
                    f"- The Arduino is not sending data\n"
                    f"- The device is in data mode (not AT command mode)\n\n"
                    f"Try connecting and sending data from your Arduino.")
        
        except Exception as e:
            if test_serial and test_serial.is_open:
                test_serial.close()
            
            error_msg = str(e)
            if "PermissionError" in str(type(e)) or "Access is denied" in error_msg:
                QMessageBox.warning(self, "Port Test Failed", 
                    f"Could not open {port}\n\n"
                    f"Error: {error_msg}\n\n"
                    f"This port may be in use by another application.")
            else:
                QMessageBox.warning(self, "Port Test Failed", f"Error testing port: {error_msg}")
    
    def connect_serial(self, port):
        """Connect to a serial port"""
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=9600,  # HC-05 default
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            self.serial_reader = SerialReader()
            self.serial_reader.set_port(self.serial_port)
            self.serial_reader.data_received.connect(self.on_data_received)
            self.serial_reader.start()
            
            self.status_label.setText(f"Status: ✅ Connected to {port}")
            self.status_label.setStyleSheet("padding: 5px; background-color: #c8e6c9;")
            self.send_btn.setEnabled(True)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.scan_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {str(e)}")
    
    def disconnect_device(self):
        """Disconnect from current device"""
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader.wait()
            self.serial_reader = None
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None
        
        self.status_label.setText("Status: Not connected")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        self.send_btn.setEnabled(False)
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)
    
    def on_data_received(self, data):
        """Handle received data"""
        self.received_text.append(data)
        self.received_text.verticalScrollBar().setValue(
            self.received_text.verticalScrollBar().maximum()
        )
    
    def send_data(self):
        """Send data to the connected device"""
        if not self.serial_port or not self.serial_port.is_open:
            QMessageBox.warning(self, "Not Connected", "No device connected")
            return
        
        text = self.send_input.text()
        if not text:
            return
        
        try:
            self.serial_port.write(text.encode('utf-8'))
            self.received_text.append(f"[SENT] {text}")
            self.send_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "Send Error", f"Failed to send data: {str(e)}")
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.disconnect_device()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = BluetoothUARTApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()