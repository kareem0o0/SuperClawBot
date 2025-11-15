"""
Virtual Bluetooth connection for testing and simulation.
Simulates serial/Bluetooth connection without physical hardware.
"""

import threading
import time
from datetime import datetime
from collections import deque


class VirtualBluetoothConnection:
    """Simulates a Bluetooth/serial connection for testing."""
    
    def __init__(self, signal_emitter):
        self.signals = signal_emitter
        self.connected = False
        self.command_history = deque(maxlen=1000)  # Store last 1000 commands
        self.lock = threading.Lock()
        self.is_virtual = True
    
    def connect(self):
        """Simulate connection."""
        self.connected = True
        self.signals.log_signal.emit("Virtual Bluetooth connected (SIMULATION MODE)", "success")
        self.signals.status_signal.emit("Connected")
        return True
    
    def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        self.signals.status_signal.emit("Disconnected")
    
    def send(self, command):
        """
        Simulate sending command.
        
        Args:
            command: Command string to send
        """
        with self.lock:
            if not self.connected:
                return
            
            # Record command with timestamp and metadata
            timestamp = datetime.now()
            command_data = {
                'command': command,
                'timestamp': timestamp,
                'timestamp_str': timestamp.strftime("%H:%M:%S.%f")[:-3],
                'mode': None  # Will be set by caller if available
            }
            
            self.command_history.append(command_data)
            
            # Emit signal for visualization
            if hasattr(self.signals, 'virtual_bt_command_signal'):
                self.signals.virtual_bt_command_signal.emit(command_data)
            
            self.signals.log_signal.emit(f"[VIRTUAL] Sent: {command}", "info")
    
    def write(self, data):
        """Compatibility method for serial-like interface."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        self.send(data)
    
    def get_history(self):
        """Get command history."""
        with self.lock:
            return list(self.command_history)
    
    def clear_history(self):
        """Clear command history."""
        with self.lock:
            self.command_history.clear()
    
    def is_connected(self):
        """Check if connected."""
        return self.connected


class VirtualBluetoothManager:
    """Manages virtual Bluetooth connection with mode tracking."""
    
    def __init__(self, signal_emitter):
        self.signals = signal_emitter
        self.connection = None
        self.lock = threading.Lock()
        self.current_mode = "KEYBOARD"
    
    def connect_virtual(self):
        """Connect virtual Bluetooth."""
        with self.lock:
            if self.connection:
                self.connection.disconnect()
            
            self.connection = VirtualBluetoothConnection(self.signals)
            return self.connection.connect()
    
    def send(self, command):
        """Send command through virtual connection."""
        with self.lock:
            if self.connection and self.connection.is_connected():
                # Add mode information
                if hasattr(self.connection, 'command_history') and len(self.connection.command_history) > 0:
                    # Update last command with current mode
                    pass
                self.connection.send(command)
    
    def set_mode(self, mode):
        """Update current mode for command tracking."""
        self.current_mode = mode
    
    def disconnect(self):
        """Disconnect virtual Bluetooth."""
        with self.lock:
            if self.connection:
                self.connection.disconnect()
                self.connection = None
    
    def is_connected(self):
        """Check if connected."""
        return self.connection is not None and self.connection.is_connected()
    
    def get_history(self):
        """Get command history."""
        if self.connection:
            return self.connection.get_history()
        return []
    
    def clear_history(self):
        """Clear history."""
        if self.connection:
            self.connection.clear_history()