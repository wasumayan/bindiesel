"""
Communication module for sending angle data to PSoC via UART/Serial
"""

import serial
import time
from typing import Optional
import struct


class PSoCCommunicator:
    """Handles serial communication with PSoC for angle control"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200,
                 timeout: float = 1.0):
        """
        Initialize PSoC serial communication
        
        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0' or '/dev/ttyACM0')
            baudrate: Serial communication baud rate
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection: Optional[serial.Serial] = None
        
    def connect(self) -> bool:
        """
        Establish serial connection with PSoC
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            # Wait for connection to stabilize
            time.sleep(0.1)
            print(f"Connected to PSoC on {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to PSoC: {e}")
            print("Available ports can be found with: ls /dev/tty*")
            return False
    
    def send_angle(self, angle: float) -> bool:
        """
        Send angle command to PSoC
        
        Args:
            angle: Angle in degrees (-90 to +90)
            
        Returns:
            True if send successful, False otherwise
        """
        if self.serial_connection is None or not self.serial_connection.is_open:
            print("Error: Not connected to PSoC")
            return False
        
        try:
            # Clamp angle to valid range
            angle = max(-90.0, min(90.0, angle))
            
            # Format: Send as float32 (4 bytes) with header byte
            # Protocol: [0xAA (header), angle as float32]
            header = b'\xAA'
            angle_bytes = struct.pack('f', angle)
            
            message = header + angle_bytes
            
            self.serial_connection.write(message)
            self.serial_connection.flush()
            
            return True
        except Exception as e:
            print(f"Error sending angle to PSoC: {e}")
            return False
    
    def send_angle_simple(self, angle: float) -> bool:
        """
        Send angle as simple text format (alternative protocol)
        Format: "ANGLE:XX.XX\n"
        
        Args:
            angle: Angle in degrees
            
        Returns:
            True if send successful, False otherwise
        """
        if self.serial_connection is None or not self.serial_connection.is_open:
            print("Error: Not connected to PSoC")
            return False
        
        try:
            angle = max(-90.0, min(90.0, angle))
            message = f"ANGLE:{angle:.2f}\n"
            self.serial_connection.write(message.encode('ascii'))
            self.serial_connection.flush()
            return True
        except Exception as e:
            print(f"Error sending angle to PSoC: {e}")
            return False
    
    def read_response(self) -> Optional[str]:
        """
        Read response from PSoC (if any)
        
        Returns:
            Response string or None if no data
        """
        if self.serial_connection is None or not self.serial_connection.is_open:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('ascii').strip()
                return response
        except Exception as e:
            print(f"Error reading from PSoC: {e}")
        
        return None
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Disconnected from PSoC")
    
    def send_navigation_command(self, angle: float, speed: float) -> bool:
        """
        Send navigation command with angle and speed
        
        Args:
            angle: Steering angle in degrees (-90 to +90)
            speed: Speed value (0.0 to 1.0)
            
        Returns:
            True if send successful, False otherwise
        """
        if self.serial_connection is None or not self.serial_connection.is_open:
            print("Error: Not connected to PSoC")
            return False
        
        try:
            angle = max(-90.0, min(90.0, angle))
            speed = max(0.0, min(1.0, speed))
            message = f"NAV:ANGLE:{angle:.2f}:SPEED:{speed:.2f}\n"
            self.serial_connection.write(message.encode('ascii'))
            self.serial_connection.flush()
            return True
        except Exception as e:
            print(f"Error sending navigation command to PSoC: {e}")
            return False
    
    def send_stop_command(self) -> bool:
        """
        Send stop command to PSoC
        
        Returns:
            True if send successful, False otherwise
        """
        if self.serial_connection is None or not self.serial_connection.is_open:
            return False
        
        try:
            message = "NAV:STOP\n"
            self.serial_connection.write(message.encode('ascii'))
            self.serial_connection.flush()
            return True
        except Exception as e:
            print(f"Error sending stop command to PSoC: {e}")
            return False
    
    def list_available_ports(self):
        """List available serial ports"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")

