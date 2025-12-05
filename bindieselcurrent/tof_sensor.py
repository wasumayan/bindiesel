"""
Time-of-Flight (TOF) Sensor Module
VL53L0X distance sensor for obstacle detection and user proximity
"""

import time
try:
    import board
    import busio
    import adafruit_vl53l0x
    TOF_AVAILABLE = True
except ImportError:
    print("WARNING: VL53L0X libraries not available")
    print("Install with: pip3 install --break-system-packages adafruit-circuitpython-vl53l0x")
    TOF_AVAILABLE = False


class TOFSensor:
    """VL53L0X Time-of-Flight distance sensor"""
    
    def __init__(self, stop_distance_mm=80, emergency_distance_mm=70):
        """
        Initialize TOF sensor
        
        Args:
            stop_distance_mm: Distance in mm to trigger stop (default 80mm = 8cm)
            emergency_distance_mm: Distance in mm for emergency stop (default 70mm = 7cm)
        """
        self.stop_distance_mm = stop_distance_mm
        self.emergency_distance_mm = emergency_distance_mm
        self.sensor = None
        self.last_distance = None
        
        if TOF_AVAILABLE:
            try:
                # Initialize I2C bus
                i2c = busio.I2C(board.SCL, board.SDA)
                
                # Initialize VL53L0X sensor
                self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
                
                # Set measurement timing budget (affects speed vs accuracy)
                # Lower = faster but less accurate, Higher = slower but more accurate
                self.sensor.measurement_timing_budget = 20000  # 20ms (good balance)
                
                print("[TOFSensor] VL53L0X initialized")
                print(f"[TOFSensor] Stop distance: {stop_distance_mm}mm ({stop_distance_mm/10:.1f}cm)")
                print(f"[TOFSensor] Emergency stop: {emergency_distance_mm}mm ({emergency_distance_mm/10:.1f}cm)")
            except Exception as e:
                print(f"[TOFSensor] ERROR: Failed to initialize sensor: {e}")
                print("[TOFSensor] Running in mock mode")
                self.sensor = None
        else:
            print("[TOFSensor] Running in mock mode (libraries not available)")
    
    def read_distance(self):
        """
        Read distance from sensor
        
        Returns:
            Distance in millimeters, or None if error
        """
        if self.sensor is None:
            # Mock mode - return a safe distance
            return 1000  # 1 meter (safe distance)
        
        try:
            distance_mm = self.sensor.range
            self.last_distance = distance_mm
            return distance_mm
        except Exception as e:
            print(f"[TOFSensor] Error reading distance: {e}")
            return None
    
    def is_too_close(self):
        """
        Check if object is too close (at stop distance)
        
        Returns:
            True if object is at or closer than stop distance
        """
        distance = self.read_distance()
        if distance is None:
            return False  # Don't stop on error
        
        return distance <= self.stop_distance_mm
    
    def is_emergency_stop(self):
        """
        Check if emergency stop is needed (object very close)
        
        Returns:
            True if object is at or closer than emergency distance
        """
        distance = self.read_distance()
        if distance is None:
            return False  # Don't stop on error
        
        return distance <= self.emergency_distance_mm
    
    def get_distance_cm(self):
        """
        Get distance in centimeters
        
        Returns:
            Distance in cm, or None if error
        """
        distance_mm = self.read_distance()
        if distance_mm is None:
            return None
        return distance_mm / 10.0
    
    def is_safe_to_move(self):
        """
        Check if it's safe to move forward
        
        Returns:
            True if no obstacle detected, False if too close
        """
        return not self.is_too_close()


if __name__ == '__main__':
    # Test TOF sensor
    import config
    
    print("Testing TOF sensor...")
    print("Move your hand in front of the sensor")
    print("Press Ctrl+C to exit")
    
    try:
        sensor = TOFSensor(
            stop_distance_mm=config.TOF_STOP_DISTANCE_MM,
            emergency_distance_mm=config.TOF_EMERGENCY_DISTANCE_MM
        )
        
        while True:
            distance_mm = sensor.read_distance()
            distance_cm = distance_mm / 10.0 if distance_mm else None
            
            if distance_cm:
                status = ""
                if sensor.is_emergency_stop():
                    status = " [EMERGENCY STOP!]"
                elif sensor.is_too_close():
                    status = " [STOP!]"
                elif sensor.is_safe_to_move():
                    status = " [SAFE]"
                
                print(f"Distance: {distance_cm:.1f}cm ({distance_mm}mm){status}")
            else:
                print("Error reading distance")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")

