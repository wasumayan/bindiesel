#!/usr/bin/env python3
"""
Test script for TOF sensor
Tests VL53L0X distance measurement
"""

import time
import sys
from tof_sensor import TOFSensor
import config

def main():
    print("=" * 70)
    print("TOF Sensor Test")
    print("=" * 70)
    print("This will test VL53L0X distance sensor")
    print("Move your hand in front of the sensor to test")
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    try:
        # Initialize TOF sensor
        sensor = TOFSensor(
            stop_distance_mm=config.TOF_STOP_DISTANCE_MM,
            emergency_distance_mm=config.TOF_EMERGENCY_DISTANCE_MM
        )
        
        print("[TEST] TOF sensor initialized")
        print(f"[TEST] Stop distance: {config.TOF_STOP_DISTANCE_MM}mm ({config.TOF_STOP_DISTANCE_MM/10:.1f}cm)")
        print(f"[TEST] Emergency stop: {config.TOF_EMERGENCY_DISTANCE_MM}mm ({config.TOF_EMERGENCY_DISTANCE_MM/10:.1f}cm)")
        print()
        print("[TEST] Starting distance readings...")
        print()
        
        reading_count = 0
        
        while True:
            distance_mm = sensor.read_distance()
            distance_cm = distance_mm / 10.0 if distance_mm else None
            
            if distance_cm:
                reading_count += 1
                
                # Determine status
                status = ""
                status_color = ""
                if sensor.is_emergency_stop():
                    status = " [EMERGENCY STOP!]"
                    status_color = "\033[91m"  # Red
                elif sensor.is_too_close():
                    status = " [STOP!]"
                    status_color = "\033[93m"  # Yellow
                elif sensor.is_safe_to_move():
                    status = " [SAFE]"
                    status_color = "\033[92m"  # Green
                
                reset_color = "\033[0m"
                
                print(f"[TEST #{reading_count}] Distance: {status_color}{distance_cm:.1f}cm ({distance_mm}mm){status}{reset_color}")
                
                # Print thresholds
                if reading_count % 10 == 0:  # Every 10 readings
                    print(f"  Thresholds: Stop={config.TOF_STOP_DISTANCE_MM/10:.1f}cm, "
                          f"Emergency={config.TOF_EMERGENCY_DISTANCE_MM/10:.1f}cm")
            else:
                print("[TEST] Error reading distance")
            
            time.sleep(0.1)  # Read every 100ms
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[TEST] Test complete")
        print()
        print("Troubleshooting:")
        print("  1. Check I2C connection: i2cdetect -y 1")
        print("  2. Should see device at address 0x29")
        print("  3. Check wiring: SDA to GPIO 2, SCL to GPIO 3")
        print("  4. Enable I2C: sudo raspi-config → Interface Options → I2C")


if __name__ == '__main__':
    main()

