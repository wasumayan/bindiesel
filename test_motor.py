#!/usr/bin/env python3
"""
Test script for motor controller
Tests PWM signals for motor speed control
"""

import time
import sys
from motor_controller import MotorController
import config

def main():
    print("=" * 70)
    print("Motor Controller Test")
    print("=" * 70)
    print("This will test motor PWM signals")
    print("WARNING: Make sure motor is properly connected!")
    print("Press Ctrl+C to exit at any time")
    print("=" * 70)
    print()
    
    try:
        # Initialize motor controller
        motor = MotorController(
            pwm_pin=config.MOTOR_PWM_PIN,
            frequency=config.PWM_FREQUENCY
        )
        
        print("[TEST] Motor controller initialized")
        print("[TEST] Starting test sequence...")
        print()
        
        # Test sequence
        speeds = [
            (0.3, "30% - Slow"),
            (0.5, "50% - Medium"),
            (0.7, "70% - Fast"),
            (0.9, "90% - Very Fast"),
        ]
        
        for speed, description in speeds:
            print(f"[TEST] Setting speed to {description}")
            motor.forward(speed)
            time.sleep(2)
            print()
        
        print("[TEST] Stopping motor...")
        motor.stop()
        time.sleep(1)
        
        print("[TEST] Test complete!")
        print()
        print("If motor didn't move, check:")
        print("  1. GPIO pin connection (should be GPIO 18)")
        print("  2. Motor controller wiring")
        print("  3. PWM values in config.py (may need adjustment)")
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'motor' in locals():
            motor.cleanup()
        print("[TEST] Cleanup complete")


if __name__ == '__main__':
    main()

