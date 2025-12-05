#!/usr/bin/env python3
"""
Test script for servo controller
Tests PWM signals for steering control
"""

import time
import sys
from servo_controller import ServoController
import config

def main():
    print("=" * 70)
    print("Servo Controller Test")
    print("=" * 70)
    print("This will test servo PWM signals")
    print("WARNING: Make sure servo is properly connected!")
    print("Press Ctrl+C to exit at any time")
    print("=" * 70)
    print()
    
    try:
        # Initialize servo controller
        servo = ServoController(
            pwm_pin=config.SERVO_PWM_PIN,
            frequency=config.PWM_FREQUENCY,
            center_duty=config.SERVO_CENTER,
            left_max_duty=config.SERVO_LEFT_MAX,
            right_max_duty=config.SERVO_RIGHT_MAX
        )
        
        print("[TEST] Servo controller initialized")
        print("[TEST] Starting test sequence...")
        print()
        
        # Test sequence
        print("[TEST] Centering servo...")
        servo.center()
        time.sleep(1)
        
        print("[TEST] Turning left (25%)...")
        servo.turn_left(0.25)
        time.sleep(1)
        
        print("[TEST] Turning left (50%)...")
        servo.turn_left(0.5)
        time.sleep(1)
        
        print("[TEST] Turning full left...")
        servo.turn_left(1.0)
        time.sleep(1)
        
        print("[TEST] Centering...")
        servo.center()
        time.sleep(1)
        
        print("[TEST] Turning right (25%)...")
        servo.turn_right(0.25)
        time.sleep(1)
        
        print("[TEST] Turning right (50%)...")
        servo.turn_right(0.5)
        time.sleep(1)
        
        print("[TEST] Turning full right...")
        servo.turn_right(1.0)
        time.sleep(1)
        
        print("[TEST] Centering...")
        servo.center()
        time.sleep(1)
        
        print("[TEST] Test complete!")
        print()
        print("If servo didn't move, check:")
        print("  1. GPIO pin connection (should be GPIO 19)")
        print("  2. Servo wiring (power, ground, signal)")
        print("  3. PWM duty cycle values in config.py (may need adjustment)")
        print("     Typical servo range: 2.5% (0°) to 12.5% (180°)")
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'servo' in locals():
            servo.cleanup()
        print("[TEST] Cleanup complete")


if __name__ == '__main__':
    main()

