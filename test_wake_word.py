#!/usr/bin/env python3
"""
Test script for wake word detection
Tests Picovoice Porcupine wake word detection without hardware
"""

import sys
import time
from wake_word_detector import WakeWordDetector
import config

def main():
    print("=" * 70)
    print("Wake Word Detection Test")
    print("=" * 70)
    print("Say 'bin diesel' to test wake word detection")
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    try:
        # Initialize wake word detector
        wake_word_path = config.WAKE_WORD_MODEL_PATH
        detector = WakeWordDetector(
            model_path=wake_word_path,
            access_key=config.WAKE_WORD_ACCESS_KEY
        )
        detector.start_listening()
        
        print("[TEST] Wake word detector initialized")
        print("[TEST] Listening...")
        print()
        
        detection_count = 0
        
        while True:
            if detector.detect():
                detection_count += 1
                print(f"✓ Wake word detected! (Count: {detection_count})")
                print()
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.stop()
        print("[TEST] Test complete")


if __name__ == '__main__':
    main()

