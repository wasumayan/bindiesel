#!/usr/bin/env python3
"""
Test script for voice recognition (manual mode)
Tests OpenAI Whisper API for command recognition
"""

import time
import sys
from voice_recognizer import VoiceRecognizer
import config

def main():
    print("=" * 70)
    print("Voice Command Recognition Test")
    print("=" * 70)
    print("This will test voice command recognition")
    print("Valid commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND")
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    try:
        # Initialize voice recognizer
        recognizer = VoiceRecognizer(
            api_key=config.OPENAI_API_KEY
        )
        
        print("[TEST] Voice recognizer initialized")
        print("[TEST] Ready to recognize commands")
        print()
        
        command_count = 0
        success_count = 0
        
        while True:
            print(f"[TEST] Waiting for command (attempt #{command_count + 1})...")
            command = recognizer.recognize_command()
            
            command_count += 1
            
            if command:
                success_count += 1
                print(f"✓ Command recognized: {command}")
                print(f"  Success rate: {success_count}/{command_count} ({success_count/command_count*100:.1f}%)")
                print()
                
                # Simulate command execution
                print(f"[TEST] Simulating: {command}")
                if command == 'FORWARD':
                    print("  → Motor would move forward")
                elif command == 'LEFT':
                    print("  → Car would turn left")
                elif command == 'RIGHT':
                    print("  → Car would turn right")
                elif command == 'STOP':
                    print("  → Car would stop")
                elif command == 'TURN_AROUND':
                    print("  → Car would turn around")
                print()
            else:
                print("✗ No command recognized, try again")
                print(f"  Success rate: {success_count}/{command_count} ({success_count/command_count*100:.1f}%)")
                print()
            
            time.sleep(1)  # Small delay between attempts
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'recognizer' in locals():
            recognizer.cleanup()
        print("[TEST] Test complete")
        print()
        print("Troubleshooting:")
        print("  1. Check OPENAI_API_KEY in .env file")
        print("  2. Check internet connection (needs API access)")
        print("  3. Speak clearly and wait for 'Recording...' prompt")
        print("  4. Use commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND")


if __name__ == '__main__':
    main()

