#!/usr/bin/env python3
"""
Record wake word reference audio file
Run this script to record your wake word (e.g., "bin diesel")
"""

import speech_recognition as sr
import time
import os
import argparse


def record_wake_word(output_file="wake_word_reference.wav", duration=3):
    """
    Record a wake word reference audio file
    
    Args:
        output_file: Output filename for the recorded wake word
        duration: Recording duration in seconds
    """
    print("[DEBUG] Initializing speech recognizer...")
    recognizer = sr.Recognizer()
    
    print("[DEBUG] Opening microphone...")
    try:
        microphone = sr.Microphone()
        print(f"[DEBUG] Microphone opened: {microphone}")
    except Exception as e:
        print(f"[DEBUG] Error opening microphone: {e}")
        return False
    
    print("=" * 70)
    print("Recording Wake Word Reference")
    print("=" * 70)
    print(f"Recording {duration} seconds of audio...")
    print("Say your wake word (e.g., 'bin diesel') when ready.")
    print("Recording will start in 2 seconds...")
    time.sleep(2)
    
    with microphone as source:
        print("[DEBUG] Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("[DEBUG] Starting to listen...")
        print("Recording now! Say your wake word...")
        
        try:
            audio = recognizer.listen(source, timeout=duration+1, phrase_time_limit=duration)
            print(f"[DEBUG] Audio captured: {len(audio.get_raw_data())} bytes")
            print(f"[DEBUG] Sample rate: {audio.sample_rate} Hz")
            print(f"[DEBUG] Duration: {len(audio.get_raw_data()) / (audio.sample_rate * 2):.2f} seconds")
            
            # Save audio to file
            print(f"[DEBUG] Saving to '{output_file}'...")
            with open(output_file, "wb") as f:
                f.write(audio.get_wav_data())
            
            file_size = os.path.getsize(output_file)
            print(f"✓ Wake word recorded and saved to '{output_file}'")
            print(f"  File size: {file_size} bytes ({file_size/1024:.2f} KB)")
            return True
            
        except sr.WaitTimeoutError:
            print("Error: Recording timeout")
            return False
        except Exception as e:
            print(f"Error recording: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Record wake word reference audio')
    parser.add_argument('--output', type=str, default='wake_word_reference.wav',
                       help='Output filename for recorded wake word (default: wake_word_reference.wav)')
    parser.add_argument('--duration', type=int, default=3,
                       help='Recording duration in seconds (default: 3)')
    
    args = parser.parse_args()
    
    record_wake_word(args.output, args.duration)


if __name__ == "__main__":
    main()

