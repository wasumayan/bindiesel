"""
Wake Word Detection Module
Uses Picovoice Porcupine to detect "bin diesel" wake word
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import pvporcupine
except ImportError:
    print("ERROR: pvporcupine not installed!")
    print("Install with: pip3 install --break-system-packages pvporcupine")
    sys.exit(1)

import pyaudio
import struct


class WakeWordDetector:
    """Detects wake word using Picovoice Porcupine"""
    
    def __init__(self, model_path, access_key=None):
        """
        Initialize wake word detector
        
        Args:
            model_path: Path to .ppn wake word model file
            access_key: Picovoice access key (or None to use PICOVOICE_ACCESS_KEY env var)
        """
        self.model_path = model_path
        self.access_key = access_key or os.getenv('PICOVOICE_ACCESS_KEY')
        
        if not self.access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not found in environment variables!")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Wake word model not found: {model_path}")
        
        # Initialize Porcupine
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[model_path]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")
        
        # Initialize audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        print(f"[WakeWord] Initialized with model: {model_path}")
    
    def start_listening(self):
        """Start listening for wake word"""
        try:
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            print("[WakeWord] Listening for 'bin diesel'...")
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}")
    
    def detect(self):
        """
        Check for wake word (non-blocking)
        
        Returns:
            True if wake word detected, False otherwise
        """
        if self.stream is None:
            return False
        
        try:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            keyword_index = self.porcupine.process(pcm)
            
            if keyword_index >= 0:
                print("[WakeWord] WAKE WORD DETECTED: 'bin diesel'")
                return True
            
            return False
        except Exception as e:
            print(f"[WakeWord] Error during detection: {e}")
            return False
    
    def stop(self):
        """Stop listening and cleanup"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
        
        if self.porcupine:
            self.porcupine.delete()
        
        print("[WakeWord] Stopped")


if __name__ == '__main__':
    # Test wake word detection
    import config
    
    print("Testing wake word detection...")
    print("Say 'bin diesel' to test")
    print("Press Ctrl+C to exit")
    
    try:
        detector = WakeWordDetector(
            model_path=config.WAKE_WORD_MODEL_PATH,
            access_key=config.WAKE_WORD_ACCESS_KEY
        )
        detector.start_listening()
        
        while True:
            if detector.detect():
                print("✓ Wake word detected!")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'detector' in locals():
            detector.stop()

