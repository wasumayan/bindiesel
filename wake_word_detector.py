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
    
    def __init__(self, model_path, access_key=None, input_device_index=None):
        """
        Initialize wake word detector
        
        Args:
            model_path: Path to .ppn wake word model file
            access_key: Picovoice access key (or None to use PICOVOICE_ACCESS_KEY env var)
            input_device_index: Optional input device index (None = use default)
        """
        self.model_path = model_path
        self.access_key = access_key or os.getenv('PICOVOICE_ACCESS_KEY')
        self.input_device_index = input_device_index
        
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
        
        # Initialize audio
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyAudio: {e}")
        
        # Find valid input device if not specified
        if self.input_device_index is None:
            self.input_device_index = self._find_input_device()
        
        self.stream = None
        
        print(f"[WakeWord] Initialized with model: {model_path}")
        if self.input_device_index is not None:
            device_info = self.audio.get_device_info_by_index(self.input_device_index)
            print(f"[WakeWord] Using audio input device: {device_info['name']} (index {self.input_device_index})")
    
    def _find_input_device(self):
        """Find a valid input device index"""
        try:
            # Try default device first
            default_device = self.audio.get_default_input_device_info()
            if default_device:
                return default_device['index']
        except Exception:
            pass
        
        # List all input devices and find first valid one
        print("[WakeWord] Searching for available audio input devices...")
        valid_devices = []
        
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    valid_devices.append((i, device_info['name']))
                    print(f"  [{i}] {device_info['name']} (channels: {device_info['maxInputChannels']})")
            except Exception as e:
                print(f"  [ERROR] Could not get info for device {i}: {e}")
        
        if not valid_devices:
            raise RuntimeError("No audio input devices found! Please check your microphone permissions and connections.")
        
        # Use first valid device
        device_index, device_name = valid_devices[0]
        print(f"[WakeWord] Selected device: [{device_index}] {device_name}")
        return device_index
    
    def start_listening(self):
        """Start listening for wake word"""
        try:
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.porcupine.frame_length
            )
            print("[WakeWord] Listening for 'bin diesel'...")
        except OSError as e:
            error_code = getattr(e, 'errno', None) or getattr(e, 'args', [None])[0]
            if error_code == 9988 or '9988' in str(e):
                error_msg = (
                    f"Audio device error 9988: Failed to access microphone.\n"
                    f"This usually means:\n"
                    f"  1. Microphone permissions not granted (check System Settings > Privacy & Security > Microphone)\n"
                    f"  2. Audio device is in use by another application\n"
                    f"  3. Invalid audio device index\n"
                    f"\n"
                    f"Available input devices:\n"
                )
                # List available devices
                for i in range(self.audio.get_device_count()):
                    try:
                        device_info = self.audio.get_device_info_by_index(i)
                        if device_info['maxInputChannels'] > 0:
                            error_msg += f"  [{i}] {device_info['name']}\n"
                    except Exception:
                        pass
                raise RuntimeError(error_msg) from e
            else:
                raise RuntimeError(f"Failed to start audio stream (error {error_code}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}") from e
    
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

