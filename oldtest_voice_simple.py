#!/usr/bin/env python3
"""
Simple voice-to-text test using Google Web Speech Recognition
Goal: Verify ReSpeaker microphone works for voice detection
No processing, just converts speech to text

Adapted from Pi5 reference: https://wiki.seeedstudio.com/respeaker_lite_pi5/
Note: Both Pi4 and Pi5 use PipeWire. 

Setup:
  1. Set sample rate: pw-metadata -n settings 0 clock.force-rate 16000
  2. (Optional) Install pipewire-jack to reduce JACK warnings:
     sudo apt-get install pipewire-jack
     systemctl --user restart pipewire pipewire-pulse
"""

import speech_recognition as sr
import sys
import os
import contextlib
from io import StringIO

# Suppress ALSA and JACK warnings (they're harmless but noisy)
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

# Store original stderr for filtering
_original_stderr = sys.stderr

# More aggressive stderr suppression for ALSA/JACK
# These warnings come from C libraries and are harmless
class StderrFilter:
    """Filter out ALSA and JACK error messages"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filtering_enabled = False
        self.filtered_keywords = [
            'ALSA lib',
            'Cannot connect to server socket',
            'jack server is not running',
            'JackShmReadWritePtr',
            'Unknown PCM',
            'Unable to find definition',
            'snd_func_refer',
            'snd_pcm_open',
            'pcm_asym',
            'pcm_dmix'
        ]
    
    def enable_filtering(self):
        """Enable filtering of ALSA/JACK warnings"""
        self.filtering_enabled = True
    
    def disable_filtering(self):
        """Disable filtering to show all output"""
        self.filtering_enabled = False
    
    def write(self, message):
        # Only filter if enabled
        if self.filtering_enabled:
            message_str = str(message)
            if any(keyword in message_str for keyword in self.filtered_keywords):
                return  # Suppress this message
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Create filter but don't enable it yet
_stderr_filter = StderrFilter(_original_stderr)
sys.stderr = _stderr_filter

def list_microphones():
    """List all available microphones"""
    print("\n" + "=" * 70)
    print("Available Microphones:")
    print("=" * 70)
    
    # List via speech_recognition (this may show warnings, but we need to see devices)
    mic_list = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_list):
        print(f"  {i}: {name}")
    
    # Also try PyAudio for more details
    try:
        import pyaudio
        print("\nDetailed PyAudio device information:")
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # It's an input device
                print(f"  Index {i}: {info['name']}")
                print(f"    Channels: {info['maxInputChannels']}, Sample Rate: {int(info['defaultSampleRate'])}")
        p.terminate()
    except Exception as e:
        print(f"\nCould not get PyAudio details: {e}")
    
    return mic_list

def find_respeaker(mic_list):
    """Find ReSpeaker microphone"""
    print("\n" + "=" * 70)
    print("Searching for ReSpeaker...")
    print("=" * 70)
    
    for i, name in enumerate(mic_list):
        name_lower = name.lower()
        if 'respeaker' in name_lower or 'seeed' in name_lower or '2886' in name_lower:
            print(f"✓ Found ReSpeaker at index {i}: {name}")
            return i
    
    print("⚠ ReSpeaker not found by name, will try default microphone")
    return None

def test_voice_recognition(mic_index=None):
    """Test voice recognition with specified microphone
    Simplified version based on Pi5 reference
    """
    recognizer = sr.Recognizer()
    
    # Select microphone (similar to Pi5 reference)
    if mic_index is not None:
        print(f"\nUsing microphone index {mic_index}")
        microphone = sr.Microphone(device_index=mic_index)
    else:
        print("\nUsing default microphone")
        microphone = sr.Microphone()
    
    # Adjust for ambient noise (shorter duration like Pi5 reference)
    print("\nAdjusting for ambient noise...")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("✓ Ready")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        print("Continuing anyway...")
    
    # Main listening loop (simplified like Pi5 reference)
    print("\n" + "=" * 70)
    print("Voice Recognition Test")
    print("=" * 70)
    print("\nSpeak clearly into the microphone")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            print("Listening...")
            
            try:
                with microphone as source:
                    # Listen for audio (like Pi5 reference)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Recognize using Google Web Speech API (same as Pi5)
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}\n")
                    
                except sr.UnknownValueError:
                    print("Could not understand audio\n")
                    
                except sr.RequestError as e:
                    print(f"Error: {e}\n")
                    
            except sr.WaitTimeoutError:
                print("No speech detected\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                print(f"Error: {e}\n")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 70)
    print("Simple Voice-to-Text Test")
    print("Using Google Web Speech Recognition")
    print("=" * 70)
    
    # Check if speech_recognition is available
    try:
        import speech_recognition as sr
    except ImportError:
        print("\n✗ ERROR: speech_recognition not installed")
        print("Install with: pip3 install SpeechRecognition")
        sys.exit(1)
    
    # List microphones
    mic_list = list_microphones()
    
    # Find ReSpeaker
    respeaker_index = find_respeaker(mic_list)
    
    # Ask user which microphone to use
    print("\n" + "=" * 70)
    if respeaker_index is not None:
        use_respeaker = input(f"Use ReSpeaker (index {respeaker_index})? [Y/n]: ").strip().lower()
        if use_respeaker in ['', 'y', 'yes']:
            mic_index = respeaker_index
        else:
            try:
                mic_index = int(input("Enter microphone index to use: "))
                if mic_index < 0 or mic_index >= len(mic_list):
                    print(f"Invalid index. Using default microphone.")
                    mic_index = None
            except ValueError:
                print("Invalid input. Using default microphone.")
                mic_index = None
    else:
        try:
            mic_index_input = input("Enter microphone index to use (or press Enter for default): ").strip()
            if mic_index_input:
                mic_index = int(mic_index_input)
                if mic_index < 0 or mic_index >= len(mic_list):
                    print(f"Invalid index. Using default microphone.")
                    mic_index = None
            else:
                mic_index = None
        except ValueError:
            print("Invalid input. Using default microphone.")
            mic_index = None
    
    # Test voice recognition
    test_voice_recognition(mic_index)

if __name__ == '__main__':
    main()

