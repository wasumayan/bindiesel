"""
Voice Recognition Module for Manual Mode
Uses real-time speech recognition + OpenAI GPT for command interpretation
Commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND, AUTOMATIC MODE
"""

import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("WARNING: speech_recognition library not available")
    print("Install with: pip3 install --break-system-packages SpeechRecognition")
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("WARNING: OpenAI library not available")
    print("Install with: pip3 install --break-system-packages openai")
    OPENAI_AVAILABLE = False


class VoiceRecognizer:
    """Recognizes voice commands using real-time speech recognition + OpenAI GPT"""
    
    # Valid commands (for reference and fallback)
    COMMANDS = {
        'forward': 'FORWARD',
        'left': 'LEFT',
        'right': 'RIGHT',
        'stop': 'STOP',
        'turn around': 'TURN_AROUND',
        'turnaround': 'TURN_AROUND',
        'automatic mode': 'AUTOMATIC_MODE',
        'automatic': 'AUTOMATIC_MODE',
        'auto mode': 'AUTOMATIC_MODE',
        'auto': 'AUTOMATIC_MODE',
        'bin diesel': 'AUTOMATIC_MODE',  # Also accept wake word to return to auto
        'manual mode': 'MANUAL_MODE',  # Enter manual mode
        'radd mode': 'RADD_MODE',  # RADD mode
        'radd': 'RADD_MODE',
        'rad mode': 'RADD_MODE'
    }
    
    # System prompt for OpenAI GPT
    SYSTEM_PROMPT = """You are a voice command interpreter for a robot car. 
Your job is to interpret spoken commands and return ONLY one of these exact commands:
- FORWARD
- LEFT
- RIGHT
- STOP
- TURN_AROUND
- AUTOMATIC_MODE
- MANUAL_MODE
- RADD_MODE

Return ONLY the command name, nothing else. If the command doesn't match any of these, return "UNKNOWN".
Be flexible with variations (e.g., "go forward", "move left", "turn right", "stop", "turn around", "automatic mode", "manual mode", "radd mode", "radd")."""
    
    def __init__(self, api_key=None, model="gpt-4o-mini", energy_threshold=4000, 
                 pause_threshold=0.8, phrase_time_limit=3.0, device_index=None):
        """
        Initialize voice recognizer
        
        Args:
            api_key: OpenAI API key (or None to use OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini for speed/cost)
            energy_threshold: Minimum energy level for speech detection
            pause_threshold: Seconds of non-speaking audio before phrase ends
            phrase_time_limit: Maximum seconds for a phrase
            device_index: Optional microphone device index (None = auto-detect, use same as wake word detector)
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("speech_recognition library not available!")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            print(f"[VoiceRecognizer] Initialized with OpenAI API (model: {model})")
        else:
            self.client = None
            print("[VoiceRecognizer] WARNING: OpenAI not available, using fallback mode")
        
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.device_index = device_index  # Store device index for later use
        
        # Try to initialize microphone with error handling
        try:
            # Try to find a valid microphone device
            microphone_list = sr.Microphone.list_microphone_names()
            print(f"[VoiceRecognizer] Found {len(microphone_list)} audio input devices")
            
            # If device_index is specified, use it (same as wake word detector)
            if device_index is not None:
                try:
                    device_name = microphone_list[device_index] if device_index < len(microphone_list) else f"Device {device_index}"
                    print(f"[VoiceRecognizer] Using specified device index {device_index}: {device_name}")
                    self.microphone = sr.Microphone(device_index=device_index)
                    print(f"[VoiceRecognizer] Successfully initialized microphone [{device_index}]: {device_name}")
                except Exception as e:
                    print(f"[VoiceRecognizer] Specified device index {device_index} failed: {e}")
                    print("[VoiceRecognizer] Falling back to auto-detection...")
                    device_index = None  # Fall back to auto-detection
            
            # If no device_index specified or specified one failed, try auto-detection
            if self.microphone is None:
                # Try default microphone first
                try:
                    self.microphone = sr.Microphone()
                    print("[VoiceRecognizer] Using default microphone")
                except Exception as e:
                    print(f"[VoiceRecognizer] Default microphone failed: {e}")
                    # Try to find any working microphone
                    for i, mic_name in enumerate(microphone_list):
                        try:
                            print(f"[VoiceRecognizer] Trying microphone [{i}]: {mic_name}")
                            self.microphone = sr.Microphone(device_index=i)
                            print(f"[VoiceRecognizer] Successfully initialized microphone [{i}]: {mic_name}")
                            break
                        except Exception as e2:
                            print(f"[VoiceRecognizer] Microphone [{i}] failed: {e2}")
                            continue
            
            if self.microphone is None:
                print("[VoiceRecognizer] WARNING: No working microphone found. Voice commands may not work.")
                print("[VoiceRecognizer] Continuing anyway - ALSA errors are often non-fatal.")
                # Don't exit - continue and let it try to work despite errors
            else:
                # Adjust for ambient noise (with timeout to avoid hanging)
                # Only do this if we have a microphone
                print("[VoiceRecognizer] Adjusting for ambient noise...")
                try:
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    print("[VoiceRecognizer] Ambient noise adjustment complete")
                except (OSError, IOError) as e:
                    # ALSA/PortAudio errors (e.g., error 9988, ALSA errors)
                    error_str = str(e)
                    if '9988' in error_str or 'alsa' in error_str.lower() or 'pa_linux_alsa' in error_str.lower():
                        print(f"[VoiceRecognizer] WARNING: ALSA error during ambient noise adjustment: {e}")
                        print("[VoiceRecognizer] These errors are often non-fatal - continuing with default settings.")
                        # Don't exit - ALSA errors are often just warnings and don't prevent functionality
                    else:
                        print(f"[VoiceRecognizer] Warning: Could not adjust for ambient noise: {e}")
                        print("[VoiceRecognizer] Continuing with default settings...")
                except Exception as e:
                    print(f"[VoiceRecognizer] Warning: Could not adjust for ambient noise: {e}")
                    print("[VoiceRecognizer] Continuing with default settings...")
            
            # Set recognition parameters
            self.recognizer.energy_threshold = energy_threshold
            self.recognizer.pause_threshold = pause_threshold
            self.recognizer.phrase_time_limit = phrase_time_limit
            
            print("[VoiceRecognizer] Initialized with real-time speech recognition")
            print(f"[VoiceRecognizer] Energy threshold: {energy_threshold}")
            print(f"[VoiceRecognizer] Pause threshold: {pause_threshold}s")
            
        except Exception as e:
            print(f"[VoiceRecognizer] WARNING: Exception during initialization: {e}")
            print("[VoiceRecognizer] Continuing anyway - ALSA errors are often non-fatal.")
            # Don't raise - let it continue and try to work despite errors
            # The microphone may still work even with ALSA warnings
    
    def interpret_command_with_gpt(self, transcribed_text):
        """
        Use OpenAI GPT to interpret the transcribed text and return a command
        
        Args:
            transcribed_text: Text from speech recognition
            
        Returns:
            Command string or None
        """
        if not self.client:
            # Fallback to keyword matching
            text_lower = transcribed_text.lower()
            for keyword, command in self.COMMANDS.items():
                if keyword in text_lower:
                    return command
            return None
        
        try:
            # Use OpenAI chat completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Interpret this command: '{transcribed_text}'"}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=10,  # Only need short response
            )
            
            command = response.choices[0].message.content.strip().upper()
            
            # Validate command
            valid_commands = ['FORWARD', 'LEFT', 'RIGHT', 'STOP', 'TURN_AROUND', 
                            'AUTOMATIC_MODE', 'MANUAL_MODE', 'RADD_MODE']
            
            if command in valid_commands:
                return command
            elif command == "UNKNOWN":
                return None
            else:
                # Try to match partial or similar
                for valid_cmd in valid_commands:
                    if valid_cmd in command or command in valid_cmd:
                        return valid_cmd
                return None
        
        except Exception as e:
            print(f"[VoiceRecognizer] Error with OpenAI API: {e}")
            # Fallback to keyword matching
            text_lower = transcribed_text.lower()
            for keyword, command in self.COMMANDS.items():
                if keyword in text_lower:
                    return command
            return None
    
    def recognize_command(self, timeout=None):
        """
        Recognize voice command in real-time
        
        Args:
            timeout: Maximum seconds to wait for speech (None = wait indefinitely)
            
        Returns:
            Command string (FORWARD, LEFT, RIGHT, STOP, TURN_AROUND, AUTOMATIC_MODE, MANUAL_MODE, RADD_MODE) or None
        """
        if self.microphone is None:
            return None
        
        try:
            # Listen for speech
            with self.microphone as source:
                if timeout:
                    print(f"[VoiceRecognizer] Listening (timeout: {timeout}s)...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3.0)
                else:
                    print("[VoiceRecognizer] Listening... (speak now)")
                    audio = self.recognizer.listen(source, phrase_time_limit=3.0)
            
            # Recognize speech using Google Speech Recognition (free, no API key needed)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"[VoiceRecognizer] Transcribed: '{text}'")
            except sr.UnknownValueError:
                print("[VoiceRecognizer] Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"[VoiceRecognizer] Error with speech recognition service: {e}")
                return None
            
            # Use OpenAI GPT to interpret the command
            command = self.interpret_command_with_gpt(text)
            
            if command:
                print(f"[VoiceRecognizer] Command recognized: {command}")
                return command
            else:
                print(f"[VoiceRecognizer] No valid command found in: '{text}'")
                print(f"[VoiceRecognizer] Valid commands: FORWARD, LEFT, RIGHT, STOP, TURN_AROUND, AUTOMATIC_MODE, MANUAL_MODE, RADD_MODE")
                return None
        
        except sr.WaitTimeoutError:
            # Don't print timeout messages - they're expected when checking for commands
            # Only print if timeout is None (waiting indefinitely)
            if timeout is None:
                print("[VoiceRecognizer] Listening timeout - no speech detected")
            return None
        except Exception as e:
            print(f"[VoiceRecognizer] Error during recognition: {e}")
            return None
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            # Microphone is automatically cleaned up when context exits
            # But we should try to close it explicitly if it exists
            if hasattr(self, 'microphone') and self.microphone is not None:
                try:
                    # Microphone cleanup is handled by speech_recognition library
                    # Just set to None to avoid further access
                    self.microphone = None
                except Exception:
                    pass  # Ignore cleanup errors
        except Exception as e:
            # PortAudio errors during cleanup are often non-fatal
            if 'PortAudio' not in str(e) and 'not initialized' not in str(e).lower():
                print(f"[VoiceRecognizer] Warning: Error during cleanup: {e}")
        
        print("[VoiceRecognizer] Cleaned up")


if __name__ == '__main__':
    # Test voice recognition
    import config
    
    print("Testing voice recognition...")
    print("Say one of: FORWARD, LEFT, RIGHT, STOP, TURN AROUND, AUTOMATIC MODE, MANUAL MODE")
    print("Press Ctrl+C to exit")
    
    try:
        recognizer = VoiceRecognizer(
            api_key=config.OPENAI_API_KEY,
            model="gpt-4o-mini"  # Fast and cost-effective
        )
        
        while True:
            command = recognizer.recognize_command(timeout=5.0)
            if command:
                print(f"✓ Command: {command}")
            else:
                print("✗ No command recognized, try again")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'recognizer' in locals():
            recognizer.cleanup()

