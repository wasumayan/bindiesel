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
        'manual mode': 'MANUAL_MODE'  # Enter manual mode
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

Return ONLY the command name, nothing else. If the command doesn't match any of these, return "UNKNOWN".
Be flexible with variations (e.g., "go forward", "move left", "turn right", "stop", "turn around", "automatic mode", "manual mode")."""
    
    def __init__(self, api_key=None, model="gpt-4o-mini", energy_threshold=4000, 
                 pause_threshold=0.8, phrase_time_limit=3.0):
        """
        Initialize voice recognizer
        
        Args:
            api_key: OpenAI API key (or None to use OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini for speed/cost)
            energy_threshold: Minimum energy level for speech detection
            pause_threshold: Seconds of non-speaking audio before phrase ends
            phrase_time_limit: Maximum seconds for a phrase
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
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        print("[VoiceRecognizer] Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Set recognition parameters
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.phrase_time_limit = phrase_time_limit
        
        print("[VoiceRecognizer] Initialized with real-time speech recognition")
        print(f"[VoiceRecognizer] Energy threshold: {energy_threshold}")
        print(f"[VoiceRecognizer] Pause threshold: {pause_threshold}s")
    
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
                            'AUTOMATIC_MODE', 'MANUAL_MODE']
            
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
            Command string (FORWARD, LEFT, RIGHT, STOP, TURN_AROUND, AUTOMATIC_MODE, MANUAL_MODE) or None
        """
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
                print(f"[VoiceRecognizer] Valid commands: FORWARD, LEFT, RIGHT, STOP, TURN_AROUND, AUTOMATIC_MODE, MANUAL_MODE")
                return None
        
        except sr.WaitTimeoutError:
            print("[VoiceRecognizer] Listening timeout - no speech detected")
            return None
        except Exception as e:
            print(f"[VoiceRecognizer] Error during recognition: {e}")
            return None
    
    def cleanup(self):
        """Cleanup audio resources"""
        # Microphone is automatically cleaned up when context exits
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

