"""
Speech recognition for natural language commands
Supports both online (Google) and offline (Vosk) recognition
"""

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: speech_recognition not installed. Voice commands disabled.")
    print("Install with: pip3 install SpeechRecognition")

from typing import Optional, Callable
import threading
import queue


class SpeechRecognizer:
    """Handles speech recognition for natural language commands"""
    
    def __init__(self, method: str = 'vosk', wake_word: str = 'bin diesel'):
        """
        Initialize speech recognizer
        
        Args:
            method: 'vosk' (offline) or 'google' (online, requires internet)
            wake_word: Word/phrase to activate command listening
        """
        self.method = method
        self.wake_word = wake_word.lower()
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None
            
        self.microphone = None
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.callback: Optional[Callable[[str], None]] = None
        
        # Initialize microphone for voice commands
        # Note: This is separate from ReSpeaker I2S (which is for TDOA)
        # Voice commands can use: built-in Pi mic, USB mic, or default system mic
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.microphone = None
            print("Speech recognition not available - voice commands disabled")
            return
            
        try:
            # Try to find ReSpeaker microphone
            mic_index = None
            if SPEECH_RECOGNITION_AVAILABLE:
                try:
                    mic_list = sr.Microphone.list_microphone_names()
                    for i, name in enumerate(mic_list):
                        if 'respeaker' in name.lower() or 'seeed' in name.lower():
                            mic_index = i
                            print(f"Found ReSpeaker at index {i}: {name}")
                            break
                except:
                    pass
            
            # Use ReSpeaker if found, otherwise use default
            if mic_index is not None:
                self.microphone = sr.Microphone(device_index=mic_index)
                print(f"Using ReSpeaker for voice commands")
            else:
                self.microphone = sr.Microphone()
                print(f"Using default microphone for voice commands")
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"Microphone initialized for speech recognition")
        except Exception as e:
            print(f"Warning: Could not initialize microphone: {e}")
            print("Voice commands will be disabled, but you can use keyboard controls")
            self.microphone = None
    
    def recognize_command(self, audio) -> Optional[str]:
        """
        Recognize command from audio
        
        Args:
            audio: Audio data from microphone
            
        Returns:
            Recognized command text or None
        """
        if not SPEECH_RECOGNITION_AVAILABLE or self.recognizer is None:
            return None
            
        try:
            if self.method == 'google':
                text = self.recognizer.recognize_google(audio)
            elif self.method == 'vosk':
                # Vosk requires different setup
                # For now, fall back to Google or use vosk library directly
                try:
                    text = self.recognizer.recognize_vosk(audio)
                except:
                    text = self.recognizer.recognize_google(audio)
            else:
                text = self.recognizer.recognize_google(audio)
            
            return text.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def listen_for_wake_word(self, timeout: float = 1.0):
        """
        Listen for wake word
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if wake word detected
        """
        if not SPEECH_RECOGNITION_AVAILABLE or self.microphone is None:
            return False
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=2)
            
            text = self.recognize_command(audio)
            if text and self.wake_word in text:
                return True
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"Error listening for wake word: {e}")
        
        return False
    
    def listen_for_command(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for command after wake word
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Command text or None
        """
        if not SPEECH_RECOGNITION_AVAILABLE or self.microphone is None:
            return None
        
        try:
            print("Listening for command...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            command = self.recognize_command(audio)
            return command
        except sr.WaitTimeoutError:
            print("No command detected")
            return None
        except Exception as e:
            print(f"Error listening for command: {e}")
            return None
    
    def process_command(self, command: str) -> dict:
        """
        Process natural language command and extract intent
        
        Args:
            command: Recognized command text
            
        Returns:
            Dictionary with intent and parameters
        """
        command = command.lower()
        
        # Remove wake word if present
        if self.wake_word in command:
            command = command.replace(self.wake_word, '').strip()
        
        intent = {
            'action': None,
            'target': None,
            'direction': None,
            'task': None
        }
        
        # Parse autonomous navigation commands
        if 'come here' in command or 'come to me' in command or 'come' in command:
            intent['action'] = 'follow'
            intent['target'] = 'person'
        elif 'pick up trash' in command or 'collect trash' in command or 'get trash' in command:
            intent['action'] = 'pickup_trash'
            intent['task'] = 'trash_collection'
        elif 'return' in command and ('origin' in command or 'start' in command or 'original' in command):
            intent['action'] = 'return'
            intent['target'] = 'origin'
        elif 'pick up' in command and 'return' in command:
            # Combined task: pick up trash and return
            intent['action'] = 'pickup_and_return'
            intent['task'] = 'trash_collection'
        
        # Parse discrete movement commands
        elif 'stop' in command or 'halt' in command:
            intent['action'] = 'stop'
        elif 'go' in command or 'move' in command:
            intent['action'] = 'move'
            if 'forward' in command or 'ahead' in command:
                intent['direction'] = 'forward'
            elif 'back' in command or 'backward' in command:
                intent['direction'] = 'backward'
            elif 'left' in command:
                intent['direction'] = 'left'
            elif 'right' in command:
                intent['direction'] = 'right'
        elif 'turn' in command:
            intent['action'] = 'turn'
            if 'left' in command:
                intent['direction'] = 'left'
            elif 'right' in command:
                intent['direction'] = 'right'
        
        return intent
    
    def start_listening_thread(self, callback: Callable[[str], None]):
        """
        Start background thread listening for commands
        
        Args:
            callback: Function to call when command detected
        """
        self.callback = callback
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                if self.listen_for_wake_word(timeout=1.0):
                    print(f"Wake word '{self.wake_word}' detected!")
                    command = self.listen_for_command(timeout=5.0)
                    if command:
                        print(f"Command: {command}")
                        if self.callback:
                            self.callback(command)
        
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
        print("Speech recognition thread started")
    
    def stop_listening(self):
        """Stop listening thread"""
        self.is_listening = False

