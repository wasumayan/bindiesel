"""
Voice Recognition Module for Manual Mode
Uses OpenAI GPT API to recognize voice commands
Commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND
"""

import os
import time
import pyaudio
import wave
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("WARNING: OpenAI library not available")
    print("Install with: pip3 install --break-system-packages openai")
    OPENAI_AVAILABLE = False


class VoiceRecognizer:
    """Recognizes voice commands using OpenAI Whisper API"""
    
    # Valid commands
    COMMANDS = {
        'forward': 'FORWARD',
        'left': 'LEFT',
        'right': 'RIGHT',
        'stop': 'STOP',
        'turn around': 'TURN_AROUND',
        'turnaround': 'TURN_AROUND',
        'go back': 'TURN_AROUND',
        'back': 'TURN_AROUND'
    }
    
    def __init__(self, api_key=None, sample_rate=16000, chunk_size=1024, 
                 record_seconds=2):
        """
        Initialize voice recognizer
        
        Args:
            api_key: OpenAI API key (or None to use OPENAI_API_KEY env var)
            sample_rate: Audio sample rate (default 16000 Hz)
            chunk_size: Audio chunk size
            record_seconds: How long to record for each command
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.record_seconds = record_seconds
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            print("[VoiceRecognizer] Initialized with OpenAI API")
        else:
            self.client = None
            print("[VoiceRecognizer] WARNING: OpenAI not available, using mock mode")
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
    
    def record_audio(self):
        """
        Record audio from microphone
        
        Returns:
            Path to temporary WAV file, or None if error
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("[VoiceRecognizer] Recording... (speak now)")
            frames = []
            
            # Record for specified duration
            for _ in range(0, int(self.sample_rate / self.chunk_size * self.record_seconds)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            print("[VoiceRecognizer] Recording complete")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # Write WAV file
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return temp_path
        
        except Exception as e:
            print(f"[VoiceRecognizer] Error recording audio: {e}")
            return None
    
    def transcribe(self, audio_file_path):
        """
        Transcribe audio using OpenAI Whisper API
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text, or None if error
        """
        if not self.client:
            # Mock mode - return a test command
            print("[VoiceRecognizer] Mock mode: returning 'FORWARD'")
            return "forward"
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            text = transcript.text.strip().lower()
            print(f"[VoiceRecognizer] Transcribed: '{text}'")
            return text
        
        except Exception as e:
            print(f"[VoiceRecognizer] Error transcribing: {e}")
            return None
    
    def recognize_command(self):
        """
        Record audio and recognize command
        
        Returns:
            Command string (FORWARD, LEFT, RIGHT, STOP, TURN_AROUND) or None
        """
        # Record audio
        audio_file = self.record_audio()
        if not audio_file:
            return None
        
        try:
            # Transcribe
            text = self.transcribe(audio_file)
            if not text:
                return None
            
            # Match to command
            for keyword, command in self.COMMANDS.items():
                if keyword in text:
                    print(f"[VoiceRecognizer] Command recognized: {command}")
                    return command
            
            print(f"[VoiceRecognizer] No valid command found in: '{text}'")
            print(f"[VoiceRecognizer] Valid commands: {', '.join(self.COMMANDS.keys())}")
            return None
        
        finally:
            # Clean up temporary file
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.audio:
            self.audio.terminate()
        print("[VoiceRecognizer] Cleaned up")


if __name__ == '__main__':
    # Test voice recognition
    import config
    
    print("Testing voice recognition...")
    print("Say one of: FORWARD, LEFT, RIGHT, STOP, TURN AROUND")
    print("Press Ctrl+C to exit")
    
    try:
        recognizer = VoiceRecognizer(
            api_key=config.OPENAI_API_KEY
        )
        
        while True:
            command = recognizer.recognize_command()
            if command:
                print(f"✓ Command: {command}")
            else:
                print("✗ No command recognized, try again")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'recognizer' in locals():
            recognizer.cleanup()

