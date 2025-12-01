#!/usr/bin/env python3
"""
Test ReSpeaker voice input with OpenAI integration
Component test: Voice input → Text → OpenAI → Response
"""

import os
import sys

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variable

# Get OpenAI API key from environment variable
# Set it with: export OPENAI_API_KEY='your-key-here'
# Or create a .env file (not committed to git)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    print("Or install python-dotenv and create a .env file")
    print("  pip3 install python-dotenv")
    print("  cp .env.example .env")
    print("  # Then edit .env with your key")
    sys.exit(1)

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("ERROR: speech_recognition not installed")
    print("Install with: pip3 install SpeechRecognition")
    sys.exit(1)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ERROR: openai not installed")
    print("Install with: pip3 install openai")
    sys.exit(1)


class VoiceAssistant:
    """Simple voice assistant with OpenAI integration"""
    
    def __init__(self):
        """Initialize voice assistant"""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.original_position = None  # Store original position for "come here" functionality
        
        # Find ReSpeaker microphone
        self._setup_microphone()
    
    def _setup_microphone(self):
        """Setup ReSpeaker microphone"""
        try:
            # List all microphones
            mic_list = sr.Microphone.list_microphone_names()
            print("\nAvailable microphones:")
            for i, name in enumerate(mic_list):
                print(f"  {i}: {name}")
            
            # Find ReSpeaker
            mic_index = None
            for i, name in enumerate(mic_list):
                if 'respeaker' in name.lower() or 'seeed' in name.lower():
                    mic_index = i
                    print(f"\n✓ Found ReSpeaker at index {i}: {name}")
                    break
            
            if mic_index is not None:
                self.microphone = sr.Microphone(device_index=mic_index)
            else:
                print("\n⚠ ReSpeaker not found, using default microphone")
                self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Microphone ready\n")
            
        except Exception as e:
            print(f"ERROR setting up microphone: {e}")
            sys.exit(1)
    
    def listen(self, timeout=5, phrase_time_limit=5):
        """Listen for voice input"""
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            return audio
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except Exception as e:
            print(f"Error listening: {e}")
            return None
    
    def recognize(self, audio):
        """Recognize speech to text"""
        try:
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def get_openai_response(self, user_input, context=""):
        """Get response from OpenAI"""
        try:
            system_prompt = """You are a helpful assistant for a robot car named "Bin Diesel". 
            
The car can:
- Navigate: go forward, backward, left, right, stop, slow down, speed up
- Answer questions: time, date, general knowledge
- Follow commands: "bin diesel, come here" (come to user), "return" (go back to original position)

For navigation commands, respond with JSON: {"action": "command", "value": "forward/left/right/stop/etc"}
For questions, respond normally.
For "come here", respond with: {"action": "come_here"}
For "return", respond with: {"action": "return"}

Context: """ + context
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return None
    
    def process_command(self, text):
        """Process command and determine action"""
        text_lower = text.lower()
        
        # Direct navigation commands (no LLM needed)
        if any(word in text_lower for word in ['go forward', 'move forward', 'forward']):
            return {"action": "navigate", "command": "forward", "text": text}
        elif any(word in text_lower for word in ['go back', 'move back', 'backward', 'back']):
            return {"action": "navigate", "command": "backward", "text": text}
        elif any(word in text_lower for word in ['go left', 'turn left', 'left']):
            return {"action": "navigate", "command": "left", "text": text}
        elif any(word in text_lower for word in ['go right', 'turn right', 'right']):
            return {"action": "navigate", "command": "right", "text": text}
        elif 'stop' in text_lower or 'halt' in text_lower:
            return {"action": "navigate", "command": "stop", "text": text}
        elif 'slow down' in text_lower:
            return {"action": "navigate", "command": "slow_down", "text": text}
        elif 'speed up' in text_lower or 'faster' in text_lower:
            return {"action": "navigate", "command": "speed_up", "text": text}
        elif 'come here' in text_lower or 'come to me' in text_lower or 'bin diesel come here' in text_lower:
            if self.original_position is None:
                self.original_position = "current_location"  # Store current position
            return {"action": "come_here", "text": text}
        elif 'return' in text_lower or 'go back to start' in text_lower:
            return {"action": "return", "text": text}
        else:
            # Use OpenAI for queries and complex commands
            return {"action": "query", "text": text}
    
    def run_interactive(self):
        """Run interactive voice assistant"""
        print("=" * 70)
        print("ReSpeaker Voice Assistant Test")
        print("=" * 70)
        print("\n✓ Microphone: Ready")
        print("✓ OpenAI: Connected")
        print("\nCommands you can try:")
        print("  - Navigation: 'go forward', 'turn left', 'stop', 'slow down'")
        print("  - Questions: 'what time is it?', 'who is the president?'")
        print("  - Main function: 'bin diesel, come here'")
        print("\n💡 Speak clearly into the ReSpeaker microphone")
        print("💡 The system will listen for 3 seconds, then process your command")
        print("\nPress Ctrl+C to exit\n")
        
        while True:
            try:
                # Listen
                print("🎤 Listening... (speak now)")
                audio = self.listen(timeout=3, phrase_time_limit=5)
                if audio is None:
                    print("   (No speech detected, trying again...)\n")
                    continue
                
                # Recognize
                print("   Processing speech...")
                text = self.recognize(audio)
                if text is None:
                    print("   (Could not understand, trying again...)\n")
                    continue
                
                print(f"\n✓ You said: '{text}'")
                
                # Process command
                result = self.process_command(text)
                
                if result["action"] == "navigate":
                    print(f"→ Navigation command: {result['command']}")
                    # In real system, this would send to PSoC
                    print(f"  [Would send to PSoC: {result['command']}]")
                
                elif result["action"] == "come_here":
                    print("→ Main function: Come here")
                    print("  [Would start person tracking and navigation]")
                    print("  [Would store current position for return]")
                
                elif result["action"] == "return":
                    print("→ Return to original position")
                    print("  [Would navigate back to stored position]")
                    self.original_position = None
                
                elif result["action"] == "query":
                    # Use OpenAI for questions
                    print("→ Querying OpenAI...")
                    response = self.get_openai_response(text)
                    if response:
                        print(f"→ Response: {response}")
                        # Try to parse JSON if it's a command
                        import json
                        try:
                            if response.strip().startswith('{'):
                                cmd = json.loads(response)
                                if cmd.get("action") == "command":
                                    print(f"  [Detected command: {cmd.get('value')}]")
                        except:
                            pass
                
                print()  # Blank line
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point"""
    if not SPEECH_AVAILABLE or not OPENAI_AVAILABLE:
        print("Required libraries not available")
        sys.exit(1)
    
    assistant = VoiceAssistant()
    assistant.run_interactive()


if __name__ == '__main__':
    main()

