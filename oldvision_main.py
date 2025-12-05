#!/usr/bin/env python3
"""
Main control script for vision-based navigation with natural language commands
"""

# Fix OpenCV import if needed
try:
    import cv2
except ImportError:
    import sys
    import os
    system_paths = [
        '/usr/lib/python3/dist-packages',
        '/usr/local/lib/python3/dist-packages',
    ]
    for path in system_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not found!")
        print("Install with: sudo apt-get install python3-opencv")
        sys.exit(1)

import signal
import sys
import time
from vision_navigator import VisionNavigator
from speech_recognizer import SpeechRecognizer


class VisionControlSystem:
    """Main system controller"""
    
    def __init__(self, psoc_port: str = '/dev/ttyUSB0',
                 psoc_baudrate: int = 115200,
                 camera_index: int = 0,
                 wake_word: str = 'bin diesel'):
        """
        Initialize vision control system
        
        Args:
            psoc_port: Serial port for PSoC
            psoc_baudrate: Serial baud rate
            camera_index: Camera device index
            wake_word: Wake word for voice commands
        """
        self.navigator = VisionNavigator(
            psoc_port=psoc_port,
            psoc_baudrate=psoc_baudrate,
            camera_index=camera_index
        )
        
        self.speech = SpeechRecognizer(wake_word=wake_word)
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down...")
        self.stop()
        sys.exit(0)
    
    def handle_command(self, command: str):
        """Handle voice command"""
        intent = self.speech.process_command(command)
        
        print(f"Command: {command}")
        print(f"Intent: {intent}")
        
        if intent['action'] == 'follow':
            self.navigator.start_following()
        elif intent['action'] == 'stop':
            self.navigator.stop()
        elif intent['action'] == 'move':
            # Handle directional movement
            if intent['direction'] == 'forward':
                # Could implement forward movement
                pass
        else:
            print(f"Unknown command: {command}")
    
    def start(self, show_video: bool = True):
        """Start the system"""
        print("=" * 70)
        print("Vision-Based Navigation System")
        print("=" * 70)
        print(f"Wake word: '{self.speech.wake_word}'")
        print("Say the wake word followed by a command (e.g., 'bin diesel, come here')")
        print("Press 'q' to quit, 'f' to start following, 's' to stop")
        print("=" * 70)
        
        # Start components
        self.navigator.start()
        self.speech.start_listening_thread(self.handle_command)
        
        self.running = True
        
        # Main loop
        while self.running:
            # Update navigation
            self.navigator.update()
            
            # Show video if requested
            if show_video:
                frame = self.navigator.get_frame_with_overlay()
                if frame is not None:
                    cv2.imshow('Vision Navigation', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('f'):
                        self.navigator.start_following()
                    elif key == ord('s'):
                        self.navigator.stop()
            
            # Small delay
            time.sleep(0.01)
        
        self.stop()
    
    def stop(self):
        """Stop the system"""
        self.running = False
        self.speech.stop_listening()
        self.navigator.cleanup()
        cv2.destroyAllWindows()
        print("System stopped.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Vision-Based Navigation with Natural Language Commands'
    )
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                       help='Serial port for PSoC')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='Serial baud rate')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--wake-word', type=str, default='bin diesel',
                       help='Wake word for voice commands')
    parser.add_argument('--no-video', action='store_true',
                       help='Disable video display')
    
    args = parser.parse_args()
    
    # Create and start system
    system = VisionControlSystem(
        psoc_port=args.port,
        psoc_baudrate=args.baudrate,
        camera_index=args.camera,
        wake_word=args.wake_word
    )
    
    try:
        system.start(show_video=not args.no_video)
    except KeyboardInterrupt:
        system.stop()


if __name__ == '__main__':
    import time
    main()

