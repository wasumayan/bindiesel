#!/usr/bin/env python3
"""
Simple implementation for Bin Diesel project:
1. Waits for wake word "bin diesel" (using Picovoice Porcupine)
2. Once wake word detected, camera looks for colored flag
3. Calculates angle relative to car
4. Sends angle to PSoC
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

import pvporcupine
import pyaudio
import numpy as np
import time
import os
from psoc_communicator import PSoCCommunicator

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[DEBUG] python-dotenv not installed, .env file won't be loaded automatically")
    print("Install with: pip3 install --break-system-packages python-dotenv")


class ColorFlagDetector:
    """Detects colored flag in camera frame"""
    
    def __init__(self, color='red', camera_index=0, horizontal_fov=102.0):
        """
        Initialize color flag detector
        
        Args:
            color: Color to detect ('red', 'green', 'blue', 'yellow')
            camera_index: Camera device index
            horizontal_fov: Horizontal field of view in degrees (default 102 for Camera Module 3 Wide)
        """
        self.color = color.lower()
        self.camera_index = camera_index
        self.horizontal_fov = horizontal_fov
        self.cap = None
        
        # Define HSV color ranges for different colors
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Lower red
                (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Upper red
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ],
            'blue': [
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))
            ]
        }
    
    def start_camera(self, width=640, height=480):
        """Initialize camera using libcamera (via V4L2 backend)"""
        # Try V4L2 backend first (works with libcamera on Raspberry Pi)
        backends_to_try = []
        if hasattr(cv2, 'CAP_V4L2'):
            backends_to_try.append((cv2.CAP_V4L2, "V4L2 (libcamera)"))
        backends_to_try.append((cv2.CAP_ANY, "ANY (auto-detect)"))
        
        for backend_id, backend_name in backends_to_try:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend_id)
                if self.cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[DEBUG] Camera started with {backend_name}: {actual_width}x{actual_height}")
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"[DEBUG] Error with {backend_name}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        print(f"[DEBUG] Error: Could not open camera {self.camera_index} with any backend")
        return False
    
    def detect_flag(self):
        """
        Detect colored flag in current frame
        
        Returns:
            (center_x, center_y) of flag or None if not detected
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range for selected color
        if self.color not in self.color_ranges:
            color_ranges = self.color_ranges['red']
        else:
            color_ranges = self.color_ranges[self.color]
        
        # Create mask for color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask += cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find largest contour (assumed to be the flag)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area to reduce noise
        if cv2.contourArea(largest_contour) < 500:
            return None
        
        # Calculate center of flag
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y)
    
    def calculate_angle(self, flag_center, frame_width):
        """
        Calculate angle of flag relative to car center
        
        Args:
            flag_center: (x, y) tuple of flag center
            frame_width: Width of camera frame
            
        Returns:
            Angle in degrees (-90 to +90, negative = left, positive = right)
        """
        if flag_center is None:
            return None
        
        center_x, center_y = flag_center
        frame_center_x = frame_width / 2
        
        # Calculate horizontal offset
        offset = center_x - frame_center_x
        
        # Convert pixel offset to angle using camera's horizontal FOV
        # Camera Module 3 Wide has 102° horizontal FOV
        angle = (offset / frame_width) * self.horizontal_fov
        
        return angle
    
    def cleanup(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()


class WakeWordListener:
    """Listens for wake word using Picovoice Porcupine"""
    
    def __init__(self, access_key, keyword_path):
        """
        Initialize Porcupine wake word listener
        
        Args:
            access_key: Picovoice AccessKey (get from https://console.picovoice.ai/)
            keyword_path: Path to custom wake word .ppn file
        """
        self.access_key = access_key
        self.keyword_path = keyword_path
        
        print("[DEBUG] Initializing Porcupine wake word engine...")
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path]
            )
            print(f"[DEBUG] Porcupine initialized successfully")
            print(f"[DEBUG] Sample rate: {self.porcupine.sample_rate} Hz")
            print(f"[DEBUG] Frame length: {self.porcupine.frame_length} samples")
        except Exception as e:
            print(f"[DEBUG] Error initializing Porcupine: {e}")
            raise
        
        # Initialize PyAudio for microphone input
        print("[DEBUG] Initializing PyAudio...")
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def listen_for_wake_word(self):
        """
        Listen for wake word
        
        Returns:
            True when wake word is detected
        """
        print(f"[DEBUG] Starting microphone stream...")
        try:
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            print(f"[DEBUG] Listening for wake word...")
            
            while True:
                pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm_array = np.frombuffer(pcm, dtype=np.int16)
                
                keyword_index = self.porcupine.process(pcm_array)
                
                if keyword_index >= 0:
                    print(f"✓ Wake word detected!")
                    return True
                    
        except KeyboardInterrupt:
            print("\n[DEBUG] Interrupted by user")
            return False
        except Exception as e:
            print(f"[DEBUG] Error during wake word detection: {e}")
            return False
    
    def cleanup(self):
        """Release resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()


def main():
    """Main function"""
    print("=" * 70)
    print("Bin Diesel Simple Implementation")
    print("=" * 70)
    
    # Configuration
    PICOVOICE_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY', '')
    KEYWORD_PATH = 'bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
    
    if not PICOVOICE_ACCESS_KEY:
        print("ERROR: PICOVOICE_ACCESS_KEY environment variable not set!")
        print("Get your AccessKey from: https://console.picovoice.ai/")
        print("Then set it with: export PICOVOICE_ACCESS_KEY='your-key-here'")
        return
    
    if not os.path.exists(KEYWORD_PATH):
        print(f"ERROR: Wake word file not found: {KEYWORD_PATH}")
        print("Make sure the bin-diesel_en_raspberry-pi_v3_0_0 folder is in the project directory")
        return
    
    # Initialize components
    print("[DEBUG] Initializing wake word listener...")
    try:
        wake_word_listener = WakeWordListener(PICOVOICE_ACCESS_KEY, KEYWORD_PATH)
    except Exception as e:
        print(f"ERROR: Failed to initialize wake word listener: {e}")
        return
    
    print("[DEBUG] Initializing color flag detector...")
    flag_detector = ColorFlagDetector(color='red', camera_index=0)
    
    print("[DEBUG] Initializing PSoC communicator...")
    psoc = PSoCCommunicator(port='/dev/ttyUSB0', baudrate=115200)
    
    # Connect to PSoC
    if not psoc.connect():
        print("Warning: Could not connect to PSoC. Continuing anyway...")
    
    try:
        while True:
            # Step 1: Wait for wake word
            print("\nWaiting for wake word...")
            if not wake_word_listener.listen_for_wake_word():
                break
            
            # Step 2: Start camera and look for flag
            print("[DEBUG] Wake word detected! Starting camera...")
            if not flag_detector.start_camera():
                print("[DEBUG] Error: Could not start camera")
                continue
            
            print("[DEBUG] Looking for colored flag (red)...")
            flag_found = False
            timeout = 10.0  # Look for flag for 10 seconds
            start_time = time.time()
            detection_count = 0
            
            while time.time() - start_time < timeout:
                flag_center = flag_detector.detect_flag()
                detection_count += 1
                
                if flag_center is not None:
                    flag_found = True
                    print(f"[DEBUG] Flag detected at position: {flag_center}")
                    
                    # Step 3: Calculate angle
                    frame_width = 640
                    angle = flag_detector.calculate_angle(flag_center, frame_width)
                    
                    if angle is not None:
                        print(f"[DEBUG] Calculated angle: {angle:.2f} degrees")
                        
                        # Step 4: Send angle to PSoC
                        print(f"[DEBUG] Sending angle to PSoC...")
                        if psoc.send_angle_simple(angle):
                            print(f"✓ Angle sent to PSoC: {angle:.2f} degrees")
                        else:
                            print("[DEBUG] Error sending angle to PSoC")
                    else:
                        print("[DEBUG] Could not calculate angle")
                    
                    # Wait a bit before next detection
                    time.sleep(0.1)
                else:
                    # Show progress every 2 seconds
                    elapsed = time.time() - start_time
                    if int(elapsed) % 2 == 0 and int(elapsed) != int(elapsed - 0.1):
                        print(f"[DEBUG] Still looking for flag... ({int(elapsed)}s elapsed, {detection_count} frames checked)")
                    time.sleep(0.1)
            
            if not flag_found:
                print("\n[DEBUG] Flag not found within timeout period")
            
            # Cleanup camera
            flag_detector.cleanup()
            print("\nReturning to wake word listening...")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        flag_detector.cleanup()
        wake_word_listener.cleanup()
        psoc.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
