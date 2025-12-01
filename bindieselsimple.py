#!/usr/bin/env python3
"""
Simple implementation for Bin Diesel project:
1. Waits for wake word "bin diesel"
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

import speech_recognition as sr
import numpy as np
import time
import os
from psoc_communicator import PSoCCommunicator


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
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Camera started: {width}x{height}")
        return True
    
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
            print(f"Warning: Unknown color '{self.color}', using red")
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
    """Listens for wake word by comparing audio to reference file"""
    
    def __init__(self, wake_word_file="wake_word_reference.wav", similarity_threshold=0.6):
        """
        Initialize wake word listener
        
        Args:
            wake_word_file: Path to recorded wake word audio file
            similarity_threshold: Similarity threshold (0.0-1.0) for detection
        """
        self.wake_word_file = wake_word_file
        self.similarity_threshold = similarity_threshold
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.reference_features = None
        
        # Load reference audio features
        self._load_reference_audio()
    
    def _load_reference_audio(self):
        """Load and process reference wake word audio file"""
        import os
        if not os.path.exists(self.wake_word_file):
            print(f"[DEBUG] Reference wake word file '{self.wake_word_file}' not found!")
            print(f"Please record a reference file first using:")
            print(f"  python3 record_wake_word.py")
            return False
        
        try:
            # Try using librosa for better audio processing
            try:
                import librosa
                print(f"[DEBUG] Loading reference audio with librosa...")
                audio_data, sample_rate = librosa.load(self.wake_word_file, sr=None)
                print(f"[DEBUG] Audio loaded: {len(audio_data)} samples, {sample_rate} Hz")
                # Extract MFCC features (good for speech recognition)
                self.reference_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                print(f"[DEBUG] MFCC features extracted: shape {self.reference_features.shape}")
                print(f"✓ Loaded reference wake word from '{self.wake_word_file}' (librosa mode)")
                return True
            except ImportError:
                # Fallback to simple audio comparison using raw audio data
                print("[DEBUG] librosa not available, using simple audio comparison")
                with sr.AudioFile(self.wake_word_file) as source:
                    audio = self.recognizer.record(source)
                    # Store raw audio data for comparison
                    self.reference_features = audio.get_raw_data()
                print(f"[DEBUG] Raw audio data length: {len(self.reference_features)} bytes")
                print(f"✓ Loaded reference wake word from '{self.wake_word_file}' (simple mode)")
                return True
        except Exception as e:
            print(f"[DEBUG] Error loading reference audio: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_features(self, audio):
        """
        Extract features from audio for comparison
        
        Args:
            audio: AudioData from speech_recognition
            
        Returns:
            Feature array for comparison
        """
        try:
            import librosa
            import numpy as np
            # Convert AudioData to numpy array
            raw_data = audio.get_raw_data()
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = audio.sample_rate
            
            # Extract MFCC features (same as reference)
            features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            return features
        except ImportError:
            # Fallback: use raw audio data
            return audio.get_raw_data()
        except Exception as e:
            # If librosa fails for any reason, fall back
            print(f"Warning: librosa feature extraction failed: {e}, using simple comparison")
            return audio.get_raw_data()
    
    def _compare_audio(self, features1, features2):
        """
        Compare two audio feature sets
        
        Args:
            features1: Reference features
            features2: Incoming audio features
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            import numpy as np
            
            # If using MFCC features (librosa)
            if isinstance(features1, np.ndarray) and isinstance(features2, np.ndarray):
                # Normalize features
                f1 = features1.flatten()
                f2 = features2.flatten()
                
                # Pad or truncate to same length
                min_len = min(len(f1), len(f2))
                f1 = f1[:min_len]
                f2 = f2[:min_len]
                
                # Calculate cosine similarity
                dot_product = np.dot(f1, f2)
                norm1 = np.linalg.norm(f1)
                norm2 = np.linalg.norm(f2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)
            
            # Fallback: simple raw audio comparison
            else:
                # Compare raw audio data
                data1 = bytes(features1) if isinstance(features1, bytes) else features1
                data2 = bytes(features2) if isinstance(features2, bytes) else features2
                
                # Simple correlation-based comparison
                min_len = min(len(data1), len(data2))
                if min_len == 0:
                    return 0.0
                
                # Normalize and compare
                arr1 = np.frombuffer(data1[:min_len], dtype=np.uint8).astype(np.float32)
                arr2 = np.frombuffer(data2[:min_len], dtype=np.uint8).astype(np.float32)
                
                # Normalize
                arr1 = (arr1 - arr1.mean()) / (arr1.std() + 1e-8)
                arr2 = (arr2 - arr2.mean()) / (arr2.std() + 1e-8)
                
                # Correlation coefficient
                correlation = np.corrcoef(arr1, arr2)[0, 1]
                return float(abs(correlation)) if not np.isnan(correlation) else 0.0
                
        except Exception as e:
            print(f"Error comparing audio: {e}")
            return 0.0
    
    def listen_for_wake_word(self):
        """
        Listen for wake word by comparing to reference audio
        
        Returns:
            True when wake word is detected
        """
        if self.reference_features is None:
            print("Error: Reference wake word not loaded. Cannot listen.")
            return False
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print(f"Listening for wake word (comparing to '{self.wake_word_file}')...")
            print(f"Similarity threshold: {self.similarity_threshold:.2f}")
            
            while True:
                try:
                    # Listen for audio (shorter duration for wake word)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    
                    # Extract features from incoming audio
                    print("[DEBUG] Extracting features from incoming audio...")
                    incoming_features = self._extract_features(audio)
                    print(f"[DEBUG] Incoming features type: {type(incoming_features)}")
                    if hasattr(incoming_features, 'shape'):
                        print(f"[DEBUG] Incoming features shape: {incoming_features.shape}")
                    else:
                        print(f"[DEBUG] Incoming features length: {len(incoming_features) if hasattr(incoming_features, '__len__') else 'N/A'}")
                    
                    # Compare to reference
                    similarity = self._compare_audio(self.reference_features, incoming_features)
                    print(f"[DEBUG] Calculated similarity: {similarity:.4f}")
                    
                    if similarity >= self.similarity_threshold:
                        print(f"✓ Wake word detected! (similarity: {similarity:.3f} >= {self.similarity_threshold:.2f})")
                        return True
                    else:
                        # Show similarity for debugging (lower threshold for visibility)
                        if similarity > 0.2:  # Show if somewhat similar
                            print(f"[DEBUG] Similarity: {similarity:.3f} (threshold: {self.similarity_threshold:.2f}) - too low")
                        else:
                            print(f"[DEBUG] Low similarity: {similarity:.3f} - not matching")
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Error during wake word detection: {e}")
                    continue


def main():
    """Main function"""
    print("=" * 70)
    print("Bin Diesel Simple Implementation")
    print("=" * 70)
    
    # Initialize components
    print("[DEBUG] Initializing wake word listener...")
    wake_word_listener = WakeWordListener()
    
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
            wake_word_listener.listen_for_wake_word()
            
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
                    # Get frame width (assuming 640 from start_camera)
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
                    # Show that we're still looking (every 2 seconds)
                    elapsed = time.time() - start_time
                    if int(elapsed) % 2 == 0 and int(elapsed) != int(elapsed - 0.1):
                        print(f"[DEBUG] Still looking for flag... ({int(elapsed)}s elapsed, {detection_count} frames checked)")
                    time.sleep(0.1)
            
            if not flag_found:
                print("\nFlag not found within timeout period")
            
            # Cleanup camera
            flag_detector.cleanup()
            print("\nReturning to wake word listening...")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        flag_detector.cleanup()
        psoc.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()

