"""
Vision-based person detection and tracking using Raspberry Pi Camera Module 3
Uses OpenCV and YOLO or MediaPipe for person detection
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class PersonTracker:
    """Tracks person position in camera frame for navigation"""
    
    def __init__(self, method: str = 'mediapipe', camera_index: int = 0):
        """
        Initialize person tracker
        
        Args:
            method: 'mediapipe' (lightweight) or 'yolo' (more accurate)
            camera_index: Camera device index (0 for default)
        """
        self.method = method
        self.camera_index = camera_index
        self.cap = None
        
        if method == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                print("Using MediaPipe for person detection")
            except ImportError:
                print("MediaPipe not installed, falling back to OpenCV")
                self.use_mediapipe = False
        else:
            self.use_mediapipe = False
        
        # Person position tracking
        self.person_center = None
        self.person_size = None
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # seconds
        
    def start_camera(self, width: int = 640, height: int = 480):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera started: {width}x{height}")
        
    def detect_person_mediapipe(self, frame) -> Optional[Tuple[int, int, float]]:
        """Detect person using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get bounding box from pose landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Find min/max x, y coordinates
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            
            h, w = frame.shape[:2]
            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            size = max(x_max - x_min, y_max - y_min)
            
            return (center_x, center_y, size)
        
        return None
    
    def detect_person_opencv(self, frame) -> Optional[Tuple[int, int, float]]:
        """Detect person using OpenCV HOG detector (fallback)"""
        # Use HOG person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people
        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05,
            hitThreshold=0.5
        )
        
        if len(boxes) > 0:
            # Use the largest detection (closest person)
            largest_idx = np.argmax([w * h for (x, y, w, h) in boxes])
            x, y, w, h = boxes[largest_idx]
            
            center_x = x + w // 2
            center_y = y + h // 2
            size = max(w, h)
            
            return (center_x, center_y, size)
        
        return None
    
    def update(self) -> Optional[Tuple[int, int, float]]:
        """
        Capture frame and detect person
        
        Returns:
            (center_x, center_y, size) or None if no person detected
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Detect person
        if self.use_mediapipe:
            detection = self.detect_person_mediapipe(frame)
        else:
            detection = self.detect_person_opencv(frame)
        
        if detection:
            self.person_center = (detection[0], detection[1])
            self.person_size = detection[2]
            self.last_detection_time = time.time()
            return detection
        else:
            # Check if detection is stale
            if time.time() - self.last_detection_time > self.detection_timeout:
                self.person_center = None
                self.person_size = None
            return None
    
    def get_person_position(self) -> Optional[Tuple[int, int]]:
        """Get last detected person center position"""
        return self.person_center
    
    def get_person_size(self) -> Optional[float]:
        """Get last detected person size (for distance estimation)"""
        return self.person_size
    
    def calculate_direction(self, frame_width: int = 640) -> Optional[float]:
        """
        Calculate direction to person relative to camera center
        
        Args:
            frame_width: Width of camera frame
            
        Returns:
            Angle in degrees (-90 to +90), None if no person detected
            Negative = person on left, Positive = person on right
        """
        if self.person_center is None:
            return None
        
        center_x, center_y = self.person_center
        frame_center_x = frame_width // 2
        
        # Calculate offset from center
        offset = center_x - frame_center_x
        
        # Convert to angle (assuming ~60 degree FOV for Pi Camera)
        # Adjust this based on your camera's actual FOV
        fov_degrees = 60.0
        angle = (offset / frame_width) * fov_degrees
        
        return angle
    
    def estimate_distance(self, frame_height: int = 480) -> Optional[float]:
        """
        Estimate distance to person based on size in frame
        
        Args:
            frame_height: Height of camera frame
            
        Returns:
            Estimated distance in meters (rough approximation)
        """
        if self.person_size is None:
            return None
        
        # Rough calibration: adjust based on your camera and typical person height
        # This is a simple linear approximation
        # Person at 1m distance might be ~200 pixels tall
        # Person at 2m distance might be ~100 pixels tall
        reference_size_1m = 200  # pixels at 1 meter
        estimated_distance = reference_size_1m / (self.person_size + 1e-6)
        
        return estimated_distance
    
    def get_frame(self):
        """Get current camera frame for display"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def draw_detection(self, frame) -> np.ndarray:
        """Draw person detection on frame"""
        if self.person_center is None:
            return frame
        
        center_x, center_y = self.person_center
        size = self.person_size if self.person_size else 50
        
        # Draw bounding box
        half_size = int(size // 2)
        cv2.rectangle(
            frame,
            (center_x - half_size, center_y - half_size),
            (center_x + half_size, center_y + half_size),
            (0, 255, 0),
            2
        )
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw frame center line
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.use_mediapipe:
            self.pose.close()

