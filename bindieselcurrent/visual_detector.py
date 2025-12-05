"""
Visual Detection Module
Detects person, tracks position, and detects arm raising gesture
Uses YOLO for object detection and custom algorithm for gesture detection
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip3 install --break-system-packages ultralytics")
    import sys
    sys.exit(1)


class VisualDetector:
    """Detects person, tracks position, and detects arm raising"""
    
    def __init__(self, model_path='yolo11n.pt', width=640, height=480, confidence=0.25):
        """
        Initialize visual detector
        
        Args:
            model_path: Path to YOLO model file
            width: Camera width
            height: Camera height
            confidence: YOLO confidence threshold
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.frame_center_x = width // 2
        
        # Load YOLO model
        print(f"[VisualDetector] Loading YOLO model: {model_path}...")
        try:
            self.yolo_model = YOLO(model_path)
            print("[VisualDetector] YOLO model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        # Find person class ID
        self.person_class_id = None
        for class_id, class_name in self.yolo_model.names.items():
            if class_name == 'person':
                self.person_class_id = class_id
                break
        
        if self.person_class_id is None:
            raise RuntimeError("Person class not found in YOLO model!")
        
        # Initialize camera
        print("[VisualDetector] Initializing camera...")
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(0.5)  # Let camera stabilize
            print(f"[VisualDetector] Camera started: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
        
        # Tracking state
        self.last_person_position = None
        self.last_arm_raised = False
    
    def detect_person(self, frame):
        """
        Detect person in frame using YOLO
        
        Args:
            frame: Camera frame (BGR format)
            
        Returns:
            (x1, y1, x2, y2, confidence) of person or None if not detected
        """
        # Run YOLO inference
        results = self.yolo_model(frame, conf=self.confidence, verbose=False)
        
        # Find person detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id == self.person_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                return (x1, y1, x2, y2, confidence)
        
        return None
    
    def detect_arm_raised(self, person_box, frame):
        """
        Detect if person has raised arm
        
        Args:
            person_box: (x1, y1, x2, y2) bounding box of person
            frame: Camera frame (BGR format)
            
        Returns:
            (left_raised, right_raised, confidence) tuple
        """
        x1, y1, x2, y2 = person_box
        
        # Extract person region
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return (False, False, 0.0)
        
        h, w = person_roi.shape[:2]
        if h < 50 or w < 30:  # Too small to analyze
            return (False, False, 0.0)
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        center_x = w // 2
        
        # Analyze upper body region (top 60%)
        upper_body_region = person_roi[:int(h * 0.6), :]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(upper_body_region, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze left and right halves
        left_half = edges[:, :center_x]
        right_half = edges[:, center_x:]
        
        # Count edge pixels in outer regions
        left_outer = left_half[:, :max(center_x//2, 1)]
        right_outer = right_half[:, center_x//2:]
        
        left_edge_density = np.sum(left_outer > 0) / max(left_outer.size, 1)
        right_edge_density = np.sum(right_outer > 0) / max(right_outer.size, 1)
        
        # Threshold for arm detection
        edge_threshold = 0.15
        
        # Check if bounding box is wider than normal (arm extended)
        width_extension = aspect_ratio > 0.7
        
        # Combine methods
        left_raised = (left_edge_density > edge_threshold) or (width_extension and x1 < frame.shape[1] * 0.3)
        right_raised = (right_edge_density > edge_threshold) or (width_extension and x2 > frame.shape[1] * 0.7)
        
        # Calculate confidence
        confidence = min(1.0, (left_edge_density + right_edge_density) / 0.3)
        if width_extension:
            confidence = max(confidence, 0.7)
        
        return (left_raised, right_raised, confidence)
    
    def get_frame(self):
        """
        Get current camera frame
        
        Returns:
            Frame in BGR format (for OpenCV processing)
        """
        array = self.picam2.capture_array()  # Returns RGB
        frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # Convert to BGR
        return frame
    
    def calculate_angle(self, person_box):
        """
        Calculate angle of person relative to camera center
        
        Args:
            person_box: (x1, y1, x2, y2) bounding box
            
        Returns:
            Angle in degrees (negative = left, positive = right, 0 = center)
        """
        x1, y1, x2, y2 = person_box
        person_center_x = (x1 + x2) / 2
        
        # Calculate offset from center
        offset = person_center_x - self.frame_center_x
        
        # Convert to angle (assuming 102° horizontal FOV)
        angle = (offset / self.width) * 102.0
        
        return angle
    
    def is_person_centered(self, person_box, threshold=30):
        """
        Check if person is centered in frame
        
        Args:
            person_box: (x1, y1, x2, y2) bounding box
            threshold: Pixels from center to consider "centered"
            
        Returns:
            True if person is centered, False otherwise
        """
        x1, y1, x2, y2 = person_box
        person_center_x = (x1 + x2) / 2
        offset = abs(person_center_x - self.frame_center_x)
        return offset < threshold
    
    def update(self):
        """
        Update detection (main detection loop)
        
        Returns:
            dict with detection results:
            {
                'person_detected': bool,
                'person_box': (x1, y1, x2, y2) or None,
                'angle': float or None,
                'is_centered': bool,
                'arm_raised': bool,
                'arm_confidence': float
            }
        """
        frame = self.get_frame()
        
        # Detect person
        person_data = self.detect_person(frame)
        
        if person_data is None:
            self.last_person_position = None
            self.last_arm_raised = False
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'arm_raised': False,
                'arm_confidence': 0.0
            }
        
        x1, y1, x2, y2, confidence = person_data
        person_box = (x1, y1, x2, y2)
        
        # Calculate angle
        angle = self.calculate_angle(person_box)
        
        # Check if centered
        is_centered = self.is_person_centered(person_box)
        
        # Detect arm raising
        left_raised, right_raised, arm_confidence = self.detect_arm_raised(person_box, frame)
        arm_raised = left_raised or right_raised
        
        # Update tracking state
        self.last_person_position = person_box
        self.last_arm_raised = arm_raised
        
        return {
            'person_detected': True,
            'person_box': person_box,
            'angle': angle,
            'is_centered': is_centered,
            'arm_raised': arm_raised,
            'arm_confidence': arm_confidence
        }
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        print("[VisualDetector] Stopped")


if __name__ == '__main__':
    # Test visual detection
    import config
    
    print("Testing visual detection...")
    print("Press Ctrl+C to exit")
    
    try:
        detector = VisualDetector(
            model_path=config.YOLO_MODEL,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=config.YOLO_CONFIDENCE
        )
        
        while True:
            result = detector.update()
            
            if result['person_detected']:
                print(f"Person detected: angle={result['angle']:.1f}°, "
                      f"centered={result['is_centered']}, "
                      f"arm_raised={result['arm_raised']}")
            else:
                print("No person detected")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'detector' in locals():
            detector.stop()

