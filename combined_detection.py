#!/usr/bin/env python3
"""
Combined object detection (YOLO) and color detection
Overlays both YOLO object detections and colored flag detections on the same video stream
"""

import cv2
import numpy as np
import time
import sys
from picamera2 import Picamera2

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Try to import Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip3 install --break-system-packages ultralytics")
    sys.exit(1)

# Gesture detection using simple image analysis (no MediaPipe/OpenPose needed)
# Uses YOLO person detection + geometric analysis of person bounding box
MEDIAPIPE_AVAILABLE = False  # Not using MediaPipe anymore


# Configuration
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
YOLO_MODEL = 'yolo11n.pt'
FPS_AVG_FRAME_COUNT = 10
HORIZONTAL_FOV = 102.0  # Camera Module 3 Wide


class ColorDetector:
    """Detects colored flags/blocks in camera frame"""
    
    def __init__(self, color='red', horizontal_fov=102.0, min_area=500):
        """
        Initialize color detector
        
        Args:
            color: Color to detect ('red', 'green', 'blue', 'yellow')
            horizontal_fov: Horizontal field of view in degrees
            min_area: Minimum contour area to consider as flag
        """
        self.color = color.lower()
        self.horizontal_fov = horizontal_fov
        self.min_area = min_area
        
        # Enhanced HSV color ranges
        self.color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'green': [
                (np.array([35, 40, 40]), np.array([85, 255, 255]))
            ],
            'blue': [
                (np.array([95, 40, 40]), np.array([135, 255, 255]))
            ],
            'yellow': [
                (np.array([15, 50, 50]), np.array([35, 255, 255]))
            ]
        }
    
    def detect_color(self, frame):
        """
        Detect colored flag/block in frame
        
        Args:
            frame: Camera frame (BGR format)
            
        Returns:
            (center_x, center_y, area, bbox) or None if not detected
        """
        if frame is None:
            return None
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Get color range for selected color
        if self.color not in self.color_ranges:
            color_ranges = self.color_ranges['red']
        else:
            color_ranges = self.color_ranges[self.color]
        
        # Create mask for color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask += cv2.inRange(hsv, lower, upper)
        
        # Enhanced morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        mask = cv2.erode(mask, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_medium, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Filter contours by area and properties
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity < 0.3:
                continue
            
            valid_contours.append((contour, area, (x, y, w, h)))
        
        if len(valid_contours) == 0:
            return None
        
        # Select largest valid contour
        largest = max(valid_contours, key=lambda x: x[1])
        largest_contour, area, bbox = largest
        
        # Calculate center
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y, area, bbox)
    
    def calculate_angle(self, flag_center, frame_width):
        """Calculate angle of flag relative to center"""
        if flag_center is None:
            return None
        
        center_x, center_y = flag_center
        frame_center_x = frame_width / 2
        offset = center_x - frame_center_x
        angle = (offset / frame_width) * self.horizontal_fov
        return angle


def detect_arm_raised_simple(person_box, frame):
    """
    Simple gesture detection: detect if person has raised arm using bounding box analysis
    and edge detection within the person's region.
    
    This is a simplified approach that works without MediaPipe/OpenPose.
    It analyzes the person's bounding box shape and detects extended regions (arms).
    
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
    
    # Calculate bounding box aspect ratio and dimensions
    aspect_ratio = w / h
    center_x = w // 2
    center_y = h // 2
    
    # Method 1: Analyze bounding box width extension
    # When arm is raised, the bounding box should be wider
    # Normal person: aspect ratio ~0.4-0.6, with raised arm: >0.7
    
    # Method 2: Detect extended regions on left/right sides
    # Look for horizontal extensions in upper body region
    upper_body_region = person_roi[:int(h * 0.6), :]  # Top 60% of person
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(upper_body_region, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Analyze left and right halves
    left_half = edges[:, :center_x]
    right_half = edges[:, center_x:]
    
    # Count edge pixels in outer regions (where arms would extend)
    left_outer = left_half[:, :center_x//2]  # Leftmost quarter
    right_outer = right_half[:, center_x//2:]  # Rightmost quarter
    
    left_edge_density = np.sum(left_outer > 0) / max(left_outer.size, 1)
    right_edge_density = np.sum(right_outer > 0) / max(right_outer.size, 1)
    
    # Threshold for arm detection
    edge_threshold = 0.15  # 15% of pixels are edges
    
    # Method 3: Check if bounding box is wider than normal
    # Normal person aspect ratio: ~0.4-0.6
    # With raised arm: >0.7
    width_extension = aspect_ratio > 0.7
    
    # Combine methods
    left_raised = (left_edge_density > edge_threshold) or (width_extension and x1 < frame.shape[1] * 0.3)
    right_raised = (right_edge_density > edge_threshold) or (width_extension and x2 > frame.shape[1] * 0.7)
    
    # Calculate confidence based on edge density and aspect ratio
    confidence = min(1.0, (left_edge_density + right_edge_density) / 0.3)
    if width_extension:
        confidence = max(confidence, 0.7)
    
    return (left_raised, right_raised, confidence)




def draw_color_detection(frame, color_data, angle, color_name, frame_width):
    """
    Draw color detection overlay on frame
    
    Args:
        frame: Camera frame
        color_data: (center_x, center_y, area, bbox) or None
        angle: Calculated angle in degrees or None
        color_name: Name of color being detected
        frame_width: Width of frame
    """
    h, w = frame.shape[:2]
    
    if color_data is not None:
        center_x, center_y, area, bbox = color_data
        x, y, bw, bh = bbox
        
        # Draw bounding box (cyan)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
        
        # Draw center point (cyan)
        cv2.circle(frame, (center_x, center_y), 10, (255, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y), 15, (255, 255, 0), 2)
        
        # Draw line from center to flag
        cv2.line(frame, (w // 2, center_y), (center_x, center_y), (255, 255, 0), 2)
        
        # Draw angle text
        if angle is not None:
            angle_text = f"{color_name.upper()} Angle: {angle:.1f}°"
            direction = "LEFT" if angle < 0 else "RIGHT" if angle > 0 else "CENTER"
            cv2.putText(frame, angle_text, (10, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, direction, (10, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw position info
        pos_text = f"{color_name.upper()} Pos: ({center_x}, {center_y})"
        area_text = f"{color_name.upper()} Area: {int(area)}"
        cv2.putText(frame, pos_text, (10, h - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, area_text, (10, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw center line (white)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.putText(frame, "CENTER", (w // 2 + 5, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw color status
    status_text = f"{color_name.upper()} DETECTED" if color_data is not None else f"NO {color_name.upper()}"
    status_color = (0, 255, 0) if color_data is not None else (0, 0, 255)
    cv2.putText(frame, status_text, (w - 250, h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combined YOLO object detection and color detection')
    parser.add_argument('--width', type=int, default=DISPLAY_WIDTH,
                       help=f'Camera width (default: {DISPLAY_WIDTH})')
    parser.add_argument('--height', type=int, default=DISPLAY_HEIGHT,
                       help=f'Camera height (default: {DISPLAY_HEIGHT})')
    parser.add_argument('--model', type=str, default=YOLO_MODEL,
                       help=f'YOLO model (default: {YOLO_MODEL})')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO confidence threshold (0.0-1.0, default: 0.25)')
    parser.add_argument('--color', type=str, default='red',
                       choices=['red', 'green', 'blue', 'yellow'],
                       help='Color to detect (default: red)')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum area for color detection (default: 500)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    parser.add_argument('--no-gesture', action='store_true',
                       help='Disable gesture detection (if MediaPipe is installed)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Combined Object Detection (YOLO + Color Detection)")
    print("=" * 70)
    print(f"YOLO Model: {args.model}")
    print(f"Color: {args.color}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"YOLO Confidence: {args.conf}")
    print("=" * 70)
    print("Press 'q' or ESC to quit")
    print("Press 'r', 'g', 'b', 'y' to switch colors")
    print("=" * 70)
    print()
    
    # Initialize components
    color_detector = ColorDetector(
        color=args.color,
        horizontal_fov=HORIZONTAL_FOV,
        min_area=args.min_area
    )
    
    # Gesture detection enabled by default (uses simple method, no MediaPipe needed)
    enable_gesture = not args.no_gesture
    if enable_gesture:
        print("[DEBUG] Gesture detection enabled (simple method - no MediaPipe/OpenPose needed)")
    else:
        print("[DEBUG] Gesture detection disabled (--no-gesture flag)")
    
    # Initialize picamera2
    print("[DEBUG] Initializing picamera2 (Camera Module 3 Wide)...")
    try:
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        picam2.start()
        time.sleep(0.5)
        print(f"[DEBUG] Camera started: {args.width}x{args.height}")
    except Exception as e:
        print(f"ERROR: Could not initialize camera: {e}")
        return
    
    # Load YOLO model
    print(f"[DEBUG] Loading YOLO model: {args.model}...")
    try:
        yolo_model = YOLO(args.model)
        print("[DEBUG] YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not load YOLO model: {e}")
        picam2.stop()
        picam2.close()
        return
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\nStarting combined detection...")
    print("YOLO detects objects, color detector finds colored flags/blocks\n")
    
    counter, fps = 0, 0
    fps_start_time = time.time()
    
    try:
        while True:
            # Capture frame from picamera2
            array = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            frame_width = frame.shape[1]
            
            counter += 1
            
            # Run YOLO inference
            yolo_results = yolo_model(frame, conf=args.conf, verbose=False)
            
            # Get YOLO annotated frame (with object detections drawn)
            annotated_frame = yolo_results[0].plot()
            
            # Run simple gesture detection on detected persons
            person_boxes = []
            gesture_detections = []
            
            if enable_gesture:
                # Find person class ID
                person_class_id = None
                for class_id, class_name in yolo_model.names.items():
                    if class_name == 'person':
                        person_class_id = class_id
                        break
                
                if person_class_id is not None:
                    # Find all person detections from YOLO
                    for box in yolo_results[0].boxes:
                        if int(box.cls[0]) == person_class_id:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            person_boxes.append((x1, y1, x2, y2))
                            
                            # Use simple gesture detection (no MediaPipe/OpenPose needed)
                            left_raised, right_raised, confidence = detect_arm_raised_simple(
                                (x1, y1, x2, y2), frame
                            )
                            gesture_detections.append((left_raised, right_raised, confidence))
            
            # Draw gesture detection
            if enable_gesture and person_boxes:
                for i, person_box in enumerate(person_boxes):
                    if i < len(gesture_detections):
                        left_raised, right_raised, confidence = gesture_detections[i]
                        x1, y1, x2, y2 = person_box
                        
                        if left_raised or right_raised:
                            # Draw green box around person with raised arm
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            gesture_text = "ARM RAISED!"
                            if left_raised and right_raised:
                                gesture_text = f"BOTH ARMS RAISED! ({confidence:.0%})"
                            elif left_raised:
                                gesture_text = f"LEFT ARM RAISED! ({confidence:.0%})"
                            elif right_raised:
                                gesture_text = f"RIGHT ARM RAISED! ({confidence:.0%})"
                            
                            cv2.putText(annotated_frame, gesture_text, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect colored flag/block
            color_data = color_detector.detect_color(frame)
            
            # Calculate angle if color detected
            angle = None
            if color_data is not None:
                center_x, center_y, _, _ = color_data
                angle = color_detector.calculate_angle((center_x, center_y), frame_width)
            
            # Draw color detection overlay on top of YOLO results
            draw_color_detection(annotated_frame, color_data, angle, 
                               color_detector.color, frame_width)
            
            # Calculate FPS
            if counter % FPS_AVG_FRAME_COUNT == 0:
                fps_end_time = time.time()
                fps = FPS_AVG_FRAME_COUNT / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                
                # Print detection info to terminal
                yolo_count = len(yolo_results[0].boxes)
                color_status = "DETECTED" if color_data is not None else "NOT DETECTED"
                
                # Count gesture detections
                gesture_count = 0
                if enable_gesture and gesture_detections:
                    for left_raised, right_raised, _ in gesture_detections:
                        if left_raised or right_raised:
                            gesture_count += 1
                
                gesture_text = f" | Gestures: {gesture_count}" if MEDIAPIPE_AVAILABLE else ""
                
                print(f"[Frame {counter}] YOLO: {yolo_count} objects | "
                      f"Color ({color_detector.color.upper()}): {color_status}{gesture_text} | "
                      f"FPS: {fps:.1f}")
                
                if yolo_count > 0:
                    for box in yolo_results[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = yolo_model.names[class_id]
                        print(f"  YOLO: {class_name}: {confidence:.2f}")
                
                if color_data is not None:
                    print(f"  Color: Center=({color_data[0]}, {color_data[1]}), "
                          f"Area={int(color_data[2])}, Angle={angle:.1f}°")
            
            # Show FPS on screen
            if not args.no_fps:
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Combined Detection (YOLO + Color) - Press q to quit', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):
                color_detector.color = 'red'
                print(f"\nSwitched to detecting: RED")
            elif key == ord('g'):
                color_detector.color = 'green'
                print(f"\nSwitched to detecting: GREEN")
            elif key == ord('b'):
                color_detector.color = 'blue'
                print(f"\nSwitched to detecting: BLUE")
            elif key == ord('y'):
                color_detector.color = 'yellow'
                print(f"\nSwitched to detecting: YELLOW")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("[DEBUG] Stopping camera...")
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        elapsed = time.time() - fps_start_time
        avg_fps = counter / elapsed if elapsed > 0 else 0
        print(f"\nTotal frames: {counter}, Average FPS: {avg_fps:.2f}")
        print("Camera closed.")


if __name__ == '__main__':
    main()

