#!/usr/bin/env python3
"""
Bin Diesel Visual Detection System
Optimized for efficiency and speed
Features:
- YOLO object detection
- Movement tracking
- Arm raising gesture detection
"""

import cv2
import numpy as np
import time
import sys
from collections import deque
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


# Configuration
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
YOLO_MODEL = 'yolo11n.pt'  # Nano model for speed
FPS_AVG_FRAME_COUNT = 10
HORIZONTAL_FOV = 102.0  # Camera Module 3 Wide
GESTURE_PROCESS_INTERVAL = 2  # Process gesture every Nth frame
TRACKER_HISTORY_SIZE = 10  # Frames to track movement


class SimpleTracker:
    """Simple object tracker using centroid distance matching"""
    
    def __init__(self, max_distance=50, history_size=TRACKER_HISTORY_SIZE):
        """
        Initialize tracker
        
        Args:
            max_distance: Maximum distance to match objects between frames (pixels)
            history_size: Number of frames to keep in history
        """
        self.max_distance = max_distance
        self.history_size = history_size
        self.tracks = {}  # {track_id: deque of (x, y, frame_num)}
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of (x1, y1, x2, y2, class_id, confidence) tuples
            
        Returns:
            List of (track_id, x1, y1, x2, y2, class_id, confidence, velocity) tuples
        """
        self.frame_count += 1
        current_centroids = []
        
        # Calculate centroids for new detections
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids.append((cx, cy, det))
        
        # Match detections to existing tracks
        matched = set()
        updated_tracks = {}
        
        for track_id, history in self.tracks.items():
            if len(history) == 0:
                continue
            
            # Get last known position
            last_pos = history[-1]
            last_cx, last_cy = last_pos[0], last_pos[1]
            
            # Find closest detection
            best_match = None
            best_distance = float('inf')
            
            for i, (cx, cy, det) in enumerate(current_centroids):
                if i in matched:
                    continue
                
                distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match = (i, det, cx, cy)
            
            if best_match is not None:
                i, det, cx, cy = best_match
                matched.add(i)
                
                # Calculate velocity (pixels per frame)
                velocity = best_distance if best_distance > 0 else 0
                
                # Update track history
                history.append((cx, cy, self.frame_count))
                if len(history) > self.history_size:
                    history.popleft()
                
                updated_tracks[track_id] = (det, velocity)
        
        # Create new tracks for unmatched detections
        for i, (cx, cy, det) in enumerate(current_centroids):
            if i not in matched:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = deque([(cx, cy, self.frame_count)], maxlen=self.history_size)
                updated_tracks[track_id] = (det, 0.0)
        
        # Remove old tracks (not seen for a while)
        tracks_to_remove = []
        for track_id, history in self.tracks.items():
            if track_id not in updated_tracks:
                # Check if track is too old
                if len(history) > 0:
                    last_frame = history[-1][2]
                    if self.frame_count - last_frame > self.history_size:
                        tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Build result list
        results = []
        for track_id, (det, velocity) in updated_tracks.items():
            x1, y1, x2, y2, class_id, confidence = det
            results.append((track_id, x1, y1, x2, y2, class_id, confidence, velocity))
        
        return results


def detect_arm_raised_simple(person_box, frame):
    """
    Simple gesture detection: detect if person has raised arm using bounding box analysis
    and edge detection within the person's region.
    
    Optimized version - only processes upper body region.
    
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
    
    # Calculate bounding box aspect ratio
    aspect_ratio = w / h
    center_x = w // 2
    
    # Analyze upper body region only (top 60% - where arms would be)
    upper_body_region = person_roi[:int(h * 0.6), :]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(upper_body_region, cv2.COLOR_BGR2GRAY)
    
    # Detect edges (optimized thresholds)
    edges = cv2.Canny(gray, 50, 150)
    
    # Analyze left and right halves
    left_half = edges[:, :center_x]
    right_half = edges[:, center_x:]
    
    # Count edge pixels in outer regions (where arms would extend)
    left_outer = left_half[:, :max(center_x//2, 1)]  # Leftmost quarter
    right_outer = right_half[:, center_x//2:]  # Rightmost quarter
    
    left_edge_density = np.sum(left_outer > 0) / max(left_outer.size, 1)
    right_edge_density = np.sum(right_outer > 0) / max(right_outer.size, 1)
    
    # Threshold for arm detection
    edge_threshold = 0.15  # 15% of pixels are edges
    
    # Check if bounding box is wider than normal (arm extended)
    # Normal person: aspect ratio ~0.4-0.6, with raised arm: >0.7
    width_extension = aspect_ratio > 0.7
    
    # Combine methods
    left_raised = (left_edge_density > edge_threshold) or (width_extension and x1 < frame.shape[1] * 0.3)
    right_raised = (right_edge_density > edge_threshold) or (width_extension and x2 > frame.shape[1] * 0.7)
    
    # Calculate confidence
    confidence = min(1.0, (left_edge_density + right_edge_density) / 0.3)
    if width_extension:
        confidence = max(confidence, 0.7)
    
    return (left_raised, right_raised, confidence)


def draw_tracking_info(frame, track_id, velocity, x1, y1, x2, y2):
    """
    Draw tracking information on frame
    
    Args:
        frame: Frame to draw on (BGR format)
        track_id: Track ID
        velocity: Velocity in pixels per frame
        x1, y1, x2, y2: Bounding box coordinates
    """
    # Draw track ID
    track_text = f"ID:{track_id}"
    if velocity > 0:
        track_text += f" V:{velocity:.1f}"
    
    # Position text above bounding box
    text_y = max(y1 - 10, 20)
    cv2.putText(frame, track_text, (x1, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_gesture_info(frame, person_box, left_raised, right_raised, confidence):
    """
    Draw gesture detection information on frame
    
    Args:
        frame: Frame to draw on (BGR format)
        person_box: (x1, y1, x2, y2) bounding box
        left_raised: Whether left arm is raised
        right_raised: Whether right arm is raised
        confidence: Detection confidence (0.0-1.0)
    """
    x1, y1, x2, y2 = person_box
    
    if left_raised or right_raised:
        # Draw green box around person with raised arm
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        gesture_text = "ARM RAISED!"
        if left_raised and right_raised:
            gesture_text = f"BOTH ARMS RAISED! ({confidence:.0%})"
        elif left_raised:
            gesture_text = f"LEFT ARM RAISED! ({confidence:.0%})"
        elif right_raised:
            gesture_text = f"RIGHT ARM RAISED! ({confidence:.0%})"
        
        cv2.putText(frame, gesture_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bin Diesel Visual Detection System (YOLO + Tracking + Gestures)')
    parser.add_argument('--width', type=int, default=DISPLAY_WIDTH,
                       help=f'Camera width (default: {DISPLAY_WIDTH})')
    parser.add_argument('--height', type=int, default=DISPLAY_HEIGHT,
                       help=f'Camera height (default: {DISPLAY_HEIGHT})')
    parser.add_argument('--model', type=str, default=YOLO_MODEL,
                       help=f'YOLO model (default: {YOLO_MODEL})')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO confidence threshold (0.0-1.0, default: 0.25)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    parser.add_argument('--no-gesture', action='store_true',
                       help='Disable gesture detection')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable movement tracking')
    parser.add_argument('--track-distance', type=float, default=50.0,
                       help='Maximum distance for tracking (pixels, default: 50.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bin Diesel Visual Detection System")
    print("=" * 70)
    print(f"YOLO Model: {args.model}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"YOLO Confidence: {args.conf}")
    print(f"Tracking: {'ENABLED' if not args.no_tracking else 'DISABLED'}")
    print(f"Gesture Detection: {'ENABLED' if not args.no_gesture else 'DISABLED'}")
    print("=" * 70)
    print("Press 'q' or ESC to quit")
    print("=" * 70)
    print()
    
    # Initialize tracker
    tracker = None
    if not args.no_tracking:
        tracker = SimpleTracker(max_distance=args.track_distance)
        print("[DEBUG] Movement tracking enabled")
    
    # Gesture detection enabled by default
    enable_gesture = not args.no_gesture
    if enable_gesture:
        print("[DEBUG] Gesture detection enabled")
    
    # Initialize picamera2
    print("[DEBUG] Initializing picamera2 (Camera Module 3 Wide)...")
    try:
        picam2 = Picamera2()
        # Optimize resolution for speed
        actual_width = min(args.width, 640) if args.width > 640 else args.width
        actual_height = min(args.height, 480) if args.height > 480 else args.height
        
        preview_config = picam2.create_preview_configuration(
            main={"size": (actual_width, actual_height), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        picam2.start()
        time.sleep(0.5)
        print(f"[DEBUG] Camera started: {actual_width}x{actual_height} (optimized for speed)")
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
    
    # Find person class ID (cache it)
    person_class_id = None
    for class_id, class_name in yolo_model.names.items():
        if class_name == 'person':
            person_class_id = class_id
            break
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\nStarting visual detection system...")
    print("Features: Object Detection | Movement Tracking | Gesture Recognition\n")
    
    counter, fps = 0, 0
    fps_start_time = time.time()
    frame_skip_count = 0
    
    try:
        while True:
            # Capture frame from picamera2 (returns RGB)
            array = picam2.capture_array()
            
            # IMPORTANT: picamera2 returns RGB, OpenCV uses BGR
            # Convert RGB to BGR for all OpenCV operations
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            frame_width = frame.shape[1]
            
            counter += 1
            frame_skip_count += 1
            
            # Run YOLO inference (always run for object detection)
            yolo_results = yolo_model(frame, conf=args.conf, verbose=False)
            
            # Get YOLO annotated frame (with object detections drawn)
            annotated_frame = yolo_results[0].plot()
            
            # Extract detections for tracking
            detections = []
            person_boxes = []
            
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detections.append((x1, y1, x2, y2, class_id, confidence))
                
                # Collect person boxes for gesture detection
                if enable_gesture and class_id == person_class_id:
                    person_boxes.append((x1, y1, x2, y2))
            
            # Update tracker
            tracked_objects = []
            if tracker is not None:
                tracked_objects = tracker.update(detections)
                
                # Draw tracking info on annotated frame
                for track_id, x1, y1, x2, y2, class_id, conf, velocity in tracked_objects:
                    draw_tracking_info(annotated_frame, track_id, velocity, x1, y1, x2, y2)
            
            # Gesture detection (only process every Nth frame for speed)
            gesture_detections = []
            if enable_gesture and frame_skip_count % GESTURE_PROCESS_INTERVAL == 0:
                for person_box in person_boxes:
                    left_raised, right_raised, confidence = detect_arm_raised_simple(
                        person_box, frame
                    )
                    gesture_detections.append((person_box, left_raised, right_raised, confidence))
            
            # Draw gesture detection
            if enable_gesture:
                for person_box, left_raised, right_raised, confidence in gesture_detections:
                    draw_gesture_info(annotated_frame, person_box, left_raised, right_raised, confidence)
            
            # Calculate FPS
            if counter % FPS_AVG_FRAME_COUNT == 0:
                fps_end_time = time.time()
                fps = FPS_AVG_FRAME_COUNT / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                
                # Print detection info to terminal
                yolo_count = len(yolo_results[0].boxes)
                tracked_count = len(tracked_objects) if tracker else 0
                gesture_count = sum(1 for _, l, r, _ in gesture_detections if l or r) if enable_gesture else 0
                
                print(f"[Frame {counter}] Objects: {yolo_count} | "
                      f"Tracked: {tracked_count} | "
                      f"Gestures: {gesture_count} | "
                      f"FPS: {fps:.1f}")
                
                if yolo_count > 0:
                    for box in yolo_results[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = yolo_model.names[class_id]
                        print(f"  - {class_name}: {confidence:.2f}")
            
            # Show FPS on screen
            if not args.no_fps:
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw center line
            h, w = annotated_frame.shape[:2]
            cv2.line(annotated_frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
            cv2.putText(annotated_frame, "CENTER", (w // 2 + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Bin Diesel Visual Detection - Press q to quit', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
    
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

