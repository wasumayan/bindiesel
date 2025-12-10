#!/usr/bin/env python3
"""
YOLO Object Detection + Pose Tracking Test
Uses YOLO11 for object detection and pose estimation with tracking
Optimized for speed and efficiency on Raspberry Pi
"""

import cv2
import numpy as np
import time
import argparse
import sys
from picamera2 import Picamera2
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

import config


class YOLOPoseTracker:
    """
    Efficient YOLO-based object detection and pose tracking
    Uses YOLO11 pose model for person detection + pose estimation + tracking
    """
    
    def __init__(self, 
                 model_path='yolo11n-pose.pt',  # Pose model (nano for speed)
                 width=640, 
                 height=480, 
                 confidence=0.25,
                 tracker='bytetrack.yaml',  # Fast tracker
                 device='cpu'):  # Use CPU on Pi
        """
        Initialize YOLO pose tracker
        
        Args:
            model_path: Path to YOLO pose model (yolo11n-pose.pt, yolo11s-pose.pt, etc.)
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold
            tracker: Tracker config file
            device: Device to run on ('cpu' or 'cuda')
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.frame_center_x = width // 2
        
        # Initialize YOLO pose model
        print(f"[YOLOPoseTracker] Loading YOLO pose model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print(f"[YOLOPoseTracker] Model loaded: {model_path}")
        except Exception as e:
            print(f"[YOLOPoseTracker] WARNING: Failed to load {model_path}, trying default...")
            # Try to download if not found
            self.model = YOLO('yolo11n-pose.pt')  # Will auto-download
            print("[YOLOPoseTracker] Default model loaded")
        
        # Initialize camera
        print("[YOLOPoseTracker] Initializing camera...")
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(0.5)  # Let camera stabilize
            print(f"[YOLOPoseTracker] Camera started: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
        
        # Tracking state
        self.tracked_persons = {}  # track_id -> person data
        self.last_frame_time = time.time()
        self.fps = 0.0
    
    def get_frame(self):
        """
        Get current camera frame with rotation and color correction
        
        Returns:
            Frame in RGB format
        """
        array = self.picam2.capture_array()  # Returns RGB
        
        # Apply camera rotation if configured
        if config.CAMERA_ROTATION == 180:
            array = cv2.rotate(array, cv2.ROTATE_180)
        elif config.CAMERA_ROTATION == 90:
            array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        elif config.CAMERA_ROTATION == 270:
            array = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply flips if configured
        if config.CAMERA_FLIP_HORIZONTAL:
            array = cv2.flip(array, 1)  # Horizontal flip
        if config.CAMERA_FLIP_VERTICAL:
            array = cv2.flip(array, 0)  # Vertical flip
        
        # Fix color channel swap (red/blue)
        if config.CAMERA_SWAP_RB:
            # Swap red and blue channels: RGB -> BGR -> RGB (swaps R and B)
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        
        return array
    
    def calculate_arm_angle(self, keypoints, arm_side='left'):
        """
        Calculate arm angle from YOLO pose keypoints (60-90 degrees from vertical)
        
        Args:
            keypoints: YOLO keypoints array (shape: [num_keypoints, 3] where 3 = [x, y, confidence])
            arm_side: 'left' or 'right'
            
        Returns:
            Angle in degrees, or None if not detected
        """
        # YOLO pose keypoint indices (17 keypoints total)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder
        # 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        # 15: left_ankle, 16: right_ankle
        
        if arm_side == 'left':
            shoulder_idx = 5
            elbow_idx = 7
            wrist_idx = 9
        else:  # right
            shoulder_idx = 6
            elbow_idx = 8
            wrist_idx = 10
        
        # Check if keypoints are visible (confidence > 0.5)
        if (keypoints[shoulder_idx][2] < 0.5 or 
            keypoints[elbow_idx][2] < 0.5 or 
            keypoints[wrist_idx][2] < 0.5):
            return None
        
        # Get keypoint positions
        shoulder = np.array([keypoints[shoulder_idx][0], keypoints[shoulder_idx][1]])
        elbow = np.array([keypoints[elbow_idx][0], keypoints[elbow_idx][1]])
        wrist = np.array([keypoints[wrist_idx][0], keypoints[wrist_idx][1]])
        
        # Calculate vectors
        upper_arm = elbow - shoulder
        lower_arm = wrist - elbow
        
        # Calculate angle of upper arm relative to horizontal
        upper_arm_angle = np.arctan2(upper_arm[1], abs(upper_arm[0])) * 180 / np.pi
        
        # Calculate elbow angle
        if np.linalg.norm(upper_arm) > 0 and np.linalg.norm(lower_arm) > 0:
            upper_arm_norm = upper_arm / np.linalg.norm(upper_arm)
            lower_arm_norm = lower_arm / np.linalg.norm(lower_arm)
            dot_product = np.clip(np.dot(upper_arm_norm, lower_arm_norm), -1.0, 1.0)
            elbow_angle = np.arccos(dot_product) * 180 / np.pi
        else:
            return None
        
        # Check if arm is raised to side (60-90 degrees from vertical)
        horizontal_extension = abs(upper_arm[0]) > abs(upper_arm[1]) * 0.5
        
        if (upper_arm_angle < 30.0 and 
            wrist[1] < shoulder[1] and  # Wrist above shoulder
            elbow_angle > 60.0 and elbow_angle < 150.0 and
            horizontal_extension):
            angle_from_vertical = 90.0 - upper_arm_angle
            return angle_from_vertical
        
        return None
    
    def detect(self, frame):
        """
        Run YOLO pose detection with tracking
        
        Args:
            frame: RGB frame
            
        Returns:
            tuple: (detection_results_dict, yolo_result_object)
            - detection_results_dict: dict with detection results
            - yolo_result_object: YOLO result object (for default overlay)
        """
        results = {
            'objects': [],
            'poses': [],
            'tracked_persons': {},
            'fps': self.fps
        }
        
        # Run YOLO inference with tracking
        # Use track mode for object tracking (per YOLO docs: https://docs.ultralytics.com/modes/track/)
        yolo_results = self.model.track(
            frame,
            conf=self.confidence,
            verbose=False,
            persist=True,  # Maintain tracking across frames
            tracker='bytetrack.yaml',  # Fast tracker (or 'botsort.yaml' for better accuracy)
            show=False  # Don't show results automatically
        )
        
        yolo_result = None
        
        # Process results
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            yolo_result = result  # Store for overlay
            
            # Get boxes (detections)
            if result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get tracking ID if available
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Get keypoints if available (pose detection)
                    keypoints = None
                    if result.keypoints is not None and len(result.keypoints) > i:
                        kpts = result.keypoints.data[i].cpu().numpy()  # [17, 3]
                        keypoints = kpts
                    
                    # Store detection
                    detection = {
                        'track_id': track_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'keypoints': keypoints
                    }
                    
                    results['objects'].append(detection)
                    
                    # Process person pose if available
                    if class_name == 'person' and keypoints is not None:
                        # Calculate arm angles (60-90 degrees raised to side)
                        left_arm_angle = self.calculate_arm_angle(keypoints, 'left')
                        right_arm_angle = self.calculate_arm_angle(keypoints, 'right')
                        
                        # Note: Hand gestures are handled separately in hand_gesture_controller.py
                        # This pose tracker only detects arm angles for autonomous following
                        
                        # Calculate person angle
                        person_center_x = (x1 + x2) / 2
                        offset = person_center_x - self.frame_center_x
                        angle = (offset / self.width) * 102.0
                        
                        pose_data = {
                            'track_id': track_id,
                            'box': (x1, y1, x2, y2),
                            'keypoints': keypoints,
                            'left_arm_raised': left_arm_angle is not None,
                            'left_arm_angle': left_arm_angle,
                            'right_arm_raised': right_arm_angle is not None,
                            'right_arm_angle': right_arm_angle,
                            'angle': angle,
                            'is_centered': abs(offset) < 30
                        }
                        
                        results['poses'].append(pose_data)
                        
                        # Store tracked person
                        if track_id is not None:
                            results['tracked_persons'][track_id] = pose_data
        
        # Calculate FPS
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            dt = current_time - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)  # Exponential moving average
        self.last_frame_time = current_time
        results['fps'] = self.fps
        
        return results, yolo_result
    
    def update(self):
        """
        Compatibility wrapper for VisualDetector interface
        Returns same format as VisualDetector.update() for easy integration
        
        Returns:
            dict with detection results:
            {
                'person_detected': bool,
                'person_box': (x1, y1, x2, y2) or None,
                'angle': float or None,
                'is_centered': bool,
                'arm_raised': bool,
                'arm_confidence': float,
                'track_id': int or None,  # New: tracking ID
                'gesture': str or None  # New: detected gesture
            }
        """
        frame = self.get_frame()
        results, _ = self.detect(frame)  # Ignore yolo_result for update() method
        
        # Find the best person (first person with pose data, or first person)
        best_person = None
        for pose in results['poses']:
            best_person = pose
            break
        
        if best_person is None:
            # No person detected
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'arm_raised': False,
                'arm_confidence': 0.0,
                'track_id': None
            }
        
        # Extract data from pose
        arm_raised = best_person.get('left_arm_raised', False) or best_person.get('right_arm_raised', False)
        arm_confidence = 1.0 if arm_raised else 0.0  # Simplified confidence
        
        return {
            'person_detected': True,
            'person_box': best_person['box'],
            'angle': best_person.get('angle'),
            'is_centered': best_person.get('is_centered', False),
            'arm_raised': arm_raised,
            'arm_confidence': arm_confidence,
            'track_id': best_person.get('track_id')
        }
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        print("[YOLOPoseTracker] Stopped")


def draw_detections(frame, yolo_result, results):
    """
    Draw detection results using YOLO's default overlay + custom arm angle info
    
    Args:
        frame: RGB frame
        yolo_result: YOLO result object (for default overlay)
        results: Detection results dict (for arm angle info)
    
    Returns:
        Annotated frame in BGR format
    """
    # Use YOLO's built-in plot() method for default overlays
    # This gives us the standard YOLO visualization with keypoints, skeleton, etc.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_frame = yolo_result.plot()  # YOLO's default overlay
    
    # Add custom arm angle information on top
    y_offset = 30
    font_scale = 0.5
    thickness = 2
    
    for pose in results['poses']:
        if pose['left_arm_raised']:
            text = f"L Arm: {pose['left_arm_angle']:.0f}°"
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            y_offset += 25
        
        if pose['right_arm_raised']:
            text = f"R Arm: {pose['right_arm_angle']:.0f}°"
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            y_offset += 25
    
    # Draw FPS
    fps_text = f'FPS: {results["fps"]:.1f}'
    cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw tracking info
    if results['tracked_persons']:
        track_text = f"Tracking: {len(results['tracked_persons'])} person(s)"
        cv2.putText(annotated_frame, track_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return annotated_frame


def main():
    parser = argparse.ArgumentParser(description='Test YOLO pose detection + tracking')
    parser.add_argument('--model', type=str, default='yolo11n-pose.pt', 
                       help='YOLO pose model (yolo11n-pose.pt, yolo11s-pose.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--fps', action='store_true', help='Show FPS counter')
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLO Pose Detection + Tracking Test")
    print("=" * 70)
    print("Features:")
    print("  - Object detection (all classes)")
    print("  - Person pose estimation (17 keypoints)")
    print("  - Multi-person tracking (BYTETracker)")
    print("  - Arm angle detection (60-90 degrees)")
    print("  - Hand gesture recognition")
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("=" * 70)
    print()
    
    try:
        # Initialize tracker
        tracker = YOLOPoseTracker(
            model_path=args.model,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=args.conf
        )
        
        print("[TEST] YOLO pose tracker initialized")
        print("[TEST] Starting camera feed...")
        print()
        
        # Start OpenCV window
        cv2.startWindowThread()
        
        while True:
            # Get frame
            frame = tracker.get_frame()
            
            # Run detection (returns both results dict and yolo_result object)
            results, yolo_result = tracker.detect(frame)
            
            # Draw detections using YOLO's default overlay + custom arm angle info
            if yolo_result is not None:
                frame_bgr = draw_detections(frame, yolo_result, results)
            else:
                # Fallback if no detections
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Display frame
            cv2.imshow('YOLO Pose Tracking - Press q to quit', frame_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Print to terminal periodically
            if results['poses']:
                for pose in results['poses']:
                    output = f"[TEST] Person ID:{pose.get('track_id', 'N/A')} "
                    if pose['left_arm_raised']:
                        output += f"L:{pose['left_arm_angle']:.0f}° "
                    if pose['right_arm_raised']:
                        output += f"R:{pose['right_arm_angle']:.0f}° "
                    print(output)
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tracker' in locals():
            tracker.stop()
        cv2.destroyAllWindows()
        print("[TEST] Test complete")


if __name__ == '__main__':
    main()

