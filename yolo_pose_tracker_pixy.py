#!/usr/bin/env python3
"""
YOLO Pose Tracker with Pixy Camera Support
Modified version of YOLOPoseTracker that uses Pixy camera instead of Picamera2
"""

import cv2
import numpy as np
import time
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

from pixy_camera import PixyCamera
import config


class YOLOPoseTrackerPixy:
    """
    YOLO-based object detection and pose tracking using Pixy camera
    Uses YOLO11 pose model for person detection + pose estimation + tracking
    """
    
    def __init__(self, 
                 model_path=config.YOLO_POSE_MODEL,
                 width=640, 
                 height=480, 
                 confidence=0.25,
                 tracker='bytetrack.yaml',
                 device='cpu'):
        """
        Initialize YOLO pose tracker with Pixy camera
        
        Args:
            model_path: Path to YOLO pose model
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold
            tracker: Tracker config file
            device: Device to run on ('cpu' or 'cuda')
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.tracker = tracker
        self.frame_center_x = width // 2
        self.debug_mode = config.DEBUG_MODE
        self._frame_counter = 0
        
        # Initialize YOLO pose model
        print(f"[YOLOPoseTrackerPixy] Loading YOLO pose model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print(f"[YOLOPoseTrackerPixy] Model loaded: {model_path}")
        except Exception as e:
            print(f"[YOLOPoseTrackerPixy] WARNING: Failed to load {model_path}: {e}")
            if config.USE_NCNN and model_path.endswith('_ncnn_model'):
                fallback_path = model_path.replace('_ncnn_model', '.pt')
                print(f"[YOLOPoseTrackerPixy] Trying PyTorch fallback: {fallback_path}...")
                try:
                    self.model = YOLO(fallback_path)
                    print(f"[YOLOPoseTrackerPixy] PyTorch model loaded: {fallback_path}")
                except Exception as e2:
                    print(f"[YOLOPoseTrackerPixy] Fallback failed, trying default...")
                    self.model = YOLO('yolo11n-pose.pt')
                    print("[YOLOPoseTrackerPixy] Default model loaded")
            else:
                self.model = YOLO('yolo11n-pose.pt')
                print("[YOLOPoseTrackerPixy] Default model loaded")
        
        # Initialize Pixy camera
        print("[YOLOPoseTrackerPixy] Initializing Pixy camera...")
        try:
            self.pixy_camera = PixyCamera(width=width, height=height)
            print(f"[YOLOPoseTrackerPixy] Pixy camera initialized: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pixy camera: {e}")
        
        # Tracking state
        self.tracked_persons = {}
        self.last_frame_time = time.time()
        self.fps = 0.0
    
    def get_frame(self):
        """
        Get current camera frame with rotation and color correction
        
        Returns:
            Frame in RGB format
        """
        # Get frame from Pixy camera (already handles rotation/flip/color swap)
        frame_rgb = self.pixy_camera.get_frame()
        return frame_rgb
    
    def calculate_arm_angle(self, keypoints, arm_side='left', debug=False):
        """Calculate arm angle from YOLO pose keypoints - same as original YOLOPoseTracker"""
        # Copy the exact same logic from test_yolo_pose_tracking.py
        if arm_side == 'left':
            shoulder_idx = 5
            elbow_idx = 7
            wrist_idx = 9
        else:
            shoulder_idx = 6
            elbow_idx = 8
            wrist_idx = 10
        
        shoulder_conf = keypoints[shoulder_idx][2]
        elbow_conf = keypoints[elbow_idx][2]
        wrist_conf = keypoints[wrist_idx][2]
        
        min_keypoint_confidence = 0.01
        
        if (shoulder_conf < min_keypoint_confidence or 
            elbow_conf < min_keypoint_confidence or 
            wrist_conf < min_keypoint_confidence):
            if debug:
                print(f"  [{arm_side.upper()} ARM] Low confidence: shoulder={shoulder_conf:.2f}, elbow={elbow_conf:.2f}, wrist={wrist_conf:.2f}")
            return None
        
        shoulder = np.array([keypoints[shoulder_idx][0], keypoints[shoulder_idx][1]], dtype=np.float32)
        elbow = np.array([keypoints[elbow_idx][0], keypoints[elbow_idx][1]], dtype=np.float32)
        wrist = np.array([keypoints[wrist_idx][0], keypoints[wrist_idx][1]], dtype=np.float32)
        
        upper_arm = elbow - shoulder
        lower_arm = wrist - elbow
        
        upper_arm_length = np.linalg.norm(upper_arm)
        lower_arm_length = np.linalg.norm(lower_arm)
        
        min_upper_arm = 15
        min_lower_arm = 10
        
        if upper_arm_length < min_upper_arm or lower_arm_length < min_lower_arm:
            if debug:
                print(f"  [{arm_side.upper()} ARM] Arm too short: upper={upper_arm_length:.1f}px, lower={lower_arm_length:.1f}px")
            return None
        
        if arm_side == 'left':
            angle_from_vertical = np.arctan2(abs(upper_arm[0]), abs(upper_arm[1])) * 180 / np.pi
        else:
            angle_from_vertical = np.arctan2(abs(upper_arm[0]), abs(upper_arm[1])) * 180 / np.pi
        
        total_arm = wrist - shoulder
        total_arm_length = np.linalg.norm(total_arm)
        
        if total_arm_length > 0:
            total_arm_angle = np.arctan2(abs(total_arm[0]), abs(total_arm[1])) * 180 / np.pi
        else:
            total_arm_angle = angle_from_vertical
        
        angle_ok = config.ARM_ANGLE_MIN <= total_arm_angle <= config.ARM_ANGLE_MAX
        horizontal_ratio = abs(upper_arm[0]) / max(abs(upper_arm[1]), 1.0)
        horizontal_extended = horizontal_ratio > config.ARM_HORIZONTAL_RATIO
        arm_extended = abs(upper_arm[0]) > config.ARM_MIN_HORIZONTAL_EXTENSION
        
        elbow_reasonable = True
        if upper_arm_length > 0 and lower_arm_length > 0:
            upper_arm_norm = upper_arm / upper_arm_length
            lower_arm_norm = lower_arm / lower_arm_length
            dot_product = np.clip(np.dot(upper_arm_norm, lower_arm_norm), -1.0, 1.0)
            elbow_angle = np.arccos(dot_product) * 180 / np.pi
            
            if elbow_angle < 5.0 and total_arm_angle > 45.0:
                elbow_reasonable = False
            if elbow_angle > 170.0 and total_arm_angle > 45.0:
                elbow_reasonable = False
        
        strict_detection = (angle_ok and horizontal_extended and arm_extended and elbow_reasonable)
        lenient_detection = (angle_ok and arm_extended and elbow_reasonable)
        
        if strict_detection or lenient_detection:
            if debug:
                print(f"  [{arm_side.upper()} ARM] ✓ DETECTED! Total arm angle: {total_arm_angle:.1f}°")
            return total_arm_angle
        
        return None
    
    def detect(self, frame):
        """Run YOLO pose detection with tracking - same as original"""
        results = {
            'objects': [],
            'poses': [],
            'tracked_persons': {},
            'fps': self.fps
        }
        
        yolo_results = self.model.track(
            frame,
            conf=0.01,
            verbose=False,
            persist=True,
            tracker=self.tracker,
            show=False,
            half=False,
            imgsz=config.YOLO_INFERENCE_SIZE,
            max_det=config.YOLO_MAX_DET,
            agnostic_nms=config.YOLO_AGNOSTIC_NMS
        )
        
        yolo_result = None
        
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            yolo_result = result
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    keypoints = None
                    if result.keypoints is not None and len(result.keypoints) > i:
                        kpts = result.keypoints.data[i].cpu().numpy()
                        keypoints = kpts
                    
                    detection = {
                        'track_id': track_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'keypoints': keypoints
                    }
                    
                    results['objects'].append(detection)
                    
                    if class_name == 'person' and keypoints is not None and confidence >= config.YOLO_PERSON_CONFIDENCE:
                        valid_keypoints = sum(1 for kpt in keypoints if kpt[2] > 0.1)
                        if valid_keypoints < 5:
                            continue
                        
                        debug_arm = self.debug_mode or (track_id is not None and track_id == 1 and self._frame_counter % 30 == 0)
                        
                        if config.CAMERA_SWAP_LEFT_RIGHT:
                            left_arm_angle = self.calculate_arm_angle(keypoints, 'right', debug=debug_arm)
                            right_arm_angle = self.calculate_arm_angle(keypoints, 'left', debug=debug_arm)
                        else:
                            left_arm_angle = self.calculate_arm_angle(keypoints, 'left', debug=debug_arm)
                            right_arm_angle = self.calculate_arm_angle(keypoints, 'right', debug=debug_arm)
                        
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
                        
                        if track_id is not None:
                            results['tracked_persons'][track_id] = pose_data
        
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            dt = current_time - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = current_time
        results['fps'] = self.fps
        
        return results, yolo_result
    
    def update(self, target_track_id=None):
        """Compatibility wrapper for VisualDetector interface"""
        frame = self.get_frame()
        results, _ = self.detect(frame)
        
        if not results['poses']:
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'arm_raised': False,
                'arm_confidence': 0.0,
                'track_id': None
            }
        
        if target_track_id is not None:
            for pose in results['poses']:
                if pose.get('track_id') == target_track_id:
                    arm_raised = pose.get('left_arm_raised', False) or pose.get('right_arm_raised', False)
                    arm_confidence = 1.0 if arm_raised else 0.0
                    return {
                        'person_detected': True,
                        'person_box': pose['box'],
                        'angle': pose.get('angle'),
                        'is_centered': pose.get('is_centered', False),
                        'arm_raised': arm_raised,
                        'arm_confidence': arm_confidence,
                        'track_id': pose.get('track_id')
                    }
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'arm_raised': False,
                'arm_confidence': 0.0,
                'track_id': None
            }
        
        for pose in results['poses']:
            arm_raised = pose.get('left_arm_raised', False) or pose.get('right_arm_raised', False)
            if arm_raised:
                arm_confidence = 1.0
                return {
                    'person_detected': True,
                    'person_box': pose['box'],
                    'angle': pose.get('angle'),
                    'is_centered': pose.get('is_centered', False),
                    'arm_raised': True,
                    'arm_confidence': arm_confidence,
                    'track_id': pose.get('track_id')
                }
        
        best_person = results['poses'][0]
        arm_raised = best_person.get('left_arm_raised', False) or best_person.get('right_arm_raised', False)
        arm_confidence = 1.0 if arm_raised else 0.0
        
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
        if hasattr(self, 'pixy_camera') and self.pixy_camera is not None:
            try:
                self.pixy_camera.stop()
            except:
                pass
        print("[YOLOPoseTrackerPixy] Stopped")

