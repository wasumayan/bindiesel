#!/usr/bin/env python3
"""
Hand Gesture Controller for Manual Mode
Uses YOLO hand keypoints model for precise hand gesture detection
Works alongside voice commands - both input methods are supported
Optimized for speed and efficiency

Note: Requires a YOLO model trained on hand-keypoints dataset.
Train with: yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100
Or use a pre-trained hand keypoints model if available.
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
    print("Install with: pip install ultralytics")
    import sys
    sys.exit(1)

import config


class HandGestureController:
    """
    Hand gesture controller using YOLO hand keypoints model
    Uses hand-keypoints dataset model (21 keypoints per hand)
    Reference: https://docs.ultralytics.com/datasets/pose/hand-keypoints/
    """
    
    # Gesture to command mapping
    GESTURE_COMMANDS = {
        'stop': 'STOP',
        'turn_left': 'LEFT',
        'turn_right': 'RIGHT',
        'turn_around': 'TURN_AROUND',
        'come': 'FORWARD',  # Beckoning = come forward
        'go_away': 'TURN_AROUND'  # Waving away = turn around
    }
    
    # Hand keypoint indices (21 keypoints per hand)
    # 0: wrist
    # 1-4: thumb (cmc, mcp, ip, tip)
    # 5-8: index (mcp, pip, dip, tip)
    # 9-12: middle (mcp, pip, dip, tip)
    # 13-16: ring (mcp, pip, dip, tip)
    # 17-20: pinky (mcp, pip, dip, tip)
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    THUMB_IP = 3
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18
    
    def __init__(self, 
                 hand_model_path=None,  # Path to hand keypoints model (trained on hand-keypoints.yaml)
                 pose_model_path='yolo11n-pose.pt',  # Fallback: use pose model to find person first
                 width=640,
                 height=480,
                 confidence=0.25,
                 gesture_hold_time=0.5):
        """
        Initialize hand gesture controller
        
        Args:
            hand_model_path: Path to YOLO model trained on hand-keypoints dataset
                           If None, uses two-stage: pose model to find person, then hand detection
            pose_model_path: Path to YOLO pose model (for person detection if hand_model not available)
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold
            gesture_hold_time: Seconds gesture must be held before executing
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.gesture_hold_time = gesture_hold_time
        self.frame_center_x = width // 2
        
        # Initialize hand keypoints model (if available)
        self.hand_model = None
        if hand_model_path:
            print(f"[HandGestureController] Loading hand keypoints model: {hand_model_path}...")
            try:
                self.hand_model = YOLO(hand_model_path)
                print("[HandGestureController] Hand keypoints model loaded")
            except Exception as e:
                print(f"[HandGestureController] WARNING: Failed to load hand model: {e}")
                self.hand_model = None
        
        # Initialize pose model (for person detection if hand model not available)
        if not self.hand_model:
            print(f"[HandGestureController] Using pose model for person detection: {pose_model_path}...")
            try:
                self.pose_model = YOLO(pose_model_path)
                print("[HandGestureController] Pose model loaded (fallback mode)")
            except Exception as e:
                print(f"[HandGestureController] WARNING: Failed to load pose model: {e}")
                self.pose_model = YOLO('yolo11n-pose.pt')  # Auto-download
        
        # Initialize camera
        print("[HandGestureController] Initializing camera...")
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(0.5)
            print(f"[HandGestureController] Camera started: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
        
        # Gesture tracking state
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_command = None
        self.gesture_history = []  # For smoothing
    
    def get_frame(self):
        """Get current camera frame in RGB"""
        return self.picam2.capture_array()
    
    def detect_gesture_from_hand_keypoints(self, keypoints):
        """
        Detect hand gesture from hand keypoints (21 keypoints per hand)
        
        Args:
            keypoints: YOLO hand keypoints array [21, 3] (x, y, confidence)
            
        Returns:
            Gesture name or None
        """
        # Check if keypoints are visible
        if keypoints[self.WRIST][2] < 0.5:  # Wrist not visible
            return None
        
        # Get keypoint positions
        wrist = np.array([keypoints[self.WRIST][0], keypoints[self.WRIST][1]])
        thumb_tip = np.array([keypoints[self.THUMB_TIP][0], keypoints[self.THUMB_TIP][1]])
        index_tip = np.array([keypoints[self.INDEX_TIP][0], keypoints[self.INDEX_TIP][1]])
        middle_tip = np.array([keypoints[self.MIDDLE_TIP][0], keypoints[self.MIDDLE_TIP][1]])
        ring_tip = np.array([keypoints[self.RING_TIP][0], keypoints[self.RING_TIP][1]])
        pinky_tip = np.array([keypoints[self.PINKY_TIP][0], keypoints[self.PINKY_TIP][1]])
        
        thumb_ip = np.array([keypoints[self.THUMB_IP][0], keypoints[self.THUMB_IP][1]])
        index_pip = np.array([keypoints[self.INDEX_PIP][0], keypoints[self.INDEX_PIP][1]])
        middle_pip = np.array([keypoints[self.MIDDLE_PIP][0], keypoints[self.MIDDLE_PIP][1]])
        ring_pip = np.array([keypoints[self.RING_PIP][0], keypoints[self.RING_PIP][1]])
        pinky_pip = np.array([keypoints[self.PINKY_PIP][0], keypoints[self.PINKY_PIP][1]])
        
        # Check if fingers are extended (tip above PIP joint)
        thumb_extended = thumb_tip[1] < thumb_ip[1] - 0.02
        index_extended = index_tip[1] < index_pip[1] - 0.02
        middle_extended = middle_tip[1] < middle_pip[1] - 0.02
        ring_extended = ring_tip[1] < ring_pip[1] - 0.02
        pinky_extended = pinky_tip[1] < pinky_pip[1] - 0.02
        
        # Count extended fingers
        fingers_up = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        num_fingers = sum(fingers_up)
        
        # Calculate hand direction (using middle finger)
        hand_vector = middle_tip - wrist
        hand_angle = np.arctan2(hand_vector[1], hand_vector[0]) * 180 / np.pi
        
        # Gesture recognition using finger states and hand direction
        
        # STOP: All 5 fingers extended (open palm)
        if num_fingers == 5:
            return 'stop'
        
        # TURN_LEFT: Index finger extended, pointing left
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if -135 < hand_angle < -45:  # Pointing left
                return 'turn_left'
        
        # TURN_RIGHT: Index finger extended, pointing right
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if 45 < hand_angle < 135:  # Pointing right
                return 'turn_right'
        
        # TURN_AROUND: Index finger extended, pointing backward/up
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if hand_angle < -135 or hand_angle > 135:  # Pointing backward
                return 'turn_around'
        
        # COME (FORWARD): Thumb + index extended, pointing forward/down
        if thumb_extended and index_extended and not middle_extended:
            if -45 < hand_angle < 45:  # Pointing forward
                return 'come'
        
        # GO_AWAY: All fingers extended but close together, pointing away
        if num_fingers >= 4:
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            finger_distances = [np.linalg.norm(finger_tips[i] - finger_tips[i+1]) 
                              for i in range(len(finger_tips)-1)]
            avg_distance = np.mean(finger_distances)
            
            if avg_distance < 0.05:  # Fingers close together
                if hand_angle < -135 or hand_angle > 135:  # Pointing away
                    return 'go_away'
        
        return None
    
    def detect_gesture_from_pose(self, keypoints):
        """
        Fallback: Detect hand gesture from pose keypoints (when hand model not available)
        
        Args:
            keypoints: YOLO pose keypoints array [17, 3]
            
        Returns:
            Gesture name or None
        """
        # Get keypoints
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        gestures = []
        
        # Check left arm gestures
        if left_wrist[2] > 0.5 and left_shoulder[2] > 0.5:
            wrist_y = left_wrist[1] - left_shoulder[1]
            wrist_x = left_wrist[0] - left_shoulder[0]
            
            if wrist_y < -60:
                gestures.append('stop')
            elif wrist_x < -50 and abs(wrist_y) < 30:
                gestures.append('turn_left')
            elif abs(wrist_x) < 30 and wrist_y < -20:
                gestures.append('come')
        
        # Check right arm gestures
        if right_wrist[2] > 0.5 and right_shoulder[2] > 0.5:
            wrist_y = right_wrist[1] - right_shoulder[1]
            wrist_x = right_wrist[0] - right_shoulder[0]
            
            if wrist_y < -60:
                gestures.append('stop')
            elif wrist_x > 50 and abs(wrist_y) < 30:
                gestures.append('turn_right')
            elif abs(wrist_x) < 30 and wrist_y < -20:
                gestures.append('come')
        
        return gestures[0] if gestures else None
    
    def detect_command(self, frame):
        """
        Detect command from hand gesture
        
        Args:
            frame: RGB frame
            
        Returns:
            Command string (FORWARD, LEFT, RIGHT, STOP, TURN_AROUND) or None
        """
        gesture = None
        
        # Use hand keypoints model if available (more accurate)
        if self.hand_model:
            results = self.hand_model(frame, conf=self.confidence, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Get hand keypoints (21 keypoints per hand)
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Process first detected hand
                    hand_keypoints = result.keypoints.data[0].cpu().numpy()  # [21, 3]
                    gesture = self.detect_gesture_from_hand_keypoints(hand_keypoints)
        
        # Fallback to pose model if hand model not available or no hand detected
        if gesture is None and hasattr(self, 'pose_model'):
            results = self.pose_model(frame, conf=self.confidence, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Get first person's pose keypoints
                    pose_keypoints = result.keypoints.data[0].cpu().numpy()  # [17, 3]
                    gesture = self.detect_gesture_from_pose(pose_keypoints)
        
        if gesture is None:
            # Reset gesture tracking
            self.current_gesture = None
            self.gesture_start_time = None
            return None
        
        # Check if gesture is held long enough
        current_time = time.time()
        
        if gesture == self.current_gesture:
            # Same gesture - check hold time
            if self.gesture_start_time is None:
                self.gesture_start_time = current_time
            
            hold_duration = current_time - self.gesture_start_time
            
            if hold_duration >= self.gesture_hold_time:
                # Gesture held long enough - return command
                command = self.GESTURE_COMMANDS.get(gesture)
                
                # Only return if command changed (avoid repeating)
                if command != self.last_command:
                    self.last_command = command
                    return command
        else:
            # New gesture - reset timer
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            self.last_command = None
        
        return None
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        print("[HandGestureController] Stopped")


# Integration function for main.py
def get_gesture_command(gesture_controller):
    """
    Get command from hand gesture (non-blocking)
    
    Args:
        gesture_controller: HandGestureController instance
        
    Returns:
        Command string or None
    """
    try:
        frame = gesture_controller.get_frame()
        command = gesture_controller.detect_command(frame)
        return command
    except Exception as e:
        if config.DEBUG_VOICE:
            print(f"[HandGestureController] Error: {e}")
        return None


if __name__ == '__main__':
    # Test hand gesture controller
    print("=" * 70)
    print("Hand Gesture Controller Test")
    print("=" * 70)
    print("Gestures:")
    print("  - STOP: Raise hand(s) high above shoulder")
    print("  - TURN_LEFT: Extend left hand to left side")
    print("  - TURN_RIGHT: Extend right hand to right side")
    print("  - FORWARD (COME): Extend hand forward (beckoning)")
    print("  - TURN_AROUND (GO_AWAY): Extend hand backward/up")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    try:
        # Try to use hand keypoints model if available
        # Otherwise falls back to pose model
        controller = HandGestureController(
            hand_model_path=None,  # Set to path of trained hand-keypoints model if available
            pose_model_path='yolo11n-pose.pt',  # Fallback model
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=config.YOLO_CONFIDENCE
        )
        
        print("[TEST] Controller initialized")
        print("[TEST] Start gesturing...")
        print()
        
        while True:
            command = controller.detect_command(controller.get_frame())
            
            if command:
                print(f"[TEST] Command detected: {command}")
            
            time.sleep(0.1)  # Check 10 times per second
    
    except KeyboardInterrupt:
        print("\n[TEST] Stopping...")
    finally:
        if 'controller' in locals():
            controller.stop()

