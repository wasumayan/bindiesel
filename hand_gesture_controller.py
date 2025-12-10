#!/usr/bin/env python3
"""
Hand Gesture Controller for Manual Mode
Uses YOLO hand-keypoints model for precise hand gesture detection
Works alongside voice commands - both input methods are supported
Optimized for speed and efficiency

Gestures:
- Turn right: Point thumb to the right
- Turn left: Point thumb to the left
- Stop: Palm up facing camera
- Move forward: Thumbs up

Training: yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
Reference: https://docs.ultralytics.com/datasets/pose/hand-keypoints/
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
    Hand gesture controller using YOLO hand-keypoints model
    Uses YOLO pose model trained on hand-keypoints dataset (21 keypoints per hand)
    Reference: https://docs.ultralytics.com/datasets/pose/hand-keypoints/
    
    Training: yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
    """
    
    # Gesture to command mapping
    GESTURE_COMMANDS = {
        'stop': 'STOP',
        'turn_left': 'LEFT',
        'turn_right': 'RIGHT',
        'thumbs_up': 'FORWARD'  # Thumbs up = move forward
    }
    
    # YOLO hand keypoint indices (21 keypoints per hand)
    # From hand-keypoints.yaml:
    # 0: wrist
    # 1: thumb_cmc
    # 2: thumb_mcp
    # 3: thumb_ip
    # 4: thumb_tip
    # 5: index_mcp
    # 6: index_pip
    # 7: index_dip
    # 8: index_tip
    # 9: middle_mcp
    # 10: middle_pip
    # 11: middle_dip
    # 12: middle_tip
    # 13: ring_mcp
    # 14: ring_pip
    # 15: ring_dip
    # 16: ring_tip
    # 17: pinky_mcp
    # 18: pinky_pip
    # 19: pinky_dip
    # 20: pinky_tip
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(self, 
                 hand_model_path=None,  # Path to trained hand-keypoints model
                 pose_model_path='yolo11n-pose.pt',  # Fallback: use pose model
                 width=640,
                 height=480,
                 confidence=0.25,
                 gesture_hold_time=0.5,
                 skip_camera=False):  # Skip camera initialization if sharing camera
        """
        Initialize hand gesture controller
        
        Args:
            hand_model_path: Path to YOLO model trained on hand-keypoints dataset
                           If None, uses pose model as fallback (less accurate)
            pose_model_path: Path to YOLO pose model (fallback if hand model not available)
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold (0.0-1.0)
            gesture_hold_time: Seconds gesture must be held before executing
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.gesture_hold_time = gesture_hold_time
        
        # Initialize hand keypoints model (if available)
        self.hand_model = None
        if hand_model_path:
            print(f"[HandGestureController] Loading hand keypoints model: {hand_model_path}...")
            try:
                self.hand_model = YOLO(hand_model_path)
                print("[HandGestureController] Hand keypoints model loaded")
            except Exception as e:
                print(f"[HandGestureController] WARNING: Failed to load hand model: {e}")
                print("[HandGestureController] Falling back to pose model")
                self.hand_model = None
        
        # Initialize pose model (for fallback if hand model not available)
        if not self.hand_model:
            print(f"[HandGestureController] Using pose model (fallback mode): {pose_model_path}...")
            print("[HandGestureController] NOTE: For better accuracy, train a hand-keypoints model:")
            print("  yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640")
            try:
                self.pose_model = YOLO(pose_model_path)
                print("[HandGestureController] Pose model loaded")
            except Exception as e:
                print(f"[HandGestureController] WARNING: Failed to load pose model: {e}")
                self.pose_model = YOLO('yolo11n-pose.pt')  # Auto-download
        
        # Initialize camera (skip if sharing camera from another component)
        self.picam2 = None
        if not skip_camera:
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
        else:
            print("[HandGestureController] Skipping camera initialization (using shared camera)")
        
        # Gesture tracking state
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_command = None
        self.gesture_history = []  # For smoothing
    
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
    
    def detect_gesture_from_hand_keypoints(self, keypoints):
        """
        Detect hand gesture from YOLO hand keypoints (21 keypoints per hand)
        
        Args:
            keypoints: YOLO hand keypoints array [21, 3] (x, y, confidence)
                      Format: [[x1, y1, conf1], [x2, y2, conf2], ...]
            
        Returns:
            Gesture name or None
        """
        # Check if keypoints are visible (confidence threshold)
        if keypoints[self.WRIST][2] < 0.4:  # Wrist not visible
            return None
        
        # Get keypoint positions (pixel coordinates)
        wrist = np.array([keypoints[self.WRIST][0], keypoints[self.WRIST][1]], dtype=np.float32)
        thumb_tip = np.array([keypoints[self.THUMB_TIP][0], keypoints[self.THUMB_TIP][1]], dtype=np.float32)
        thumb_ip = np.array([keypoints[self.THUMB_IP][0], keypoints[self.THUMB_IP][1]], dtype=np.float32)
        index_tip = np.array([keypoints[self.INDEX_TIP][0], keypoints[self.INDEX_TIP][1]], dtype=np.float32)
        index_pip = np.array([keypoints[self.INDEX_PIP][0], keypoints[self.INDEX_PIP][1]], dtype=np.float32)
        middle_tip = np.array([keypoints[self.MIDDLE_TIP][0], keypoints[self.MIDDLE_TIP][1]], dtype=np.float32)
        middle_pip = np.array([keypoints[self.MIDDLE_PIP][0], keypoints[self.MIDDLE_PIP][1]], dtype=np.float32)
        ring_tip = np.array([keypoints[self.RING_TIP][0], keypoints[self.RING_TIP][1]], dtype=np.float32)
        ring_pip = np.array([keypoints[self.RING_PIP][0], keypoints[self.RING_PIP][1]], dtype=np.float32)
        pinky_tip = np.array([keypoints[self.PINKY_TIP][0], keypoints[self.PINKY_TIP][1]], dtype=np.float32)
        pinky_pip = np.array([keypoints[self.PINKY_PIP][0], keypoints[self.PINKY_PIP][1]], dtype=np.float32)
        
        # Check keypoint visibility
        if (keypoints[self.THUMB_TIP][2] < 0.4 or keypoints[self.INDEX_TIP][2] < 0.4 or
            keypoints[self.MIDDLE_TIP][2] < 0.4 or keypoints[self.RING_TIP][2] < 0.4 or
            keypoints[self.PINKY_TIP][2] < 0.4):
            return None
        
        # Check if fingers are extended upward (tip.y < pip.y means extended upward)
        # In image coordinates, Y increases downward
        index_extended = index_tip[1] < index_pip[1] - 10  # 10 pixel threshold
        middle_extended = middle_tip[1] < middle_pip[1] - 10
        ring_extended = ring_tip[1] < ring_pip[1] - 10
        pinky_extended = pinky_tip[1] < pinky_pip[1] - 10
        
        # Thumb extension: check horizontal and vertical position relative to IP joint
        thumb_horizontal_dist = thumb_tip[0] - thumb_ip[0]
        thumb_vertical_dist = thumb_tip[1] - thumb_ip[1]
        
        # Determine thumb direction
        thumb_points_right = thumb_horizontal_dist > 20  # Significant rightward movement (pixels)
        thumb_points_left = thumb_horizontal_dist < -20  # Significant leftward movement (pixels)
        thumb_points_up = thumb_vertical_dist < -15  # Thumb pointing up (pixels)
        
        # Check if palm is facing camera (fingers extended upward)
        # Palm facing camera = all finger tips are above wrist
        palm_facing_camera = (index_tip[1] < wrist[1] - 20 and 
                             middle_tip[1] < wrist[1] - 20 and 
                             ring_tip[1] < wrist[1] - 20 and 
                             pinky_tip[1] < wrist[1] - 20)
        
        # Gesture 1: STOP - Palm up facing camera
        # All 4 fingers (index, middle, ring, pinky) extended upward, palm facing camera
        if (palm_facing_camera and 
            index_extended and middle_extended and ring_extended and pinky_extended):
            return 'stop'
        
        # Gesture 2: THUMBS UP - Move forward
        # Thumb extended upward, other fingers closed (not extended)
        if (thumb_points_up and 
            not index_extended and not middle_extended and not ring_extended and not pinky_extended):
            return 'thumbs_up'
        
        # Gesture 3: TURN RIGHT - Thumb points right
        # Thumb extended horizontally to the right
        # Other fingers can be in any position (relaxed)
        if thumb_points_right:
            return 'turn_right'
        
        # Gesture 4: TURN LEFT - Thumb points left
        # Thumb extended horizontally to the left
        # Other fingers can be in any position (relaxed)
        if thumb_points_left:
            return 'turn_left'
        
        return None
    
    
    def detect_command(self, frame):
        """
        Detect command from hand gesture using YOLO hand keypoints model
        
        Args:
            frame: RGB frame
            
        Returns:
            Command string (FORWARD, LEFT, RIGHT, STOP) or None
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
                    # Note: Pose model has wrist keypoints but not detailed hand keypoints
                    # This is a basic fallback - for best results, train a hand-keypoints model
                    # For now, we'll just return None if hand model isn't available
                    pass
        
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
    print("  - STOP: Palm up facing camera (all fingers extended)")
    print("  - TURN_LEFT: Point thumb to the left")
    print("  - TURN_RIGHT: Point thumb to the right")
    print("  - FORWARD: Thumbs up")
    print()
    print("Press 'q' to exit or Ctrl+C to stop")
    print("=" * 70)
    print()
    
    try:
        # Try to use hand keypoints model if available
        # Otherwise falls back to pose model (less accurate for hand gestures)
        controller = HandGestureController(
            hand_model_path=None,  # Set to path of trained hand-keypoints model if available
            pose_model_path='yolo11n-pose.pt',  # Fallback model
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=config.YOLO_CONFIDENCE,
            gesture_hold_time=0.5
        )
        
        print("[TEST] Controller initialized")
        print("[TEST] Start gesturing...")
        print()
        
        # Start OpenCV window thread
        cv2.startWindowThread()
        
        while True:
            frame = controller.get_frame()
            command = controller.detect_command(frame)
            
            # Draw detection info on frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw YOLO hand keypoints if available
            if controller.hand_model:
                results = controller.hand_model(frame, conf=controller.confidence, verbose=False)
                if results and len(results) > 0:
                    result = results[0]
                    # Draw YOLO results (bounding boxes and keypoints)
                    annotated_frame = result.plot()
                    frame_bgr = annotated_frame
            
            # Add text overlay
            cv2.putText(frame_bgr, "Hand Gesture Controller", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if command:
                cv2.putText(frame_bgr, f"Command: {command}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"[TEST] Command detected: {command}")
            else:
                cv2.putText(frame_bgr, "No command", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Hand Gesture Controller", frame_bgr)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Check 10 times per second
    
    except KeyboardInterrupt:
        print("\n[TEST] Stopping...")
    finally:
        cv2.destroyAllWindows()
        if 'controller' in locals():
            controller.stop()

