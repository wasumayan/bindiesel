#!/usr/bin/env python3
"""
Home Marker Tracking Test
Tests red square (home marker) detection with OpenCV MOSSE tracker,
servo steering control, and motor speed control.

Usage:
    python test_home_tracking.py              # Use default config
    python test_home_tracking.py --help       # Show all options

Controls:
    'q'     - Quit
    's'     - Restart (back to scan mode from stop)
    'm'     - Toggle motor on/off during lock
    'r'     - Reset tracker and return to scan mode
"""

import sys
import time
import argparse
import cv2
import config
import numpy as np
from pathlib import Path
from home_marker_detector import detect_red_box, check_color_match_red
from logger import setup_logger, log_info, log_warning, log_error
from servo_controller import ServoController
from motor_controller import MotorController
from ultralytics import YOLO
from tof_sensor import ToFSensor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))




class CentroidTracker:
    """Simple centroid-based tracker (works on all platforms without cv2.legacy)"""
    yolo_model = None  # Set by parent class
    
    def __init__(self, max_distance=100, max_lost_frames=5):
        self.last_bbox = None
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.lost_frames = 0
    
    def init(self, frame, bbox):
        """Initialize tracker with initial bounding box"""
        self.last_bbox = bbox
        self.lost_frames = 0
        return True
    
    def update(self, frame):
        """Update tracker (re-run YOLO detection to find marker)"""
        if not self.yolo_model:
            return False, self.last_bbox
        
        # Re-detect marker in frame (use relaxed thresholds for lock mode reliability)
        marker = detect_red_box(
            self.yolo_model,
            frame,
            confidence_threshold=0.20,  # Lowered for lock mode reliability
            color_threshold=0.12,       # Lowered to catch darkened reds
            square_aspect_ratio_tolerance=0.60  # Tolerant of irregular shapes
        )
        
        if marker['detected']:
            # Calculate new bbox from marker
            x1 = marker['center_x'] - marker['width'] // 2
            y1 = marker['center_y'] - marker['height'] // 2
            new_bbox = (x1, y1, marker['width'], marker['height'])
            
            # Check distance from last known position
            if self.last_bbox:
                last_cx = self.last_bbox[0] + self.last_bbox[2] // 2
                last_cy = self.last_bbox[1] + self.last_bbox[3] // 2
                new_cx = marker['center_x']
                new_cy = marker['center_y']
                distance = np.sqrt((new_cx - last_cx)**2 + (new_cy - last_cy)**2)
                
                if distance > self.max_distance:
                    self.lost_frames += 1
                    if self.lost_frames >= self.max_lost_frames:
                        return False, self.last_bbox
                    return True, self.last_bbox
            
            self.last_bbox = new_bbox
            self.lost_frames = 0
            return True, new_bbox
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames or not self.last_bbox:
                return False, self.last_bbox
            return True, self.last_bbox

#####################################################################################################
class HomeMarkerTracker:
    """Tracks red square home marker with servo/motor integration"""
    #####################################################################################################
    def __init__(self, use_camera=True, use_servo=True, use_motor=True):
        self.logger = setup_logger(__name__)
        self.use_camera = use_camera
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(config.YOLO_MODEL)
            log_info(self.logger, f"YOLO: {config.YOLO_MODEL}")
        except Exception as e:
            if config.USE_NCNN and str(config.YOLO_MODEL).endswith('_ncnn_model'):
                fallback = str(config.YOLO_MODEL).replace('_ncnn_model', '.pt')
                self.yolo_model = YOLO(fallback)
                log_info(self.logger, f"YOLO fallback: {fallback}")
            else:
                raise
        
        # Camera
        if use_camera:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config_obj = self.picam2.create_preview_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": config.CAMERA_FPS}
            )
            self.picam2.configure(config_obj)
            self.picam2.start()
            time.sleep(1.5)
        else:
            self.picam2 = None
        
        # Servo
        self.servo = ServoController(
            pwm_pin=config.SERVO_PWM_PIN,
            frequency=config.PWM_FREQUENCY,
            center_duty=config.SERVO_CENTER,
            left_max_duty=config.SERVO_LEFT_MAX,
            right_max_duty=config.SERVO_RIGHT_MAX
        ) if use_servo else None
        
        self.servo.center()
        
        # Motor
        self.motor = MotorController(
            pwm_pin=config.MOTOR_PWM_PIN,
            frequency=config.PWM_FREQUENCY
        ) if use_motor else None
        
        self.motor.stop()
        
        # Tracking state
        self.is_scanning = True
        self.is_locked = False
        self.tracker = None
        self.last_detection = None
        self.lost_count = 0
        self.lost_threshold = 10
        
        CentroidTracker.yolo_model = self.yolo_model
        
        # Performance tracking
        self.motor_enabled = True
        
        # Detection parameters (loosened for shadowed red cube)
        self.confidence_threshold = 0.25
        self.color_threshold = 0.15  # Lowered from 0.25 to catch darker reds
        self.square_tolerance = 0.60  # Raised from 0.45 to accept irregular cube shapes
        self.stop_distance = config.HOME_MARKER_STOP_DISTANCE
        self.slow_threshold = config.HOME_MARKER_SLOW_DISTANCE
        self.center_tolerance = config.CAMERA_WIDTH * 0.1
        
        # Servo/Motor parameters
        self.steering_gain = config.ANGLE_TO_STEERING_GAIN
        self.slow_speed = config.MOTOR_SLOW
        self.medium_speed = config.MOTOR_MEDIUM
        self.fast_speed = config.MOTOR_FAST
       
        # Stop confirmation to avoid immediate stop on a single frame
        self.stop_confirm_count = 0
        self.stop_confirm_threshold = 2
        self.is_stopped = False
        
        # Slow approach confirmation (separate counter for smooth deceleration)
        self.slow_confirm_count = 0
        self.slow_confirm_threshold = 2
        
        # Servo state tracking (to reduce jitter from redundant center commands)
        self.last_servo_angle = 0.0  # Track last commanded angle

        self.run_flag = False
        self.approach_flag = False
#####################################################################################################
    def safe_center_servo(self):
        """Center servo only if not already centered (to reduce jitter)"""
        if self.servo and self.last_servo_angle != 0.0:
            self.servo.center()
            self.last_servo_angle = 0.0
    #####################################################################################################
    def get_frame(self):
        """Get next frame from camera or dummy source"""
        if self.use_camera and self.picam2:
            frame = self.picam2.capture_array(wait=True)  # Returns RGB
            
            # Apply camera rotation if configured
            if config.CAMERA_ROTATION == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif config.CAMERA_ROTATION == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif config.CAMERA_ROTATION == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Apply flips if configured
            if config.CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            if config.CAMERA_FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)
            
            return frame
        else:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8) + 50
#####################################################################################################
    def handle_scan_mode(self, frame_bgr):
        """Scan for home marker (no lock yet)"""

        if self.run_flag == False:
            self.motor.stop()
            self.servo.center()

        marker = detect_red_box(
            self.yolo_model,
            frame_bgr,
            confidence_threshold=self.confidence_threshold,
            color_threshold=self.color_threshold,
            square_aspect_ratio_tolerance=self.square_tolerance
        )
        
        if marker['detected']:
            x1 = marker['center_x'] - marker['width'] // 2
            y1 = marker['center_y'] - marker['height'] // 2
            bbox = (x1, y1, marker['width'], marker['height'])
            
            self.tracker = CentroidTracker(max_distance=150, max_lost_frames=10)
            ok = self.tracker.init(frame_bgr, bbox)
            
            if ok:
                self.is_locked = True
                self.is_scanning = False
                self.last_detection = marker
                self.lost_count = 0
                log_info(self.logger, f"LOCKED: conf={marker['confidence']:.2f}, color={marker['color_match']:.1%}")
                return marker
        else:
            self.servo.turn_left(0.3)
            self.motor.forward(self.slow_speed)
        return None
#####################################################################################################
    def handle_lock_mode(self, frame_bgr):
        """Track locked marker with servo/motor steering"""
        self.run_flag = True

        if not self.tracker:
            log_warning(self.logger, "Tracker is None in lock mode", "Reverting to scan")
            self.is_locked = False
            self.is_scanning = True
            return None
        
        # Track using centroid tracker (re-detects marker each frame)
        ok, bbox = self.tracker.update(frame_bgr)
        
        if not ok:
            log_info(self.logger, "Lost lock, returning to scan mode")
            self.is_locked = False
            self.is_scanning = True
            self.tracker = None
            
            self.motor.stop()
            self.safe_center_servo()
            return None
        
        # Tracker succeeded - bbox is (x, y, w, h)
        x, y, w, h = map(int, bbox)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Verify color match (sanity check)
        color_match = check_color_match_red(frame_bgr, (x, y, x + w, y + h))

        # If color match drops, count misses and only give up after threshold
        if color_match < 0.20:
            self.lost_count += 1
            if self.lost_count >= self.lost_threshold:
                log_info(self.logger, "Lost color match, returning to scan mode")
                self.is_locked = False
                self.is_scanning = True
                self.tracker = None
                
                self.motor.stop()
                self.safe_center_servo()
                return None
        else:
            self.lost_count = 0
        
        # Calculate steering offset
        frame_center_x = config.CAMERA_WIDTH // 2
        offset = center_x - frame_center_x
        
        # Compute steering angle (proportional to offset)
        steering_angle = (offset / frame_center_x) * 45.0  # Map to [-45, 45] degrees
        steering_angle = max(-45.0, min(45.0, steering_angle))
        self.servo.set_angle(steering_angle)
        self.last_servo_angle = steering_angle  # Track for centering check
        
        # Determine speed based on approach mode and centering
        if self.approach_flag:
            # In slow approach mode: lock to slow speed
            speed = self.slow_speed
            speed_str = "SLOW (approach)"
        else:
            # Normal mode: speed based on centering
            if abs(offset) < self.center_tolerance:
                speed = self.medium_speed
                speed_str = "MEDIUM"
            else:
                # Marker off-center - move slower
                speed = self.slow_speed
                speed_str = "SLOW"
        
        # Move forward if motor enabled
        if self.motor_enabled and self.motor:
            self.motor.forward(speed)
            log_info(self.logger, f"STEERING: {steering_angle:.1f}°, SPEED: {speed_str}")
        
        # Build detection dict for consistency
        detection = {
            'detected': True,
            'center_x': center_x,
            'center_y': center_y,
            'width': w,
            'height': h,
            'area': w * h,
            'confidence': 1.0,  # Tracker doesn't have confidence
            'color_match': color_match,
            'aspect_ratio': w / h if h > 0 else 0.0
        }
        self.last_detection = detection
        
        # Check slow approach condition (width-based)
        if w >= self.slow_threshold:
            self.slow_confirm_count += 1
            if self.slow_confirm_count >= self.slow_confirm_threshold and not self.approach_flag:
                log_info(self.logger, f"SLOW APPROACH TRIGGERED! Width: {w} >= {self.slow_threshold}")
                self.approach_flag = True
                self.slow_confirm_count = 0
        else:
            self.slow_confirm_count = 0
        
        # Check stopping condition 
        centered = abs(offset) <= (self.center_tolerance * 0.5)
        if w >= self.stop_distance and centered:
            self.stop_confirm_count += 1
            log_info(self.logger, f"Stop check: width={w} >= {self.stop_distance} and centered={centered} (count={self.stop_confirm_count})")
        else:
            if w >= self.stop_distance and not centered:
                log_info(self.logger, f"Stop candidate ignored: width={w} >= {self.stop_distance} but not centered (offset={offset:.0f}px)")
            self.stop_confirm_count = 0

        if self.stop_confirm_count >= self.stop_confirm_threshold:
            log_info(self.logger, f"STOP CONDITION MET! Width: {w} >= {self.stop_distance} (confirmed)")
            self.motor.stop()

            self.safe_center_servo()
            
            # mark stopped and keep last_detection for overlay
            self.is_locked = False
            self.tracker = None
            self.is_stopped = True
            self.stop_confirm_count = 0
            self.approach_flag = False
        
        return detection
#####################################################################################################
    def draw_ui(self, frame, detection, mode_str):
        """Draw UI overlay on frame"""
        annotated = frame.copy()
        h, w_img = annotated.shape[:2]
        
        # Draw frame center line
        cv2.line(annotated, (config.CAMERA_WIDTH // 2, 0), (config.CAMERA_WIDTH // 2, h), (100, 100, 255), 1)
        
        if detection and detection['detected']:
            # Draw bounding box
            x1 = detection['center_x'] - detection['width'] // 2
            y1 = detection['center_y'] - detection['height'] // 2
            x2 = detection['center_x'] + detection['width'] // 2
            y2 = detection['center_y'] + detection['height'] // 2
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (detection['center_x'], detection['center_y']), 5, (0, 255, 0), -1)
            
            # Draw offset indicator
            offset = detection['center_x'] - config.CAMERA_WIDTH // 2
            cv2.line(annotated, (config.CAMERA_WIDTH // 2, h - 30), (detection['center_x'], h - 30), (0, 255, 0), 2)
            
            # Status text
            status_lines = [
                f"MODE: {mode_str}",
                f"WIDTH: {detection['width']:.0f}px (stop at {self.stop_distance})",
                f"COLOR MATCH: {detection['color_match']:.1%}",
                f"OFFSET: {offset:.0f}px",
                f"MOTOR: {'ON' if self.motor_enabled else 'OFF'}"
            ]
        else:
            status_lines = [
                f"MODE: {mode_str}",
                "STATUS: No marker detected",
                "MOTOR: OFF"
            ]
        
        # Draw status background and text
        y_offset = 20
        for i, line in enumerate(status_lines):
            cv2.rectangle(annotated, (5, y_offset + i * 25 - 20), (400, y_offset + i * 25 + 5), (0, 0, 0), -1)
            cv2.putText(annotated, line, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw controls
        controls = [
            "q: quit | s: restart (scan) | m: motor | r: reset"
        ]
        y_control = h - 30
        cv2.rectangle(annotated, (5, y_control - 20), (700, y_control + 5), (0, 0, 0), -1)
        cv2.putText(annotated, controls[0], (10, y_control), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # (FPS display removed)
        
        return annotated
    #####################################################################################################
    def run(self):
        """Main test loop"""
        log_info(self.logger, "="*50)
        log_info(self.logger, "Home Marker Tracker Test")
        log_info(self.logger, f"YOLO: {config.YOLO_MODEL} | Camera: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        log_info(self.logger, f"Servo: {bool(self.servo)} | Motor: {bool(self.motor)}")
        log_info(self.logger, "="*50)
        
        # Import TOF sensor if available
        tof_sensor = ToFSensor()
   
        try:
            while True:
              
                # Get frame
                frame_rgb = self.get_frame()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process based on mode
                if self.is_stopped:
                    # Stopped state: only respond to 's' key to restart
                    detection = self.last_detection
                    mode_str = "STOPPED (press 's' to restart)"
                elif self.is_scanning:
                    detection = self.handle_scan_mode(frame_bgr)
                    mode_str = "SCAN (searching)"
                elif self.is_locked:
                    detection = self.handle_lock_mode(frame_bgr)
                    mode_str = "LOCK (tracking)"
                else:
                    detection = None
                    mode_str = "IDLE"
                
                # Draw UI
                annotated = self.draw_ui(frame_bgr, detection, mode_str)
                
                # Display
                cv2.imshow('Home Marker Tracker Test', annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_info(self.logger, "Quit requested")
                    break
                elif key == ord('s'):
                    # 's' key: restart scan mode from any state
                    log_info(self.logger, "Restarting: switching to SCAN mode")
                    self.is_scanning = True
                    self.is_locked = False
                    self.is_stopped = False
                    self.tracker = None
                    self.lost_count = 0
                    if self.motor:
                        self.motor.stop()
                    self.safe_center_servo()
                elif key == ord('m'):
                    self.motor_enabled = not self.motor_enabled
                    log_info(self.logger, f"Motor: {'ENABLED' if self.motor_enabled else 'DISABLED'}")
                elif key == ord('r'):
                    log_info(self.logger, "Reset requested")
                    self.is_scanning = True
                    self.is_locked = False
                    self.tracker = None
                    self.lost_count = 0
                    self.is_stopped = False
                    if self.motor:
                        self.motor.stop()
                    self.safe_center_servo()
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            log_info(self.logger, "Interrupted by user")
        except Exception as e:
            log_error(self.logger, e, "Fatal error in main loop")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        log_info(self.logger, "Cleaning up...")
        
        # Stop motors and servo
        if self.motor:
            try:
                self.motor.stop()
                self.motor.cleanup()
            except Exception as e:
                log_warning(self.logger, f"Error stopping motor: {e}", "Cleanup")
        
        if self.servo:
            try:
                self.safe_center_servo()
                self.servo.cleanup()
            except Exception as e:
                log_warning(self.logger, f"Error stopping servo: {e}", "Cleanup")
        
        # Stop camera
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception as e:
                log_warning(self.logger, f"Error stopping camera: {e}", "Cleanup")
        
        # Close windows
        cv2.destroyAllWindows()
        
        log_info(self.logger, "Test complete.")
        log_info(self.logger, "Cleanup finished")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Test home marker tracking with servo/motor integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_home_tracking.py                    # Full test with camera, servo, motor
    python test_home_tracking.py --no-camera        # Simulation mode (no camera)
    python test_home_tracking.py --no-servo         # Test without servo
    python test_home_tracking.py --no-motor         # Test without motor
    python test_home_tracking.py --no-camera --no-servo --no-motor  # Simulation only
        """
    )
    
    parser.add_argument('--no-camera', action='store_true', help='Disable camera (simulation mode)')
    parser.add_argument('--no-servo', action='store_true', help='Disable servo controller')
    parser.add_argument('--no-motor', action='store_true', help='Disable motor controller')
    
    args = parser.parse_args()
    
    # Create and run tracker
    tracker = HomeMarkerTracker(
        use_camera=not args.no_camera,
        use_servo=not args.no_servo,
        use_motor=not args.no_motor
    )
    tracker.run()


if __name__ == '__main__':
    main()
