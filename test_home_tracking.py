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
    's'     - Toggle scan mode (searching for marker)
    'l'     - Toggle lock mode (tracking with servo/motor)
    'm'     - Toggle motor on/off during lock
    'r'     - Reset tracker and return to scan mode
    'f'     - Show FPS
"""

import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config
    from home_marker_detector import detect_red_box, check_color_match_red
    from logger import setup_logger, log_info, log_warning, log_error
    from servo_controller import ServoController
    from motor_controller import MotorController
    from ultralytics import YOLO
except ImportError as e:
    print(f"ERROR: Missing required module: {e}")
    print("Required: config, home_marker_detector, logger, servo_controller, motor_controller, ultralytics")
    sys.exit(1)


class HomeMarkerTracker:
    """Tracks red square home marker with servo/motor integration""" 
    def __init__(self, use_camera=True, use_servo=True, use_motor=True):
        """
        Initialize tracker.
        
        Args:
            use_camera: Use camera (True) or read from video file (False)
            use_servo: Initialize servo controller
            use_motor: Initialize motor controller
        """
        self.logger = setup_logger(__name__)
        self.use_camera = use_camera
        self.use_servo = use_servo
        self.use_motor = use_motor
        
        # Initialize YOLO model
        log_info(self.logger, "Initializing YOLO model...")
        self.yolo_model = YOLO(config.YOLO_MODEL)
        
        try:
            self.yolo_model = YOLO(config.YOLO_MODEL)
            log_info(self.logger, f"YOLO model loaded: {config.YOLO_MODEL}")
        except Exception as e1:
            # Fallback: if NCNN failed, try PyTorch
            if config.USE_NCNN and str(config.YOLO_MODEL).endswith('_ncnn_model'):
                fallback_path = str(config.YOLO_MODEL).replace('_ncnn_model', '.pt')
                log_info(self.logger, f"NCNN model not found, trying PyTorch: {fallback_path}")
                self.yolo_model = YOLO(fallback_path)
                log_info(self.logger, f"PyTorch model loaded: {fallback_path}")
            else:
                raise RuntimeError(f"Failed to load YOLO model: {e1}") from e1
        
        # Initialize camera/video
        if self.use_camera:
            try:
                from picamera2 import Picamera2
                log_info(self.logger, "Initializing Picamera2...")
                self.picam2 = Picamera2()
                preview_config = self.picam2.create_preview_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                    controls={"FrameRate": config.CAMERA_FPS}
                )
                self.picam2.configure(preview_config)
                self.picam2.start()
                time.sleep(1.5)
                log_info(self.logger, f"Camera started: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
                self.video_source = None
            except Exception as e:
                log_error(self.logger, e, "Failed to initialize camera")
                sys.exit(1)
        else:
            log_info(self.logger, "Using dummy frame source (no camera)")
            self.picam2 = None
            self.video_source = None
        
        # Initialize servo
        if self.use_servo:
            log_info(self.logger, "Initializing servo controller...")
            try:
                self.servo = ServoController(
                    pwm_pin=config.SERVO_PWM_PIN,
                    frequency=config.PWM_FREQUENCY,
                    center_duty=config.SERVO_CENTER,
                    left_max_duty=config.SERVO_LEFT_MAX,
                    right_max_duty=config.SERVO_RIGHT_MAX
                )
                self.servo.center()
                log_info(self.logger, "Servo initialized and centered")
            except Exception as e:
                log_warning(self.logger, f"Failed to initialize servo: {e}", "Servo disabled")
                self.servo = None
        else:
            self.servo = None
        
        # Initialize motor
        if self.use_motor:
            log_info(self.logger, "Initializing motor controller...")
            try:
                self.motor = MotorController(
                    pwm_pin=config.MOTOR_PWM_PIN,
                    frequency=config.PWM_FREQUENCY
                )
                self.motor.stop()
                log_info(self.logger, "Motor initialized and stopped")
            except Exception as e:
                log_warning(self.logger, f"Failed to initialize motor: {e}", "Motor disabled")
                self.motor = None
        else:
            self.motor = None
        
        # Tracking state
        self.is_scanning = True
        self.is_locked = False
        self.tracker = None
        self.last_detection = None
        self.lost_count = 0
        self.lost_threshold = 5
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.show_fps = False
        self.motor_enabled = True
        
        # Detection parameters
        self.confidence_threshold = 0.35
        self.color_threshold = 0.40
        self.square_tolerance = 0.25
        self.stop_distance = config.HOME_MARKER_STOP_DISTANCE
        self.center_tolerance = config.CAMERA_WIDTH * 0.05  # 5% of width
        
        # Servo/Motor parameters
        self.steering_gain = config.ANGLE_TO_STEERING_GAIN
        self.slow_speed = config.MOTOR_SLOW
        self.medium_speed = config.MOTOR_MEDIUM
        self.fast_speed = config.MOTOR_FAST
        
###############################################################################################################################################    
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
            
            return frame  # RGB format
        else:
            # Dummy frame: dark gray
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8) + 50
###############################################################################################################################################    
    def handle_scan_mode(self, frame_bgr):
        """Scan for home marker (no lock yet)"""
        # Run full-frame YOLO detection to find marker
        marker = detect_red_box(
            self.yolo_model,
            frame_bgr,
            confidence_threshold=self.confidence_threshold,
            color_threshold=self.color_threshold,
            square_aspect_ratio_tolerance=self.square_tolerance
        )
        
        if marker['detected']:
            # Marker found! Initialize tracker and lock on
            x1 = marker['center_x'] - marker['width'] // 2
            y1 = marker['center_y'] - marker['height'] // 2
            bbox = (x1, y1, marker['width'], marker['height'])
            
            # Initialize MOSSE tracker (very fast)
            self.tracker = cv2.TrackerMOSSE_create()
            ok = self.tracker.init(frame_bgr, bbox)
            
            if ok:
                self.is_locked = True
                self.is_scanning = False
                self.last_detection = marker
                self.lost_count = 0
                log_info(self.logger, f"Marker LOCKED! Bbox: {bbox}, Confidence: {marker['confidence']:.2f}")
                return marker
            else:
                log_warning(self.logger, "Tracker initialization failed", "Continuing scan")
                return None
        else:
            # No marker found - perform search sweep
            if self.servo:
                self.servo.turn_left(0.3)  # Small left turn while scanning
            if self.motor:
                self.motor.forward(self.slow_speed)
            return None
        
###############################################################################################################################################
    def handle_lock_mode(self, frame_bgr):
        """Track locked marker with servo/motor steering"""
        if not self.tracker:
            log_warning(self.logger, "Tracker is None in lock mode", "Reverting to scan")
            self.is_locked = False
            self.is_scanning = True
            return None
        
        # Track using MOSSE
        ok, bbox = self.tracker.update(frame_bgr)
        
        if not ok:
            self.lost_count += 1
            log_warning(self.logger, f"Tracker update failed (lost_count: {self.lost_count})", "Tracking")
            if self.lost_count >= self.lost_threshold:
                log_info(self.logger, "Lost lock, returning to scan mode")
                self.is_locked = False
                self.is_scanning = True
                self.tracker = None
                if self.motor:
                    self.motor.stop()
                if self.servo:
                    self.servo.center()
                return None
            else:
                return self.last_detection  # Use cached detection
        
        # Tracker succeeded
        x, y, w, h = map(int, bbox)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Verify color match (quick check to confirm still red)
        color_match = check_color_match_red(frame_bgr, (x, y, x + w, y + h))
        
        if color_match < 0.2:
            self.lost_count += 1
            log_warning(self.logger, f"Color match failed: {color_match:.2f} (lost_count: {self.lost_count})", "Tracking")
            if self.lost_count >= self.lost_threshold:
                log_info(self.logger, "Lost color match, returning to scan mode")
                self.is_locked = False
                self.is_scanning = True
                self.tracker = None
                if self.motor:
                    self.motor.stop()
                if self.servo:
                    self.servo.center()
                return None
        else:
            self.lost_count = 0  # Reset lost count on successful track
        
        # Calculate steering offset
        frame_center_x = config.CAMERA_WIDTH // 2
        offset = center_x - frame_center_x
        
        # Compute steering angle (proportional to offset)
        steering_angle = (offset / frame_center_x) * 45.0  # Map to [-45, 45] degrees
        steering_angle = max(-45.0, min(45.0, steering_angle))
        
        # Set servo angle
        if self.servo:
            self.servo.set_angle(steering_angle)
        
        # Determine speed based on centering and distance
        if abs(offset) < self.center_tolerance:
            # Marker is centered - move faster
            speed = self.fast_speed
            speed_str = "FAST"
        else:
            # Marker off-center - move slower
            speed = self.medium_speed
            speed_str = "MEDIUM"
        
        # Move forward if motor enabled
        if self.motor_enabled and self.motor:
            self.motor.forward(speed)
        
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
        
        # Check stopping condition
        if w >= self.stop_distance:
            log_info(self.logger, f"STOP CONDITION MET! Width: {w} >= {self.stop_distance}")
            if self.motor:
                self.motor.stop()
            if self.servo:
                self.servo.center()
            self.is_locked = False
            self.tracker = None
        
        return detection
    
    ###############################################################################################################################################
    
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
            "q: quit | s: scan | l: lock | m: motor | r: reset | f: fps"
        ]
        y_control = h - 30
        cv2.rectangle(annotated, (5, y_control - 20), (700, y_control + 5), (0, 0, 0), -1)
        cv2.putText(annotated, controls[0], (10, y_control), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw FPS if enabled
        if self.show_fps and len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(annotated, fps_text, (config.CAMERA_WIDTH - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    ###############################################################################################################################################
    def run(self):
        """Main test loop"""
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "Home Marker Tracker Test")
        log_info(self.logger, "=" * 70)
        log_info(self.logger, f"YOLO Model: {config.YOLO_MODEL}")
        log_info(self.logger, f"Camera: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        log_info(self.logger, f"Servo: {'Enabled' if self.servo else 'Disabled'}")
        log_info(self.logger, f"Motor: {'Enabled' if self.motor else 'Disabled'}")
        log_info(self.logger, "=" * 70)
        
        try:
            while True:
                frame_start = time.time()
                
                # Get frame
                frame_rgb = self.get_frame()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process based on mode
                if self.is_scanning:
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
                
                # Update FPS
                self.frame_count += 1
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    self.fps_history.append(fps)
                
                # Display
                cv2.imshow('Home Marker Tracker Test', annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_info(self.logger, "Quit requested")
                    break
                elif key == ord('s'):
                    if not self.is_scanning:
                        log_info(self.logger, "Switching to SCAN mode")
                        self.is_scanning = True
                        self.is_locked = False
                        self.tracker = None
                        if self.motor:
                            self.motor.stop()
                        if self.servo:
                            self.servo.center()
                elif key == ord('l'):
                    if not self.is_locked and self.last_detection and self.last_detection['detected']:
                        log_info(self.logger, "Switching to LOCK mode")
                        self.is_locked = True
                        self.is_scanning = False
                elif key == ord('m'):
                    self.motor_enabled = not self.motor_enabled
                    log_info(self.logger, f"Motor: {'ENABLED' if self.motor_enabled else 'DISABLED'}")
                elif key == ord('r'):
                    log_info(self.logger, "Reset requested")
                    self.is_scanning = True
                    self.is_locked = False
                    self.tracker = None
                    self.lost_count = 0
                    if self.motor:
                        self.motor.stop()
                    if self.servo:
                        self.servo.center()
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    log_info(self.logger, f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                
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
                self.servo.center()
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
        
        log_info(self.logger, f"Test complete. Frames processed: {self.frame_count}")
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
