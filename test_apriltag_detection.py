#!/usr/bin/env python3
"""
ArUco Marker Detection Test
Tests ArUco marker detection and navigation similar to user following logic in main.py
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config
    from logger import setup_logger, log_error, log_info
except ImportError as e:
    print(f"ERROR: Missing required module: {e}")
    sys.exit(1)

# Initialize logger first
logger = setup_logger(__name__)

# Try to import Picamera2 (Raspberry Pi only)
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    Picamera2 = None
    logger.info("Picamera2 not available - will use webcam instead")

# Try to import motor/servo controllers (Raspberry Pi only)
try:
    from motor_controller import MotorController
    from servo_controller import ServoController
    HAS_GPIO_CONTROLLERS = True
except ImportError:
    HAS_GPIO_CONTROLLERS = False
    MotorController = None
    ServoController = None

# Using ArUco markers (built into OpenCV) - no external dependencies needed

logger = setup_logger(__name__)


class ArUcoDetector:
    """ArUco marker detection and navigation controller"""
    
    def __init__(self, tag_size_m=0.047, dictionary_id=cv2.aruco.DICT_6X6_250):
        """
        Initialize ArUco marker detector
        
        Args:
            tag_size_m: Physical size of marker in meters (default: 0.047 = 47mm)
            dictionary_id: ArUco dictionary to use (default: DICT_6X6_250)
        """
        self.tag_size_m = tag_size_m
        
        # Use OpenCV ArUco markers (built-in, no extra dependencies)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimized detection parameters for better detection, especially tilted markers
        # These help with reflections, lighting, various tag sizes, and perspective distortion
        
        # Adaptive thresholding - more lenient for tilted markers
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 5  # Good for printed tags and screens
        
        # Marker size constraints - allow more variation for perspective distortion
        self.aruco_params.minMarkerPerimeterRate = 0.02  # Allow smaller markers (when tilted)
        self.aruco_params.maxMarkerPerimeterRate = 4.0   # Allow larger markers (when close/tilted)
        
        # Polygonal approximation - more lenient for perspective distortion
        self.aruco_params.polygonalApproxAccuracyRate = 0.08  # Increased from 0.05 - more tolerant of distortion
        
        # Corner detection - more lenient for tilted markers
        self.aruco_params.minCornerDistanceRate = 0.02  # Reduced from 0.03 - allow closer corners when tilted
        self.aruco_params.minDistanceToBorder = 1  # Reduced from 2 - allow markers near edges
        
        # Marker spacing - more lenient
        self.aruco_params.minMarkerDistanceRate = 0.02  # Reduced from 0.03
        
        # Corner refinement - important for tilted markers
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 7  # Increased from 5 - better for tilted markers
        self.aruco_params.cornerRefinementMaxIterations = 50  # Increased from 30 - more refinement
        self.aruco_params.cornerRefinementMinAccuracy = 0.01  # Reduced from 0.05 - more lenient
        
        # Perspective removal - critical for tilted markers
        self.aruco_params.perspectiveRemovePixelPerCell = 8  # Increased from 4 - better for perspective
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.2  # Increased from 0.13 - more margin
        
        # Additional parameters for tilted markers
        # Allow more perspective distortion
        self.aruco_params.maxErroneousBitsInBorderRate = 0.35  # Increased tolerance for border errors
        self.aruco_params.errorCorrectionRate = 0.6  # Higher error correction for tilted markers
        
        # Use new API for OpenCV 4.7+ (ArucoDetector) or fallback to old API
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
        else:
            self.aruco_detector = None
            self.use_new_aruco_api = False
        
        logger.info(f"ArUco detector initialized (dictionary: {dictionary_id}, tag size: {tag_size_m}m)")
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_camera_calibration()
        
    def _load_camera_calibration(self):
        """Load camera calibration if available"""
        calibration_file = Path('camera_calibration/calibration_savez.npz')
        if calibration_file.exists():
            try:
                data = np.load(str(calibration_file))
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                logger.info("Camera calibration loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load camera calibration: {e}")
                logger.info("Using default camera parameters")
        else:
            logger.warning("No camera calibration file found. Using default parameters.")
            logger.info("For better accuracy, run camera calibration first.")
            # Default camera matrix (approximate for Raspberry Pi Camera)
            # These will be replaced with actual calibration values
            self.camera_matrix = np.array([
                [640, 0, 320],  # fx, 0, cx
                [0, 640, 240],  # 0, fy, cy
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1))
    
    def detect_tag(self, frame):
        """
        Detect ArUco marker in frame
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            dict with detection results:
            {
                'detected': bool,
                'center_x': float,  # Marker center in pixels
                'center_y': float,
                'angle': float,     # Angle to steer (-45 to +45 degrees)
                'is_centered': bool, # Whether marker is centered in frame
                'distance_m': float, # Estimated distance in meters
                'corners': np.array, # Marker corners for visualization
                'tag_id': int,      # Marker ID
                'pose': dict        # Pose information (rotation, translation)
            }
        """
        # Convert BGR to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use ArUco detection (new API for OpenCV 4.7+)
        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            # Old API (OpenCV < 4.7)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        if ids is None or len(ids) == 0:
            # Debug: log rejection info if available
            if hasattr(self, 'debug_detection') and self.debug_detection and rejected:
                logger.debug(f"ArUco detection: {len(rejected)} rejected candidates")
            return {
                'detected': False,
                'center_x': None,
                'center_y': None,
                'angle': None,
                'is_centered': False,
                'distance_m': None,
                'corners': None,
                'tag_id': None,
                'pose': None
            }
        
        # Use the first detected marker
        marker_corners = corners[0][0]  # Shape: (4, 2)
        tag_id = int(ids[0][0])
        
        # Get marker center
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))
        corners_array = marker_corners
        
        # Calculate angle based on tag position relative to frame center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        # Calculate offset from center
        offset_x = center_x - frame_center_x
        offset_y = center_y - frame_center_y
        
        # Convert pixel offset to angle (similar to main.py logic)
        # Maximum angle when tag is at edge of frame
        max_offset = frame.shape[1] / 2
        angle = (offset_x / max_offset) * 45.0  # Scale to ±45 degrees
        angle = max(-45.0, min(45.0, angle))  # Clamp to servo range
        
        # Check if tag is centered (within threshold)
        center_threshold = config.PERSON_CENTER_THRESHOLD  # Reuse from config
        is_centered = abs(offset_x) < center_threshold
        
        # Estimate distance based on tag size
        # Calculate tag size in pixels
        tag_width_px = np.linalg.norm(corners_array[1] - corners_array[0])
        tag_height_px = np.linalg.norm(corners_array[2] - corners_array[1])
        tag_size_px = (tag_width_px + tag_height_px) / 2
        
        # Estimate distance using similar triangles
        # distance = (real_size * focal_length) / pixel_size
        focal_length = self.camera_matrix[0, 0]  # fx
        if tag_size_px > 0:
            distance_m = (self.tag_size_m * focal_length) / tag_size_px
        else:
            distance_m = None
        
        # Calculate pose (rotation and translation) if camera calibration is available
        pose = None
        if self.camera_matrix is not None:
            try:
                # ArUco marker pose estimation
                # Note: This requires proper camera calibration for accurate results
                object_points = np.array([
                    [-self.tag_size_m/2, -self.tag_size_m/2, 0],
                    [self.tag_size_m/2, -self.tag_size_m/2, 0],
                    [self.tag_size_m/2, self.tag_size_m/2, 0],
                    [-self.tag_size_m/2, self.tag_size_m/2, 0]
                ], dtype=np.float32)
                
                image_points = corners_array.astype(np.float32)
                
                # Solve PnP to get pose
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    rmat, _ = cv2.Rodrigues(rvec)
                    
                    # Extract translation (distance in camera frame)
                    # tvec[2] is the Z distance (forward)
                    pose = {
                        'rotation': rmat,
                        'translation': tvec.flatten(),
                        'distance_z': float(tvec[2, 0]),  # Forward distance
                        'x': float(tvec[0, 0]),  # Left/right offset
                        'y': float(tvec[1, 0])   # Up/down offset
                    }
                    
                    # Use pose-based distance if available (more accurate)
                    if pose['distance_z'] > 0:
                        distance_m = pose['distance_z']
            except Exception as e:
                logger.debug(f"Pose estimation failed: {e}")
        
        return {
            'detected': True,
            'center_x': center_x,
            'center_y': center_y,
            'angle': angle,
            'is_centered': is_centered,
            'distance_m': distance_m,
            'corners': corners_array,
            'tag_id': tag_id,
            'pose': pose
        }
    
    def draw_overlay(self, frame, detection):
        """
        Draw ArUco marker detection overlay on frame
        
        Args:
            frame: BGR frame
            detection: Detection result from detect_tag()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if not detection['detected']:
            cv2.putText(annotated, "No ArUco marker detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        # Draw tag corners and edges
        corners = detection['corners'].astype(int)
        # Ensure corners are in correct format for polylines: (N, 1, 2)
        if len(corners.shape) == 2 and corners.shape[1] == 2:
            corners = corners.reshape(-1, 1, 2)
        cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
        
        # Draw center point
        center = (detection['center_x'], detection['center_y'])
        cv2.circle(annotated, center, 5, (0, 255, 0), -1)
        
        # Draw marker ID
        tag_id_text = f"ArUco ID: {detection['tag_id']}"
        cv2.putText(annotated, tag_id_text, (center[0] - 50, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw angle and distance info
        angle_text = f"Angle: {detection['angle']:.1f}°"
        cv2.putText(annotated, angle_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if detection['distance_m']:
            dist_text = f"Distance: {detection['distance_m']:.2f}m"
            cv2.putText(annotated, dist_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        centered_text = "CENTERED" if detection['is_centered'] else "NOT CENTERED"
        color = (0, 255, 0) if detection['is_centered'] else (0, 165, 255)
        cv2.putText(annotated, centered_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw pose info if available
        if detection['pose']:
            pose = detection['pose']
            pose_text = f"Pose Z: {pose['distance_z']:.2f}m, X: {pose['x']:.2f}m"
            cv2.putText(annotated, pose_text, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw frame center crosshair
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        cv2.line(annotated, (frame_center[0] - 20, frame_center[1]),
                 (frame_center[0] + 20, frame_center[1]), (255, 0, 0), 2)
        cv2.line(annotated, (frame_center[0], frame_center[1] - 20),
                 (frame_center[0], frame_center[1] + 20), (255, 0, 0), 2)
        
        return annotated


class ArUcoNavigationTest:
    """Test ArUco marker detection and navigation"""
    
    def __init__(self, tag_size_m=0.047, stop_distance_m=0.3, use_webcam=False, camera_index=0, 
                 test_rotation=0, test_flip_horizontal=False, test_flip_vertical=False):
        """
        Initialize navigation test
        
        Args:
            tag_size_m: Physical size of ArUco marker in meters (default: 0.047 = 47mm)
            stop_distance_m: Stop when marker is this close (meters)
            use_webcam: Use webcam instead of Picamera2 (default: False, auto-detect)
            camera_index: Webcam index (default: 0)
            test_rotation: Override camera rotation for testing (0, 90, 180, 270)
            test_flip_horizontal: Override horizontal flip for testing
            test_flip_vertical: Override vertical flip for testing
        """
        # Store test overrides for frame processing
        self.test_rotation = test_rotation
        self.test_flip_horizontal = test_flip_horizontal
        self.test_flip_vertical = test_flip_vertical
        self.tag_detector = ArUcoDetector(tag_size_m=tag_size_m)
        self.stop_distance_m = stop_distance_m
        self.use_webcam = use_webcam
        self.camera_index = camera_index
        
        # Initialize hardware (if GPIO enabled and available)
        if config.USE_GPIO and HAS_GPIO_CONTROLLERS:
            self.motor = MotorController(
                config.MOTOR_PWM_PIN,
                config.PWM_FREQUENCY_MOTOR
            )
            self.servo = ServoController(
                config.SERVO_PWM_PIN,
                config.PWM_FREQUENCY_SERVO,
                config.SERVO_CENTER,
                config.SERVO_LEFT_MAX,
                config.SERVO_RIGHT_MAX
            )
        else:
            self.motor = None
            self.servo = None
            if not HAS_GPIO_CONTROLLERS:
                logger.info("GPIO controllers not available - motor and servo will not be controlled")
            else:
                logger.info("GPIO disabled - motor and servo will not be controlled")
        
        # Initialize camera
        self.picam2 = None
        self.webcam = None
        
        # Auto-detect: use webcam if Picamera2 not available or explicitly requested
        if use_webcam or not HAS_PICAMERA2:
            # Use webcam
            logger.info("Initializing webcam...")
            self.webcam = cv2.VideoCapture(camera_index)
            if not self.webcam.isOpened():
                raise RuntimeError(f"Failed to open webcam at index {camera_index}")
            
            # Set webcam resolution
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.webcam.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            # Read a test frame to ensure camera is working
            ret, _ = self.webcam.read()
            if not ret:
                raise RuntimeError("Webcam opened but failed to read frame")
            
            logger.info(f"Webcam initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        else:
            # Use Picamera2 (Raspberry Pi)
            logger.info("Initializing Picamera2...")
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": config.CAMERA_FPS}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(1.5)  # Allow camera to stabilize
            logger.info(f"Picamera2 initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        
        logger.info("ArUco navigation test initialized")
        logger.info(f"Marker size: {tag_size_m}m, Stop distance: {stop_distance_m}m")
    
    def _get_frame(self):
        """Get frame from camera with rotation/flip applied"""
        if self.webcam is not None:
            # Webcam: returns BGR
            ret, frame = self.webcam.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam")
            array = frame  # Already BGR
        else:
            # Picamera2: returns RGB
            array = self.picam2.capture_array(wait=True)  # Returns RGB
        
        # Apply camera transformations
        # For webcam testing, you might need to rotate if camera is upside down
        # Default: use config settings, but allow override for testing
        rotation = getattr(self, 'test_rotation', config.CAMERA_ROTATION)
        if rotation == 180:
            array = cv2.rotate(array, cv2.ROTATE_180)
        elif rotation == 90:
            array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 270:
            array = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        flip_h = getattr(self, 'test_flip_horizontal', config.CAMERA_FLIP_HORIZONTAL)
        flip_v = getattr(self, 'test_flip_vertical', config.CAMERA_FLIP_VERTICAL)
        
        if flip_h:
            array = cv2.flip(array, 1)
        if flip_v:
            array = cv2.flip(array, 0)
        
        # Handle color format
        if self.webcam is None:
            # Picamera2 returns RGB, convert to BGR if needed
            if config.CAMERA_SWAP_RB:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        # Webcam already returns BGR, no conversion needed
        
        return array
    
    def run_test(self, display=True, control_hardware=True):
        """
        Run ArUco marker detection and navigation test
        
        Args:
            display: Show video feed with overlays
            control_hardware: Control motor and servo (requires GPIO)
        """
        logger.info("Starting ArUco navigation test...")
        logger.info("Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        last_detection_time = time.time()
        tag_lost_timeout = 2.0  # Stop if tag lost for 2 seconds
        
        try:
            while True:
                frame = self._get_frame()
                
                # Detect tag
                detection = self.tag_detector.detect_tag(frame)
                
                if detection['detected']:
                    last_detection_time = time.time()
                    
                    # Control hardware if enabled
                    if control_hardware and self.motor and self.servo:
                        angle = detection['angle']
                        distance = detection['distance_m']
                        is_centered = detection['is_centered']
                        
                        # Check if we should stop (too close)
                        if distance and distance < self.stop_distance_m:
                            logger.info(f"Marker reached! Distance: {distance:.2f}m < {self.stop_distance_m}m")
                            self.motor.stop()
                            self.servo.center()
                            
                            if display:
                                annotated = self.tag_detector.draw_overlay(frame, detection)
                                cv2.putText(annotated, "STOPPED - Marker reached!", (10, 180),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.imshow('ArUco Navigation', annotated)
                                cv2.waitKey(0)  # Wait for keypress
                            break
                        
                        # Set servo angle (same logic as main.py)
                        self.servo.set_angle(angle)
                        
                        # Adjust motor speed based on centering (same logic as main.py)
                        if is_centered:
                            speed = config.MOTOR_FAST
                        else:
                            speed = config.MOTOR_MEDIUM
                        
                        self.motor.forward(speed)
                        
                        # Log navigation info
                        logger.debug(f"Marker detected: angle={angle:.1f}°, "
                                   f"distance={distance:.2f}m, centered={is_centered}")
                    else:
                        # Just log detection info
                        logger.info(f"Marker detected: ID={detection['tag_id']}, "
                                  f"angle={detection['angle']:.1f}°, "
                                  f"distance={detection['distance_m']:.2f}m")
                else:
                    # Marker not detected
                    if control_hardware and self.motor and self.servo:
                        # Check if marker has been lost for too long
                        if time.time() - last_detection_time > tag_lost_timeout:
                            logger.warning("Marker lost for too long, stopping")
                            self.motor.stop()
                            self.servo.center()
                        else:
                            # Slow search movement
                            self.motor.forward(config.MOTOR_SLOW)
                            self.servo.center()
                
                # Display frame if requested
                if display:
                    annotated = self.tag_detector.draw_overlay(frame, detection)
                    
                    # Add FPS counter
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps = frame_count / elapsed
                        fps_text = f'FPS: {fps:.1f}'
                        cv2.putText(annotated, fps_text,
                                   (annotated.shape[1] - 150, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('ArUco Navigation', annotated)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested by user")
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            log_error(logger, e, "Error in ArUco navigation test")
        finally:
            # Cleanup
            if control_hardware and self.motor:
                self.motor.stop()
            if control_hardware and self.servo:
                self.servo.center()
            
            if self.picam2 is not None:
                self.picam2.stop()
            if self.webcam is not None:
                self.webcam.release()
            
            if display:
                cv2.destroyAllWindows()
            
            logger.info("Test complete")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.motor:
            self.motor.cleanup()
        if self.servo:
            self.servo.cleanup()
        if self.picam2 is not None:
            self.picam2.stop()
        if self.webcam is not None:
            self.webcam.release()


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ArUco marker detection and navigation')
    parser.add_argument('--tag-size', type=float, default=0.047,
                       help='Physical marker size in meters (default: 0.047 = 47mm)')
    parser.add_argument('--stop-distance', type=float, default=0.3,
                       help='Stop distance in meters (default: 0.3)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--no-control', action='store_true',
                       help='Disable motor/servo control (test detection only)')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of Picamera2 (auto-detected if Picamera2 unavailable)')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Webcam index (default: 0)')
    parser.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                       help='Rotate camera feed (0, 90, 180, 270 degrees) - useful if camera is upside down')
    parser.add_argument('--flip-h', action='store_true',
                       help='Flip camera feed horizontally')
    parser.add_argument('--flip-v', action='store_true',
                       help='Flip camera feed vertically')
    parser.add_argument('--debug-detection', action='store_true',
                       help='Show debug info about detection failures')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ArUco Marker Detection and Navigation Test")
    print("=" * 70)
    print(f"Marker size: {args.tag_size}m")
    print(f"Stop distance: {args.stop_distance}m")
    print(f"Display: {not args.no_display}")
    print(f"Hardware control: {not args.no_control}")
    print(f"Camera: {'Webcam' if (args.webcam or not HAS_PICAMERA2) else 'Picamera2'}")
    if args.webcam or not HAS_PICAMERA2:
        print(f"Webcam index: {args.camera_index}")
    if args.rotate != 0:
        print(f"Camera rotation: {args.rotate}°")
    if args.flip_h:
        print("Horizontal flip: enabled")
    if args.flip_v:
        print("Vertical flip: enabled")
    print("Press 'q' to quit")
    print("=" * 70)
    print()
    
    test = ArUcoNavigationTest(
        tag_size_m=args.tag_size,
        stop_distance_m=args.stop_distance,
        use_webcam=args.webcam or not HAS_PICAMERA2,
        camera_index=args.camera_index,
        test_rotation=args.rotate,
        test_flip_horizontal=args.flip_h,
        test_flip_vertical=args.flip_v
    )
    
    # Store debug flag for detection
    test.debug_detection = args.debug_detection
    
    try:
        test.run_test(
            display=not args.no_display,
            control_hardware=not args.no_control
        )
    finally:
        test.cleanup()


if __name__ == '__main__':
    main()

