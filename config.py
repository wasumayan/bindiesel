"""
Configuration file for Bin Diesel system
Modify GPIO pins, PWM values, and thresholds here
"""
import os

USE_GPIO = True    # disables GPIO for local testing


# GPIO Pin Assignments
MOTOR_PWM_PIN = 12  # GPIO12 pin 32 motor speed control
SERVO_PWM_PIN = 13  # GPIO13 pin 33 for servo steering control
ToF_DIGITAL_PIN = 23 # GPIO23 pin 16 
ToF_ACTIVE_HIGH = True 

# State machine behavior timing
STOP_SECONDS = 4.0      # pause time at user
RETURN_MARGIN = 0.5     # buffer for return time 


# Motor Control Values (PWM duty cycle percentages
PWM_FREQUENCY_MOTOR = 40  #Hz 
PWM_FREQUENCY = 40  # Alias for motor frequency

MOTOR_STOP = 100.0      # 0% duty cycle = stopped
MOTOR_MAX = 76.0       # 100% duty cycle = maximum speed 93.7 

# Motor Speed Configuration
FOLLOW_SPEED = 1.0  # Speed when following user (0.0-1.0)
MOTOR_SUPER_SLOW = 1.07  # Super slow speed for fine adjustments
MOTOR_SLOW = 1.05  # low speed for turning
MOTOR_MEDIUM = 1.02  # Medium speed
MOTOR_FAST = 1.0  # Fast speed
MOTOR_TURN = 0.91

# Servo Control Values (PWM duty cycle percentages)
PWM_FREQUENCY_SERVO = 50 #HZ

SERVO_CENTER = 92.600
SERVO_LEFT_MAX = 95.422 
SERVO_RIGHT_MAX = 89.318 

# Visual Detection Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30  # Target camera FPS (Raspberry Pi Camera Module 3 Wide supports up to 50 FPS at 640x480)
CAMERA_ROTATION = 180  # Rotate camera 180 degrees (0, 90, 180, 270) - set to 180 if camera is upside down
CAMERA_FLIP_HORIZONTAL = False  # Flip horizontally (mirror)
CAMERA_FLIP_VERTICAL = False  # Flip vertically
CAMERA_SWAP_RB = True  # Swap red and blue channels (fixes color swap issue)
CAMERA_SWAP_LEFT_RIGHT = True  # Swap left/right arm detection (needed when camera is rotated 180°)
# YOLO Model Configuration
# Use NCNN format for better performance on Raspberry Pi (ARM architecture)
# Run convert_to_ncnn.py to convert PyTorch models to NCNN format
USE_NCNN = True  # Set to False to use PyTorch models (.pt) instead of NCNN

# Model paths (will use NCNN if USE_NCNN=True and NCNN model exists)
YOLO_MODEL = 'yolo11n_ncnn_model' if USE_NCNN else 'yolo11n.pt'  # Object detection model
YOLO_POSE_MODEL = 'yolo11n-pose_ncnn_model' if USE_NCNN else 'yolo11n-pose.pt'  # Pose estimation + tracking
YOLO_OBB_MODEL = 'yolo11n-obb_ncnn_model' if USE_NCNN else 'yolo11n-obb.pt'  # Oriented bounding boxes
YOLO_CLOTHING_MODEL = None  # Path to trained clothing detection model (for RADD mode)
                                  # See: https://github.com/kesimeg/YOLO-Clothing-Detection
                                  # Set to model path after training/downloading
                                  # Example: 'models/clothing/best.pt' or 'models/clothing/best_ncnn_model'
RADD_VIOLATION_TIMEOUT = 2.0  # Seconds before removing violator from tracking if not seen
YOLO_HAND_MODEL = 'models/hand_keypoints/weights/best.pt'  # Hand keypoints for manual mode
                                  # Note: Custom trained models need to be converted separately
                                  # Example: 'models/hand_keypoints/weights/best_ncnn_model'
HAND_MODEL_PATH = YOLO_HAND_MODEL  # Alias for compatibility
HAND_GESTURE_HOLD_TIME = 0.5  # Seconds gesture must be held before executing
YOLO_CONFIDENCE = 0.01  # Very low confidence for arm detection (confidence boundaries removed)
YOLO_PERSON_CONFIDENCE = 0.3  # Minimum confidence for person detection (higher threshold to avoid false positives)
PERSON_CENTER_THRESHOLD = 30  # Pixels from center to consider "centered"
ANGLE_TO_STEERING_GAIN = 0.65  # How much to turn based on angle
TRACKING_TIMEOUT = 30.0  # Seconds before returning to idle if no user detected

# Arm Angle Detection Configuration
ARM_ANGLE_MIN = 60.0  # Minimum angle from vertical (degrees) - 0° = straight down, 90° = T-pose (horizontal)
ARM_ANGLE_MAX = 90.0  # Maximum angle from vertical (degrees) - 0° = straight down, 90° = T-pose (horizontal)
ARM_KEYPOINT_CONFIDENCE = 0.01  # Minimum keypoint confidence (very low to accept all detections)
ARM_MIN_HORIZONTAL_EXTENSION = 30  # Minimum horizontal extension in pixels
ARM_HORIZONTAL_RATIO = 0.8  # Minimum horizontal/vertical ratio (0.8 = more horizontal)
ARM_WRIST_ABOVE_SHOULDER_TOLERANCE = 30  # Pixels tolerance for wrist above shoulder
ARM_ELBOW_ANGLE_MIN = 45.0  # Minimum elbow bend angle (degrees)
ARM_ELBOW_ANGLE_MAX = 160.0  # Maximum elbow bend angle (degrees)

# Wake Word Configuration
WAKE_WORD_MODEL_PATH = 'bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
WAKE_WORD_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY')  # Read from environment / .env

# Voice Recognition (Manual Mode)
OPENAI_API_KEY = None  # Set in .env file or environment variable
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for command interpretation (gpt-4o-mini is fast/cheap)
VOICE_COMMAND_TIMEOUT = 5.0  # Seconds to wait for voice command


# Safety Configuration
EMERGENCY_STOP_ENABLED = False  # Enable TOF emergency stop
TOF_STOP_DISTANCE_MM = 200  # Stop when within 30cm
TOF_LOW_VOLTAGE_THRESHOLD = 2.2  # Voltage threshold below which signal is treated as LOW (V) - note: GPIO can't read exact voltage, this is for reference
TOF_DEBOUNCE_TIME = 0.05  # Debounce time in seconds (50ms) to filter noise
TOF_HIGH_COUNT_THRESHOLD = 1  # Number of consecutive HIGH readings required to trigger (set to 1 for immediate response; use 2-3 if noise is an issue)

# Home Marker Configuration (for return to home)
# ArUco Marker Configuration
ARUCO_TAG_SIZE_M = 0.2  # Physical size of ArUco marker in meters (default: 0.047 = 47mm)
HOME_MARKER_STOP_DISTANCE_M = 0.4  # Stop when marker is this close in meters (default: 0.3m = 30cm)

# Legacy YOLO-based detection (deprecated - using ArUco now, but still used by test_home_tracking.py)
HOME_MARKER_OBJECT_CLASS = 'box'  # YOLO object class to detect as home marker (deprecated)
HOME_MARKER_COLOR = 'red'  # Color of home marker (deprecated - using ArUco now)
HOME_MARKER_STOP_DISTANCE = 100  # Stop when marker is this many pixels wide (deprecated - use HOME_MARKER_STOP_DISTANCE_M)
HOME_MARKER_SLOW_DISTANCE = 50  # Slow down when marker is this many pixels wide (for YOLO-based detection in test_home_tracking.py)
TURN_180_DURATION = 3.4  # Seconds to turn 180 degrees

SLEEP_TIMER = 0.15

# Performance Configuration
ENABLE_FRAME_CACHING = True  # Cache frames to reduce redundant captures
FRAME_CACHE_TTL = 0.05  # Frame cache time-to-live (seconds)
VISUAL_UPDATE_INTERVAL = 0.033  # Visual detection update interval (seconds) - lower = higher FPS (0.033 = 30 FPS max, 0.05 = 20 FPS max)
ENABLE_PERFORMANCE_MONITORING = True  # Track FPS and performance metrics
FRAME_SKIP_INTERVAL = 3  # Process every Nth frame (1 = all frames, 2 = every other, etc.)

# YOLO Performance Optimization
YOLO_INFERENCE_SIZE = 640  # YOLO input image size (matches camera 640x480, no resize needed)
YOLO_MAX_DET = 5  # Maximum detections per image (lower = faster, default is 300)
YOLO_AGNOSTIC_NMS = True  # Class-agnostic NMS (faster, slight accuracy tradeoff)
# Note: imgsz resizes internally - doesn't reduce field of view, but resizing has CPU overhead
# Better to match camera resolution (640) and use other optimizations:
# - max_det: Limits detections (biggest speedup)
# - agnostic_nms: Faster NMS processing
# - frame skipping: Process every Nth frame

# Debug Configuration
DEBUG_MODE = True  # Enable debug logging throughout system
DEBUG_VISUAL = True  # Debug visual detection specifically
DEBUG_MOTOR = True  # Debug motor commands
DEBUG_SERVO = False  # Debug servo commands
DEBUG_TOF = False  # Debug TOF sensor readings
DEBUG_VOICE = True  # Debug voice recognition
DEBUG_STATE = True  # Debug state machine transitions
DEBUG_PERFORMANCE = False  # Debug performance metrics

