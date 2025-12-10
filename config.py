"""
Configuration file for Bin Diesel system
Modify GPIO pins, PWM values, and thresholds here
"""

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
MOTOR_MAX = 92.7        # 100% duty cycle = maximum speed

# Servo Control Values (PWM duty cycle percentages)
PWM_FREQUENCY_SERVO = 50 #HZ

SERVO_CENTER = 92.600  
SERVO_LEFT_MAX = 95.422 
SERVO_RIGHT_MAX = 89.318 

# Visual Detection Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 120  # Target camera FPS (Raspberry Pi Camera Module 3 Wide supports up to 50 FPS at 640x480)
CAMERA_ROTATION = 180  # Rotate camera 180 degrees (0, 90, 180, 270) - set to 180 if camera is upside down
CAMERA_FLIP_HORIZONTAL = False  # Flip horizontally (mirror)
CAMERA_FLIP_VERTICAL = False  # Flip vertically
CAMERA_SWAP_RB = True  # Swap red and blue channels (fixes color swap issue)
CAMERA_SWAP_LEFT_RIGHT = True  # Swap left/right arm detection (needed when camera is rotated 180°)
YOLO_MODEL = 'yolo11n.pt'  # YOLO nano model for speed (object detection)
YOLO_POSE_MODEL = 'yolo11n-pose.pt'  # YOLO pose model (for pose estimation + tracking)
YOLO_OBB_MODEL = 'yolo11n-obb.pt'  # YOLO OBB model (oriented bounding boxes for trash detection)
YOLO_CLOTHING_MODEL = None  # Path to trained clothing detection model (for RADD mode)
                                  # See: https://github.com/kesimeg/YOLO-Clothing-Detection
                                  # Set to model path after training/downloading
                                  # Example: 'models/clothing/best.pt'
RADD_VIOLATION_TIMEOUT = 2.0  # Seconds before removing violator from tracking if not seen
YOLO_HAND_MODEL = 'models/hand_keypoints/weights/best.pt' #YOLO for hand keypoints for manual mode!
HAND_MODEL_PATH = YOLO_HAND_MODEL  # Alias for compatibility
HAND_GESTURE_HOLD_TIME = 0.5  # Seconds gesture must be held before executing
YOLO_CONFIDENCE = 0.01  # Very low confidence to detect everything (confidence boundaries removed)
PERSON_CENTER_THRESHOLD = 30  # Pixels from center to consider "centered"
ANGLE_TO_STEERING_GAIN = 0.5  # How much to turn based on angle
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

# Motor Speed Configuration
FOLLOW_SPEED = 1.0  # Speed when following user (0.0-1.0)
MOTOR_SLOW = 0.3  # Slow speed for turning
MOTOR_MEDIUM = 0.5  # Medium speed
MOTOR_FAST = 1.0  # Fast speed

# Wake Word Configuration
WAKE_WORD_MODEL_PATH = 'bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
WAKE_WORD_ACCESS_KEY = None  # Set in .env file or environment variable

# Voice Recognition (Manual Mode)
OPENAI_API_KEY = None  # Set in .env file or environment variable
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for command interpretation (gpt-4o-mini is fast/cheap)
VOICE_COMMAND_TIMEOUT = 5.0  # Seconds to wait for voice command


# Safety Configuration
EMERGENCY_STOP_ENABLED = False  # Enable TOF emergency stop
TOF_STOP_DISTANCE_MM = 200  # Stop when within 30cm
TOF_EMERGENCY_DISTANCE_MM = 100  # Emergency stop when within 10cm

# Performance Configuration
ENABLE_FRAME_CACHING = True  # Cache frames to reduce redundant captures
FRAME_CACHE_TTL = 0.05  # Frame cache time-to-live (seconds)
VISUAL_UPDATE_INTERVAL = 0.05  # Visual detection update interval (seconds) - lower = higher FPS (0.05 = 20 FPS max, 0.033 = 30 FPS max)
ENABLE_PERFORMANCE_MONITORING = True  # Track FPS and performance metrics
FRAME_SKIP_INTERVAL = 1  # Process every Nth frame (1 = all frames, 2 = every other, etc.)

# Debug Configuration
DEBUG_MODE = False  # Enable debug logging throughout system
DEBUG_VISUAL = True  # Debug visual detection specifically
DEBUG_MOTOR = True  # Debug motor commands
DEBUG_SERVO = True  # Debug servo commands
DEBUG_TOF = True  # Debug TOF sensor readings
DEBUG_VOICE = True  # Debug voice recognition
DEBUG_STATE = True  # Debug state machine transitions
DEBUG_PERFORMANCE = False  # Debug performance metrics

