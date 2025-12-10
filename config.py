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

SERVO_CENTER = 92.675   # 7.5% = center position (straight)
SERVO_LEFT_MAX = 95.422  # 5% = full left
SERVO_RIGHT_MAX = 89.318 # 10% = full right

# Visual Detection Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_ROTATION = 180  # Rotate camera 180 degrees (0, 90, 180, 270) - set to 180 if camera is upside down
CAMERA_FLIP_HORIZONTAL = False  # Flip horizontally (mirror)
CAMERA_FLIP_VERTICAL = False  # Flip vertically
CAMERA_SWAP_RB = True  # Swap red and blue channels (fixes color swap issue)
CAMERA_SWAP_LEFT_RIGHT = True  # Swap left/right arm detection (needed when camera is rotated 180°)
YOLO_MODEL = 'yolo11n.pt'  # YOLO nano model for speed (object detection)
YOLO_POSE_MODEL = 'yolo11n-pose.pt'  # YOLO pose model (for pose estimation + tracking)
YOLO_CONFIDENCE = 0.25
PERSON_CENTER_THRESHOLD = 30  # Pixels from center to consider "centered"
ANGLE_TO_STEERING_GAIN = 0.5  # How much to turn based on angle
TRACKING_TIMEOUT = 30.0  # Seconds before returning to idle if no user detected

# Motor Speed Configuration
FOLLOW_SPEED = 0.6  # Speed when following user (0.0-1.0)
MOTOR_SLOW = 0.3  # Slow speed for turning
MOTOR_MEDIUM = 0.5  # Medium speed
MOTOR_FAST = 0.8  # Fast speed

# Wake Word Configuration
WAKE_WORD_MODEL_PATH = 'bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
WAKE_WORD_ACCESS_KEY = None  # Set in .env file or environment variable

# Voice Recognition (Manual Mode)
OPENAI_API_KEY = None  # Set in .env file or environment variable
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for command interpretation (gpt-4o-mini is fast/cheap)
VOICE_COMMAND_TIMEOUT = 5.0  # Seconds to wait for voice command


# Safety Configuration
EMERGENCY_STOP_ENABLED = True  # Enable TOF emergency stop
TOF_STOP_DISTANCE_MM = 300  # Stop when within 30cm
TOF_EMERGENCY_DISTANCE_MM = 100  # Emergency stop when within 10cm

# Hand Gesture Configuration
HAND_GESTURE_HOLD_TIME = 0.5  # Seconds gesture must be held before executing
HAND_MODEL_PATH = None  # Path to hand keypoints model (if trained), None to use pose model

# Debug Configuration
DEBUG_MODE = True  # Enable debug logging throughout system
DEBUG_VISUAL = True  # Debug visual detection specifically
DEBUG_MOTOR = True  # Debug motor commands
DEBUG_SERVO = True  # Debug servo commands
DEBUG_TOF = True  # Debug TOF sensor readings
DEBUG_VOICE = True  # Debug voice recognition
DEBUG_STATE = True  # Debug state machine transitions

