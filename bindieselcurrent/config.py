"""
Configuration file for Bin Diesel system
Modify GPIO pins, PWM values, and thresholds here
"""

# GPIO Pin Assignments (BCM numbering)
MOTOR_PWM_PIN = 18  # GPIO 18 for motor speed control
SERVO_PWM_PIN = 19  # GPIO 19 for servo steering control

# I2C pins for TOF sensor (default Raspberry Pi I2C)
# SDA: GPIO 2 (Physical pin 3)
# SCL: GPIO 3 (Physical pin 5)

# PWM Configuration
PWM_FREQUENCY = 50  # Hz (standard for servos, also works for motors)

# Motor Control Values (PWM duty cycle percentages)
# These are PLACEHOLDER values - adjust based on your motor controller
MOTOR_STOP = 0.0      # 0% duty cycle = stopped
MOTOR_SLOW = 0.3       # 30% duty cycle = slow speed
MOTOR_MEDIUM = 0.5     # 50% duty cycle = medium speed
MOTOR_FAST = 0.7       # 70% duty cycle = fast speed
MOTOR_MAX = 1.0        # 100% duty cycle = maximum speed

# Servo Control Values (PWM duty cycle percentages)
# These are PLACEHOLDER values - adjust based on your servo
# Typical servo range: 2.5% (0°) to 12.5% (180°)
SERVO_CENTER = 0.075   # 7.5% = center position (straight)
SERVO_LEFT_MAX = 0.05  # 5% = full left
SERVO_RIGHT_MAX = 0.10 # 10% = full right
SERVO_LEFT_SLIGHT = 0.065  # Slight left turn
SERVO_RIGHT_SLIGHT = 0.085 # Slight right turn

# TOF Sensor Configuration
TOF_STOP_DISTANCE_MM = 80  # Stop when object is 8cm (80mm) away
TOF_EMERGENCY_DISTANCE_MM = 70  # Emergency stop at 7cm (70mm)
TOF_READ_INTERVAL = 0.1  # Read sensor every 100ms

# Visual Detection Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_MODEL = 'yolo11n.pt'  # YOLO nano model for speed
YOLO_CONFIDENCE = 0.25
PERSON_CENTER_THRESHOLD = 30  # Pixels from center to consider "centered"
ANGLE_TO_STEERING_GAIN = 0.5  # How much to turn based on angle

# Wake Word Configuration
WAKE_WORD_MODEL_PATH = '../bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
WAKE_WORD_ACCESS_KEY = None  # Set in .env file or environment variable

# Voice Recognition (Manual Mode)
OPENAI_API_KEY = None  # Set in .env file or environment variable
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for command interpretation (gpt-4o-mini is fast/cheap)
VOICE_COMMAND_TIMEOUT = 5.0  # Seconds to wait for voice command

# State Machine Configuration
TRACKING_TIMEOUT = 30.0  # Seconds before returning to idle if no user detected
RETURN_SPEED = MOTOR_MEDIUM  # Speed when returning to start
FOLLOW_SPEED = MOTOR_SLOW    # Speed when following user

# Safety Configuration
EMERGENCY_STOP_ENABLED = True  # Enable TOF emergency stop
MAX_SPEED_LIMIT = MOTOR_MEDIUM  # Maximum allowed speed (safety limit)

# Debug Configuration
DEBUG_MODE = True  # Enable debug logging throughout system
DEBUG_VISUAL = True  # Debug visual detection specifically
DEBUG_MOTOR = True  # Debug motor commands
DEBUG_SERVO = True  # Debug servo commands
DEBUG_TOF = True  # Debug TOF sensor readings
DEBUG_VOICE = True  # Debug voice recognition
DEBUG_STATE = True  # Debug state machine transitions

