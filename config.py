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

MOTOR_STOP = 99.9      # 0% duty cycle = stopped
MOTOR_SLOW = 98.0       # 30% duty cycle = slow speed
MOTOR_MAX = 92.7        # 100% duty cycle = maximum speed

# Servo Control Values (PWM duty cycle percentages)
PWM_FREQUENCY_SERVO = 50 #HZ

SERVO_CENTER = 92.675   # 7.5% = center position (straight)
SERVO_LEFT_MAX = 95.422  # 5% = full left
SERVO_RIGHT_MAX = 89.318 # 10% = full right

# Visual Detection Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_MODEL = 'yolo11n.pt'  # YOLO nano model for speed
YOLO_CONFIDENCE = 0.25
PERSON_CENTER_THRESHOLD = 30  # Pixels from center to consider "centered"
ANGLE_TO_STEERING_GAIN = 0.5  # How much to turn based on angle

# Wake Word Configuration
WAKE_WORD_MODEL_PATH = 'bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn'
WAKE_WORD_ACCESS_KEY = None  # Set in .env file or environment variable

# Voice Recognition (Manual Mode)
OPENAI_API_KEY = None  # Set in .env file or environment variable
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for command interpretation (gpt-4o-mini is fast/cheap)
VOICE_COMMAND_TIMEOUT = 5.0  # Seconds to wait for voice command


# Safety Configuration
EMERGENCY_STOP_ENABLED = True  # Enable TOF emergency stop

# Debug Configuration
DEBUG_MODE = True  # Enable debug logging throughout system
DEBUG_VISUAL = True  # Debug visual detection specifically
DEBUG_MOTOR = True  # Debug motor commands
DEBUG_SERVO = True  # Debug servo commands
DEBUG_TOF = True  # Debug TOF sensor readings
DEBUG_VOICE = True  # Debug voice recognition
DEBUG_STATE = True  # Debug state machine transitions

