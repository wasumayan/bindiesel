# Code Structure Overview

## Main Entry Points

### 1. `main.py` - **PRIMARY MAIN FILE** ⭐
**Purpose**: Full-featured main control system with all features

**Key Features**:
- Complete state machine (IDLE → ACTIVE → TRACKING → FOLLOWING → STOPPED → RETURNING → MANUAL_MODE)
- Wake word detection ("bin diesel")
- Visual detection (person tracking, arm raising)
- Voice commands (manual mode)
- Path tracking (records path for return navigation)
- TOF sensor (emergency stop)
- Motor and servo control

**States Handled**:
- `IDLE`: Waiting for wake word
- `ACTIVE`: Awake, waiting for mode selection
- `TRACKING_USER`: Detecting and tracking user
- `FOLLOWING_USER`: Moving toward user
- `STOPPED`: At target distance, waiting
- `RETURNING_TO_START`: Executing reverse path
- `MANUAL_MODE`: Voice command control

**Usage**: `python main.py`

---

### 2. `main_statecontrol.py` - **SIMPLIFIED VERSION**
**Purpose**: Minimal version with basic functionality

**Key Features**:
- Simplified state machine (IDLE → DRIVING_TO_USER → STOPPED_AT_USER → RETURNING)
- Wake word detection
- Visual detection
- TOF sensor
- Motor and servo control
- **No voice commands, no path tracking**

**Usage**: `python main_statecontrol.py`

**When to use**: Testing basic functionality, simpler debugging

---

## Core Modules

### 3. `config.py` - **Configuration Hub**
**Purpose**: Central configuration file for all system parameters

**Contains**:
- GPIO pin assignments (motor, servo, TOF)
- PWM values (motor speeds, servo positions)
- Camera settings (width, height, YOLO model)
- Wake word settings
- Voice recognition settings (OpenAI API)
- Debug flags
- Safety thresholds

**Key Settings**:
```python
MOTOR_PWM_PIN = 12
SERVO_PWM_PIN = 13
YOLO_MODEL = 'yolo11n.pt'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

**Usage**: Imported by all modules, modify values here to tune system

---

### 4. `state_machine.py` - **State Management**
**Purpose**: Manages system states and transitions

**Current States** (simplified version):
- `IDLE`
- `DRIVING_TO_USER`
- `STOPPED_AT_USER`
- `RETURNING`

**Note**: `main.py` uses a more complex state machine with additional states

**Key Methods**:
- `get_state()`: Get current state
- `transition_to(new_state)`: Change state
- `get_time_in_state()`: How long in current state

---

### 5. `visual_detector.py` - **Person Detection & Tracking**
**Purpose**: Detects person, tracks position, detects arm raising

**Technology**: YOLO11 (object detection)

**Key Features**:
- Person detection using YOLO
- Arm raising detection (custom algorithm)
- Angle calculation (person position relative to camera)
- Centering detection

**Key Methods**:
- `update()`: Main detection loop, returns detection results
- `detect_person()`: Find person in frame
- `detect_arm_raised()`: Check if arm is raised
- `calculate_angle()`: Calculate person angle
- `is_person_centered()`: Check if person is centered

**Returns**:
```python
{
    'person_detected': bool,
    'person_box': (x1, y1, x2, y2),
    'angle': float,
    'is_centered': bool,
    'arm_raised': bool,
    'arm_confidence': float
}
```

---

### 6. `motor_controller.py` - **Motor Speed Control**
**Purpose**: Controls motor speed via PWM

**Key Methods**:
- `forward(speed)`: Move forward (speed: 0.0-1.0)
- `stop()`: Stop motor
- `cleanup()`: Cleanup GPIO

**PWM Configuration**:
- Pin: `config.MOTOR_PWM_PIN` (GPIO 12)
- Frequency: `config.PWM_FREQUENCY_MOTOR` (40 Hz)
- Duty cycle: Inverted if `config.PWM_INVERT = True`

**Note**: PWM values go through inversion circuit (see config)

---

### 7. `servo_controller.py` - **Steering Control**
**Purpose**: Controls steering servo via PWM

**Key Methods**:
- `set_angle(angle)`: Set steering angle (-45° to +45°)
- `center()`: Center steering
- `turn_left(amount)`: Turn left (0.0-1.0)
- `turn_right(amount)`: Turn right (0.0-1.0)
- `cleanup()`: Cleanup GPIO

**PWM Configuration**:
- Pin: `config.SERVO_PWM_PIN` (GPIO 13)
- Frequency: `config.PWM_FREQUENCY_SERVO` (50 Hz)
- Center: `config.SERVO_CENTER` (92.675%)
- Left max: `config.SERVO_LEFT_MAX` (95.422%)
- Right max: `config.SERVO_RIGHT_MAX` (89.318%)

**Note**: PWM values go through inversion circuit

---

### 8. `wake_word_detector.py` - **Wake Word Detection**
**Purpose**: Detects "bin diesel" wake word

**Technology**: Picovoice Porcupine

**Key Methods**:
- `start_listening()`: Start listening for wake word
- `detect()`: Check if wake word detected (non-blocking)
- `stop()`: Stop listening

**Configuration**:
- Model: `bin-diesel_en_raspberry-pi_v3_0_0.ppn`
- Access key: `config.WAKE_WORD_ACCESS_KEY` (from .env)

---

### 9. `voice_recognizer.py` - **Voice Commands**
**Purpose**: Recognizes voice commands for manual mode

**Technology**: Speech Recognition + OpenAI GPT

**Key Methods**:
- `recognize_command(timeout)`: Listen for and interpret voice command
- `interpret_command_with_gpt(text)`: Use GPT to interpret command

**Commands Supported**:
- FORWARD, LEFT, RIGHT, STOP, TURN_AROUND
- AUTOMATIC_MODE, MANUAL_MODE

**Configuration**:
- API key: `config.OPENAI_API_KEY` (from .env)
- Model: `config.OPENAI_MODEL` (gpt-4o-mini)

---

### 10. `tof_sensor.py` - **Distance Sensor**
**Purpose**: Measures distance using VL53L0X TOF sensor

**Key Methods**:
- `read_distance()`: Read distance in mm
- `is_too_close()`: Check if too close (stop distance)
- `is_emergency_stop()`: Check if emergency stop needed

**Configuration**:
- Stop distance: `config.TOF_STOP_DISTANCE_MM`
- Emergency distance: `config.TOF_EMERGENCY_DISTANCE_MM`

---

### 11. `path_tracker.py` - **Path Recording**
**Purpose**: Records movement path for return navigation

**Key Methods**:
- `start_tracking()`: Start recording path
- `add_segment(speed, steering, duration)`: Record movement segment
- `get_reverse_path()`: Get reverse path for return
- `stop_tracking()`: Stop recording

**Usage**: Records path during FOLLOWING_USER state, plays back in reverse during RETURNING state

---

## New YOLO-Based Files

### 12. `test_yolo_pose_tracking.py` - **YOLO Pose + Tracking Test**
**Purpose**: Test YOLO pose detection with built-in tracking

**Features**:
- YOLO11 pose model (17 keypoints)
- BYTETracker for multi-person tracking
- Arm angle detection (60-90 degrees)
- Hand gesture recognition from pose
- Real-time visualization

**Usage**: `python test_yolo_pose_tracking.py --fps`

**Technology**: Single YOLO model (replaces YOLO + MediaPipe)

---

### 13. `hand_gesture_controller.py` - **Hand Gesture Control**
**Purpose**: Hand gesture recognition for manual mode

**Features**:
- Hand keypoints model (21 keypoints per hand) - if trained
- Falls back to pose model (17 body keypoints)
- Gesture hold time (prevents accidental commands)
- Non-blocking detection

**Gestures**:
- STOP, TURN_LEFT, TURN_RIGHT, FORWARD, TURN_AROUND

**Usage**: 
```python
from hand_gesture_controller import HandGestureController
controller = HandGestureController()
command = controller.detect_command(frame)
```

**Integration**: See `manual_mode_with_gestures.py` for integration example

---

### 14. `test_mediapipe_combined.py` - **Legacy Combined Test**
**Purpose**: Combined YOLO + MediaPipe test (legacy)

**Note**: This uses MediaPipe (requires Python 3.12). Consider using `test_yolo_pose_tracking.py` instead for better performance.

---

## Test Files

### 15. `test_visual_detection.py`
Tests visual detector (person detection, arm raising)

### 16. `test_motor.py`
Tests motor controller (speed control)

### 17. `test_servo.py`
Tests servo controller (steering)

### 18. `test_tof.py`
Tests TOF sensor (distance measurement)

### 19. `test_voice_commands.py`
Tests voice recognition

### 20. `test_full_system.py`
Tests complete system integration

---

## Data Flow

```
Wake Word → State Machine (IDLE → ACTIVE)
    ↓
Visual Detector → Person Detection → Arm Raised?
    ↓
State Machine (ACTIVE → TRACKING → FOLLOWING)
    ↓
Motor/Servo Controllers → Movement
    ↓
TOF Sensor → Emergency Stop Check
    ↓
Path Tracker → Record Movement
    ↓
State Machine (STOPPED → RETURNING)
    ↓
Path Tracker → Reverse Path → Return to Start
```

## Manual Mode Flow

```
Wake Word → State Machine (IDLE → ACTIVE)
    ↓
Voice: "manual mode" → State Machine (ACTIVE → MANUAL_MODE)
    ↓
Voice Recognizer OR Hand Gesture Controller → Command
    ↓
Motor/Servo Controllers → Execute Command
```

---

## Key Differences: main.py vs main_statecontrol.py

| Feature | main.py | main_statecontrol.py |
|---------|---------|----------------------|
| **States** | 7 states (full workflow) | 4 states (simplified) |
| **Voice Commands** | ✅ Yes | ❌ No |
| **Path Tracking** | ✅ Yes | ❌ No |
| **Manual Mode** | ✅ Yes | ❌ No |
| **Return Navigation** | ✅ Yes | ❌ No |
| **Complexity** | High | Low |

**Recommendation**: Use `main.py` for full system, `main_statecontrol.py` for basic testing

---

## Integration Status

### ✅ Implemented
- YOLO pose tracking (`test_yolo_pose_tracking.py`)
- Hand gesture controller (`hand_gesture_controller.py`)
- Tracking mode integration

### ⏳ Not Yet Integrated
- YOLO pose tracking in `main.py` (still uses `visual_detector.py`)
- Hand gestures in manual mode (see `manual_mode_with_gestures.py` for example)

### 🔄 Migration Path
1. Replace `visual_detector.py` with YOLO pose tracking
2. Add hand gesture controller to `main.py`
3. Update state handlers to use new detection methods

---

## Quick Reference

**Run full system**: `python main.py`  
**Run simplified**: `python main_statecontrol.py`  
**Test pose tracking**: `python test_yolo_pose_tracking.py --fps`  
**Test hand gestures**: `python hand_gesture_controller.py`  
**Configure system**: Edit `config.py`

