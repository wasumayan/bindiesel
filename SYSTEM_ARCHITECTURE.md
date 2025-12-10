# Bin Diesel System Architecture - Complete Technical Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [State Machine Deep Dive](#state-machine-deep-dive)
3. [Main Control Loop](#main-control-loop)
4. [Component Interactions](#component-interactions)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [State Handlers Explained](#state-handlers-explained)
7. [Error Handling & Robustness](#error-handling--robustness)
8. [Configuration System](#configuration-system)

---

## System Overview

### Architecture Pattern
The system uses a **State Machine Pattern** with **Event-Driven Control Loop**:
- **Central Controller**: `main.py` (BinDieselSystem class)
- **State Manager**: `state_machine.py` (StateMachine class)
- **Modular Components**: Each subsystem is independent and communicates via defined interfaces

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    BinDieselSystem                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ StateMachine │  │  PathTracker  │  │   Config     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ WakeWord     │  │ YOLOPose     │  │   Voice      │  │
│  │ Detector     │  │ Tracker      │  │ Recognizer   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ HandGesture  │  │     TOF      │  │   Motor      │  │
│  │ Controller   │  │   Sensor     │  │ Controller   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────┐                                        │
│  │   Servo      │                                        │
│  │ Controller   │                                        │
│  └──────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

---

## State Machine Deep Dive

### State Definitions

```python
State.IDLE              # Waiting for wake word
State.ACTIVE            # Awake, waiting for mode selection
State.TRACKING_USER     # Detecting and tracking user
State.FOLLOWING_USER    # Moving toward user
State.STOPPED           # At target distance, waiting
State.RETURNING_TO_START # Navigating back using reverse path
State.MANUAL_MODE       # Voice/gesture control mode
```

### State Transition Diagram

```
                    ┌─────────┐
                    │  IDLE   │◄─────────────────────────────┐
                    └────┬────┘                              │
                         │ Wake Word Detected                │
                         ▼                                   │
                    ┌─────────┐                             │
                    │ ACTIVE  │                             │
                    └────┬────┘                             │
                         │                                  │
         ┌───────────────┼───────────────┐                 │
         │               │               │                 │
         │               │               │                 │
    Arm Raised    "Manual Mode"    Voice Command           │
         │               │               │                 │
         ▼               ▼               ▼                 │
    ┌─────────┐    ┌─────────────┐  ┌─────────────┐       │
    │TRACKING │───►│ MANUAL_MODE  │  │  (stays)    │       │
    │  USER   │    └─────────────┘  └─────────────┘       │
    └────┬────┘                                            │
         │ Arm Still Raised                                │
         ▼                                                 │
    ┌─────────────┐                                        │
    │  FOLLOWING  │                                        │
    │    USER     │                                        │
    └────┬────────┘                                        │
         │                                                  │
    ┌────┼────┐                                            │
    │    │    │                                            │
    │    │    │ User Too Close                             │
    │    │    │                                            │
    │    │    ▼                                            │
    │    │ ┌─────────┐                                    │
    │    │ │ STOPPED │                                    │
    │    │ └────┬────┘                                    │
    │    │      │ Wait Timeout                            │
    │    │      ▼                                         │
    │    │ ┌──────────────────┐                          │
    │    │ │ RETURNING_TO_    │                          │
    │    │ │      START       │                          │
    │    │ └────┬─────────────┘                          │
    │    │      │                                          │
    │    │      └─────────────────────────────────────────┘
    │    │
    │    │ User Lost + Timeout
    │    │
    │    └─────────────────────────────────────────────────┘
    │
    └───────────────────────────────────────────────────────┘
```

### State Machine Class

**Key Methods:**
- `get_state()` - Returns current state
- `transition_to(new_state)` - Changes state and resets timer
- `get_time_in_state()` - Returns seconds since state entry
- `is_timeout()` - Checks if tracking timeout exceeded
- `set_start_position()` - Stores starting position for return navigation

**State Entry Timing:**
- Each state transition resets `state_enter_time`
- Used for timeout detection (e.g., 30 seconds in TRACKING_USER)
- Used for wait timers (e.g., 10 seconds in STOPPED)

---

## Main Control Loop

### The `run()` Method

```python
def run(self):
    while self.running:
        state = self.state_machine.get_state()
        
        # Route to appropriate handler
        if state == State.IDLE:
            self.handle_idle_state()
        elif state == State.ACTIVE:
            self.handle_active_state()
        # ... etc
        
        time.sleep(0.01)  # 10ms delay = ~100Hz loop rate
```

**Loop Characteristics:**
- **Frequency**: ~100Hz (10ms delay)
- **Non-blocking**: All operations use timeouts
- **Event-driven**: State changes trigger handler execution
- **Graceful shutdown**: Signal handlers set `self.running = False`

### Signal Handling

```python
signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, self.signal_handler) # System shutdown
```

**Shutdown Flow:**
1. Signal received → `signal_handler()` called
2. Sets `self.running = False`
3. Loop exits → `finally` block executes
4. `cleanup()` called → All components stopped gracefully

---

## Component Interactions

### 1. Wake Word Detector → State Machine

**Flow:**
```
WakeWordDetector.detect() → returns bool
    ↓
handle_idle_state() checks result
    ↓
If True → state_machine.transition_to(State.ACTIVE)
```

**Interface:**
- `detect()` - Non-blocking, returns True if wake word detected
- `start_listening()` - Starts audio capture
- `stop()` - Stops listening and cleans up

### 2. YOLO Pose Tracker → State Machine

**Flow:**
```
YOLOPoseTracker.update() → returns dict
    ↓
State handler processes result
    ↓
State transitions based on detection results
```

**Return Format:**
```python
{
    'person_detected': bool,
    'person_box': (x1, y1, x2, y2) or None,
    'angle': float or None,  # Person position angle
    'is_centered': bool,
    'arm_raised': bool,
    'arm_confidence': float,
    'track_id': int or None,  # Person tracking ID
    'gesture': str or None    # Detected gesture
}
```

**Key Features:**
- **Tracking**: BYTETracker maintains person IDs across frames
- **Pose Detection**: 17 keypoints (shoulders, elbows, wrists, etc.)
- **Arm Angle**: Calculates 60-90° arm raise detection
- **Gesture Recognition**: Infers gestures from pose keypoints

### 3. Motor/Servo Controllers → Movement

**Motor Controller:**
- `forward(speed)` - Speed: 0.0-1.0 (percentage)
- `stop()` - Stops motor
- Uses PWM on GPIO 12

**Servo Controller:**
- `set_angle(angle)` - Angle: -45° to +45°
- `center()` - Centers steering
- `turn_left(amount)` / `turn_right(amount)` - Relative turning
- Uses PWM on GPIO 13

**Control Logic:**
```python
# In FOLLOWING_USER state:
if result['is_centered']:
    speed = config.FOLLOW_SPEED  # 0.6 (60%)
else:
    speed = config.FOLLOW_SPEED * 0.7  # 42% (slower when turning)

self.servo.set_angle(result['angle'])  # Steer toward person
self.motor.forward(speed)              # Move forward
```

### 4. TOF Sensor → Safety System

**Flow:**
```
TOFSensor.read_distance() → returns mm
    ↓
Check emergency_stop() → returns bool
    ↓
If True → motor.stop(), servo.center()
```

**Safety Levels:**
- **Emergency Stop**: < 10cm (100mm) - Immediate stop
- **Normal Stop**: < 30cm (300mm) - Transition to STOPPED state

**Integration:**
- Checked every loop in FOLLOWING_USER state
- Takes priority over visual detection
- Prevents collisions

### 5. Path Tracker → Return Navigation

**Flow:**
```
During FOLLOWING_USER:
    path_tracker.add_segment(speed, steering, duration)
    
During RETURNING_TO_START:
    reverse_path = path_tracker.get_reverse_path()
    Execute segments in reverse order
```

**Path Recording:**
- Records: speed, steering position, duration
- Stored as list of segments
- Reversed for return journey

### 6. Voice Recognizer → Manual Mode

**Flow:**
```
VoiceRecognizer.recognize_command(timeout=0.1)
    ↓
Returns: 'FORWARD', 'LEFT', 'RIGHT', 'STOP', etc.
    ↓
execute_manual_command_continuous(command)
```

**Features:**
- Uses OpenAI GPT for natural language interpretation
- Non-blocking with timeout
- Continuous command execution until new command/stop

### 7. Hand Gesture Controller → Manual Mode

**Flow:**
```
Get frame from YOLOPoseTracker (shared camera)
    ↓
HandGestureController.detect_command(frame)
    ↓
Returns: 'FORWARD', 'LEFT', 'RIGHT', 'STOP', etc.
    ↓
Same execution path as voice commands
```

**Features:**
- Uses hand keypoints model (21 keypoints) or pose fallback
- Gesture hold time prevents accidental commands
- Shares camera frame with pose tracker

---

## Data Flow Diagrams

### Autonomous Mode Flow

```
┌─────────────┐
│ Wake Word   │──► IDLE → ACTIVE
│ Detector    │
└─────────────┘
       │
       ▼
┌─────────────┐
│ YOLO Pose   │──► Person Detected + Arm Raised
│ Tracker     │    → ACTIVE → TRACKING_USER
└─────────────┘
       │
       ▼
┌─────────────┐
│ Arm Still   │──► TRACKING_USER → FOLLOWING_USER
│ Raised?     │
└─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Calculate   │──►  │ Motor/Servo │──► Movement
│ Angle &     │     │ Controllers │
│ Position    │     └─────────────┘
└─────────────┘
       │
       ▼
┌─────────────┐
│ TOF Sensor  │──► Too Close? → STOPPED
│ Check       │
└─────────────┘
       │
       ▼
┌─────────────┐
│ Path        │──► Record movement segments
│ Tracker     │
└─────────────┘
```

### Manual Mode Flow

```
┌─────────────┐     ┌─────────────┐
│ Voice       │ OR  │ Hand       │
│ Recognizer  │     │ Gesture    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Command       │
         │ Interpreter   │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Execute       │
         │ Command       │
         │ Continuously  │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Motor/Servo  │──► Movement
         │ Controllers   │
         └───────────────┘
```

---

## State Handlers Explained

### `handle_idle_state()`

**Purpose**: Wait for wake word activation

**Logic:**
```python
if self.wake_word.detect():
    self.state_machine.transition_to(State.ACTIVE)
```

**Characteristics:**
- Minimal CPU usage
- Only checks wake word
- No movement, no camera processing

### `handle_active_state()`

**Purpose**: Mode selection (autonomous vs manual)

**Logic:**
1. Check for "manual mode" voice command (if voice available)
2. Check visual detection for arm raising
3. If arm raised → autonomous mode
4. If "manual mode" → manual mode

**Key Features:**
- Dual input paths (voice + visual)
- Visual update throttled (100ms interval)
- Non-blocking voice check (0.1s timeout)

### `handle_tracking_user_state()`

**Purpose**: Confirm user detection and arm position

**Logic:**
1. Update visual detection
2. If no person → check timeout → return to IDLE
3. If person + arm raised → transition to FOLLOWING_USER
4. If person but no arm → wait (stay in state)

**Timeout Handling:**
- If no person detected for 30 seconds → return to IDLE
- Prevents system from getting stuck

### `handle_following_user_state()`

**Purpose**: Active following with movement control

**Logic Flow:**
```
1. Check TOF emergency stop (highest priority)
   └─► If emergency → stop immediately, return

2. Update visual detection
   └─► If no person → check timeout → return to IDLE

3. Check TOF normal stop
   └─► If too close → transition to STOPPED

4. Calculate steering from person angle
   └─► Set servo angle

5. Calculate speed based on centering
   └─► If centered: full speed
       If not centered: 70% speed

6. Record path segment
   └─► For return navigation
```

**Control Algorithm:**
```python
angle = result['angle']  # Person position relative to center
steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
self.servo.set_angle(angle)

if result['is_centered']:
    speed = config.FOLLOW_SPEED  # 0.6
else:
    speed = config.FOLLOW_SPEED * 0.7  # 0.42

self.motor.forward(speed)
```

**Path Tracking:**
- Records: speed, steering_position, duration
- Used for reverse path navigation

### `handle_stopped_state()`

**Purpose**: Wait at user location for trash placement

**Logic:**
```python
wait_time = 10.0  # seconds
if self.state_machine.get_time_in_state() > wait_time:
    self.state_machine.transition_to(State.RETURNING_TO_START)
```

**Future Enhancements:**
- Could wait for button press
- Could wait for voice command
- Could wait for gesture signal

### `handle_returning_to_start_state()`

**Purpose**: Navigate back using recorded path

**Logic:**
1. Get reverse path from path tracker
2. Execute segments sequentially
3. Track current segment index
4. When all segments done → return to IDLE

**Path Execution:**
```python
reverse_path = self.path_tracker.get_reverse_path()
for segment in reverse_path:
    self.motor.forward(segment['motor_speed'])
    self.servo.set_position(segment['servo_position'])
    time.sleep(segment['duration'])
```

**Error Handling:**
- If no path recorded → stop and return to IDLE
- Prevents crashes if path tracking failed

### `handle_manual_mode_state()`

**Purpose**: Voice and gesture control

**Logic:**
1. Check voice command (priority)
2. Check hand gesture (fallback)
3. Process command (voice OR gesture)
4. Execute command continuously
5. Handle special commands (AUTOMATIC_MODE, STOP)

**Command Execution:**
- Commands execute continuously until new command/stop
- TURN_AROUND is one-time action, then continues forward
- Voice takes priority over gestures

**Command Mapping:**
```python
'FORWARD' → motor.forward(MOTOR_MEDIUM), servo.center()
'LEFT' → motor.forward(MOTOR_SLOW), servo.turn_left(0.5)
'RIGHT' → motor.forward(MOTOR_SLOW), servo.turn_right(0.5)
'STOP' → motor.stop(), servo.center()
'TURN_AROUND' → turn left 180°, then continue forward
'AUTOMATIC_MODE' → return to ACTIVE state
```

---

## Error Handling & Robustness

### Initialization Errors

**Critical Components** (system exits if fail):
- Wake word detector
- YOLO pose tracker
- Motor controller
- Servo controller

**Optional Components** (system continues if fail):
- TOF sensor → `self.tof = None`
- Voice recognizer → `self.voice = None`
- Hand gesture controller → `self.gesture_controller = None`

**Error Handling Pattern:**
```python
try:
    self.component = Component(...)
except Exception as e:
    print(f"[Main] WARNING: Failed to initialize: {e}")
    print("[Main] Continuing without component")
    self.component = None
```

### Runtime Error Handling

**Visual Detection Errors:**
- Wrapped in try/except in manual mode
- Debug mode logs errors
- System continues if detection fails

**Command Execution Errors:**
- Commands checked before execution
- Invalid commands ignored
- System doesn't crash on bad input

### Timeout Protection

**Tracking Timeout:**
- 30 seconds in TRACKING_USER or FOLLOWING_USER
- Prevents infinite waiting
- Returns to IDLE if user lost

**State-Specific Timeouts:**
- STOPPED: 10 seconds wait time
- Manual mode: 0.1s voice check timeout (non-blocking)

### Resource Cleanup

**Graceful Shutdown:**
```python
def cleanup(self):
    # Stop all movement first
    self.motor.stop()
    self.servo.center()
    
    # Stop all components
    self.wake_word.stop()
    self.visual.stop()
    # ... etc
```

**Signal Handling:**
- SIGINT (Ctrl+C) → graceful shutdown
- SIGTERM (system shutdown) → graceful shutdown
- All resources cleaned up in `finally` block

---

## Configuration System

### Config File Structure

**GPIO Configuration:**
```python
MOTOR_PWM_PIN = 12
SERVO_PWM_PIN = 13
ToF_DIGITAL_PIN = 23
```

**PWM Values:**
```python
PWM_FREQUENCY_MOTOR = 40  # Hz
PWM_FREQUENCY_SERVO = 50  # Hz
MOTOR_STOP = 100.0        # Duty cycle %
SERVO_CENTER = 92.675     # Duty cycle %
```

**Detection Configuration:**
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_POSE_MODEL = 'yolo11n-pose.pt'
YOLO_CONFIDENCE = 0.25
PERSON_CENTER_THRESHOLD = 30  # pixels
ANGLE_TO_STEERING_GAIN = 0.5
```

**Safety Configuration:**
```python
EMERGENCY_STOP_ENABLED = True
TOF_STOP_DISTANCE_MM = 300      # 30cm
TOF_EMERGENCY_DISTANCE_MM = 100 # 10cm
TRACKING_TIMEOUT = 30.0         # seconds
```

**Debug Configuration:**
```python
DEBUG_MODE = True
DEBUG_VISUAL = True
DEBUG_MOTOR = True
DEBUG_SERVO = True
DEBUG_TOF = True
DEBUG_VOICE = True
DEBUG_STATE = True
```

### Environment Variables

Loaded from `.env` file:
- `PICOVOICE_ACCESS_KEY` - Wake word detection
- `OPENAI_API_KEY` - Voice recognition

**Loading:**
```python
from dotenv import load_dotenv
load_dotenv()
config.WAKE_WORD_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY')
config.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```

---

## Key Design Patterns

### 1. State Machine Pattern
- Centralized state management
- Clear state transitions
- Timeout handling per state

### 2. Observer Pattern (Implicit)
- Components notify state machine via return values
- State machine reacts to component outputs

### 3. Strategy Pattern
- Different handlers for different states
- Same interface, different implementations

### 4. Dependency Injection
- Components initialized with config values
- Easy to swap implementations

### 5. Error Resilience
- Optional components can fail
- System degrades gracefully
- Timeouts prevent infinite waits

---

## Performance Characteristics

### Loop Frequency
- **Main Loop**: ~100Hz (10ms delay)
- **Visual Updates**: 10Hz (100ms interval)
- **Voice Checks**: 10Hz (0.1s timeout)

### Resource Usage
- **CPU**: Moderate (YOLO inference is main load)
- **Memory**: ~200-500MB (YOLO models)
- **Camera**: Single instance shared between components

### Latency
- **Wake Word**: ~100-200ms detection latency
- **Visual Detection**: ~50-100ms per frame
- **Motor Response**: Immediate (PWM update)
- **State Transitions**: < 10ms

---

## Future Enhancement Points

1. **Path Tracking**: More sophisticated path following
2. **Obstacle Avoidance**: Additional sensors
3. **Multi-Person Handling**: Track multiple users
4. **Learning**: Improve following behavior over time
5. **Remote Control**: Network interface for remote commands
6. **Logging**: Comprehensive system logs
7. **Telemetry**: Real-time status reporting

---

**This architecture provides a robust, maintainable, and extensible foundation for the Bin Diesel system.**


