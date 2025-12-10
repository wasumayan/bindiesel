# Navigation Logic & PWM Signal Flow

This document explains how the car navigates and sends PWM signals in different modes.

---

## Overview: PWM Signal Flow

```
Detection → Angle Calculation → Steering Position → PWM Duty Cycle → Hardware
     ↓              ↓                    ↓                ↓              ↓
  YOLO Pose    Person Angle      Servo Position    GPIO PWM      Servo Motor
  Tracking     (degrees)         (-1.0 to 1.0)     (duty %)      Movement
```

---

## 1. Motor Controller (PWM for Speed)

**File**: `motor_controller.py`

### PWM Signal Generation

```python
# Motor speed: 0.0 to 1.0 (percentage)
motor.forward(0.5)  # 50% speed

# Converts to PWM duty cycle:
duty_cycle = config.MOTOR_MAX * speed
# Example: 92.7% * 0.5 = 46.35% duty cycle

# Sends to GPIO:
GPIO.PWM(pin, frequency).ChangeDutyCycle(duty_cycle)
```

### Motor Commands

| Method | Speed Parameter | PWM Duty Cycle | Effect |
|--------|----------------|----------------|--------|
| `motor.forward(speed)` | 0.0 - 1.0 | `MOTOR_MAX * speed` | Move forward at speed |
| `motor.stop()` | N/A | `MOTOR_STOP` (100.0%) | Stop motor |

**Config Values**:
- `MOTOR_MAX = 92.7%` (maximum speed duty cycle)
- `MOTOR_STOP = 100.0%` (stop duty cycle)
- `PWM_FREQUENCY_MOTOR = 40 Hz`

---

## 2. Servo Controller (PWM for Steering)

**File**: `servo_controller.py`

### PWM Signal Generation

```python
# Steering angle: -45° to +45° (negative = left, positive = right)
servo.set_angle(30.0)  # Turn 30° right

# Converts to PWM duty cycle:
if angle >= 0:  # Right turn
    duty = center_duty + (right_max - center_duty) * (angle / 45.0)
else:  # Left turn
    duty = center_duty + (left_max - center_duty) * (angle / -45.0)

# Sends to GPIO:
GPIO.PWM(pin, frequency).ChangeDutyCycle(duty)
```

### Servo Commands

| Method | Parameter | PWM Duty Cycle | Effect |
|--------|-----------|----------------|--------|
| `servo.set_angle(angle)` | -45° to +45° | Calculated from angle | Steer to angle |
| `servo.center()` | N/A | `SERVO_CENTER` | Center steering |
| `servo.turn_left(amount)` | 0.0 - 1.0 | Converts to angle: `-45° * amount` | Turn left |
| `servo.turn_right(amount)` | 0.0 - 1.0 | Converts to angle: `+45° * amount` | Turn right |
| `servo.set_position(position)` | -1.0 to 1.0 | Converts to angle: `position * 45°` | Set position |

**Config Values**:
- `SERVO_CENTER = 7.5%` (center position)
- `SERVO_LEFT_MAX = 10.0%` (full left)
- `SERVO_RIGHT_MAX = 5.0%` (full right)
- `PWM_FREQUENCY_SERVO = 50 Hz`

---

## 3. Navigation Logic by Mode

### Mode 1: FOLLOWING_USER (Autonomous Following)

**Location**: `main.py` → `handle_following_user_state()`

**Logic Flow**:

```
1. Get person detection from YOLO pose tracker
   ↓
2. Calculate person angle from center
   angle = (person_center_x - frame_center_x) / frame_width * FOV
   ↓
3. Convert angle to steering position
   steering_position = (angle / 45.0) * ANGLE_TO_STEERING_GAIN
   ↓
4. Check if person is centered
   if abs(offset) < PERSON_CENTER_THRESHOLD:
       → Person centered
   else:
       → Person not centered
   ↓
5. Send PWM signals:
   if centered:
       motor.forward(FOLLOW_SPEED)      # Full speed forward
       servo.center()                    # Center steering
   else:
       motor.forward(FOLLOW_SPEED * 0.7) # Slower while turning
       servo.set_angle(angle)            # Steer towards person
```

**Code Location**: `main.py` lines 346-388

**Key Calculations**:
```python
# Person angle calculation
person_center_x = (x1 + x2) / 2
frame_center_x = config.CAMERA_WIDTH / 2
offset = person_center_x - frame_center_x
angle = (offset / config.CAMERA_WIDTH) * 102.0  # Camera FOV

# Steering position
steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
steering_position = max(-1.0, min(1.0, steering_position))  # Clamp

# Speed adjustment
if result['is_centered']:
    speed = config.FOLLOW_SPEED  # Full speed
else:
    speed = config.FOLLOW_SPEED * 0.7  # Slower while turning
```

**PWM Signals Sent**:
- **Motor**: `motor.forward(speed)` → `MOTOR_MAX * speed` duty cycle
- **Servo**: `servo.set_angle(angle)` → Calculated duty cycle based on angle

---

### Mode 2: MANUAL_MODE (Voice/Gesture Control)

**Location**: `main.py` → `handle_manual_mode_state()` → `execute_manual_command_continuous()`

**Logic Flow**:

```
1. Receive command (voice or gesture)
   ↓
2. Map command to motor/servo actions
   ↓
3. Execute continuously until new command/stop
```

**Command Mappings**:

| Command | Motor Action | Servo Action | PWM Signals |
|---------|-------------|--------------|-------------|
| `FORWARD` | `motor.forward(MOTOR_MEDIUM)` | `servo.center()` | Motor: MEDIUM speed, Servo: CENTER |
| `LEFT` | `motor.forward(MOTOR_SLOW)` | `servo.turn_left(0.5)` | Motor: SLOW speed, Servo: LEFT 50% |
| `RIGHT` | `motor.forward(MOTOR_SLOW)` | `servo.turn_right(0.5)` | Motor: SLOW speed, Servo: RIGHT 50% |
| `STOP` | `motor.stop()` | `servo.center()` | Motor: STOP, Servo: CENTER |
| `TURN_AROUND` | `motor.stop()` → `motor.forward(MOTOR_MEDIUM)` | `servo.turn_left(1.0)` → `servo.center()` | Special sequence |

**Code Location**: `main.py` lines 500-528

**PWM Signals Sent**:
- **FORWARD**: Motor duty = `MOTOR_MAX * MOTOR_MEDIUM`, Servo duty = `SERVO_CENTER`
- **LEFT**: Motor duty = `MOTOR_MAX * MOTOR_SLOW`, Servo duty = `SERVO_LEFT_MAX * 0.5`
- **RIGHT**: Motor duty = `MOTOR_MAX * MOTOR_SLOW`, Servo duty = `SERVO_RIGHT_MAX * 0.5`
- **STOP**: Motor duty = `MOTOR_STOP`, Servo duty = `SERVO_CENTER`

---

### Mode 3: RADD_MODE (Follow Violators)

**Location**: `main.py` → `handle_radd_mode_state()`

**Logic Flow**:

```
1. Detect RADD violations in tracked persons
   ↓
2. Select target violator (prioritize in-frame violators)
   ↓
3. Calculate violator position and angle
   ↓
4. Navigate towards violator (same logic as FOLLOWING_USER)
```

**Code Location**: `main.py` lines 656-690

**Key Calculations**:
```python
# Violator position
person_box = target_violator_info.get('current_box')
person_center_x = (x1 + x2) / 2
frame_center_x = config.CAMERA_WIDTH / 2
offset = person_center_x - frame_center_x
angle = (offset / config.CAMERA_WIDTH) * 102.0  # Camera FOV

# Steering and speed (same as FOLLOWING_USER)
if abs(offset) < config.PERSON_CENTER_THRESHOLD:
    speed = config.FOLLOW_SPEED
    motor.forward(speed)
    servo.center()
else:
    speed = config.FOLLOW_SPEED * 0.7
    motor.forward(speed)
    servo.set_angle(angle)
```

**PWM Signals Sent**:
- Same as FOLLOWING_USER mode
- Motor: `MOTOR_MAX * speed` duty cycle
- Servo: Calculated duty cycle based on angle

---

### Mode 4: RETURNING_TO_START (Path Reversal)

**Location**: `main.py` → `handle_returning_to_start_state()`

**Logic Flow**:

```
1. Get reverse path from path tracker
   ↓
2. Execute path segments in reverse order
   ↓
3. For each segment:
   - motor.forward(segment['motor_speed'])
   - servo.set_position(segment['servo_position'])
   - Wait for segment duration
   ↓
4. When all segments complete → Stop and return to IDLE
```

**Code Location**: `main.py` lines 401-445

**PWM Signals Sent**:
- Motor: `MOTOR_MAX * segment['motor_speed']` duty cycle
- Servo: `servo.set_position(segment['servo_position'])` → Converts position to duty cycle

---

## 4. Angle to Steering Conversion

**Key Formula**:
```python
# Person angle from center (degrees)
angle = (person_center_x - frame_center_x) / frame_width * camera_fov

# Convert to steering position (-1.0 to 1.0)
steering_position = (angle / 45.0) * ANGLE_TO_STEERING_GAIN

# Clamp to valid range
steering_position = max(-1.0, min(1.0, steering_position))

# Convert to servo angle
servo_angle = steering_position * 45.0  # -45° to +45°

# Convert to PWM duty cycle
if servo_angle >= 0:  # Right
    duty = SERVO_CENTER + (SERVO_RIGHT_MAX - SERVO_CENTER) * (servo_angle / 45.0)
else:  # Left
    duty = SERVO_CENTER + (SERVO_LEFT_MAX - SERVO_CENTER) * (servo_angle / -45.0)
```

**Config Values**:
- `ANGLE_TO_STEERING_GAIN = 0.5` (sensitivity adjustment)
- `PERSON_CENTER_THRESHOLD = 30` pixels (centering tolerance)
- `CAMERA_WIDTH = 640` pixels
- Camera FOV = 102.0 degrees

---

## 5. Speed Control Logic

### Speed Levels

| Speed Variable | Value | Usage |
|---------------|-------|-------|
| `MOTOR_SLOW` | 0.3 | Slow movement (turning, precise) |
| `MOTOR_MEDIUM` | 0.5 | Medium speed (manual mode forward) |
| `FOLLOW_SPEED` | 0.4 | Following user (autonomous) |
| `FOLLOW_SPEED * 0.7` | 0.28 | Slower while turning |

### Speed Adjustment Logic

```python
# In FOLLOWING_USER and RADD_MODE:
if person_is_centered:
    speed = FOLLOW_SPEED  # Full following speed
else:
    speed = FOLLOW_SPEED * 0.7  # Slower while turning

# In MANUAL_MODE:
if command == 'FORWARD':
    speed = MOTOR_MEDIUM  # Medium speed
elif command in ['LEFT', 'RIGHT']:
    speed = MOTOR_SLOW  # Slow speed for turning
```

---

## 6. Complete Navigation Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    STATE MACHINE                         │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   FOLLOWING_USER    MANUAL_MODE      RADD_MODE
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐   ┌──────────────┐  ┌──────────────┐
│ YOLO Pose    │   │ Voice/Gesture│  │ RADD Detector│
│ Detection    │   │ Commands     │  │ + Pose Track │
└──────────────┘   └──────────────┘  └──────────────┘
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│         ANGLE CALCULATION & POSITION                 │
│  - Calculate person angle from center                │
│  - Check if centered                                 │
│  - Determine speed based on centering                │
└─────────────────────────────────────────────────────┘
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│         PWM SIGNAL GENERATION                        │
│  Motor: MOTOR_MAX * speed → GPIO duty cycle         │
│  Servo: angle → duty cycle → GPIO duty cycle        │
└─────────────────────────────────────────────────────┘
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│              HARDWARE (GPIO)                         │
│  - Motor PWM pin (GPIO 12)                          │
│  - Servo PWM pin (GPIO 13)                          │
└─────────────────────────────────────────────────────┘
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│              PHYSICAL MOVEMENT                       │
│  - Car moves forward/backward                        │
│  - Car steers left/right                            │
└─────────────────────────────────────────────────────┘
```

---

## 7. Key Code Locations

### Motor Control
- **File**: `motor_controller.py`
- **Methods**: `forward(speed)`, `stop()`
- **PWM Pin**: GPIO 12 (configurable in `config.py`)

### Servo Control
- **File**: `servo_controller.py`
- **Methods**: `set_angle(angle)`, `center()`, `turn_left(amount)`, `turn_right(amount)`
- **PWM Pin**: GPIO 13 (configurable in `config.py`)

### Navigation Logic
- **FOLLOWING_USER**: `main.py` lines 301-388
- **MANUAL_MODE**: `main.py` lines 447-528
- **RADD_MODE**: `main.py` lines 530-690
- **RETURNING_TO_START**: `main.py` lines 401-445

### Configuration
- **File**: `config.py`
- **Motor**: `MOTOR_MAX`, `MOTOR_STOP`, `FOLLOW_SPEED`, `MOTOR_SLOW`, `MOTOR_MEDIUM`
- **Servo**: `SERVO_CENTER`, `SERVO_LEFT_MAX`, `SERVO_RIGHT_MAX`
- **Navigation**: `ANGLE_TO_STEERING_GAIN`, `PERSON_CENTER_THRESHOLD`

---

## 8. Debugging Navigation

### Enable Debug Output

Edit `config.py`:
```python
DEBUG_MODE = True
DEBUG_MOTOR = True
DEBUG_SERVO = True
DEBUG_VISUAL = True
```

### What to Check

1. **Motor PWM Signals**:
   - Check `[Motor] forward speed = X.XX (duty = XX.X%)` messages
   - Verify duty cycle is reasonable (0-92.7%)

2. **Servo PWM Signals**:
   - Check `[Servo] set_angle(XX.X) degrees -> duty = XX.XX%` messages
   - Verify duty cycle is within range (5.0% - 10.0%)

3. **Angle Calculations**:
   - Check `Person angle: XX.X°, centered: True/False` messages
   - Verify angles are reasonable (-45° to +45°)

4. **State Transitions**:
   - Check state machine transitions
   - Verify correct handler is called

---

## 9. Common Issues & Fixes

### Issue: Car doesn't move
- **Check**: Motor PWM signals in debug output
- **Check**: GPIO permissions (`sudo usermod -a -G gpio $USER`)
- **Check**: Motor connections to GPIO pin 12

### Issue: Car doesn't steer
- **Check**: Servo PWM signals in debug output
- **Check**: Servo connections to GPIO pin 13
- **Check**: Servo duty cycle values (should be 5.0% - 10.0%)

### Issue: Car steers wrong direction
- **Check**: `CAMERA_SWAP_LEFT_RIGHT` in config
- **Check**: Servo left/right max values might be swapped

### Issue: Car moves too fast/slow
- **Adjust**: `FOLLOW_SPEED` in config (0.0 - 1.0)
- **Adjust**: `MOTOR_SLOW`, `MOTOR_MEDIUM` in config

### Issue: Car doesn't center properly
- **Adjust**: `PERSON_CENTER_THRESHOLD` in config (pixels)
- **Adjust**: `ANGLE_TO_STEERING_GAIN` in config (sensitivity)

---

## 10. PWM Signal Details

### Motor PWM
- **Pin**: GPIO 12
- **Frequency**: 40 Hz
- **Duty Cycle Range**: 0% (stopped) to 92.7% (max speed)
- **Stop Duty**: 100.0% (special stop value)

### Servo PWM
- **Pin**: GPIO 13
- **Frequency**: 50 Hz
- **Duty Cycle Range**: 5.0% (right max) to 10.0% (left max)
- **Center Duty**: 7.5%

### Duty Cycle Calculation Examples

**Motor**:
```python
speed = 0.5  # 50% speed
duty = 92.7 * 0.5 = 46.35%
```

**Servo**:
```python
angle = 30.0  # 30° right
# Right turn: duty = center + (right_max - center) * (angle / 45.0)
duty = 7.5 + (5.0 - 7.5) * (30.0 / 45.0)
duty = 7.5 + (-2.5) * 0.667
duty = 7.5 - 1.667 = 5.83%
```

---

**This is the complete navigation and PWM signal flow for all modes!**

