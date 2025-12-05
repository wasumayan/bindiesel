# Bin Diesel System Architecture

## Overview

The Bin Diesel system is a Raspberry Pi-based autonomous trash collection vehicle with both autonomous and manual control modes.

## System Components

### 1. Input Systems
- **Wake Word Detection**: Picovoice Porcupine for "bin diesel" (offline, real-time)
- **Visual Detection**: YOLO + custom gesture detection for arm raising (person tracking)
- **Distance Sensing**: VL53L0X Time-of-Flight sensor (7-8cm stop threshold)
- **Voice Recognition**: Real-time speech recognition + OpenAI GPT for command interpretation

### 2. Control Systems
- **Motor Controller**: PWM-based speed control via GPIO (forward only, no reverse)
- **Servo Controller**: PWM-based steering control via GPIO (left/right/center)
- **State Machine**: Manages workflow and mode switching
- **Path Tracker**: Records movement path for reverse navigation to start position

### 3. Processing
- **Raspberry Pi**: Main processing unit
- **Camera Module 3 Wide**: Visual input
- **GPIO Pins**: PWM output for motor and servo

## Workflow

### Autonomous Mode

```
[IDLE] 
  ↓ (Wake word: "bin diesel")
[ACTIVE]
  ↓ (User raises arm detected)
[TRACKING USER]
  ↓ (Path tracking starts)
  ↓ (Calculate angle, move in direction)
[FOLLOWING USER]
  ↓ (Keep user centered, record path segments)
  ↓ (TOF sensor monitors distance)
[APPROACHING USER]
  ↓ (TOF sensor detects 7-8cm)
[STOPPED]
  ↓ (Wait for trash placement ~10 seconds)
[RETURNING TO START]
  ↓ (Execute reverse path - opposite movements)
[IDLE]
```

### Manual Mode

```
[IDLE]
  ↓ (Wake word: "bin diesel")
[ACTIVE]
  ↓ (User says "manual mode")
[MANUAL MODE]
  ↓ (Voice commands: FORWARD/LEFT/RIGHT/STOP/TURN AROUND)
  ↓ (Commands execute continuously until changed)
  ↓ (Say "STOP" to stop current command)
  ↓ (Say "AUTOMATIC MODE" or "bin diesel" to return to auto)
[MANUAL MODE] (loops until mode change)
```

## State Machine States

1. **IDLE**: Waiting for wake word "bin diesel"
2. **ACTIVE**: Awake, waiting for mode selection (arm raise or "manual mode")
3. **TRACKING_USER**: Detecting and tracking user position (path tracking starts)
4. **FOLLOWING_USER**: Moving toward user, keeping centered (path segments recorded)
5. **STOPPED**: At target distance (7-8cm), waiting for trash placement
6. **RETURNING_TO_START**: Executing reverse path back to origin
7. **MANUAL_MODE**: Waiting for voice commands (commands execute continuously)

## Communication Flow

```
Wake Word Detector → State Machine (triggers ACTIVE state)
Visual Detector → State Machine → Motor/Servo Controllers (autonomous mode)
TOF Sensor → State Machine → Motor Controller (emergency stop at 7cm)
Voice Recognizer → State Machine → Motor/Servo Controllers (manual mode)
Path Tracker → Records movements → Reverse path for return navigation
```

## GPIO Pin Assignments

- **Motor PWM**: GPIO 18 (BCM)
- **Servo PWM**: GPIO 19 (BCM)
- **TOF Sensor**: I2C (SDA: GPIO 2, SCL: GPIO 3)

## Safety Features

1. **TOF Emergency Stop**: Stops immediately if object < 7cm detected (configurable)
2. **Timeout Protection**: Returns to idle after inactivity (30s default)
3. **Manual Override**: Manual mode always available via voice
4. **Safe Shutdown**: Graceful cleanup on exit (stops motors, centers servo)
5. **Path Tracking**: Records movements for safe return navigation
6. **No Reverse Movement**: Car only moves forward (simpler, safer)

## Voice Commands (Manual Mode)

- **FORWARD**: Move forward continuously
- **LEFT**: Turn left continuously
- **RIGHT**: Turn right continuously
- **STOP**: Stop current command
- **TURN AROUND**: Rotate 180° then continue forward
- **AUTOMATIC MODE**: Return to autonomous mode
- **MANUAL MODE**: Enter manual control mode

Commands execute continuously until a new command is given or STOP is said.

