# Bin Diesel System Architecture

## Overview

The Bin Diesel system is a Raspberry Pi-based autonomous trash collection vehicle with both autonomous and manual control modes.

## System Components

### 1. Input Systems
- **Wake Word Detection**: Picovoice Porcupine for "bin diesel"
- **Visual Detection**: YOLO + custom gesture detection for arm raising
- **Distance Sensing**: VL53L0X Time-of-Flight sensor
- **Voice Recognition**: GPT API for manual mode commands

### 2. Control Systems
- **Motor Controller**: PWM-based speed control via GPIO
- **Servo Controller**: PWM-based steering control via GPIO
- **State Machine**: Manages workflow and mode switching

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
  ↓ (User raises arm)
[TRACKING USER]
  ↓ (Calculate angle, move in direction)
[FOLLOWING USER]
  ↓ (Keep user centered)
[APPROACHING USER]
  ↓ (TOF sensor detects 7-8cm)
[STOPPED]
  ↓ (User places trash)
[RETURNING TO START]
  ↓ (Navigate back)
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
[EXECUTING COMMAND]
  ↓ (Command complete or new command)
[MANUAL MODE]
```

## State Machine States

1. **IDLE**: Waiting for wake word
2. **ACTIVE**: Awake, waiting for mode selection
3. **TRACKING_USER**: Detecting and tracking user position
4. **FOLLOWING_USER**: Moving toward user, keeping centered
5. **STOPPED**: At target distance, waiting for trash
6. **RETURNING_TO_START**: Navigating back to origin
7. **MANUAL_MODE**: Waiting for voice commands
8. **EXECUTING_COMMAND**: Executing manual command

## Communication Flow

```
Wake Word Detector → State Machine
Visual Detector → State Machine → Motor/Servo Controllers
TOF Sensor → State Machine → Motor Controller (emergency stop)
Voice Recognizer → State Machine → Motor/Servo Controllers
```

## GPIO Pin Assignments

- **Motor PWM**: GPIO 18 (BCM)
- **Servo PWM**: GPIO 19 (BCM)
- **TOF Sensor**: I2C (SDA: GPIO 2, SCL: GPIO 3)

## Safety Features

1. **TOF Emergency Stop**: Stops immediately if object < 7cm detected
2. **Timeout Protection**: Returns to idle after inactivity
3. **Manual Override**: Manual mode always available
4. **Safe Shutdown**: Graceful cleanup on exit

