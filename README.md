# Bin Diesel System

Complete autonomous and manual control system for Bin Diesel car using YOLO pose estimation and tracking.

## Quick Start

**See `QUICK_START.md` for complete setup instructions.**

1. **Install Python 3.12** (see `PYTHON_3.12_SETUP.md`)
2. **Install dependencies:**
```bash
pip install ultralytics opencv-contrib-python numpy
sudo apt install -y python3-picamera2
```
3. **Test components:**
```bash
python test_yolo_pose_tracking.py --fps  # Test pose detection + tracking
python hand_gesture_controller.py        # Test hand gestures
```
4. **Run full system:**
```bash
python main.py
```

## Features

### Autonomous Mode
1. Wake word: "bin diesel"
2. YOLO pose detection + tracking → car follows user
3. Arm angle detection (60-90 degrees raised to side)
4. TOF sensor stops car at 7-8cm from user
5. Auto-return to starting position after trash collection

### Manual Mode
1. Say "manual mode" after wake word
2. **Voice commands**: FORWARD, LEFT, RIGHT, STOP, TURN AROUND
3. **Hand gestures**: STOP, TURN_LEFT, TURN_RIGHT, FORWARD, TURN_AROUND
4. Uses GPT API for voice recognition
5. Uses YOLO hand keypoints for gesture recognition

## File Structure

### Core System
- `main.py` - Main entry point and state machine
- `wake_word_detector.py` - Wake word detection (Picovoice)
- `visual_detector.py` - Person detection and tracking (YOLO)
- `motor_controller.py` - PWM speed control
- `servo_controller.py` - PWM steering control
- `tof_sensor.py` - Distance measurement (VL53L0X)
- `voice_recognizer.py` - Voice commands for manual mode
- `hand_gesture_controller.py` - Hand gesture control (YOLO hand keypoints)
- `config.py` - Configuration and GPIO pin assignments
- `state_machine.py` - System state management

### Test Files
- `test_yolo_pose_tracking.py` - YOLO pose detection + tracking test
- `test_mediapipe_combined.py` - MediaPipe + YOLO combined (legacy)
- `test_visual_detection.py` - Visual detection test

### Documentation
- `QUICK_START.md` - Quick setup guide
- `PYTHON_3.12_SETUP.md` - Python 3.12 installation
- `YOLO_IMPROVEMENTS_REPORT.md` - Detailed YOLO capabilities
- `HAND_KEYPOINTS_TRAINING.md` - Hand keypoints model training guide
- `ARCHITECTURE.md` - System architecture

## GPIO Pin Configuration

Default pins (configurable in `config.py`):
- Motor PWM: GPIO 18
- Servo PWM: GPIO 19
- TOF Sensor: I2C (SDA/SCL)

## Key Technologies

- **YOLO11 Pose**: Person detection + pose estimation (17 keypoints)
- **YOLO Tracking**: BYTETracker for persistent person tracking
- **Hand Keypoints**: 21 keypoints per hand for gesture recognition
- **Picamera2**: Raspberry Pi camera interface
- **Picovoice**: Wake word detection

## Documentation

- `QUICK_START.md` - Setup and quick start
- `PYTHON_3.12_SETUP.md` - Python 3.12 installation (required for MediaPipe)
- `YOLO_IMPROVEMENTS_REPORT.md` - Comprehensive YOLO capabilities analysis
- `HAND_KEYPOINTS_TRAINING.md` - Training hand keypoints model
- `ARCHITECTURE.md` - System architecture details

## Troubleshooting

- **Python version**: Must be 3.12 (see `PYTHON_3.12_SETUP.md`)
- **Camera**: `libcamera-hello --list-cameras`
- **GPIO permissions**: `sudo usermod -a -G gpio $USER`
- **Test components**: See individual test files

