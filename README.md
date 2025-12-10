# Bin Diesel System

Complete autonomous and manual control system for Bin Diesel car using YOLO pose estimation and tracking.

## Quick Start

**See `COMPLETE_SETUP.md` for full setup instructions.**

1. **Clone repository:**
```bash
git clone https://github.com/wasumayan/bindiesel.git
cd bindiesel
```

2. **Set up environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Pi: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure:**
```bash
cp .env.example .env  # Edit .env with your API keys
# Edit config.py for GPIO pins and camera settings
```

4. **Test components:**
```bash
python test_yolo_pose_tracking.py --fps  # Test pose detection + tracking
python hand_gesture_controller.py        # Test hand gestures
python test_motor.py                     # Test motor
python test_servo.py                     # Test servo
```

5. **Run full system:**
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

### RADD Mode (Dress Code Enforcement)
1. Say "radd mode" after wake word
2. Car automatically drives towards users violating dress code:
   - Not wearing full pants (shorts, skirt, or bare legs detected)
   - Not wearing closed-toe shoes (sandals, flip-flops, or barefoot detected)
3. Uses YOLO pose keypoints to analyze clothing/footwear
4. Follows violating users until they comply or timeout

## File Structure

### Core System
- `main.py` - Main entry point and state machine
- `test_yolo_pose_tracking.py` - YOLO pose detection + tracking (YOLOPoseTracker class)
- `wake_word_detector.py` - Wake word detection (Picovoice)
- `hand_gesture_controller.py` - Hand gesture control (YOLO hand keypoints)
- `motor_controller.py` - PWM speed control
- `servo_controller.py` - PWM steering control
- `tof_sensor.py` - Distance measurement (VL53L0X)
- `voice_recognizer.py` - Voice commands for manual mode (OpenAI GPT)
- `radd_detector.py` - RADD mode: dress code violation detection
- `path_tracker.py` - Path recording for auto-return
- `logger.py` - Structured logging system
- `config.py` - Configuration and GPIO pin assignments
- `state_machine.py` - System state management

### Test Files
- `test_yolo_pose_tracking.py` - YOLO pose detection + tracking test
- `test_yolo_obb.py` - YOLO OBB (Oriented Bounding Boxes) test for trash detection
- `test_visual_detection.py` - Visual detection test
- `test_motor.py` - Motor controller test
- `test_servo.py` - Servo controller test
- `test_tof.py` - TOF sensor test
- `test_voice_commands.py` - Voice recognition test

### Documentation
- `COMPLETE_SETUP.md` - Complete setup guide (start here!)
- `AFTER_CLONE.md` - Post-clone setup instructions
- `PI_GIT_SETUP.md` - Git setup on Raspberry Pi
- `TRAINING_GUIDE.md` - Hand keypoints model training guide
- `SYSTEM_ARCHITECTURE.md` - System architecture details
- `TESTING_GUIDE.md` - Component testing guide
- `INSTALL_PACKAGES.md` - Package installation guide
- `AUTO_START_SETUP.md` - Auto-start on boot setup

## GPIO Pin Configuration

Default pins (configurable in `config.py`):
- Motor PWM: GPIO 12 (pin 32)
- Servo PWM: GPIO 13 (pin 33)
- TOF Sensor: GPIO 23 (pin 16) - I2C interface

## Key Technologies

- **YOLO11 Pose**: Person detection + pose estimation (17 keypoints)
- **YOLO Tracking**: BYTETracker for persistent person tracking
- **Hand Keypoints**: 21 keypoints per hand for gesture recognition
- **Picamera2**: Raspberry Pi camera interface
- **Picovoice**: Wake word detection

## Documentation

- `COMPLETE_SETUP.md` - Complete setup guide (start here!)
- `AFTER_CLONE.md` - Post-clone setup instructions
- `TRAINING_GUIDE.md` - Train hand keypoints model on MacBook
- `SYSTEM_ARCHITECTURE.md` - System architecture details
- `TESTING_GUIDE.md` - Component testing guide
- `AUTO_START_SETUP.md` - Auto-start on boot

## Troubleshooting

- **Python version**: Python 3.9+ (tested on 3.11)
- **Camera**: `libcamera-hello --list-cameras`
- **GPIO permissions**: `sudo usermod -a -G gpio $USER` (logout/login)
- **Camera upside down**: Set `CAMERA_ROTATION = 180` in `config.py`
- **Colors flipped**: Set `CAMERA_SWAP_RB = True` in `config.py`
- **Test components**: See `TESTING_GUIDE.md`

## Requirements

- Raspberry Pi (tested on Pi 4/5)
- Raspberry Pi Camera Module
- Python 3.9+ (venv recommended)
- See `requirements.txt` for Python packages
- See `COMPLETE_SETUP.md` for full requirements

