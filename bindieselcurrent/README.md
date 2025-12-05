# Bin Diesel Current System

Complete autonomous and manual control system for Bin Diesel car.

## System Architecture

See `ARCHITECTURE.md` for detailed system design.

## Quick Start

1. Install dependencies:
```bash
pip3 install --break-system-packages -r requirements.txt
```

2. Configure GPIO pins in `config.py` (if needed)

3. Run the system:
```bash
python3 main.py
```

## Features

### Autonomous Mode
1. Wake word: "bin diesel"
2. Arm raising detection → car follows user
3. TOF sensor stops car at 7-8cm from user
4. Auto-return to starting position after trash collection

### Manual Mode
1. Say "manual mode" after wake word
2. Voice commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND
3. Uses GPT API for voice recognition

## File Structure

- `main.py` - Main entry point and state machine
- `wake_word_detector.py` - Wake word detection (Picovoice)
- `visual_detector.py` - Arm raising and person tracking
- `motor_controller.py` - PWM speed control
- `servo_controller.py` - PWM steering control
- `tof_sensor.py` - Distance measurement (VL53L0X)
- `voice_recognizer.py` - Voice commands for manual mode
- `config.py` - Configuration and GPIO pin assignments
- `state_machine.py` - System state management

## GPIO Pin Configuration

Default pins (configurable in `config.py`):
- Motor PWM: GPIO 18
- Servo PWM: GPIO 19
- TOF Sensor: I2C (SDA/SCL)

## Troubleshooting

- Check GPIO permissions: `sudo usermod -a -G gpio $USER`
- Verify camera: `libcamera-hello --list-cameras`
- Test TOF sensor: `python3 test_tof.py`
- Test motors: `python3 test_motors.py`

