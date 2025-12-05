# Testing Guide for Bin Diesel System

This guide helps you test each component individually without connecting to the car hardware.

## Prerequisites

1. Install all dependencies:
```bash
pip3 install --break-system-packages -r requirements.txt
```

2. Set up environment variables in `.env`:
```
PICOVOICE_ACCESS_KEY=your_key
OPENAI_API_KEY=your_key
```

## Testing Individual Components

### 1. Wake Word Detection

**Test Script**: `test_wake_word.py`

```bash
python3 test_wake_word.py
```

**What it does**:
- Listens for "bin diesel" wake word
- Prints debug messages when detected
- No hardware needed (just microphone)

**Expected Output**:
```
[WakeWord] Listening for 'bin diesel'...
[WakeWord] WAKE WORD DETECTED: 'bin diesel'
✓ Wake word detected!
```

**Troubleshooting**:
- If no audio input: Check microphone permissions
- If not detecting: Speak clearly, check access key

---

### 2. Visual Detection (Person + Arm Raising)

**Test Script**: `test_visual_detection.py`

```bash
python3 test_visual_detection.py
```

**What it does**:
- Shows camera feed with detection overlays
- Detects person in frame
- Detects arm raising gesture
- Displays angle and position info

**Expected Output**:
- Window showing camera feed
- Green boxes around detected persons
- "ARM RAISED!" text when arm is raised
- Terminal output with detection data

**Controls**:
- Press 'q' to quit
- Press 'd' to toggle debug mode

**Troubleshooting**:
- If no camera: Check `libcamera-hello --list-cameras`
- If slow: Lower resolution in config.py

---

### 3. Motor Controller (Mock Mode)

**Test Script**: `test_motor.py`

```bash
python3 test_motor.py
```

**What it does**:
- Tests motor PWM signals (in mock mode if not on Pi)
- Simulates speed changes
- Shows what PWM values would be sent

**Expected Output**:
```
[MotorController] Mock mode - GPIO 18 would be used
[MotorController] Speed set to 30.0% (duty cycle: 30.0%)
[MotorController] Speed set to 50.0% (duty cycle: 50.0%)
[MotorController] Motor stopped
```

**On Real Hardware**:
- Connect motor to GPIO 18
- Watch motor speed change
- Adjust PWM values in config.py if needed

---

### 4. Servo Controller (Mock Mode)

**Test Script**: `test_servo.py`

```bash
python3 test_servo.py
```

**What it does**:
- Tests servo PWM signals (in mock mode if not on Pi)
- Sweeps left to right
- Shows steering positions

**Expected Output**:
```
[ServoController] Mock mode - GPIO 19 would be used
[ServoController] Position: CENTER (duty cycle: 7.5%)
[ServoController] Position: LEFT 50.0% (duty cycle: 6.25%)
[ServoController] Position: RIGHT 50.0% (duty cycle: 8.75%)
```

**On Real Hardware**:
- Connect servo to GPIO 19
- Watch servo move
- Adjust duty cycle values in config.py if needed

---

### 5. TOF Sensor (Distance Measurement)

**Test Script**: `test_tof.py`

```bash
python3 test_tof.py
```

**What it does**:
- Reads distance from VL53L0X sensor
- Shows real-time distance readings
- Indicates when stop/emergency thresholds are reached

**Expected Output**:
```
[TOFSensor] VL53L0X initialized
Distance: 45.2cm (452mm) [SAFE]
Distance: 8.5cm (85mm) [STOP!]
Distance: 6.8cm (68mm) [EMERGENCY STOP!]
```

**Troubleshooting**:
- If sensor not found: Check I2C connection
- Run: `i2cdetect -y 1` to see I2C devices
- Should see device at address 0x29

---

### 6. Voice Recognition (Manual Mode)

**Test Script**: `test_voice_commands.py`

```bash
python3 test_voice_commands.py
```

**What it does**:
- Records audio from microphone
- Sends to OpenAI Whisper API
- Recognizes commands: FORWARD, LEFT, RIGHT, STOP, TURN AROUND

**Expected Output**:
```
[VoiceRecognizer] Recording... (speak now)
[VoiceRecognizer] Recording complete
[VoiceRecognizer] Transcribed: 'forward'
[VoiceRecognizer] Command recognized: FORWARD
✓ Command: FORWARD
```

**Troubleshooting**:
- If no API key: Set OPENAI_API_KEY in .env
- If transcription fails: Check internet connection
- Speak clearly and wait for "Recording..." prompt

---

### 7. Full System Test (No Hardware)

**Test Script**: `test_full_system.py`

```bash
python3 test_full_system.py
```

**What it does**:
- Tests entire workflow without hardware
- Simulates all states
- Shows state transitions
- Useful for debugging logic

**Expected Output**:
```
[StateMachine] Transition: idle → active
[StateMachine] Transition: active → tracking_user
[StateMachine] Transition: tracking_user → following_user
[StateMachine] Transition: following_user → stopped
```

---

## Debug Mode

Enable debug mode in any test script:

```bash
python3 test_visual_detection.py --debug
```

Or set in config.py:
```python
DEBUG_MODE = True
```

Debug mode provides:
- Detailed logging
- Frame-by-frame analysis
- Performance metrics
- State transition details

## Testing Workflows

### Autonomous Mode Workflow Test

1. Start visual detection test
2. Stand in front of camera
3. Raise your arm
4. Watch detection and angle calculation
5. Verify car would move in correct direction

### Manual Mode Workflow Test

1. Start voice recognition test
2. Say "FORWARD"
3. Verify command is recognized
4. Test all commands: LEFT, RIGHT, STOP, TURN AROUND

### TOF Sensor Integration Test

1. Start TOF sensor test
2. Move hand toward sensor
3. Verify stop at 8cm
4. Verify emergency stop at 7cm

## Performance Testing

### FPS Measurement

```bash
python3 test_visual_detection.py --fps
```

Shows frames per second for visual detection.

### Latency Measurement

```bash
python3 test_voice_commands.py --latency
```

Shows time from speech to command recognition.

## Common Issues

### "Module not found"
- Install dependencies: `pip3 install --break-system-packages -r requirements.txt`
- Check Python version: `python3 --version` (should be 3.6+)

### "Camera not found"
- Enable camera: `sudo raspi-config` → Interface Options → Camera
- Check: `libcamera-hello --list-cameras`

### "GPIO permission denied"
- Add user to gpio group: `sudo usermod -a -G gpio $USER`
- Log out and back in

### "I2C not found"
- Enable I2C: `sudo raspi-config` → Interface Options → I2C
- Check: `i2cdetect -y 1`

## Next Steps

Once all tests pass:
1. Connect hardware (motor, servo, TOF sensor)
2. Run full system: `python3 main.py`
3. Test with real car
4. Adjust PWM values in config.py as needed

