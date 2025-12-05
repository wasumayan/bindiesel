# Setup and Testing Guide

Complete guide for setting up and testing the Bin Diesel system.

## Initial Setup

### 1. Environment Variables

Create your `.env` file:

```bash
cd bindieselcurrent
cp .env.example .env
nano .env  # or use your preferred editor
```

Fill in your keys:
```
PICOVOICE_ACCESS_KEY=your_actual_key_here
OPENAI_API_KEY=your_actual_key_here
```

**Getting Keys:**
- **Picovoice**: https://console.picovoice.ai/ (free account)
- **OpenAI**: https://platform.openai.com/api-keys (requires payment method)

### 2. Install Dependencies

```bash
# System dependencies (Raspberry Pi)
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-pip

# Python dependencies
pip3 install --break-system-packages -r requirements.txt
```

### 3. Hardware Setup

**GPIO Connections:**
- Motor PWM: GPIO 18 (BCM)
- Servo PWM: GPIO 19 (BCM)
- TOF Sensor: I2C (SDA: GPIO 2, SCL: GPIO 3)

**Enable Interfaces:**
```bash
sudo raspi-config
# Interface Options → I2C → Enable
# Interface Options → Camera → Enable
```

**GPIO Permissions:**
```bash
sudo usermod -a -G gpio $USER
# Log out and back in for changes to take effect
```

## Testing Individual Components

### Test 1: Wake Word Detection

**No hardware needed** (just microphone)

```bash
python3 test_wake_word.py
```

**What to do:**
1. Say "bin diesel" clearly
2. Should see: `[WakeWord] WAKE WORD DETECTED: 'bin diesel'`

**Troubleshooting:**
- If no detection: Check microphone permissions
- If error: Verify PICOVOICE_ACCESS_KEY in .env
- Test microphone: `arecord -d 3 test.wav && aplay test.wav`

---

### Test 2: Visual Detection (Person + Arm)

**Needs:** Camera Module 3 Wide

```bash
python3 test_visual_detection.py
```

**What to do:**
1. Stand in front of camera
2. Raise your arm to the side
3. Should see:
   - Green box around you
   - "ARM RAISED!" text
   - Angle displayed
   - Terminal output with detection data

**Controls:**
- Press 'q' to quit
- Press 'd' to toggle debug mode

**Troubleshooting:**
- If no camera: `libcamera-hello --list-cameras`
- If slow: Lower resolution in config.py (CAMERA_WIDTH, CAMERA_HEIGHT)
- If no person detected: Check lighting, move closer

---

### Test 3: Motor Controller

**Needs:** Motor connected to GPIO 18

```bash
python3 test_motor.py
```

**What to do:**
1. Watch motor speed change (30% → 50% → 70% → 90%)
2. Motor should stop at end

**Troubleshooting:**
- If motor doesn't move: Check GPIO 18 connection
- If wrong speed: Adjust PWM values in config.py (MOTOR_SLOW, MOTOR_MEDIUM, etc.)
- If permission error: `sudo usermod -a -G gpio $USER` then logout/login

**Mock Mode:** Works without hardware, shows what PWM values would be sent

---

### Test 4: Servo Controller

**Needs:** Servo connected to GPIO 19

```bash
python3 test_servo.py
```

**What to do:**
1. Watch servo sweep left → center → right
2. Should return to center

**Troubleshooting:**
- If servo doesn't move: Check GPIO 19 connection
- If wrong position: Adjust duty cycle in config.py (SERVO_CENTER, SERVO_LEFT_MAX, etc.)
- Typical servo range: 2.5% (0°) to 12.5% (180°)

**Mock Mode:** Works without hardware, shows what PWM values would be sent

---

### Test 5: TOF Sensor (Distance)

**Needs:** VL53L0X sensor connected via I2C

```bash
python3 test_tof.py
```

**What to do:**
1. Move hand toward sensor
2. Watch distance readings
3. Should show:
   - `[SAFE]` when > 8cm
   - `[STOP!]` when 7-8cm
   - `[EMERGENCY STOP!]` when < 7cm

**Troubleshooting:**
- If sensor not found: `i2cdetect -y 1` (should see device at 0x29)
- If no I2C: Enable in raspi-config
- Check wiring: SDA to GPIO 2, SCL to GPIO 3

---

### Test 6: Voice Commands (Manual Mode)

**Needs:** Microphone + Internet (for OpenAI API)

```bash
python3 test_voice_commands.py
```

**What to do:**
1. Say one of: "FORWARD", "LEFT", "RIGHT", "STOP", "TURN AROUND"
2. Wait for "Listening..." prompt
3. Should see transcription and recognized command

**Troubleshooting:**
- If no API key: Check OPENAI_API_KEY in .env
- If transcription fails: Check internet connection
- If wrong command: Speak clearly, wait for prompt
- Test microphone: `arecord -d 3 test.wav && aplay test.wav`

---

### Test 7: Full System (Simulation)

**No hardware needed** - tests logic only

```bash
python3 test_full_system.py
```

**What to do:**
1. Watch state transitions
2. Simulates full workflow

---

## Testing Full System

### Prerequisites

All individual tests should pass first!

### Run Full System

```bash
python3 main.py
```

**Workflow:**
1. Say "bin diesel" → System activates
2. Raise arm → Car starts tracking
3. Car follows you → Keeps you centered
4. Car stops at 7-8cm → Wait 10 seconds
5. Car returns to start → Executes reverse path

**Manual Mode:**
1. Say "bin diesel" → System activates
2. Say "manual mode" → Enter manual control
3. Say commands: "FORWARD", "LEFT", "RIGHT", "STOP"
4. Commands execute continuously until changed
5. Say "automatic mode" → Return to auto

**Debug Mode:**
Enable in config.py:
```python
DEBUG_MODE = True
DEBUG_VISUAL = True
DEBUG_MOTOR = True
# etc.
```

---

## Common Issues

### "Module not found"
```bash
pip3 install --break-system-packages -r requirements.txt
```

### "Camera not found"
```bash
sudo raspi-config  # Enable camera
libcamera-hello --list-cameras  # Verify
```

### "GPIO permission denied"
```bash
sudo usermod -a -G gpio $USER
# Log out and back in
```

### "I2C not found"
```bash
sudo raspi-config  # Enable I2C
i2cdetect -y 1  # Verify
```

### "OpenAI API error"
- Check API key in .env
- Verify internet connection
- Check OpenAI account has credits

### "Picovoice error"
- Check PICOVOICE_ACCESS_KEY in .env
- Verify wake word model file exists

---

## Configuration Tuning

### Motor Speed
Edit `config.py`:
```python
MOTOR_SLOW = 0.3    # 30% - adjust as needed
MOTOR_MEDIUM = 0.5  # 50% - adjust as needed
FOLLOW_SPEED = MOTOR_SLOW  # Speed when following user
```

### Servo Position
Edit `config.py`:
```python
SERVO_CENTER = 0.075   # 7.5% - adjust for your servo
SERVO_LEFT_MAX = 0.05  # 5% - adjust for your servo
SERVO_RIGHT_MAX = 0.10 # 10% - adjust for your servo
```

### TOF Thresholds
Edit `config.py`:
```python
TOF_STOP_DISTANCE_MM = 80  # 8cm - when to stop
TOF_EMERGENCY_DISTANCE_MM = 70  # 7cm - emergency stop
```

### Camera Resolution
Edit `config.py`:
```python
CAMERA_WIDTH = 640   # Lower = faster
CAMERA_HEIGHT = 480  # Lower = faster
```

---

## Performance Tips

1. **Lower camera resolution** for better FPS
2. **Increase frame skipping** in visual detection
3. **Use gpt-4o-mini** for faster voice recognition
4. **Disable debug mode** in production
5. **Optimize YOLO confidence** threshold

---

## Next Steps

Once all tests pass:
1. Connect all hardware
2. Calibrate PWM values in config.py
3. Test full system
4. Adjust thresholds as needed
5. Deploy!

