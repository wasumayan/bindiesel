# Complete Testing Guide

This guide walks you through testing each component individually, then the full system.

---

## Prerequisites

**Make sure you're in the project directory and venv is activated:**
```bash
cd ~/Desktop/bindiesel
source venv/bin/activate
```

**Verify .env file exists:**
```bash
cat .env
```

Should show:
```
PICOVOICE_ACCESS_KEY=your_key
OPENAI_API_KEY=your_key
```

---

## 1. Test Wake Word Detection

**Purpose**: Verify "bin diesel" wake word detection works

**Command:**
```bash
python -c "from wake_word_detector import WakeWordDetector; import config; import os; from dotenv import load_dotenv; load_dotenv(); w = WakeWordDetector('bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn', os.getenv('PICOVOICE_ACCESS_KEY')); w.start_listening(); print('Listening for wake word: \"bin diesel\"... Press Ctrl+C to exit'); import time; [time.sleep(1) for _ in iter(int, 1) if not w.detect()]"
```

**OR use the test script:**
```bash
python test_voice_commands.py
```
*(This tests voice recognition, but wake word is part of it)*

**Expected Behavior:**
- System starts listening
- Say "bin diesel" clearly
- Should print "[Main] System activated!" or similar
- Press Ctrl+C to exit

**Troubleshooting:**
- **No audio input**: Check microphone with `arecord -l`
- **Permission denied**: `sudo usermod -a -G audio $USER` (then logout/login)
- **No wake word detected**: Speak clearly, check microphone volume

---

## 2. Test Pose Detection & Tracking

**Purpose**: Verify YOLO pose detection and person tracking works

**Command:**
```bash
python test_yolo_pose_tracking.py --fps
```

**Expected Behavior:**
- Camera feed opens (or shows in terminal if no display)
- Detects person in frame
- Shows bounding boxes, pose keypoints, tracking IDs
- Displays FPS in terminal
- Press `q` or Ctrl+C to exit

**What to Look For:**
- ✅ Person detected with bounding box
- ✅ Pose keypoints visible (17 keypoints)
- ✅ Tracking ID shown (same person = same ID)
- ✅ Arm angle detection (60-90° raise)
- ✅ FPS > 10 (should be 15-30 FPS on Pi 4/5)

**Troubleshooting:**
- **Camera not found**: `libcamera-hello --list-cameras`
- **Camera permission error**: `sudo raspi-config` → Interface Options → Camera → Enable
- **Low FPS**: Normal on Pi, but should be > 10 FPS
- **No person detected**: Make sure you're in frame, good lighting

**Test Arm Raising:**
- Raise one arm to the side at 60-90° angle
- Should detect `arm_raised: True`
- Should show arm angle in output

---

## 3. Test Hand Gesture Detection

**Purpose**: Verify hand gesture recognition works

**Command:**
```bash
python hand_gesture_controller.py
```

**Expected Behavior:**
- Camera feed starts
- Detects hand gestures
- Prints detected commands when gesture held for 0.5 seconds
- Press Ctrl+C to exit

**Gestures to Test:**
- **STOP**: Raise hand(s) high above shoulder
- **TURN_LEFT**: Extend left hand to left side
- **TURN_RIGHT**: Extend right hand to right side
- **FORWARD (COME)**: Extend hand forward (beckoning motion)
- **TURN_AROUND**: Extend hand backward/up

**What to Look For:**
- ✅ Hand detected in frame
- ✅ Gesture recognized after holding 0.5s
- ✅ Command printed: `[TEST] Command detected: STOP`

**Troubleshooting:**
- **No hand detected**: Make sure hand is clearly visible
- **Gestures not recognized**: Try more exaggerated gestures
- **False positives**: Gesture hold time prevents accidental commands

**Note**: If hand keypoints model not available, falls back to pose model (less accurate but works)

---

## 4. Test Voice Recognition

**Purpose**: Verify voice command recognition works

**Command:**
```bash
python test_voice_commands.py
```

**Expected Behavior:**
- Starts listening for voice commands
- Say a command (e.g., "go forward", "turn left", "stop")
- System interprets via OpenAI GPT
- Prints recognized command
- Press Ctrl+C to exit

**Commands to Test:**
- "go forward" or "move forward" → Should return `FORWARD`
- "turn left" → Should return `LEFT`
- "turn right" → Should return `RIGHT`
- "stop" → Should return `STOP`
- "turn around" → Should return `TURN_AROUND`
- "manual mode" → Should return `MANUAL_MODE`
- "automatic mode" → Should return `AUTOMATIC_MODE`

**What to Look For:**
- ✅ Microphone picks up audio
- ✅ Speech transcribed correctly
- ✅ OpenAI interprets command correctly
- ✅ Returns correct command string

**Troubleshooting:**
- **No audio input**: Check microphone permissions
- **OpenAI API error**: Check API key in .env file
- **Internet required**: OpenAI needs internet connection
- **Slow response**: Normal (1-3 seconds for API call)

**Test .env Loading:**
```bash
python test_env_loading.py
```

Should show:
- ✅ .env file loaded
- ✅ OPENAI_API_KEY found
- ✅ OpenAI client initialized

---

## 5. Test Individual Hardware Components

### Test Motor
```bash
python test_motor.py
```

**Expected**: Motor runs at different speeds, then stops

### Test Servo
```bash
python test_servo.py
```

**Expected**: Servo moves left, center, right

### Test TOF Sensor
```bash
python test_tof.py
```

**Expected**: Distance readings printed, changes when object moves

---

## 6. Test Full System

**Purpose**: Test complete system integration

**Command:**
```bash
python main.py
```

**Expected Behavior:**

1. **Initialization:**
   - All components initialize
   - "System Ready!" message
   - "Waiting for wake word: 'bin diesel'"

2. **Wake Word Activation:**
   - Say "bin diesel"
   - System activates → "System activated!"
   - Transitions to ACTIVE state

3. **Autonomous Mode:**
   - Raise arm to side (60-90°)
   - System detects → "Autonomous mode: User detected with raised arm"
   - Transitions to TRACKING_USER
   - Keep arm raised → Transitions to FOLLOWING_USER
   - Car follows you, steering based on your position
   - When you get close (TOF sensor) → Stops at STOPPED state
   - After 10 seconds → Returns to start (RETURNING_TO_START)

4. **Manual Mode:**
   - In ACTIVE state, say "manual mode"
   - Enters MANUAL_MODE
   - Use voice commands or hand gestures
   - Car responds to commands
   - Say "automatic mode" to return

**What to Look For:**
- ✅ All components initialize without errors
- ✅ Wake word detection works
- ✅ Person detection and tracking works
- ✅ Arm raising detection works
- ✅ Car follows correctly
- ✅ TOF sensor stops car at correct distance
- ✅ Manual mode voice/gesture commands work
- ✅ State transitions happen correctly

**Troubleshooting:**
- **Component fails to initialize**: Check error message, verify dependencies
- **Wake word not detected**: Check microphone, speak clearly
- **Person not detected**: Check camera, lighting, position in frame
- **Car doesn't move**: Check GPIO permissions, motor connections
- **State machine stuck**: Check debug output, verify state transitions

---

## 7. Test with Debug Mode

**Enable debug mode in config.py:**
```python
DEBUG_MODE = True
DEBUG_VISUAL = True
DEBUG_MOTOR = True
DEBUG_SERVO = True
DEBUG_TOF = True
DEBUG_VOICE = True
DEBUG_STATE = True
```

**Then run:**
```bash
python main.py
```

**You'll see detailed output:**
- State transitions
- Visual detection results
- Motor/servo commands
- TOF sensor readings
- Voice recognition results

---

## Testing Checklist

### Component Tests
- [ ] Wake word detection works
- [ ] Pose detection works (person detected)
- [ ] Pose tracking works (same person = same ID)
- [ ] Arm raising detection works (60-90°)
- [ ] Hand gesture detection works
- [ ] Voice recognition works
- [ ] Motor controller works
- [ ] Servo controller works
- [ ] TOF sensor works

### Integration Tests
- [ ] System initializes completely
- [ ] Wake word activates system
- [ ] Autonomous mode follows user
- [ ] Manual mode responds to commands
- [ ] State machine transitions correctly
- [ ] Safety systems work (TOF emergency stop)
- [ ] Path tracking records movement
- [ ] Return navigation works

### System Tests
- [ ] Full autonomous cycle works (wake → follow → stop → return)
- [ ] Manual mode works (voice + gestures)
- [ ] Mode switching works (autonomous ↔ manual)
- [ ] Error handling works (component failures)
- [ ] Graceful shutdown works (Ctrl+C)

---

## Quick Test Commands Summary

```bash
# Wake word
python -c "from wake_word_detector import WakeWordDetector; ..."

# Pose detection
python test_yolo_pose_tracking.py --fps

# Hand gestures
python hand_gesture_controller.py

# Voice recognition
python test_voice_commands.py

# Hardware
python test_motor.py
python test_servo.py
python test_tof.py

# Full system
python main.py
```

---

## Common Issues & Solutions

### Camera Issues
- **Permission denied**: `sudo raspi-config` → Enable camera
- **Camera busy**: Kill other processes: `killall pipewire wireplumber`
- **No camera found**: Check connection, run `libcamera-hello --list-cameras`

### Audio Issues
- **No microphone**: Check with `arecord -l`
- **Permission denied**: `sudo usermod -a -G audio $USER` (logout/login)
- **ALSA warnings**: Usually harmless, can ignore

### GPIO Issues
- **Permission denied**: `sudo usermod -a -G gpio $USER` (logout/login)
- **Pin conflicts**: Check config.py for pin assignments

### YOLO Issues
- **Model download slow**: Normal, wait for first download
- **Low FPS**: Normal on Pi, should be > 10 FPS
- **No detections**: Check lighting, person position

### API Issues
- **OpenAI errors**: Check API key, internet connection
- **Picovoice errors**: Check access key in .env

---

**Test each component individually first, then test the full system!**


