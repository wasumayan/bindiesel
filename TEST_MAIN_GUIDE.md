# Step-by-Step Guide: Testing main.py

This guide walks you through testing the complete Bin Diesel system step by step.

---

## Prerequisites

**1. Make sure you're in the project directory:**
```bash
cd ~/Desktop/bindiesel
```

**2. Activate virtual environment:**
```bash
source venv/bin/activate
```

**3. Verify .env file exists (for voice/wake word):**
```bash
cat .env
```

Should show:
```
PICOVOICE_ACCESS_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

**4. Check camera is working:**
```bash
libcamera-hello --list-cameras
```

**5. (Optional) Enable debug mode for detailed output:**
Edit `config.py` and set:
```python
DEBUG_MODE = True
```

---

## Step 1: Start the System

**Run main.py:**
```bash
python main.py
```

**Expected Initial Output:**
```
======================================================================
Bin Diesel System Initializing...
======================================================================
INFO - Initializing wake word detector...
INFO - Wake word detector initialized successfully
INFO - Initializing YOLO pose tracker...
INFO - YOLO pose tracker initialized with tracking enabled
INFO - Initializing motor controller...
INFO - Motor controller initialized successfully
INFO - Initializing servo controller...
INFO - Servo controller initialized successfully
INFO - Initializing TOF sensor...
INFO - TOF sensor initialized successfully
INFO - Initializing voice recognizer...
INFO - Voice recognizer initialized successfully
INFO - Initializing hand gesture controller...
INFO - Hand gesture controller initialized (using shared camera frame)
INFO - Initializing RADD detector...
INFO - RADD detector initialized successfully
======================================================================
System Ready!
======================================================================
Waiting for wake word: 'bin diesel'
Available modes: autonomous, manual, radd
Press Ctrl+C to exit
======================================================================
```

**What to check:**
- ✅ All components initialize without errors
- ✅ "System Ready!" message appears
- ✅ System is waiting for wake word

**If you see errors:**
- Component failed to initialize → Check error message
- Camera error → Run `libcamera-hello` to test camera
- GPIO error → Check permissions: `sudo usermod -a -G gpio $USER`
- Missing .env → Create `.env` file with API keys (voice/wake word won't work)

---

## Step 2: Test Wake Word Detection

**What to do:**
1. System is now listening for wake word
2. Say **"bin diesel"** clearly into microphone
3. Wait for activation

**Expected Output:**
```
INFO - Wake word detected!
INFO - System activated!
INFO - Transitioning to ACTIVE state
```

**What to check:**
- ✅ Wake word detected
- ✅ System transitions to ACTIVE state
- ✅ No errors

**If wake word doesn't work:**
- Check microphone: `arecord -l`
- Check .env has `PICOVOICE_ACCESS_KEY`
- Speak clearly and wait 1-2 seconds
- Check audio permissions: `sudo usermod -a -G audio $USER`

---

## Step 3: Test Autonomous Mode (Following User)

**After wake word activation, system is in ACTIVE state.**

**What to do:**
1. **Stand in front of the camera** (2-3 feet away)
2. **Raise one arm to the side** at 60-90° angle (like a T-pose)
3. Hold arm raised for 1-2 seconds

**Expected Output:**
```
DEBUG - Person detected
DEBUG - Left arm raised: 75.0°
DEBUG - Transitioning to TRACKING_USER state
DEBUG - User detected with raised arm
DEBUG - Transitioning to FOLLOWING_USER state
```

**What should happen:**
- ✅ Person detected
- ✅ Arm raise detected (60-90°)
- ✅ Car starts following you
- ✅ Car steers based on your position
- ✅ Car moves forward when you're centered

**What to check:**
- ✅ Person detection works
- ✅ Arm angle detection works (60-90°)
- ✅ Car follows your position
- ✅ Car steers left/right to center you
- ✅ Car moves forward when centered

**If following doesn't work:**
- Make sure you're in frame
- Raise arm clearly to side (60-90°)
- Check motor/servo connections
- Check debug output for errors

---

## Step 4: Test TOF Sensor (Stopping at Target)

**While car is following you:**

**What to do:**
1. Walk forward slowly
2. Let car get close to you (within 7-8cm)
3. Car should stop automatically

**Expected Output:**
```
DEBUG - TOF distance: 8.5 cm
DEBUG - Target reached (TOF sensor), stopping
DEBUG - Transitioning to STOPPED state
```

**What should happen:**
- ✅ Car stops when close to you
- ✅ Transitions to STOPPED state
- ✅ Waits 10 seconds for trash placement

**What to check:**
- ✅ TOF sensor reads distance
- ✅ Car stops at correct distance (7-8cm)
- ✅ System waits in STOPPED state

**If TOF doesn't work:**
- Check TOF sensor connection (I2C)
- Check distance readings in debug output
- Verify sensor is facing forward

---

## Step 5: Test Return to Start

**After car stops (STOPPED state):**

**What to do:**
1. Wait 10 seconds (system auto-returns)
2. Car should reverse and return to starting position

**Expected Output:**
```
DEBUG - Wait time complete, returning to start
DEBUG - Transitioning to RETURNING_TO_START state
DEBUG - Reversing path segment: speed=0.5, steering=0.0, duration=2.3
```

**What should happen:**
- ✅ Car reverses recorded path
- ✅ Returns to starting position
- ✅ Transitions back to IDLE or ACTIVE

**What to check:**
- ✅ Path was recorded during following
- ✅ Car reverses correctly
- ✅ Returns to approximate starting position

---

## Step 6: Test Manual Mode (Voice Commands)

**Activate manual mode:**

**What to do:**
1. Say wake word: **"bin diesel"**
2. Wait for activation
3. Say: **"manual mode"**

**Expected Output:**
```
INFO - Manual mode activated via voice
DEBUG - Transitioning to MANUAL_MODE state
```

**Test voice commands:**
1. Say **"go forward"** or **"move forward"**
   - Car should move forward
   
2. Say **"turn left"**
   - Car should turn left
   
3. Say **"turn right"**
   - Car should turn right
   
4. Say **"stop"**
   - Car should stop
   
5. Say **"turn around"**
   - Car should turn around

**Expected Output:**
```
DEBUG - Voice command detected: FORWARD
DEBUG - Executing command: FORWARD
DEBUG - Motor forward at 50%
```

**What to check:**
- ✅ Manual mode activates
- ✅ Voice commands are recognized
- ✅ Car responds to commands
- ✅ Commands execute correctly

**If voice doesn't work:**
- Check .env has `OPENAI_API_KEY`
- Check internet connection (OpenAI needs internet)
- Speak clearly and wait 1-3 seconds for response
- Check microphone permissions

---

## Step 7: Test Manual Mode (Hand Gestures)

**In manual mode, test hand gestures:**

**What to do:**
1. Make sure you're in MANUAL_MODE
2. Hold gestures for 0.5 seconds

**Gestures to test:**
- **Thumb pointing right** → Should turn right
- **Thumb pointing left** → Should turn left
- **Palm up facing camera** → Should stop
- **Thumbs up** → Should move forward

**Expected Output:**
```
DEBUG - Gesture command detected: RIGHT
DEBUG - Executing command: RIGHT
```

**What to check:**
- ✅ Gestures are detected
- ✅ Commands execute
- ✅ Car responds to gestures

**If gestures don't work:**
- Make sure hand is clearly visible
- Hold gesture for 0.5+ seconds
- Try more exaggerated gestures
- Check camera feed (if available)

---

## Step 8: Test RADD Mode (Dress Code Enforcement)

**Activate RADD mode:**

**What to do:**
1. Say wake word: **"bin diesel"**
2. Wait for activation
3. Say: **"radd mode"**

**Expected Output:**
```
INFO - RADD mode activated via voice
DEBUG - Transitioning to RADD_MODE state
```

**Test RADD detection:**
1. **Wear shorts** (or no pants) → Should detect violation
2. **Wear sandals** (or no closed-toe shoes) → Should detect violation
3. Car should follow violators

**Expected Output:**
```
🚨 NEW RADD VIOLATOR: Person ID=1
   Violations: SHORTS/NO PANTS + NON-CLOSED-TOE SHOES
   Confidence: 0.85
DEBUG - RADD: Violator 1 centered, moving forward
```

**What to check:**
- ✅ RADD mode activates
- ✅ Violations are detected
- ✅ Car follows violators
- ✅ Violation messages appear

**Note**: RADD mode uses heuristics (pose-based) if clothing model not available. Less accurate but functional.

---

## Step 9: Test Mode Switching

**Test switching between modes:**

**What to do:**
1. In MANUAL_MODE, say **"automatic mode"**
   - Should return to autonomous mode
   
2. In ACTIVE state, say **"manual mode"**
   - Should enter manual mode
   
3. In ACTIVE state, say **"radd mode"**
   - Should enter RADD mode

**Expected Output:**
```
INFO - Returning to automatic mode
DEBUG - Transitioning to ACTIVE state
```

**What to check:**
- ✅ Mode switching works
- ✅ State transitions correctly
- ✅ No errors during transitions

---

## Step 10: Test Emergency Stop

**Test safety features:**

**What to do:**
1. While car is moving (following or manual)
2. Place object very close to TOF sensor (< 5cm)
3. Car should emergency stop

**Expected Output:**
```
WARNING - EMERGENCY STOP: Object too close!
DEBUG - Motor stopped
```

**What to check:**
- ✅ Emergency stop triggers
- ✅ Car stops immediately
- ✅ Safety system works

---

## Step 11: Test Graceful Shutdown

**Stop the system:**

**What to do:**
1. Press **Ctrl+C**

**Expected Output:**
```
INFO - Shutdown signal received, cleaning up...
INFO - Stopping motors...
INFO - Cleaning up components...
INFO - Cleanup complete
```

**What to check:**
- ✅ System shuts down gracefully
- ✅ Motors stop
- ✅ All components cleaned up
- ✅ No errors

---

## Troubleshooting Common Issues

### System Won't Start
- **Check**: All dependencies installed? `pip list`
- **Check**: Virtual environment activated? `which python`
- **Check**: Camera permissions? `sudo raspi-config`

### Wake Word Not Working
- **Check**: .env file exists with `PICOVOICE_ACCESS_KEY`
- **Check**: Microphone working? `arecord -l`
- **Check**: Audio permissions? `sudo usermod -a -G audio $USER`

### Person Not Detected
- **Check**: Good lighting
- **Check**: Person in frame
- **Check**: Camera working? `libcamera-hello`

### Car Doesn't Move
- **Check**: Motor connected to GPIO pins
- **Check**: GPIO permissions? `sudo usermod -a -G gpio $USER`
- **Check**: Motor controller initialized? Check debug output

### Voice Commands Not Working
- **Check**: .env has `OPENAI_API_KEY`
- **Check**: Internet connection (OpenAI needs internet)
- **Check**: Wait 1-3 seconds after speaking

### State Machine Stuck
- **Check**: Debug output for state transitions
- **Check**: No errors in logs
- **Restart**: Press Ctrl+C and restart

---

## Testing Checklist

### Initialization
- [ ] All components initialize
- [ ] No errors during startup
- [ ] System ready message appears

### Wake Word
- [ ] Wake word detected
- [ ] System activates
- [ ] Transitions to ACTIVE state

### Autonomous Mode
- [ ] Person detected
- [ ] Arm raise detected (60-90°)
- [ ] Car follows user
- [ ] Car steers correctly
- [ ] TOF sensor stops car

### Manual Mode
- [ ] Manual mode activates
- [ ] Voice commands work
- [ ] Hand gestures work
- [ ] Car responds to commands

### RADD Mode
- [ ] RADD mode activates
- [ ] Violations detected
- [ ] Car follows violators

### Safety
- [ ] Emergency stop works
- [ ] TOF sensor works
- [ ] Graceful shutdown works

---

## Quick Reference Commands

```bash
# Start system
python main.py

# With debug output (edit config.py first)
DEBUG_MODE = True
python main.py

# Check logs
tail -f logs/*.log

# Test individual components first
python test_yolo_pose_tracking.py  # Test pose detection
python test_motor.py               # Test motor
python test_servo.py               # Test servo
python test_tof.py                 # Test TOF sensor
```

---

## Expected Behavior Summary

1. **Startup**: All components initialize → System ready
2. **Wake Word**: Say "bin diesel" → System activates
3. **Autonomous**: Raise arm → Car follows you
4. **Stop**: Get close → Car stops
5. **Return**: Wait 10s → Car returns to start
6. **Manual**: Say "manual mode" → Use voice/gestures
7. **RADD**: Say "radd mode" → Car follows violators
8. **Shutdown**: Ctrl+C → Graceful cleanup

---

**Start with Step 1 and work through each step. Check each expected behavior before moving to the next step!**

