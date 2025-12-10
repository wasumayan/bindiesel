# Quick Test Guide for Raspberry Pi

## ✅ Ready to Test RIGHT NOW (No Extra Setup)

### 1. **YOLO Pose Detection & Tracking** ⭐ RECOMMENDED FIRST
```bash
cd ~/Desktop/bindiesel
source venv/bin/activate
python test_yolo_pose_tracking.py
```

**What it tests:**
- ✅ Camera works
- ✅ YOLO pose model loads (auto-downloads if needed)
- ✅ Person detection
- ✅ Pose keypoints (17 keypoints)
- ✅ Person tracking (same person = same ID)
- ✅ Arm angle detection (60-90° raise)

**What to do:**
- Stand in front of camera
- Raise arm to side (60-90° angle)
- Should see bounding box, keypoints, tracking ID
- Press `q` to quit

**Expected output:**
- Camera feed with overlays
- Terminal shows: "Person detected", "Arm raised", FPS

---

### 2. **Hardware Components** (If Connected)

#### Motor Test
```bash
python test_motor.py
```
- Motor should run at different speeds, then stop
- **Note**: Make sure motor is connected to GPIO pins

#### Servo Test
```bash
python test_servo.py
```
- Servo should move left, center, right
- **Note**: Make sure servo is connected to GPIO pins

#### TOF Sensor Test
```bash
python test_tof.py
```
- Should print distance readings
- Move object closer/farther, readings should change
- **Note**: Make sure TOF sensor is connected via I2C

---

### 3. **YOLO OBB (Oriented Bounding Boxes)**
```bash
python test_yolo_obb.py
```

**What it tests:**
- ✅ YOLO OBB model for object detection
- ✅ Rotation-aware bounding boxes
- ✅ Trash detection (bottles, cans, etc.)

**What to do:**
- Point camera at objects
- Should detect objects with rotated bounding boxes
- Press `q` to quit

---

### 4. **Basic Visual Detection**
```bash
python test_visual_detection.py
```

**What it tests:**
- ✅ Person detection
- ✅ Angle calculation
- ✅ Centering detection

---

## ⚠️ Needs Setup (But Can Test)

### 5. **Voice Recognition** (Needs .env file)
```bash
# First, make sure .env exists with OPENAI_API_KEY
cat .env  # Should show your API keys

# Then test
python test_voice_commands.py
```

**What it tests:**
- ✅ Microphone input
- ✅ Speech-to-text
- ✅ OpenAI command interpretation

**Commands to try:**
- "go forward"
- "turn left"
- "stop"

---

### 6. **Hand Gesture Controller** (Works with fallback)
```bash
python hand_gesture_controller.py
```

**What it tests:**
- ✅ Hand detection (uses pose model if hand model not available)
- ✅ Gesture recognition
- ✅ Command output

**Gestures:**
- Thumb right → TURN_RIGHT
- Thumb left → TURN_LEFT
- Palm up → STOP
- Thumbs up → FORWARD

**Note**: Works with pose model fallback, but better with trained hand model

---

### 7. **Full System** (Needs .env file)
```bash
python main.py
```

**What it tests:**
- ✅ All components together
- ✅ Wake word detection (needs .env)
- ✅ State machine
- ✅ Autonomous following
- ✅ Manual mode

**Note**: Will work even if some components fail (graceful degradation)

---

## 🚫 Not Ready Yet (Needs Model Training/Download)

### 8. **RADD Mode** (Needs clothing model)
- Currently uses heuristics (pose-based)
- Will work but less accurate
- For full accuracy, need to train/download clothing model

### 9. **Hand Gesture Model** (Optional)
- Works with pose model fallback
- For better accuracy, train hand keypoints model
- See `TRAINING_GUIDE.md`

---

## Quick Test Order (Recommended)

1. **Start with pose detection** (easiest, no dependencies):
   ```bash
   python test_yolo_pose_tracking.py
   ```

2. **Test hardware** (if connected):
   ```bash
   python test_motor.py
   python test_servo.py
   python test_tof.py
   ```

3. **Test voice** (if .env set up):
   ```bash
   python test_voice_commands.py
   ```

4. **Test full system**:
   ```bash
   python main.py
   ```

---

## Troubleshooting

### Camera Issues
```bash
# Check camera
libcamera-hello --list-cameras

# Enable camera if needed
sudo raspi-config  # Interface Options → Camera → Enable
```

### Permission Issues
```bash
# GPIO permissions
sudo usermod -a -G gpio $USER

# Audio permissions
sudo usermod -a -G audio $USER

# Then logout/login or reboot
```

### Model Download
- First run will download YOLO models (yolo11n-pose.pt, etc.)
- This can take a few minutes
- Models are cached after first download

### Low FPS
- Normal on Raspberry Pi
- Should get 10-20 FPS with pose detection
- Lower resolution = higher FPS

---

## What's Working vs What Needs Setup

| Component | Status | Notes |
|-----------|--------|-------|
| YOLO Pose Detection | ✅ Ready | Auto-downloads model |
| Person Tracking | ✅ Ready | Built into pose tracker |
| Arm Angle Detection | ✅ Ready | 60-90° detection |
| Motor Control | ⚠️ If connected | Needs GPIO setup |
| Servo Control | ⚠️ If connected | Needs GPIO setup |
| TOF Sensor | ⚠️ If connected | Needs I2C setup |
| Voice Recognition | ⚠️ Needs .env | Needs OpenAI API key |
| Wake Word | ⚠️ Needs .env | Needs Picovoice key |
| Hand Gestures | ✅ Works (fallback) | Better with trained model |
| RADD Mode | ⚠️ Heuristics only | Needs clothing model for accuracy |
| Full System | ⚠️ Partial | Works with graceful degradation |

---

## Next Steps After Testing

1. **If pose detection works**: You're good to go! System core is working
2. **If hardware works**: Full autonomous mode is ready
3. **If voice works**: Manual mode is ready
4. **If everything works**: Full system integration is ready!

---

**Start with `python test_yolo_pose_tracking.py` - it's the easiest and most important test!**

