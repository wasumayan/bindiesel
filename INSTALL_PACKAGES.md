# Individual Package Installation Commands

Install these packages **one at a time** or in small groups. This helps identify any issues and is often faster.

## Essential Packages (Install These First)

```bash
pip install --upgrade pip
```

```bash
pip install numpy
```

```bash
pip install opencv-python
```

```bash
pip install ultralytics
```

```bash
pip install python-dotenv
```

---

## Core Dependencies (Install Next)

```bash
pip install pvporcupine
```

```bash
pip install pyaudio
```

```bash
pip install openai
```

---

## Optional but Recommended

```bash
pip install mediapipe
```

*(Only if you want to use MediaPipe features - not required for YOLO-only setup)*

---

## Quick Install (All at Once - If You Prefer)

```bash
pip install numpy opencv-python ultralytics python-dotenv pvporcupine pyaudio openai
```

---

## System Packages (via apt)

```bash
sudo apt install -y python3-picamera2 libcamera-dev
```

```bash
sudo apt install -y portaudio19-dev python3-pyaudio
```

---

## Verify Installation

```bash
python -c "import numpy; import cv2; from ultralytics import YOLO; print('Core packages OK!')"
```

```bash
python -c "import pvporcupine; import pyaudio; print('Audio packages OK!')"
```

```bash
python -c "import openai; from dotenv import load_dotenv; print('API packages OK!')"
```

---

## If Packages Fail

**If numpy fails:**
```bash
pip install numpy --upgrade
```

**If opencv fails:**
```bash
pip install opencv-contrib-python
```

**If ultralytics fails:**
```bash
pip install ultralytics --upgrade --no-cache-dir
```

**If pyaudio fails:**
```bash
sudo apt install -y portaudio19-dev
pip install pyaudio
```

---

## Minimal Installation (Just YOLO + Basic)

If you only want the core functionality:

```bash
pip install numpy opencv-python ultralytics python-dotenv
```

This gives you:
- ✅ YOLO pose tracking
- ✅ Camera support
- ✅ Basic functionality

You'll be missing:
- ❌ Wake word detection (needs pvporcupine)
- ❌ Voice commands (needs openai, pyaudio)
- ❌ Hand gestures (works with YOLO pose model)

---

## Recommended Order

1. **Core first:**
```bash
pip install numpy opencv-python ultralytics python-dotenv
```

2. **Test core:**
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

3. **Audio packages:**
```bash
sudo apt install -y portaudio19-dev python3-pyaudio
pip install pvporcupine pyaudio
```

4. **API packages:**
```bash
pip install openai
```

5. **Test everything:**
```bash
python test_yolo_pose_tracking.py --fps
```

---

**This approach is often faster and helps identify any problematic packages!**


