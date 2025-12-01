# Vision System Installation Guide for Raspberry Pi

## Problem: Python Version and Package Compatibility

If you're seeing errors with OpenCV and MediaPipe, follow these steps:

## Step 1: Use System OpenCV (Recommended)

Instead of installing OpenCV via pip, use the system package:

```bash
# Exit virtual environment first
deactivate

# Install system OpenCV
sudo apt-get update
sudo apt-get install -y python3-opencv python3-opencv-contrib

# Verify installation
python3 -c "import cv2; print(cv2.__version__)"
```

## Step 2: Re-enter Virtual Environment and Install Other Packages

```bash
# Re-enter venv
source venv/bin/activate

# Install remaining packages (OpenCV already installed system-wide)
pip3 install numpy scipy pyaudio pyserial SpeechRecognition
```

## Step 3: Make OpenCV Available in Virtual Environment

The system OpenCV should be accessible, but if not:

```bash
# Create symlink in venv (if needed)
ln -s /usr/lib/python3/dist-packages/cv2* venv/lib/python3.*/site-packages/
```

Or add system packages to Python path in your code.

## Step 4: MediaPipe (Optional)

MediaPipe may not be available for all Python versions on ARM. The system will automatically fall back to OpenCV HOG detector if MediaPipe is not available.

If you want to try installing MediaPipe:

```bash
# For Python 3.9-3.11 (MediaPipe supports these)
pip3 install mediapipe

# For Python 3.12+, MediaPipe may not be available
# The system will use OpenCV HOG detector instead (works fine)
```

## Alternative: Use System Python (Not Virtual Environment)

If virtual environment causes issues:

```bash
# Exit venv
deactivate

# Install packages system-wide
sudo apt-get install python3-opencv python3-opencv-contrib python3-pip
pip3 install numpy scipy pyaudio pyserial SpeechRecognition

# Run with system Python
python3 vision_main.py
```

## Quick Fix Script

Run this to fix the installation:

```bash
#!/bin/bash
# Fix vision system dependencies

echo "Installing system OpenCV..."
sudo apt-get update
sudo apt-get install -y python3-opencv python3-opencv-contrib

echo "Installing Python packages..."
pip3 install numpy scipy pyaudio pyserial SpeechRecognition

echo "Testing OpenCV..."
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

echo "Done! Try running: python3 vision_main.py"
```

## Verify Installation

```bash
# Test OpenCV
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test other imports
python3 -c "import numpy, scipy, serial; print('All imports OK')"

# Test vision system
python3 -c "from vision_person_tracker import PersonTracker; print('Vision tracker OK')"
```

## If OpenCV Still Not Found

If `cv2` is still not found after installing system package:

1. **Check Python version compatibility:**
   ```bash
   python3 --version
   # OpenCV system package works with Python 3.9-3.11 typically
   ```

2. **Check if OpenCV is installed:**
   ```bash
   dpkg -l | grep opencv
   ```

3. **Find OpenCV location:**
   ```bash
   find /usr -name "cv2*.so" 2>/dev/null
   ```

4. **Add to Python path manually:**
   Create a file `fix_opencv.py`:
   ```python
   import sys
   sys.path.insert(0, '/usr/lib/python3/dist-packages')
   import cv2
   ```

## Notes

- **MediaPipe is optional** - The system works fine with just OpenCV HOG detector
- **System OpenCV is more stable** on Raspberry Pi than pip-installed version
- **Python 3.13** may have compatibility issues - consider using Python 3.11 or system Python

## After Installation

Once OpenCV is working:

```bash
# Test camera (if connected)
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera not found'); cap.release()"

# Run vision system
python3 vision_main.py
```

