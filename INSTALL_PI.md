# Installation Instructions for Raspberry Pi

This guide will help you install all dependencies for the Bin Diesel project on your Raspberry Pi.

## Quick Installation

### Option 1: Automated Script (Recommended)

```bash
# Make script executable
chmod +x install_dependencies.sh

# Run installation script
./install_dependencies.sh
```

The script will:
- Update package lists
- Install all system packages
- Install Python packages from requirements.txt
- Optionally install librosa for better wake word detection
- Set up camera and audio permissions

### Option 2: Manual Installation

#### Step 1: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

#### Step 2: Install System Packages

```bash
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-opencv-contrib \
    libopencv-dev \
    portaudio19-dev \
    python3-pyaudio \
    python3-numpy \
    python3-serial
```

#### Step 3: Install Python Packages

```bash
# Install from requirements.txt
pip3 install -r requirements.txt

# Optional: For better wake word detection
pip3 install librosa
```

#### Step 4: Set Up Permissions

```bash
# Add user to video group (for camera access)
sudo usermod -a -G video $USER

# Add user to audio group (for microphone access)
sudo usermod -a -G audio $USER

# Logout and login again for changes to take effect
```

#### Step 5: Enable Camera (if using Raspberry Pi Camera)

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot after enabling
```

## Verify Installation

### Test Camera

```bash
python3 test_camera_basic.py
```

### Test Microphone

```bash
# List audio devices
arecord -l

# Test recording
arecord -d 5 test.wav && aplay test.wav
```

### Test Wake Word Recording

```bash
python3 record_wake_word.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'speech_recognition'"

**Solution**: Install Python packages:
```bash
pip3 install -r requirements.txt
```

### "Could not open camera"

**Solutions**:
1. Enable camera in raspi-config:
   ```bash
   sudo raspi-config
   # Interface Options → Camera → Enable
   ```

2. Check permissions:
   ```bash
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

3. Check camera is detected:
   ```bash
   ls /dev/video*
   vcgencmd get_camera
   ```

### "Error opening microphone"

**Solutions**:
1. Check microphone is connected:
   ```bash
   lsusb
   arecord -l
   ```

2. Check permissions:
   ```bash
   sudo usermod -a -G audio $USER
   # Logout and login again
   ```

3. Test microphone:
   ```bash
   arecord -d 5 test.wav
   aplay test.wav
   ```

### "OpenCV not found"

**Solution**: Install OpenCV system package:
```bash
sudo apt-get install python3-opencv python3-opencv-contrib
```

### "Permission denied" errors

**Solution**: Make sure you're in the correct groups:
```bash
groups  # Check current groups
sudo usermod -a -G video,audio $USER
# Logout and login again
```

## Required Packages Summary

### System Packages
- `python3-pip` - Python package manager
- `python3-opencv` - Computer vision library
- `python3-opencv-contrib` - OpenCV contrib modules
- `portaudio19-dev` - Audio I/O library (for microphone)
- `python3-pyaudio` - Python audio interface
- `python3-numpy` - Numerical computing
- `python3-serial` - Serial communication

### Python Packages (from requirements.txt)
- `numpy>=1.21.0` - Numerical computing
- `pyaudio>=0.2.11` - Audio I/O
- `pyserial>=3.5` - Serial communication
- `SpeechRecognition>=3.10.0` - Speech recognition
- `openai>=1.0.0` - OpenAI API (optional, for full system)
- `python-dotenv>=1.0.0` - Environment variables

### Optional Packages
- `librosa` - Better audio processing for wake word detection
- `mediapipe` - Person detection (alternative to OpenCV)

## After Installation

1. **Record wake word**:
   ```bash
   python3 record_wake_word.py
   ```

2. **Test the system**:
   ```bash
   python3 bindieselsimple.py
   ```

3. **See setup guide**: [SETUP_WAKE_WORD.md](SETUP_WAKE_WORD.md)

## Notes

- If you're using SSH, you may need X11 forwarding for video display
- Camera Module 3 Wide has 102° horizontal FOV (already configured)
- Default wake word file: `wake_word_reference.wav`
- Default similarity threshold: 0.6 (60% match required)

