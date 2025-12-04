#!/bin/bash
# Installation script for Bin Diesel project dependencies on Raspberry Pi

echo "=========================================="
echo "Installing Bin Diesel Project Dependencies"
echo "=========================================="
echo ""

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install system packages
echo ""
echo "Installing system packages..."
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    portaudio19-dev \
    python3-pyaudio \
    python3-numpy \
    python3-serial \
    python3-picamera2 \
    libcamera-apps \
    v4l-utils \
    libcamera-apps \
    v4l-utils

# Try to install opencv-contrib if available (optional)
echo ""
echo "Attempting to install python3-opencv-contrib (optional)..."
sudo apt-get install -y python3-opencv-contrib 2>/dev/null || echo "  python3-opencv-contrib not available, skipping..."

# Install Python packages from requirements.txt using --break-system-packages
# (Required for newer Python versions with externally-managed-environment)
echo ""
echo "Installing Python packages from requirements.txt..."
echo "Note: Using --break-system-packages flag for system-wide installation"
pip3 install --break-system-packages -r requirements.txt

# Note: Picovoice Porcupine will be installed from requirements.txt
echo ""
echo "Note: Picovoice Porcupine (pvporcupine) will be installed from requirements.txt"
echo "You'll need a Picovoice AccessKey from https://console.picovoice.ai/"

# Check camera permissions
echo ""
echo "Checking camera permissions..."
if groups $USER | grep -q video; then
    echo "✓ User is in video group"
else
    echo "Adding user to video group..."
    sudo usermod -a -G video $USER
    echo "⚠ You may need to logout and login again for camera permissions to take effect"
fi

# Verify libcamera setup
echo ""
echo "Verifying libcamera setup..."
if command -v libcamera-hello &> /dev/null; then
    echo "✓ libcamera is installed"
    echo "  Test camera with: libcamera-hello"
else
    echo "⚠ libcamera-apps not found, but OpenCV should still work with V4L2"
fi

# Check for video devices
echo ""
echo "Checking for video devices..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✓ Video devices found:"
    ls -l /dev/video*
else
    echo "⚠ No video devices found. Make sure camera is connected and enabled."
    echo "  Enable camera: sudo raspi-config → Interface Options → Camera → Enable"
fi

# Check audio permissions
echo ""
echo "Checking audio permissions..."
if groups $USER | grep -q audio; then
    echo "✓ User is in audio group"
else
    echo "Adding user to audio group..."
    sudo usermod -a -G audio $USER
    echo "⚠ You may need to logout and login again for audio permissions to take effect"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If you were added to video/audio groups, logout and login again"
echo "2. Record your wake word: python3 record_wake_word.py"
echo "3. Run the system: python3 bindieselsimple.py"
echo ""

