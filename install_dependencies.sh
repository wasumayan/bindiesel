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
    python3-opencv-contrib \
    libopencv-dev \
    portaudio19-dev \
    python3-pyaudio \
    python3-numpy \
    python3-serial

# Install Python packages from requirements.txt
echo ""
echo "Installing Python packages from requirements.txt..."
pip3 install -r requirements.txt

# Optional: Install librosa for better wake word detection
echo ""
read -p "Install librosa for better wake word detection? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing librosa..."
    pip3 install librosa
else
    echo "Skipping librosa (will use simple audio comparison)"
fi

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

