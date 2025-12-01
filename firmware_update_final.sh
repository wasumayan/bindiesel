#!/bin/bash
# ReSpeaker Lite Firmware Update - Final Version
# Based on official DFU guide: https://github.com/respeaker/ReSpeaker_Lite/blob/master/xmos_firmwares/dfu_guide.md

echo "=========================================="
echo "ReSpeaker Lite USB Firmware Update"
echo "=========================================="
echo ""

# Step 1: Clone repository to find firmware files
echo "Step 1: Cloning repository to find firmware files..."
cd ~/Downloads
rm -rf ReSpeaker_Lite  # Clean up if exists
git clone https://github.com/respeaker/ReSpeaker_Lite.git
cd ReSpeaker_Lite/xmos_firmwares

echo ""
echo "Available firmware files:"
ls -lh *.bin 2>/dev/null || echo "No .bin files found in xmos_firmwares directory"

echo ""
echo "Looking for USB firmware..."
USB_FIRMWARE=$(ls *usb*.bin 2>/dev/null | head -1)

if [ -z "$USB_FIRMWARE" ]; then
    echo "⚠ USB firmware not found with 'usb' in name"
    echo "Listing all .bin files:"
    ls -lh *.bin
    echo ""
    echo "Please identify the USB firmware file and update the script"
    exit 1
fi

echo "✓ Found USB firmware: $USB_FIRMWARE"
cp "$USB_FIRMWARE" ~/Downloads/
cd ~/Downloads

echo ""
echo "=========================================="
echo "Step 2: Enter DFU Mode"
echo "=========================================="
echo ""
echo "MANUAL STEPS:"
echo "  1. Unplug USB cable from ReSpeaker Lite"
echo "  2. Hold down the button on ReSpeaker Lite"
echo "  3. While holding button, plug in USB cable"
echo "  4. Release button after 2-3 seconds"
echo ""
echo "IMPORTANT: Use the XMOS USB-C port (near 3.5mm jack)"
echo ""
read -p "Press Enter after device is in DFU mode..."

echo ""
echo "Checking DFU mode..."
dfu-util -l

echo ""
read -p "Do you see 'DFU' devices listed? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Device not in DFU mode. Please try again."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Flash Firmware"
echo "=========================================="
echo ""
echo "Flashing: $USB_FIRMWARE"
echo ""

# Use the command format from the official DFU guide
sudo dfu-util -R -e -a 1 -D "$USB_FIRMWARE"

echo ""
echo "=========================================="
echo "Step 4: Restart Device"
echo "=========================================="
echo ""
echo "MANUAL STEPS:"
echo "  1. Unplug USB cable"
echo "  2. Wait 2 seconds"
echo "  3. Plug it back in"
echo ""
read -p "Press Enter after you've replugged the device..."

echo ""
echo "=========================================="
echo "Step 5: Verify Update"
echo "=========================================="
echo ""

# Check device
echo "Checking if device is recognized..."
arecord -l

echo ""
echo "Testing stereo recording..."
cd ~/angleestimationbindiesel
python3 test_stereo_recording.py

echo ""
echo "=========================================="
echo "Update Complete!"
echo "=========================================="

