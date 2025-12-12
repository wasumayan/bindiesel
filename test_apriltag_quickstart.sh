#!/bin/bash
# Quick start script for AprilTag testing on laptop

echo "=========================================="
echo "AprilTag Detection Test - Quick Start"
echo "=========================================="
echo ""

# Check if dt-apriltags is installed
if ! python3 -c "import apriltag" 2>/dev/null; then
    echo "Installing dt-apriltags..."
    pip3 install dt-apriltags
    echo ""
fi

# Check if opencv-python is installed
if ! python3 -c "import cv2" 2>/dev/null; then
    echo "Installing opencv-python..."
    pip3 install opencv-python
    echo ""
fi

echo "1. First, generate an AprilTag:"
echo "   python3 generate_apriltag.py --tag-id 0 --size 200"
echo ""
echo "2. Print the generated tag and measure its size in meters"
echo ""
echo "3. Test detection with webcam:"
echo "   python3 test_apriltag_detection.py --webcam --tag-size 0.047 --no-control"
echo ""
echo "   (Replace 0.047 with your measured tag size)"
echo ""
echo "4. If detection works, test with hardware control:"
echo "   python3 test_apriltag_detection.py --webcam --tag-size 0.047"
echo ""

