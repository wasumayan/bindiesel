#!/usr/bin/env python3
"""
Quick test to see if Pixy2 works as USB video device
If this works, you can use Pixy2 with all your existing YOLO/ArUco code!
"""

import cv2
import sys

print("=" * 70)
print("Pixy2 USB Video Device Test")
print("=" * 70)
print()

# Check for video devices
print("Checking for video devices...")
import subprocess
result = subprocess.run(['ls', '/dev/video*'], capture_output=True, text=True)
if result.returncode == 0:
    devices = result.stdout.strip().split('\n')
    print(f"Found {len(devices)} video device(s):")
    for dev in devices:
        print(f"  {dev}")
else:
    print("No /dev/video* devices found")
print()

# Try to open each video device
for device_idx in range(8):
    print(f"Trying /dev/video{device_idx}...", end=" ")
    cap = cv2.VideoCapture(device_idx)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✓ SUCCESS! Resolution: {width}x{height}")
            print(f"  This is likely your Pixy2 camera!")
            print()
            print("=" * 70)
            print("GREAT NEWS: Pixy2 works as USB video device!")
            print("=" * 70)
            print()
            print("You can use it with all your existing code:")
            print("  - YOLO pose detection")
            print("  - ArUco marker detection")
            print("  - All your current functionality")
            print()
            print(f"Just change camera initialization to:")
            print(f"  cap = cv2.VideoCapture({device_idx})")
            print()
            cap.release()
            sys.exit(0)
        else:
            print("✗ Opened but can't read frames")
            cap.release()
    else:
        print("✗ Cannot open")
        if cap:
            cap.release()

print()
print("=" * 70)
print("Pixy2 not found as USB video device")
print("=" * 70)
print()
print("This means Pixy2 doesn't support USB video mode.")
print("You'll need to use the native Pixy2 library instead.")
print("See: test_pixy2_native.py")
print()

