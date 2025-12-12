#!/usr/bin/env python3
"""
Pixy2 USB Video Mode Test Script
Tests if Pixy2 can be accessed as a standard USB video device
This allows full frame capture for YOLO/ArUco processing
"""

import cv2
import sys
import time

print("=" * 70)
print("Pixy2 USB Video Mode Test")
print("=" * 70)
print()
print("This script tests if Pixy2 can be accessed as a USB video device")
print("(like /dev/video0, /dev/video1, etc.)")
print()
print("If successful, you can use Pixy2 for full frame capture with YOLO/ArUco")
print()

# Try to find Pixy2 as USB video device
cap = None
camera_found = False
camera_index = None

print("Searching for Pixy2 camera...")
print()

for device_idx in range(8):  # Try first 8 video devices
    try:
        print(f"Trying /dev/video{device_idx}...", end=" ")
        test_cap = cv2.VideoCapture(device_idx)
        
        if test_cap.isOpened():
            # Test if we can read a frame
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                # Check if this might be Pixy2 (resolution hints)
                height, width = test_frame.shape[:2]
                
                # Pixy2 typical resolutions: 320x200, 640x480
                if (width, height) in [(320, 200), (200, 320), (640, 480), (480, 640)]:
                    print(f"✓ FOUND! Possible Pixy2 at /dev/video{device_idx}")
                    print(f"  Resolution: {width}x{height}")
                    cap = test_cap
                    camera_found = True
                    camera_index = device_idx
                    
                    # Set desired resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Get actual resolution
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"  Set resolution: {actual_width}x{actual_height}")
                    print(f"  FPS: {actual_fps}")
                    break
                else:
                    print(f"✗ Camera found but resolution ({width}x{height}) doesn't match Pixy2")
                    test_cap.release()
            else:
                print("✗ Camera opened but cannot read frames")
                test_cap.release()
        else:
            print("✗ Cannot open")
            if test_cap:
                test_cap.release()
    except Exception as e:
        print(f"✗ Error: {e}")
        if test_cap:
            test_cap.release()
        continue

print()

if not camera_found:
    print("=" * 70)
    print("WARNING: Could not find Pixy2 as USB video device")
    print("=" * 70)
    print()
    print("This means:")
    print("  - Pixy2 may not support USB video mode")
    print("  - Or Pixy2 is not connected")
    print("  - Or Pixy2 needs different firmware")
    print()
    print("Alternative: Use Pixy2's native library for color-based detection")
    print("  Run: python3 test_pixy2_native.py")
    print()
    print("Troubleshooting:")
    print("  1. Check USB connection: lsusb | grep -i pixy")
    print("  2. Check video devices: ls -l /dev/video*")
    print("  3. Check permissions: sudo chmod 666 /dev/video*")
    print("  4. Try running with sudo: sudo python3 test_pixy2_video.py")
    print()
    sys.exit(1)

# Display video feed
print("=" * 70)
print("Starting video display...")
print("Press 'q' to quit")
print("=" * 70)
print()
print("If you see video, Pixy2 supports USB video mode!")
print("You can use this for YOLO pose detection and ArUco marker detection.")
print()

frame_count = 0
start_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR: Failed to read frame")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
        
        # Add info text to frame
        cv2.putText(frame, f"Pixy2 USB Video Mode", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: /dev/video{camera_index}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Pixy2 USB Video Test", frame)
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    print()
    if camera_found:
        print("✓ SUCCESS: Pixy2 supports USB video mode!")
        print("  You can use this camera for:")
        print("    - YOLO pose detection")
        print("    - ArUco marker detection")
        print("    - Full frame processing")
        print()
        print("  Update your code to use:")
        print(f"    cap = cv2.VideoCapture({camera_index})")
        print()

