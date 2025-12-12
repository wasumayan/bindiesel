#!/usr/bin/env python3
"""
Simple Pixy Camera Test Program
Tests Pixy camera connection and displays video feed
"""

import cv2
import sys
import time

print("=" * 70)
print("Pixy Camera Test Program")
print("=" * 70)
print()

# Method 1: Try OpenCV VideoCapture (Pixy2 connected via USB)
print("Attempting to connect to Pixy camera via USB (OpenCV)...")
print("Note: Pixy2 connected via USB appears as /dev/video0, /dev/video1, etc.")
print()

cap = None
camera_found = False

# Try to find Pixy camera device
for device_idx in range(4):  # Try first 4 video devices
    try:
        print(f"Trying /dev/video{device_idx}...", end=" ")
        test_cap = cv2.VideoCapture(device_idx)
        
        if test_cap.isOpened():
            # Test if we can read a frame
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                print(f"✓ SUCCESS! Found camera at /dev/video{device_idx}")
                cap = test_cap
                camera_found = True
                
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Get actual resolution
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"  Resolution: {actual_width}x{actual_height}")
                print(f"  FPS: {actual_fps}")
                break
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
    print("ERROR: Could not find Pixy camera!")
    print("=" * 70)
    print()
    print("TROUBLESHOOTING:")
    print("1. Check USB connection:")
    print("   - Ensure Pixy2 is connected via USB cable")
    print("   - Try a different USB port")
    print("   - Try a different USB cable")
    print()
    print("2. Check if camera is detected by system:")
    print("   Run: ls -l /dev/video*")
    print("   You should see /dev/video0, /dev/video1, etc.")
    print()
    print("3. Check camera permissions:")
    print("   Run: ls -l /dev/video*")
    print("   If permissions are wrong, run: sudo chmod 666 /dev/video*")
    print()
    print("4. Install required packages:")
    print("   sudo apt-get update")
    print("   sudo apt-get install python3-opencv python3-numpy")
    print()
    print("5. If using native Pixy library instead:")
    print("   git clone https://github.com/charmedlabs/pixy2")
    print("   cd pixy2/scripts")
    print("   ./build_libpixyusb2.sh")
    print("   pip3 install ./build/python_demos")
    print()
    sys.exit(1)

# Method 2: Try native Pixy library (optional)
try_pixy_library = False
if not camera_found:
    print("Attempting to use native Pixy library...")
    try:
        from pixy import *
        from ctypes import *
        HAS_PIXY = True
        
        status = pixy_init()
        if status == 0:
            print("✓ Native Pixy library initialized successfully")
            try_pixy_library = True
        else:
            print(f"✗ pixy_init() returned error code: {status}")
    except ImportError:
        print("✗ Pixy library not installed")
        print()
        print("To install native Pixy library:")
        print("  git clone https://github.com/charmedlabs/pixy2")
        print("  cd pixy2/scripts")
        print("  ./build_libpixyusb2.sh")
        print("  pip3 install ./build/python_demos")
        print()
    except Exception as e:
        print(f"✗ Error initializing Pixy library: {e}")

if not camera_found and not try_pixy_library:
    print("=" * 70)
    print("ERROR: Could not initialize Pixy camera with any method!")
    print("=" * 70)
    sys.exit(1)

# Display video feed
print("=" * 70)
print("Starting video display...")
print("Press 'q' to quit")
print("=" * 70)
print()

frame_count = 0
start_time = time.time()
fps = 0.0

try:
    while True:
        if camera_found and cap is not None:
            # Use OpenCV VideoCapture
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from camera")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
            
            # Add FPS text to frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Pixy Camera Test", frame)
            
        elif try_pixy_library:
            # Use native Pixy library (if it supports full frame capture)
            # Note: Standard Pixy2 firmware doesn't support full frame capture
            # This is a placeholder for future implementation
            print("ERROR: Native Pixy library full-frame capture not yet implemented")
            print("Please use USB VideoCapture mode (OpenCV)")
            break
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    if cap is not None:
        cap.release()
    
    if try_pixy_library:
        try:
            pixy_close()
        except:
            pass
    
    cv2.destroyAllWindows()
    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)

