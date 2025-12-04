#!/usr/bin/env python3
"""
Simple camera stream display using picamera2
Displays the camera feed in an OpenCV window
"""

import cv2
from picamera2 import Picamera2

def main():
    """Display camera stream"""
    print("=" * 70)
    print("Camera Stream Display")
    print("=" * 70)
    print("Press 'q' to quit")
    print("=" * 70)
    
    # Initialize picamera2
    print("[DEBUG] Initializing picamera2...")
    try:
        picam2 = Picamera2()
    except Exception as e:
        print(f"ERROR: Could not initialize picamera2: {e}")
        print("Make sure:")
        print("  1. Camera is enabled: sudo raspi-config → Interface Options → Camera → Enable")
        print("  2. picamera2 is installed: sudo apt install python3-picamera2")
        return
    
    # Configure camera
    print("[DEBUG] Configuring camera...")
    try:
        # Configure for preview with RGB format for OpenCV
        preview_config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
        picam2.configure(preview_config)
        print("[DEBUG] Camera configured: 640x480, XRGB8888")
    except Exception as e:
        print(f"ERROR: Could not configure camera: {e}")
        return
    
    # Start camera
    print("[DEBUG] Starting camera...")
    try:
        picam2.start()
        print("[DEBUG] Camera started successfully")
    except Exception as e:
        print(f"ERROR: Could not start camera: {e}")
        return
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\nDisplaying camera stream...")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            # Capture frame from picamera2
            # capture_array() returns numpy array in RGB format
            array = picam2.capture_array()
            
            # picamera2 returns XRGB8888, but OpenCV expects BGR
            # Convert RGB to BGR for OpenCV display
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            
            # Display frame
            cv2.imshow("Camera Stream", frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        # Cleanup
        print("[DEBUG] Stopping camera...")
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stream closed.")


if __name__ == "__main__":
    main()

