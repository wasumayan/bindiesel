#!/usr/bin/env python3
"""
Simple camera stream display using picamera2
Displays the camera feed using picamera2's built-in preview (no OpenCV needed)
"""

import time
from picamera2 import Picamera2, Preview

def main():
    """Display camera stream"""
    print("=" * 70)
    print("Camera Stream Display")
    print("=" * 70)
    print("Press Ctrl+C to quit")
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
    
    # Start preview (uses GPU-accelerated QtGL preview)
    print("[DEBUG] Starting preview...")
    try:
        picam2.start_preview(Preview.QTGL)
    except Exception as e:
        print(f"ERROR: Could not start preview: {e}")
        print("Trying alternative preview method...")
        try:
            picam2.start_preview(Preview.QT)
        except Exception as e2:
            print(f"ERROR: Could not start QT preview either: {e2}")
            print("Preview may not be available in this environment (e.g., SSH without X11)")
            return
    
    # Configure camera
    print("[DEBUG] Configuring camera...")
    try:
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 480)}
        )
        picam2.configure(preview_config)
        print("[DEBUG] Camera configured: 640x480")
    except Exception as e:
        print(f"ERROR: Could not configure camera: {e}")
        return
    
    # Start camera
    print("[DEBUG] Starting camera...")
    try:
        picam2.start()
        print("[DEBUG] Camera started successfully")
        print("\nDisplaying camera stream...")
        print("Press Ctrl+C to quit\n")
    except Exception as e:
        print(f"ERROR: Could not start camera: {e}")
        return
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("[DEBUG] Stopping camera...")
        picam2.stop()
        picam2.close()
        print("Camera stream closed.")


if __name__ == "__main__":
    main()
