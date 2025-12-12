#!/usr/bin/env python3
"""
Pixy2 Native Library Test Script
Tests Pixy2 connection using native libpixyusb2 library
Shows detected blocks (color signatures) in real-time
"""

import sys
import time
from ctypes import *

# Try to import Pixy library
try:
    from pixy import *
    HAS_PIXY = True
except ImportError:
    print("=" * 70)
    print("ERROR: Pixy library not found!")
    print("=" * 70)
    print()
    print("Please install the Pixy2 library first:")
    print("  1. git clone https://github.com/charmedlabs/pixy2")
    print("  2. cd pixy2/scripts")
    print("  3. ./build_libpixyusb2.sh")
    print("  4. cd ../build/python_demos")
    print("  5. pip3 install .")
    print()
    print("Or add the build directory to your Python path:")
    print("  export PYTHONPATH=$PYTHONPATH:/path/to/pixy2/build/python_demos")
    print()
    sys.exit(1)

# Define Block structure (from Pixy2 API)
class Block(Structure):
    _fields_ = [
        ("m_signature", c_uint),
        ("m_x", c_uint),
        ("m_y", c_uint),
        ("m_width", c_uint),
        ("m_height", c_uint),
        ("m_angle", c_uint),
        ("m_index", c_uint),
        ("m_age", c_uint)
    ]

def main():
    print("=" * 70)
    print("Pixy2 Native Library Test")
    print("=" * 70)
    print()
    
    # Initialize Pixy2
    print("Initializing Pixy2...")
    status = pixy_init()
    
    if status != 0:
        print(f"ERROR: pixy_init() returned {status}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure Pixy2 is connected via USB")
        print("  2. Check USB permissions: sudo chmod 666 /dev/bus/usb/*/*")
        print("  3. Try running with sudo: sudo python3 test_pixy2_native.py")
        print("  4. Check if PixyMon can detect the camera")
        sys.exit(1)
    
    print("✓ Pixy2 initialized successfully!")
    print()
    
    # Get firmware version
    try:
        version = pixy_get_firmware_version()
        print(f"Firmware Version: {version}")
    except:
        print("Could not retrieve firmware version")
    print()
    
    # Set camera brightness (optional, 0-255)
    try:
        pixy_set_camera_brightness(50)  # Medium brightness
        print("Camera brightness set to 50")
    except:
        print("Could not set camera brightness")
    print()
    
    print("=" * 70)
    print("Starting block detection...")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    print("Note: You need to teach Pixy2 color signatures first using PixyMon")
    print("      or configure signatures programmatically.")
    print()
    
    # Create block array (max 100 blocks)
    blocks = BlockArray(100)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get blocks (detected color signatures)
            count = pixy_ccc_get_blocks(100, blocks)
            
            if count > 0:
                print(f"\nFrame {frame_count}: Detected {count} block(s)")
                print("-" * 70)
                
                for i in range(count):
                    block = blocks[i]
                    print(f"  Block {i+1}:")
                    print(f"    Signature: {block.m_signature}")
                    print(f"    Position: ({block.m_x}, {block.m_y})")
                    print(f"    Size: {block.m_width}x{block.m_height}")
                    print(f"    Angle: {block.m_angle}")
                    print(f"    Index: {block.m_index}")
                    print(f"    Age: {block.m_age}")
                    
                    # Calculate center position
                    center_x = block.m_x
                    center_y = block.m_y
                    
                    # Calculate angle from center (for navigation)
                    frame_center_x = 160  # Pixy2 resolution is 320x200
                    offset = center_x - frame_center_x
                    angle = (offset / frame_center_x) * 45.0  # Approximate angle
                    
                    print(f"    Center offset: {offset} pixels")
                    print(f"    Estimated angle: {angle:.1f}°")
                    print()
            else:
                # Show status every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\rNo blocks detected... (FPS: {fps:.1f})", end="", flush=True)
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        pixy_close()
        print()
        print("=" * 70)
        print("Test complete!")
        print("=" * 70)

if __name__ == '__main__':
    main()

