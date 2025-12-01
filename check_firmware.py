#!/usr/bin/env python3
"""
Check ReSpeaker Lite firmware version and USB device status
"""

import subprocess
import sys

def check_dfu_util():
    """Check if dfu-util is installed"""
    try:
        result = subprocess.run(['dfu-util', '--version'], 
                              capture_output=True, text=True)
        print(f"✓ dfu-util is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ dfu-util is not installed")
        print("  Install with: sudo apt-get install dfu-util")
        return False

def check_firmware():
    """Check current firmware version"""
    print("\n" + "=" * 70)
    print("Checking Firmware Version")
    print("=" * 70)
    
    try:
        result = subprocess.run(['dfu-util', '-l'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout
            print(output)
            
            # Check for USB firmware
            if 'ReSpeaker' in output or 'XMOS' in output:
                if 'USB' in output or '2.0' in output:
                    print("\n✓ USB firmware detected")
                    # Try to extract version
                    lines = output.split('\n')
                    for line in lines:
                        if '2.0' in line or 'v2' in line.lower():
                            print(f"  Version info: {line.strip()}")
                else:
                    print("\n⚠ May be I2S firmware - need USB firmware for USB audio")
            else:
                print("\n⚠ No ReSpeaker device found in DFU mode")
                print("  Device must be in DFU mode to check firmware")
                print("  (Unplug, hold button, plug in, release button)")
        else:
            print("✗ Error running dfu-util")
            print(result.stderr)
            
    except Exception as e:
        print(f"✗ Error: {e}")

def check_usb_device():
    """Check USB audio device"""
    print("\n" + "=" * 70)
    print("Checking USB Audio Device")
    print("=" * 70)
    
    try:
        result = subprocess.run(['arecord', '-l'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout
            print(output)
            
            if 'ReSpeaker' in output:
                print("\n✓ ReSpeaker USB audio device found")
                
                # Check for stereo support
                if '2 channels' in output.lower() or 'stereo' in output.lower():
                    print("  ✓ Stereo support indicated")
                else:
                    print("  ⚠ Stereo support not clearly indicated")
            else:
                print("\n✗ ReSpeaker USB audio device not found")
                print("  Make sure:")
                print("    1. Device is plugged in via USB")
                print("    2. Firmware is USB version (not I2S)")
                print("    3. Firmware version is 2.0.5 or higher")
        else:
            print("✗ Error running arecord")
            print(result.stderr)
            
    except FileNotFoundError:
        print("✗ arecord not found")
        print("  Install with: sudo apt-get install alsa-utils")
    except Exception as e:
        print(f"✗ Error: {e}")

def check_lsusb():
    """Check USB device via lsusb"""
    print("\n" + "=" * 70)
    print("Checking USB Device (lsusb)")
    print("=" * 70)
    
    try:
        result = subprocess.run(['lsusb'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout
            print(output)
            
            if 'Seeed' in output or 'XMOS' in output or 'ReSpeaker' in output:
                print("\n✓ ReSpeaker device found in USB device list")
            else:
                print("\n⚠ ReSpeaker device not found in USB list")
                print("  Device may not be properly connected")
        else:
            print("✗ Error running lsusb")
            
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("=" * 70)
    print("ReSpeaker Lite Firmware and Device Check")
    print("=" * 70)
    
    # Check dfu-util
    dfu_available = check_dfu_util()
    
    # Check firmware (if device in DFU mode)
    if dfu_available:
        check_firmware()
    
    # Check USB device
    check_usb_device()
    
    # Check lsusb
    check_lsusb()
    
    print("\n" + "=" * 70)
    print("Summary and Recommendations")
    print("=" * 70)
    print("\nIf channels are identical:")
    print("  1. Check firmware version: dfu-util -l")
    print("  2. Update to USB firmware v2.0.7 if needed")
    print("  3. Verify device shows up in: arecord -l")
    print("  4. Test recording: arecord -D hw:3,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav")
    print("\nFor firmware update instructions, see: RESPEAKER_FIRMWARE_FIX.md")

if __name__ == '__main__':
    main()

