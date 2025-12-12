# Pixy2 CMUcam5 Setup Guide

This guide will help you set up and test your Pixy2 CMUcam5 (Rev 2.3) camera for use with the Bin Diesel system.

## Prerequisites

- Raspberry Pi (or Linux system)
- Pixy2 CMUcam5 camera connected via USB
- Python 3 installed

## Installation Methods

### Method 1: Native Pixy2 Library (Recommended for Color-Based Detection)

Follow the official Pixy2 setup instructions: [Hooking Up Pixy2 to a Raspberry Pi](https://docs.pixycam.com/wiki/doku.php?id=wiki:v2:hooking_up_pixy_to_a_raspberry_pi)

#### Install dependencies:

```bash
sudo apt-get update
sudo apt-get install git libusb-1.0-0-dev g++ build-essential python3-pip
```

#### Build libpixyusb2:

```bash
git clone https://github.com/charmedlabs/pixy2
cd pixy2/scripts
./build_libpixyusb2.sh
cd ../build/python_demos
pip3 install .
```

#### Test the native library:

```bash
python3 test_pixy2_native.py
```

This will show detected color signatures (blocks) in real-time.

### Method 2: USB Video Mode (For Full Frame Capture)

The Pixy2 camera may be accessible as a standard USB video device using OpenCV. This allows full frame capture for YOLO/ArUco processing.

#### Install OpenCV:

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy
```

#### Test USB video mode:

```bash
python3 test_pixy2_video.py
```

The test program will automatically search for the camera at `/dev/video0`, `/dev/video1`, etc.

**Note**: Not all Pixy2 models/firmware versions support USB video mode. If this doesn't work, use Method 1 (native library).

### Method 2: Native Pixy Library (Optional - Advanced)

If you want to use the native Pixy2 library for color-based object tracking:

#### Install dependencies:

```bash
sudo apt-get install git libusb-1.0-0-dev g++ build-essential python3-pip
```

#### Clone and build Pixy2 library:

```bash
git clone https://github.com/charmedlabs/pixy2
cd pixy2/scripts
./build_libpixyusb2.sh
pip3 install ./build/python_demos
```

#### Test the library:

```python
from pixy import *
from ctypes import *

status = pixy_init()
if status == 0:
    print("Pixy initialized successfully!")
    pixy_close()
else:
    print(f"Error: {status}")
```

## Troubleshooting

### Camera Not Detected

1. **Check USB connection:**
   ```bash
   lsusb
   ```
   You should see a device from "Charmed Labs" or similar.

2. **Check video devices:**
   ```bash
   ls -l /dev/video*
   ```
   You should see `/dev/video0` or similar.

3. **Check permissions:**
   ```bash
   ls -l /dev/video*
   ```
   If permissions are wrong, add your user to the `video` group:
   ```bash
   sudo usermod -a -G video $USER
   ```
   Then log out and log back in.

4. **Try different USB port/cable:**
   - Some USB ports may not provide enough power
   - Try a different USB cable

### Camera Opens But No Frames

1. **Check if another program is using the camera:**
   ```bash
   lsof | grep video
   ```
   Kill any processes using the camera.

2. **Try a different video device index:**
   The test program tries `/dev/video0` through `/dev/video3`. If your camera is at a different index, you may need to modify the test program.

3. **Check camera firmware:**
   - Ensure Pixy2 firmware is up to date
   - Some firmware versions may not support full-frame capture

### Permission Denied Errors

```bash
sudo chmod 666 /dev/video0
```

Or add your user to the video group (see above).

## Testing

Run the test program:

```bash
python3 test_pixy_camera.py
```

You should see:
- A window displaying the camera feed
- FPS counter in the top-left corner
- "Press 'q' to quit" message

Press 'q' to exit.

## Next Steps

Once the camera is working:
1. The test program confirms the camera is accessible
2. You can integrate it into `maincpixy.py` using the `PixyCamera` class
3. The camera will be used for YOLO pose detection and ArUco marker detection

## Notes

- **USB VideoCapture mode** (OpenCV) is recommended because:
  - It's simpler to set up
  - Provides full-frame capture
  - Works with standard OpenCV functions
  - No additional libraries needed

- **Native Pixy library** is useful if you want:
  - Color-based object tracking (signatures)
  - Line tracking
  - Custom firmware features

For the Bin Diesel system, USB VideoCapture mode is sufficient since we use YOLO for person detection and ArUco for marker detection.

