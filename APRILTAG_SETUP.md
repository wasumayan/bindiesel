# AprilTag Detection Setup Guide

This guide explains how to use AprilTag detection for returning to home instead of red box object detection.

## Installation

**No additional installation needed!** The code uses OpenCV's built-in ArUco markers, which are already included with OpenCV.

If you have OpenCV installed (which you should already have), you're ready to go:

```bash
# Verify OpenCV is installed
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

ArUco markers work very similarly to AprilTags and are more reliable since they're built into OpenCV.

## Quick Start

### Testing on Laptop (Webcam)

1. **Generate an AprilTag:**

```bash
python3 generate_apriltag.py --tag-id 0 --size 200
```

This creates `apriltag_tag36h11_000.png` (ArUco marker) that you can print or display on a screen.

**Note:** The script uses ArUco markers by default (built into OpenCV). ArUco markers work identically to AprilTags for this use case - no extra dependencies needed!

**Important:** After printing/displaying, measure the actual size of the tag (the edge between white and black border) in meters. You'll need this for accurate distance estimation.

2. **Test Detection with Webcam (No Hardware Control):**

```bash
python3 test_apriltag_detection.py --webcam --no-control --tag-size 0.047
```

Replace `0.047` with your measured tag size in meters (e.g., 0.047 = 47mm).

- The script will automatically use webcam if Picamera2 is not available
- Press 'q' to quit
- You should see the tag detection overlay with angle and distance info

3. **Test with Hardware Control (Raspberry Pi only):**

Once detection works, test with motor and servo control on Raspberry Pi:

```bash
python3 test_apriltag_detection.py --tag-size 0.047 --stop-distance 0.3
```

The car will:
- Detect the AprilTag
- Calculate angle to steer
- Move towards the tag
- Stop when within `--stop-distance` meters

### Quick Start Script

For convenience, use the quick start script:

```bash
./test_apriltag_quickstart.sh
```

This will check dependencies and show you the commands to run.

## Command Line Options

### `generate_apriltag.py`

- `--tag-id`: Tag ID to generate (0-586 for tag36h11, default: 0)
- `--tag-family`: Tag family (default: tag36h11)
- `--size`: Output image size in pixels (default: 200)
- `--output`: Output file path
- `--multiple`: Generate multiple tags (provide IDs as arguments)
- `--aruco`: Use ArUco markers instead of AprilTag (if AprilTag generation fails)

### `test_apriltag_detection.py`

- `--tag-size`: Physical tag size in meters (required for distance estimation)
- `--stop-distance`: Stop when tag is this close in meters (default: 0.3)
- `--no-display`: Disable video display
- `--no-control`: Disable motor/servo control (test detection only)
- `--tag-family`: AprilTag family (default: tag36h11)
- `--webcam`: Use webcam instead of Picamera2 (auto-detected if Picamera2 unavailable)
- `--camera-index`: Webcam index (default: 0, use 1, 2, etc. if you have multiple cameras)

## Camera Calibration (Optional but Recommended)

For accurate distance and pose estimation, calibrate your camera:

1. Print a chessboard pattern (download from OpenCV tutorials)
2. Run camera calibration script (if available)
3. Save calibration to `camera_calibration/calibration_savez.npz`

The test script will automatically use calibration if available, otherwise it uses default parameters.

## Integration with main.py

To integrate AprilTag detection into the home return logic in `main.py`:

1. Import `AprilTagDetector` from `test_apriltag_detection.py`
2. Replace the red box detection in `handle_home_state()` with AprilTag detection
3. Use the same navigation logic (angle calculation, servo/motor control)

Example integration:

```python
from test_apriltag_detection import AprilTagDetector

# In __init__:
self.apriltag_detector = AprilTagDetector(tag_size_m=0.047)

# In handle_home_state:
detection = self.apriltag_detector.detect_tag(frame)
if detection['detected']:
    # Use detection['angle'] and detection['distance_m']
    # Control servo and motor similar to handle_following_user_state
```

## How It Works

1. **Detection**: AprilTag library detects tags in grayscale camera frames
2. **Angle Calculation**: Tag center position relative to frame center → steering angle (-45° to +45°)
3. **Distance Estimation**: Tag size in pixels vs. known physical size → distance in meters
4. **Navigation**: Same logic as user following:
   - Set servo angle based on tag position
   - Adjust motor speed based on whether tag is centered
   - Stop when tag is close enough

## Troubleshooting

**Tag not detected:**
- Ensure good lighting
- Tag should be clearly visible (not too far, not too close)
- Check tag is in focus
- Try different tag families (tag25h9, tag16h5)

**Distance inaccurate:**
- Measure tag size accurately after printing
- Use camera calibration for better accuracy
- Ensure tag is perpendicular to camera (not at angle)

**Low FPS:**
- AprilTag detection is fast, but if slow:
  - Reduce camera resolution
  - Process every Nth frame (add frame skipping)

## References

- [AprilTag GitHub](https://github.com/AprilRobotics/apriltag)
- [dt-apriltags Python Library](https://github.com/daniilidis-group/apriltag)
- [OpenCV ArUco Markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) (alternative to AprilTag)

