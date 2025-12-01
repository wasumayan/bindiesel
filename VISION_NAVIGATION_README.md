# Vision-Based Navigation System

This system enables your car to follow a person using a Raspberry Pi Camera Module 3, with natural language voice commands and obstacle avoidance.

## Features

- **Person Detection & Tracking**: Uses MediaPipe or OpenCV to detect and track people
- **Obstacle Avoidance**: Detects obstacles and finds safe paths
- **Natural Language Commands**: Voice commands like "bin diesel, come here"
- **PSoC Integration**: Sends speed and direction commands to PSoC for car control

## System Architecture

```
Camera → Person Detection → Navigation Logic → PSoC → Car Motors
         ↓
    Obstacle Detection
         ↓
    Safe Path Planning
```

## Installation

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    portaudio19-dev \
    python3-pyaudio
```

### 2. Install Python Packages

```bash
pip3 install -r requirements.txt
```

Or install individually:
```bash
pip3 install opencv-python opencv-contrib-python mediapipe SpeechRecognition
```

### 3. Camera Setup

Enable camera on Raspberry Pi:
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
```

Test camera:
```bash
raspistill -o test.jpg
```

## Usage

### Basic Usage

```bash
python3 vision_main.py
```

### With Custom Settings

```bash
python3 vision_main.py --port /dev/ttyUSB0 --camera 0 --wake-word "bin diesel"
```

### Command Line Options

- `--port`: Serial port for PSoC (default: `/dev/ttyUSB0`)
- `--baudrate`: Serial baud rate (default: `115200`)
- `--camera`: Camera device index (default: `0`)
- `--wake-word`: Wake word for voice commands (default: `bin diesel`)
- `--no-video`: Disable video display (for headless operation)

## Voice Commands

Say the wake word followed by a command:

- **"bin diesel, come here"** - Start following the person
- **"bin diesel, come to me"** - Start following
- **"bin diesel, stop"** - Stop the car
- **"bin diesel, follow"** - Start following mode

## Keyboard Controls

When video window is open:
- **'q'** - Quit
- **'f'** - Start following mode
- **'s'** - Stop

## PSoC Communication Protocol

The system sends commands to PSoC in the format:

```
NAV:ANGLE:XX.XX:SPEED:XX.XX\n
```

Example:
```
NAV:ANGLE:15.50:SPEED:0.75\n
```

For stop:
```
NAV:STOP\n
```

### PSoC Code Requirements

Your PSoC should:
1. Parse the `NAV:` commands
2. Extract angle and speed values
3. Convert angle to PWM for steering servo
4. Convert speed to PWM for motor control
5. Execute the commands

Example PSoC pseudocode:
```c
// Parse: "NAV:ANGLE:15.50:SPEED:0.75\n"
if (strncmp(buffer, "NAV:ANGLE:", 10) == 0) {
    float angle = atof(buffer + 10);
    // Find speed
    char* speed_ptr = strstr(buffer, "SPEED:");
    if (speed_ptr) {
        float speed = atof(speed_ptr + 6);
        // Update PWM outputs
        update_steering(angle);
        update_motor(speed);
    }
} else if (strncmp(buffer, "NAV:STOP", 8) == 0) {
    stop_motor();
}
```

## Components

### 1. Person Tracker (`vision_person_tracker.py`)
- Detects people in camera frame
- Tracks person position
- Calculates direction and distance

### 2. Obstacle Detector (`obstacle_detector.py`)
- Detects obstacles using depth estimation
- Creates obstacle map
- Finds safe navigation paths

### 3. Speech Recognizer (`speech_recognizer.py`)
- Listens for wake word
- Recognizes voice commands
- Processes natural language

### 4. Vision Navigator (`vision_navigator.py`)
- Combines all components
- Implements navigation logic
- Sends commands to PSoC

## Calibration

### Camera Field of View
Adjust in `vision_person_tracker.py`:
```python
fov_degrees = 60.0  # Adjust based on your camera
```

### Distance Estimation
Calibrate in `vision_person_tracker.py`:
```python
reference_size_1m = 200  # Pixels at 1 meter distance
```

### Obstacle Detection
Adjust in `obstacle_detector.py`:
```python
self.obstacle_threshold = 0.3  # Minimum safe distance (meters)
```

## Troubleshooting

### Camera Not Working
```bash
# Check if camera is detected
ls /dev/video*

# Test with raspistill
raspistill -o test.jpg

# Check permissions
sudo usermod -a -G video $USER
# Logout and login again
```

### Person Not Detected
- Ensure good lighting
- Person should be clearly visible
- Try adjusting detection confidence in `PersonTracker`

### Obstacles Not Detected
- Obstacle detection uses simple depth estimation
- May need calibration for your environment
- Consider using stereo camera or depth sensor for better results

### Speech Recognition Not Working
- Check microphone: `arecord -l`
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`
- For offline recognition, install Vosk:
  ```bash
  pip3 install vosk
  ```

## Advanced: Using YOLO for Better Detection

For more accurate person detection, you can use YOLOv8:

```bash
pip3 install ultralytics
```

Then modify `vision_person_tracker.py` to use YOLO instead of MediaPipe.

## Performance Tips

- Lower camera resolution for faster processing
- Reduce frame rate if needed
- Use MediaPipe (lighter) instead of YOLO for real-time performance
- Disable video display with `--no-video` for headless operation

## Next Steps

1. **Calibrate** the system for your car and environment
2. **Test** person following in a safe area
3. **Adjust** obstacle detection parameters
4. **Fine-tune** PSoC communication protocol
5. **Add** more voice commands as needed

## Notes

- The system runs at ~10 Hz update rate
- Person detection works best with clear view of person
- Obstacle avoidance is basic - consider adding depth sensor for better results
- Voice commands require clear speech and quiet environment

