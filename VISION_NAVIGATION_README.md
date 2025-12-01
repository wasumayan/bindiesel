# Vision-Based Navigation System

This system enables your car to follow a person using a Raspberry Pi Camera Module 3, with natural language voice commands and obstacle avoidance.

## Features

- **Person Detection & Tracking**: Uses MediaPipe or OpenCV to detect and track people
- **Obstacle Avoidance**: Detects obstacles and finds safe paths
- **Natural Language Commands**: Voice commands via ReSpeaker mic array + OpenAI LLM
- **PSoC Integration**: Sends speed and direction commands to PSoC for car control
- **OpenAI Integration**: Handles navigation commands, general queries, and "come here" functionality

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
pip3 install opencv-python opencv-contrib-python mediapipe SpeechRecognition openai python-dotenv
```

### 3. Setup ReSpeaker Microphone

The ReSpeaker Lite should be connected via USB. On Raspberry Pi 4, it will appear as a USB audio device.

**Check ReSpeaker connection:**
```bash
# List audio devices
arecord -l

# Test recording
arecord -d 5 -f cd test.wav
aplay test.wav
```

**On Raspberry Pi 4 (PipeWire):**
Both Pi4 and Pi5 use PipeWire. Configure sample rate:
```bash
pw-metadata -n settings 0 clock.force-rate 16000
```

For permanent change, edit `/etc/pipewire/pipewire.conf` and set `default.clock.rate = 16000`

**Troubleshooting:**
- If ReSpeaker not detected: `lsusb` to check USB connection
- Check audio devices: `aplay -l` and `arecord -l`
- Check PipeWire: `systemctl --user status pipewire`
- Permissions: `sudo usermod -a -G audio $USER` (logout/login)
- ALSA warnings are usually harmless (PipeWire handles audio)

### 4. Setup OpenAI API Key

**Option 1: Environment Variable**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Option 2: .env File (Recommended)**
```bash
pip3 install python-dotenv
cp .env.example .env
# Edit .env and add your API key
```

The `.env` file is in `.gitignore` and won't be committed.

### 5. Camera Setup

Enable camera on Raspberry Pi:
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
```

Test camera:
```bash
raspistill -o test.jpg
# Or use test script
python3 test_camera_basic.py
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

The system uses OpenAI to process natural language commands. Say the wake word followed by a command:

### Navigation Commands
- **"bin diesel, come here"** - Follow the person and return to original spot
- **"bin diesel, come to me"** - Start following
- **"bin diesel, stop"** - Stop the car
- **"bin diesel, go forward"** - Move forward
- **"bin diesel, turn left/right"** - Turn direction
- **"bin diesel, slow down"** - Reduce speed
- **"bin diesel, speed up"** - Increase speed

### General Queries
- **"bin diesel, what time is it?"** - Get current time
- **"bin diesel, who's the president?"** - General knowledge queries

### Main Functionality
- **"bin diesel, come here"** - The car will:
  1. Store its current position
  2. Follow you while avoiding obstacles
  3. Return to original position when done

**Test voice commands:**
```bash
python3 test_respeaker_openai.py
```

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
- Verify ReSpeaker is detected: `python3 test_respeaker_openai.py` (will list microphones)
- Check OpenAI API key is set: `echo $OPENAI_API_KEY`
- For offline recognition (optional), install Vosk:
  ```bash
  pip3 install vosk
  ```

### OpenAI API Issues
- Verify API key is set correctly
- Check internet connection (OpenAI requires internet)
- Test with: `python3 test_respeaker_openai.py`
- Check API quota/limits on OpenAI dashboard

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

