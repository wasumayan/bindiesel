# Vision-Based Autonomous Navigation System

An autonomous car navigation system using Raspberry Pi Camera Module 3 for person following with obstacle avoidance, controlled via natural language voice commands.

## System Overview

```
Camera → Person Detection → Navigation Logic → PSoC → Car Motors
         ↓
    Obstacle Detection
         ↓
    Safe Path Planning
```

**Features:**
- **Person Following**: Detects and tracks people using computer vision
- **Obstacle Avoidance**: Avoids obstacles while navigating
- **Natural Language Commands**: Voice control via ReSpeaker mic array + OpenAI
- **PSoC Integration**: Sends speed and direction commands to PSoC for car control

## Hardware Requirements

- Raspberry Pi 4
- Raspberry Pi Camera Module 3
- ReSpeaker Lite 2-mic array (USB)
- PSoC 5 (for PWM control)
- USB-to-Serial adapter (if needed)

## Quick Start

### 1. Install Dependencies

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv libopencv-dev portaudio19-dev python3-pyaudio

# Python packages
pip3 install -r requirements.txt

# Optional: For better wake word detection
pip3 install librosa
```

### 2. Enable Camera

```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

### 3. Set Up Wake Word (NEW - Simple Implementation)

The simple implementation (`bindieselsimple.py`) uses audio file comparison instead of Google Speech Recognition:

```bash
# Record your wake word (e.g., "bin diesel")
python3 record_wake_word.py

# Run the simple implementation
python3 bindieselsimple.py
```

See [SETUP_WAKE_WORD.md](SETUP_WAKE_WORD.md) for detailed instructions.

### 4. Run System

```bash
# Simple implementation (recommended for testing)
python3 bindieselsimple.py

# Or full vision navigation system
python3 vision_main.py

# Test components individually:
python3 test_camera_basic.py        # Test camera
python3 camerasimple.py             # Test camera with color detection
python3 test_respeaker_openai.py    # Test voice + OpenAI
```

## Usage

### Basic Usage

```bash
python3 vision_main.py
```

### Command Line Options

```bash
python3 vision_main.py \
    --port /dev/ttyUSB0 \      # PSoC serial port
    --baudrate 115200 \         # Serial baud rate
    --camera 0 \                # Camera device index
    --wake-word "bin diesel" \  # Voice wake word
    --no-video                  # Disable video display
```

## Voice Commands

Say the wake word followed by a command:

- **"bin diesel, come here"** - Follow the person
- **"bin diesel, stop"** - Stop the car
- **"bin diesel, go forward"** - Move forward
- **"bin diesel, turn left/right"** - Turn direction
- **General queries** - "what time is it?", "who's the president?"

## Project Structure

```
.
├── bindieselsimple.py          # Simple implementation (wake word + flag detection)
├── record_wake_word.py         # Record wake word reference audio
├── camerasimple.py             # Camera test with color flag detection
├── vision_main.py              # Main entry point (full system)
├── vision_navigator.py         # Navigation controller
├── vision_person_tracker.py    # Person detection & tracking
├── obstacle_detector.py        # Obstacle detection
├── speech_recognizer.py        # Voice command recognition
├── psoc_communicator.py       # PSoC serial communication
├── test_camera_basic.py        # Camera test script
├── test_respeaker_openai.py    # Voice + OpenAI test
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SETUP_WAKE_WORD.md          # Wake word setup guide
├── VISION_NAVIGATION_README.md # Detailed system guide
└── OBSTACLE_AVOIDANCE_RESEARCH.md # Obstacle avoidance research
```

## Documentation

- **[VISION_NAVIGATION_README.md](VISION_NAVIGATION_README.md)** - Complete system documentation
- **[OBSTACLE_AVOIDANCE_RESEARCH.md](OBSTACLE_AVOIDANCE_RESEARCH.md)** - Obstacle avoidance methods and research

## PSoC Communication Protocol

Commands sent to PSoC:
```
NAV:ANGLE:XX.XX:SPEED:XX.XX\n    # Navigation command
NAV:STOP\n                        # Stop command
```

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
ls /dev/video*
raspistill -o test.jpg

# Fix permissions
sudo usermod -a -G video $USER
```

### Voice Recognition Issues
```bash
# Check microphone
arecord -l
arecord -d 5 test.wav && aplay test.wav

# Install speech recognition
pip3 install SpeechRecognition
```

### OpenCV Import Issues
The code automatically handles OpenCV import paths. If issues persist:
```bash
sudo apt-get install python3-opencv python3-opencv-contrib
```

## License

Educational/research purposes.
