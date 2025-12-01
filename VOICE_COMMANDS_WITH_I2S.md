# Using Voice Commands with I2S ReSpeaker

## Overview

When using ReSpeaker Lite via **I2S for TDOA direction estimation**, you can still use **natural voice commands** with a **separate microphone**.

## How It Works

- **ReSpeaker via I2S**: Used for TDOA direction estimation (raw stereo channels)
- **Separate Microphone**: Used for voice command recognition
  - Built-in Raspberry Pi microphone (if available)
  - USB microphone
  - Any other audio input device

## Microphone Options for Voice Commands

### Option 1: Built-in Pi Microphone (if available)

Some Raspberry Pi models have built-in microphones. Check:

```bash
arecord -l
# Look for built-in microphone
```

### Option 2: USB Microphone

Connect a USB microphone for voice commands:

```bash
# Plug in USB mic
arecord -l
# Should show the USB microphone

# Test it
arecord -d 5 -f cd test_voice.wav
aplay test_voice.wav
```

### Option 3: Use ReSpeaker for Both (Advanced)

You could potentially use the ReSpeaker for both TDOA and voice commands, but this requires:
- Configuring the system to use ReSpeaker I2S for TDOA
- Configuring speech recognition to use a different audio source
- More complex setup

**Recommendation**: Use a separate USB microphone for voice commands - simpler and more reliable.

## System Architecture

```
┌─────────────────────────────────────────┐
│         Raspberry Pi                    │
│                                          │
│  ┌──────────────┐    ┌──────────────┐   │
│  │ ReSpeaker    │    │ USB Mic      │   │
│  │ (I2S GPIO)   │    │ (Voice Cmds) │   │
│  └──────┬───────┘    └──────┬───────┘   │
│         │                   │            │
│         ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐   │
│  │ TDOA System  │    │ Speech       │   │
│  │ (Direction)  │    │ Recognition  │   │
│  └──────┬───────┘    └──────┬───────┘   │
│         │                   │            │
│         └─────────┬─────────┘            │
│                   ▼                      │
│            Navigation Controller         │
│                   ▼                      │
│                 PSoC                     │
└─────────────────────────────────────────┘
```

## Setup Instructions

### 1. Connect ReSpeaker via I2S (for TDOA)

Follow `RESPEAKER_I2S_CONNECTION.md` to connect ReSpeaker to GPIO pins.

### 2. Connect USB Microphone (for voice commands)

Plug in a USB microphone. Test it:

```bash
arecord -l
# Should show both ReSpeaker (I2S) and USB mic

# Test USB mic
arecord -D hw:X,0 -d 5 test.wav
# Replace X with USB mic device number
```

### 3. Configure Speech Recognition

The speech recognition system will automatically use the default system microphone. If you want to specify a device:

```python
# In speech_recognizer.py, you can modify to use specific device:
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone(device_index=2)  # Specify device index
```

### 4. Run the System

```bash
# The system will:
# - Use ReSpeaker I2S for person direction (TDOA)
# - Use USB mic for voice commands
python3 vision_main.py
```

## Testing

### Test TDOA (ReSpeaker I2S)

```bash
python3 test_i2s_connection.py
# Should show different channels (not identical)
```

### Test Voice Commands (USB Mic)

```bash
python3 << 'EOF'
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    print("Say something...")
    audio = r.listen(source, timeout=5)
    try:
        text = r.recognize_google(audio)
        print(f"Heard: {text}")
    except:
        print("Could not recognize")
EOF
```

## Benefits of Separate Microphones

1. **Better TDOA**: ReSpeaker I2S gives raw stereo channels for accurate direction
2. **Better Voice Recognition**: USB mic can be positioned optimally for voice
3. **Independent Operation**: Each system works independently
4. **Flexibility**: Can use any microphone for voice commands

## Alternative: Keyboard Controls

If you don't have a separate microphone, you can still use the system with keyboard controls:

- **'f'** - Start following
- **'s'** - Stop
- **'q'** - Quit

Voice commands are optional - the system works fine without them!

## Summary

**Yes, you can use voice commands with I2S!**

- ReSpeaker I2S → TDOA direction estimation
- Separate USB mic → Voice commands
- Both work independently and simultaneously

This gives you the best of both worlds:
- Accurate direction estimation (I2S raw channels)
- Natural voice commands (separate mic)

