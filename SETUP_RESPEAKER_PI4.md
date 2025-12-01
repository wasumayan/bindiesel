# ReSpeaker Lite Setup for Raspberry Pi 4

## Quick Setup

### 1. Install Dependencies

```bash
sudo apt update
sudo apt install python3-pip python3-dev portaudio19-dev
pip3 install pyaudio speechrecognition
```

### 2. Configure PipeWire (Important!)

Raspberry Pi 4 uses PipeWire. Set the sample rate for ReSpeaker:

```bash
pw-metadata -n settings 0 clock.force-rate 16000
```

**For permanent change:**
```bash
# Copy config file
sudo cp /usr/share/pipewire/pipewire.conf /etc/pipewire/pipewire.conf

# Edit config
sudo nano /etc/pipewire/pipewire.conf

# Find and set:
default.clock.rate = 16000

# Restart PipeWire
systemctl --user restart pipewire
```

### 2a. Enable PipeWire JACK Compatibility (Optional - Reduces Warnings)

PipeWire can act as a JACK server, which reduces JACK connection warnings:

```bash
# Check if PipeWire JACK is installed
dpkg -l | grep pipewire-jack

# If not installed, install it:
sudo apt-get install pipewire-jack

# Restart PipeWire services
systemctl --user restart pipewire pipewire-pulse pipewire-media-session
```

This allows PyAudio to connect via JACK interface to PipeWire, eliminating JACK warnings.

### 3. Verify ReSpeaker Connection

```bash
# Check USB
lsusb | grep -i seeed

# Check audio devices
arecord -l

# Test recording
arecord -d 3 -f cd test.wav
aplay test.wav
```

### 4. Adjust Volume (if needed)

```bash
alsamixer
# Select ReSpeaker device and adjust volume
```

### 5. Test Voice Recognition

```bash
python3 test_voice_simple.py
```

## Troubleshooting

### ALSA Errors
The ALSA warnings you see are usually harmless. PipeWire handles the audio, and these are just legacy ALSA messages.

### Microphone Not Working
1. Check PipeWire is running:
   ```bash
   systemctl --user status pipewire
   ```

2. Check permissions:
   ```bash
   sudo usermod -a -G audio $USER
   # Logout and login again
   ```

3. Try specifying device index in Python code

### No Speech Detected
- Speak louder or closer to microphone
- Check microphone volume with `alsamixer`
- Ensure internet connection (Google Speech Recognition requires internet)

## Reference
- Pi5 Guide: https://wiki.seeedstudio.com/respeaker_lite_pi5/
- Same setup applies to Pi4 (both use PipeWire)

