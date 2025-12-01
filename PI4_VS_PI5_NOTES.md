# Pi4 and Pi5 ReSpeaker Setup

## Both Use PipeWire

Both Raspberry Pi 4 and Pi 5 use **PipeWire** audio system (modern Raspberry Pi OS).

## Setup for ReSpeaker Lite

### 1. Configure PipeWire Sample Rate

For ReSpeaker Lite, set the sample rate to 16000 Hz:

```bash
pw-metadata -n settings 0 clock.force-rate 16000
```

**For permanent change:**
1. Copy the config file:
   ```bash
   sudo cp /usr/share/pipewire/pipewire.conf /etc/pipewire/pipewire.conf
   ```

2. Edit the config:
   ```bash
   sudo nano /etc/pipewire/pipewire.conf
   ```

3. Find and uncomment/edit this line:
   ```
   default.clock.rate = 16000
   ```

4. Restart PipeWire:
   ```bash
   systemctl --user restart pipewire
   ```

### 2. Check ReSpeaker Detection
```bash
lsusb | grep -i seeed
arecord -l
```

### 3. Test Direct Recording
```bash
arecord -d 3 -f cd test.wav
aplay test.wav
```

### 4. Adjust Volume (if needed)
```bash
alsamixer
# Navigate to ReSpeaker device and adjust volume
```

### 5. Python Code
The code structure is the same for both Pi4 and Pi5:
- Use `speech_recognition` library
- Auto-detect or specify ReSpeaker device index
- Google Web Speech Recognition for voice-to-text

### 6. Troubleshooting

**If ALSA errors appear:**
- These are usually harmless warnings
- PipeWire handles audio, ALSA errors can be ignored
- If audio doesn't work, check PipeWire status:
  ```bash
  systemctl --user status pipewire
  ```

**If microphone not detected:**
```bash
# Check USB connection
lsusb | grep -i seeed

# Check PipeWire devices
pw-cli list-objects | grep -i audio

# Check permissions
sudo usermod -a -G audio $USER
# Logout and login again
```

## Reference
- Pi5 Guide: https://wiki.seeedstudio.com/respeaker_lite_pi5/
- Our Pi4 implementation uses same code structure, just different audio backend

