# Audio/Microphone Troubleshooting Guide

## Problem: No Audio Input Devices Found

If you see "No audio input devices found!" when running `test_wake_word.py`, follow these steps:

### Step 1: Check Physical Connection

**USB Microphone:**
```bash
# Check if USB microphone is connected
lsusb
# Should show your microphone (e.g., ReSpeaker, USB mic, etc.)
```

**If no USB device appears:**
- Check USB cable connection
- Try a different USB port
- Try a different USB cable
- Check if microphone works on another computer

### Step 2: Check Audio Devices

```bash
# List ALSA audio devices
arecord -l

# Should show something like:
# card 1: DeviceName [Device Name], device 0: USB Audio [USB Audio]
```

**If `arecord -l` shows nothing:**
- No microphone is detected by the system
- Check physical connection
- Check if microphone requires drivers

**If `arecord -l` shows devices but PyAudio doesn't:**
- This is a PyAudio/ALSA configuration issue
- See Step 3

### Step 3: Test Microphone Directly

```bash
# Test recording (5 seconds)
arecord -d 5 -f cd test.wav

# Play it back
aplay test.wav
```

**If `arecord` works:**
- Microphone hardware is fine
- Issue is with PyAudio configuration
- See Step 4

**If `arecord` fails:**
- Hardware or driver issue
- Check microphone on another device
- Check if microphone needs specific drivers

### Step 4: Check Permissions

```bash
# Check current groups
groups

# Add to audio group if not already
sudo usermod -a -G audio $USER

# Logout and login again (or reboot)
# Then check again
groups  # Should show 'audio' in the list
```

### Step 5: Check ALSA Configuration

```bash
# Check ALSA configuration
cat /proc/asound/cards

# Should list audio cards
```

### Step 6: Common Solutions

#### Solution 1: Set Default Audio Device

```bash
# List audio devices
arecord -l

# Create/edit ~/.asoundrc to set default device
nano ~/.asoundrc
```

Add (replace with your card/device numbers from `arecord -l`):
```
pcm.!default {
    type hw
    card 1
    device 0
}

ctl.!default {
    type hw
    card 1
}
```

#### Solution 2: Use PulseAudio (if available)

```bash
# Check if PulseAudio is running
pulseaudio --check

# Start PulseAudio if not running
pulseaudio --start

# List PulseAudio sources
pactl list sources short
```

#### Solution 3: ReSpeaker Specific

If using ReSpeaker microphone:

```bash
# Check if ReSpeaker is detected
lsusb | grep -i respeaker

# Check ReSpeaker documentation for setup
# May need specific drivers or configuration
```

### Step 7: Test with Different Tools

```bash
# Test with arecord (ALSA)
arecord -d 5 test.wav

# Test with Python directly
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0])"
```

## Quick Diagnostic Commands

Run these to gather information:

```bash
# 1. Check USB devices
lsusb

# 2. Check ALSA devices
arecord -l
cat /proc/asound/cards

# 3. Check audio groups
groups | grep audio

# 4. Test recording
arecord -d 3 test.wav && aplay test.wav

# 5. Check PyAudio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

## Common Issues

### Issue: "No default input device found"

**Cause:** System doesn't have a default audio input configured

**Solution:**
1. Connect a microphone
2. Set default device in ALSA config (see Solution 1 above)
3. Or specify device index in code

### Issue: "Permission denied"

**Cause:** User not in audio group

**Solution:**
```bash
sudo usermod -a -G audio $USER
# Logout and login again
```

### Issue: "Device busy"

**Cause:** Another application is using the microphone

**Solution:**
```bash
# Find processes using audio
lsof | grep snd

# Kill processes if needed
# Or close other applications using microphone
```

### Issue: ReSpeaker not detected

**Cause:** ReSpeaker may need specific setup

**Solution:**
- Check ReSpeaker documentation
- May need to install specific drivers
- Check if ReSpeaker has a setup script

## Still Not Working?

1. **Try a different microphone** - Test if it's hardware-specific
2. **Check Raspberry Pi model** - Some models have built-in audio, others don't
3. **Check OS version** - Older Raspberry Pi OS versions may have different audio setups
4. **Check if running in virtual environment** - May need to install pyaudio in venv
5. **Check system logs** - `dmesg | grep -i audio` or `journalctl -u pulseaudio`

## For Wake Word Testing

If you can't get a microphone working, you can:
1. Test the camera/flag detection separately with `camerasimple.py`
2. Skip wake word for now and modify `bindieselsimple.py` to start camera immediately
3. Use a different audio input method (e.g., network audio, Bluetooth)

