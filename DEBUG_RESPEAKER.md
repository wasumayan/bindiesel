# Debugging ReSpeaker Detection

## Issue: ReSpeaker Not Showing in Microphone List

If ReSpeaker is detected via USB but not in Python microphone list:

### 1. Check USB Connection
```bash
lsusb | grep -i seeed
# Should show: Bus 001 Device 003: ID 2886:0019 Seeed Technology Co., Ltd. ReSpeaker Lite
```

### 2. Check ALSA Devices
```bash
arecord -l
# Should show ReSpeaker as a card
```

### 3. Check PipeWire Devices
```bash
pw-cli list-objects | grep -i audio
# Or
pw-cli list-objects | grep -i input
```

### 4. Check if ReSpeaker is Default Input
```bash
# Check default input
pactl list sources short

# Set ReSpeaker as default (if found)
pactl set-default-source <source_name>
```

### 5. Test Direct Recording
```bash
# Try recording with specific device
arecord -D hw:1,0 -d 3 test.wav  # Adjust hw:X,0 based on arecord -l output
aplay test.wav
```

### 6. Check Permissions
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again

# Check groups
groups
```

### 7. Restart Audio Services
```bash
systemctl --user restart pipewire pipewire-pulse
```

### 8. Check if ReSpeaker Needs Firmware Update
Sometimes ReSpeaker needs firmware update for proper USB audio:
- Check firmware version
- Update if needed (see ReSpeaker documentation)

## Common Issues

### ReSpeaker Shows as Output Only
- Some ReSpeaker modes only output audio
- May need to switch to input mode
- Check ReSpeaker documentation for mode switching

### ReSpeaker Not in Python List
- PyAudio may not see PipeWire devices directly
- Try using device index from `arecord -l`
- May need to specify device explicitly

### Multiple Audio Devices
- Pi's built-in audio (bcm2835) may be default
- ReSpeaker may be at different index
- Use the detailed device listing in test script

