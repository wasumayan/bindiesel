# ReSpeaker Lite Firmware and Configuration Fix

## The Problem

Your ReSpeaker Lite is outputting **mono audio to both channels**, which prevents TDOA from working. This is likely due to:

1. **Firmware version** - Older USB firmware versions may not support proper stereo output
2. **Audio processing mode** - The XMOS chip may be outputting processed/beamformed audio instead of raw stereo
3. **Firmware type** - Need USB firmware (not I2S firmware) for USB audio

## Solution Steps

### 1. Check Current Firmware Version

```bash
# Install dfu-util if not already installed
sudo apt-get install dfu-util

# Check firmware version
dfu-util -l
```

Look for:
- **USB Firmware**: Should be **v2.0.7** or higher (minimum v2.0.5 for proper USB support)
- If you see I2S firmware, you need to flash USB firmware

### 2. Download and Flash USB Firmware

According to the [ReSpeaker Lite GitHub](https://github.com/respeaker/ReSpeaker_Lite/):

**Latest USB Firmware: v2.0.7**

1. **Download firmware** from the [ReSpeaker Lite releases](https://github.com/respeaker/ReSpeaker_Lite/tree/master/xmos_firmwares)

2. **Enter DFU mode** on ReSpeaker Lite:
   - Unplug USB
   - Hold down a button (check device documentation for which button)
   - Plug in USB while holding button
   - Release button

3. **Flash firmware**:
   ```bash
   dfu-util -a 0 -D respeaker_lite_usb_v2.0.7.bin
   ```

4. **Unplug and replug** the device

### 3. Verify Device After Flashing

```bash
# Check if device is recognized
arecord -l

# Should show: ReSpeaker Lite: USB Audio

# Test recording
arecord -D hw:3,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav

# Check if it's truly stereo
file test.wav
# Should show: WAVE audio, 2 channels, 16000 Hz
```

### 4. Check Audio Processing Modes

The XMOS XU316 chip has built-in audio processing:
- **Beamforming** - Combines mic signals into directional beam
- **DoA (Direction of Arrival)** - Already calculates direction
- **Noise Suppression** - Processes audio
- **Echo Cancellation** - Processes audio

**The problem**: If the firmware is outputting **processed/beamformed audio**, both channels might be the same processed signal.

**Solution**: We may need to:
1. Disable processing via I2C configuration (if using I2S firmware)
2. Use a firmware version that outputs raw stereo channels
3. Access raw microphone data before processing

### 5. Alternative: Use I2S Firmware with Raw Audio

If USB firmware continues to output mono, consider:

1. **Switch to I2S firmware** - Connect via I2S to get raw audio
2. **Access raw channels** - I2S firmware may provide access to individual mic channels
3. **Configure via I2C** - Use I2C interface to configure audio processing

**Note**: I2S requires different hardware connections (not USB)

## Configuration via I2C (I2S Firmware Only)

If using I2S firmware, you can configure the device via I2C:

- **I2C Address**: `0x42`
- **Resource ID 241 (0xF1)**: Configuration service
- Can read/write registers to control audio processing

See: [ReSpeaker Lite I2C Configuration](https://github.com/respeaker/ReSpeaker_Lite/#resid-2410xf1-configuration-service)

## Testing After Firmware Update

Run the diagnostic again:

```bash
python3 debug_audio_channels.py
```

**Expected results after fix**:
- Channels should be **different** (correlation < 0.9)
- Should see different values when speaking near one mic vs the other
- TDOA should work and show non-zero delays

## If Problem Persists

If channels are still identical after firmware update:

1. **Check firmware version**: Must be USB v2.0.5 or higher
2. **Try different USB port**: Some USB ports may have issues
3. **Check device in Windows/Mac**: Test if issue is Linux-specific
4. **Contact Seeed support**: May be a hardware issue or need special firmware

## References

- [ReSpeaker Lite GitHub](https://github.com/respeaker/ReSpeaker_Lite/)
- [ReSpeaker Lite Wiki](https://wiki.seeedstudio.com/reSpeaker_usb_v3/)
- [XMOS DFU Guide](https://github.com/respeaker/ReSpeaker_Lite/#latest-xmos-firmware)

