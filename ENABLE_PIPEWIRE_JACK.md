# Enable PipeWire JACK Compatibility

## Why This Helps

PyAudio tries to connect to JACK first, then falls back to ALSA/PipeWire. By enabling PipeWire's JACK compatibility, PyAudio can connect directly via JACK interface to PipeWire, eliminating JACK connection warnings.

## Installation

```bash
# Install PipeWire JACK compatibility layer
sudo apt-get install pipewire-jack

# Restart PipeWire services
systemctl --user restart pipewire pipewire-pulse pipewire-media-session
```

## Verify It's Working

```bash
# Check PipeWire JACK is running
pw-cli list-objects | grep jack

# Or check status
systemctl --user status pipewire-jack
```

## After Installation

Run your voice test again:
```bash
python3 test_voice_simple.py
```

You should see fewer (or no) JACK connection warnings. The ALSA warnings may still appear but are harmless.

## How It Works

- **PipeWire JACK**: Provides a JACK-compatible interface
- **PyAudio**: Connects via JACK → PipeWire → ReSpeaker
- **Result**: No JACK connection errors, cleaner output

## Note

Even without this, your audio works fine - PipeWire handles it. This just makes the output cleaner by eliminating JACK connection attempts.

