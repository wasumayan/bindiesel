# Suppressing Audio Warnings

## Common Warnings (All Harmless)

### ALSA Warnings
```
ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) Unknown PCM surround51
ALSA lib conf.c:5205:(_snd_config_evaluate) function snd_func_refer returned error
```
**These are harmless** - PipeWire handles audio, these are just legacy ALSA messages.

### JACK Warnings
```
Cannot connect to server socket err = No such file or directory
jack server is not running or cannot be started
```
**These are harmless** - JACK is not needed when using PipeWire. PyAudio tries to connect to JACK but falls back to PipeWire/ALSA.

## Solution

The `test_voice_simple.py` script now suppresses these warnings automatically by redirecting stderr during audio operations.

## Manual Suppression (if needed)

If you want to suppress warnings when running other scripts:

```bash
# Suppress all stderr
python3 script.py 2>/dev/null

# Or redirect to a log file
python3 script.py 2>audio_warnings.log
```

## What Actually Matters

- **FLAC must be installed**: `sudo apt-get install flac`
- **Microphone must be detected**: Check with `arecord -l`
- **Internet connection**: Required for Google Speech Recognition

The ALSA and JACK warnings can be completely ignored - they don't affect functionality.

