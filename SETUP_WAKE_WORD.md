# Wake Word Setup and Testing Guide

This guide explains how to set up and test the wake word detection system for the Bin Diesel project.

## Overview

The wake word system uses audio comparison instead of Google Speech Recognition. You record a reference audio file of your wake word, and the system compares incoming audio to that reference.

## Prerequisites

1. **Microphone**: ReSpeaker or any USB microphone connected
2. **Python packages**: Already installed via `requirements.txt`
3. **Optional**: `librosa` for better audio comparison (falls back to simple mode if not available)

## Step 1: Record Your Wake Word

### Basic Recording

```bash
python3 record_wake_word.py
```

This will:
- Wait 2 seconds
- Adjust for ambient noise
- Record 3 seconds of audio
- Save to `wake_word_reference.wav`

### Custom Options

```bash
# Custom filename and duration
python3 record_wake_word.py --output my_wake_word.wav --duration 5

# Record a longer wake word phrase
python3 record_wake_word.py --duration 4
```

### Tips for Recording

1. **Speak clearly**: Say your wake word naturally but clearly
2. **Quiet environment**: Record in a quiet room to reduce background noise
3. **Consistent volume**: Speak at a normal volume (not too loud or quiet)
4. **Multiple attempts**: You can record multiple times and test which works best

### Example

```bash
$ python3 record_wake_word.py
======================================================================
Recording Wake Word Reference
======================================================================
Recording 3 seconds of audio...
Say your wake word (e.g., 'bin diesel') when ready.
Recording will start in 2 seconds...
[DEBUG] Adjusting for ambient noise...
[DEBUG] Starting to listen...
Recording now! Say your wake word...
[DEBUG] Audio captured: 96000 bytes
[DEBUG] Sample rate: 16000 Hz
[DEBUG] Duration: 3.00 seconds
[DEBUG] Saving to 'wake_word_reference.wav'...
✓ Wake word recorded and saved to 'wake_word_reference.wav'
  File size: 96044 bytes (93.79 KB)
```

## Step 2: Test Wake Word Detection

### Run the Main Program

```bash
python3 bindieselsimple.py
```

The program will:
1. Load the reference wake word file
2. Start listening for the wake word
3. Compare incoming audio to the reference
4. When detected, start camera and look for colored flag

### Debugging Output

The program includes extensive debugging output:

```
[DEBUG] Loading reference audio with librosa...
[DEBUG] Audio loaded: 48000 samples, 16000 Hz
[DEBUG] MFCC features extracted: shape (13, 94)
✓ Loaded reference wake word from 'wake_word_reference.wav' (librosa mode)
[DEBUG] Initializing wake word listener...
[DEBUG] Initializing color flag detector...
[DEBUG] Initializing PSoC communicator...
Listening for wake word (comparing to 'wake_word_reference.wav')...
Similarity threshold: 0.60
[DEBUG] Extracting features from incoming audio...
[DEBUG] Incoming features type: <class 'numpy.ndarray'>
[DEBUG] Incoming features shape: (13, 87)
[DEBUG] Calculated similarity: 0.7234
✓ Wake word detected! (similarity: 0.723 >= 0.60)
```

### Understanding Similarity Scores

- **0.0 - 0.3**: Very different, not matching
- **0.3 - 0.5**: Somewhat similar, but below threshold
- **0.5 - 0.7**: Good match, likely the wake word
- **0.7 - 1.0**: Very good match, definitely the wake word

### Adjusting Sensitivity

If the wake word is not being detected:

1. **Lower the threshold** (edit `bindieselsimple.py`):
   ```python
   wake_word_listener = WakeWordListener(similarity_threshold=0.5)  # Lower = more sensitive
   ```

2. **Re-record the wake word** with clearer pronunciation

3. **Check microphone**: Make sure microphone is working and positioned correctly

If false positives (detecting when you didn't say the wake word):

1. **Raise the threshold**:
   ```python
   wake_word_listener = WakeWordListener(similarity_threshold=0.7)  # Higher = less sensitive
   ```

## Step 3: Testing Workflow

### Complete Test Sequence

1. **Record wake word**:
   ```bash
   python3 record_wake_word.py
   ```

2. **Test wake word detection** (without camera/PSoC):
   - You can modify `bindieselsimple.py` to skip camera/PSoC for testing
   - Or just watch the debug output to see if wake word is detected

3. **Full system test**:
   ```bash
   python3 bindieselsimple.py
   ```
   - Say your wake word
   - Camera should start
   - Flag detection should begin

### Troubleshooting

#### "Reference wake word file not found"

**Solution**: Record the wake word first:
```bash
python3 record_wake_word.py
```

#### "Error opening microphone"

**Solution**: 
- Check microphone is connected: `lsusb` or `arecord -l`
- Check permissions: `sudo usermod -a -G audio $USER` (then logout/login)
- Try different microphone index in code

#### "Low similarity" or wake word not detected

**Solutions**:
1. Re-record the wake word more clearly
2. Lower the similarity threshold
3. Check microphone quality/position
4. Reduce background noise
5. Speak at consistent volume

#### "librosa not available"

**Solution**: This is optional. The system will use simple audio comparison:
```bash
pip3 install librosa  # Optional, for better accuracy
```

## File Structure

```
.
├── record_wake_word.py          # Script to record wake word
├── bindieselsimple.py           # Main program (uses wake word)
├── wake_word_reference.wav      # Your recorded wake word (created after recording)
└── SETUP_WAKE_WORD.md           # This file
```

## Advanced Usage

### Using Different Wake Word Files

Edit `bindieselsimple.py`:
```python
wake_word_listener = WakeWordListener(wake_word_file="my_custom_wake_word.wav")
```

### Testing Similarity Thresholds

Create a test script:
```python
from bindieselsimple import WakeWordListener
import speech_recognition as sr

listener = WakeWordListener(similarity_threshold=0.6)
recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)
    while True:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
        # Test similarity without triggering
        features = listener._extract_features(audio)
        similarity = listener._compare_audio(listener.reference_features, features)
        print(f"Similarity: {similarity:.3f}")
```

## Notes

- The wake word file should be in the same directory as `bindieselsimple.py`
- Default filename: `wake_word_reference.wav`
- Default similarity threshold: 0.6 (60% match required)
- Audio is processed at 16kHz sample rate
- With librosa: Uses MFCC features (more accurate)
- Without librosa: Uses raw audio correlation (simpler, less accurate)

