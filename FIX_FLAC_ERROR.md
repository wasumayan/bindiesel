# Fix FLAC Error

## Error Message
```
Error: FLAC conversion utility not available - consider installing the FLAC command line application
```

## Solution

Install FLAC on your Raspberry Pi:

```bash
sudo apt-get install flac
```

## Why This Happens

The `speech_recognition` library uses FLAC to convert audio before sending it to Google's Speech Recognition API. FLAC is a lossless audio codec that Google requires.

## After Installing

Run the test again:
```bash
python3 test_voice_simple.py
```

The ALSA warnings are harmless and can be ignored - they're just legacy messages from the audio system.

