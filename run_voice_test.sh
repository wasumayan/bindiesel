#!/bin/bash
# Run voice test with stderr redirected to suppress ALSA/JACK warnings

# Redirect stderr to /dev/null to suppress ALSA/JACK warnings
# These are harmless warnings from the audio system
python3 test_voice_simple.py 2>/dev/null

