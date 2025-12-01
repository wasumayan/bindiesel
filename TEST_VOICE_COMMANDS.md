# Testing Voice Commands

Quick guide for testing ReSpeaker voice commands with OpenAI integration.

## Prerequisites

1. **ReSpeaker connected via USB** to Raspberry Pi 4
2. **OpenAI API key** set as environment variable
3. **Internet connection** (for Google Speech Recognition and OpenAI)

## Setup

### 1. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```bash
pip3 install python-dotenv
cp .env.example .env
# Edit .env and add your key
```

### 2. Install Dependencies

```bash
pip3 install SpeechRecognition openai python-dotenv
```

### 3. Verify ReSpeaker Connection

```bash
# Check USB connection
lsusb | grep -i seeed

# Check audio devices
arecord -l

# Test recording
arecord -d 3 -f cd test.wav
aplay test.wav
```

## Running the Test

```bash
python3 test_respeaker_openai.py
```

The script will:
1. List all available microphones
2. Auto-detect ReSpeaker (or use default)
3. Adjust for ambient noise
4. Start listening for voice commands

## Test Commands

### Navigation Commands
- **"go forward"** - Move forward
- **"turn left"** - Turn left
- **"turn right"** - Turn right
- **"stop"** - Stop movement
- **"slow down"** - Reduce speed
- **"speed up"** - Increase speed

### General Queries
- **"what time is it?"** - Get current time
- **"who is the president?"** - General knowledge
- **"what's the weather?"** - Weather query

### Main Functionality
- **"bin diesel, come here"** - Main command (stores position, follows, returns)

## Expected Output

```
======================================================================
ReSpeaker Voice Assistant Test
======================================================================

Available microphones:
  0: Built-in Microphone
  1: ReSpeaker 4 Mic Array (UAC1.0)

✓ Found ReSpeaker at index 1: ReSpeaker 4 Mic Array (UAC1.0)
Adjusting for ambient noise...
✓ Microphone ready

✓ Microphone: Ready
✓ OpenAI: Connected

Commands you can try:
  - Navigation: 'go forward', 'turn left', 'stop', 'slow down'
  - Questions: 'what time is it?', 'who is the president?'
  - Main function: 'bin diesel, come here'

💡 Speak clearly into the ReSpeaker microphone
💡 The system will listen for 3 seconds, then process your command

Press Ctrl+C to exit

🎤 Listening... (speak now)
   Processing speech...

✓ You said: 'go forward'
→ Navigation command: forward
  [Would send to PSoC: forward]
```

## Troubleshooting

### ReSpeaker Not Detected
```bash
# Check USB connection
lsusb

# Check ALSA devices
aplay -l
arecord -l

# Check permissions
sudo usermod -a -G audio $USER
# Logout and login again
```

### No Speech Detected
- Speak louder or closer to microphone
- Check microphone volume: `alsamixer`
- Test with: `arecord -d 5 test.wav && aplay test.wav`

### Speech Recognition Errors
- Check internet connection (Google Speech Recognition requires internet)
- Try speaking more clearly
- Reduce background noise

### OpenAI Errors
- Verify API key: `echo $OPENAI_API_KEY`
- Check internet connection
- Check API quota on OpenAI dashboard

## Next Steps

Once voice commands work:
1. Test with the full vision system: `python3 vision_main.py --no-video`
2. Integrate commands with PSoC communication
3. Test "come here" functionality with person tracking

