#!/usr/bin/env python3
"""
Debug script to check if ReSpeaker is providing true stereo channels
or if it's duplicating mono to both channels
"""

import numpy as np
import pyaudio
import time

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

def check_device_channels():
    """Check device channel configuration"""
    p = pyaudio.PyAudio()
    
    # Find ReSpeaker
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '').lower()
        if 'respeaker' in name or 'seeed' in name:
            device_index = i
            print(f"Found ReSpeaker: {info['name']} at index {i}")
            print(f"  Max Input Channels: {info.get('maxInputChannels', 0)}")
            print(f"  Default Sample Rate: {info.get('defaultSampleRate', 0)}")
            break
    
    if device_index is None:
        print("ReSpeaker not found!")
        p.terminate()
        return
    
    # Try to open stream and check what we get
    print("\n" + "=" * 70)
    print("Testing Audio Channels")
    print("=" * 70)
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("\nRecording 2 seconds... Speak into ONE microphone at a time!")
        print("(Try covering one mic, then the other)\n")
        
        chunks = []
        for _ in range(30):  # ~2 seconds
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            chunks.append(data)
            time.sleep(0.067)  # ~67ms per chunk
        
        stream.stop_stream()
        stream.close()
        
        # Analyze the data
        print("\n" + "=" * 70)
        print("Analysis:")
        print("=" * 70)
        
        all_left = []
        all_right = []
        identical_count = 0
        different_count = 0
        
        for chunk_data in chunks:
            audio = np.frombuffer(chunk_data, dtype=np.int16)
            audio = audio.reshape(-1, 2)
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            all_left.extend(left)
            all_right.extend(right)
            
            # Check if channels are identical
            if np.array_equal(left, right):
                identical_count += 1
            else:
                different_count += 1
        
        all_left = np.array(all_left)
        all_right = np.array(all_right)
        
        print(f"\nChunks analyzed: {len(chunks)}")
        print(f"Identical chunks: {identical_count}")
        print(f"Different chunks: {different_count}")
        
        # Correlation between channels
        correlation = np.corrcoef(all_left, all_right)[0, 1]
        print(f"\nChannel correlation: {correlation:.6f}")
        
        if correlation > 0.99:
            print("⚠️  PROBLEM: Channels are nearly identical!")
            print("   This means the device is likely outputting MONO audio")
            print("   (same signal to both channels)")
        elif correlation > 0.9:
            print("⚠️  WARNING: Channels are very similar")
            print("   This might indicate poor stereo separation")
        else:
            print("✓ Channels are different - stereo is working!")
        
        # Check if one channel is just a copy of the other
        if np.array_equal(all_left, all_right):
            print("\n❌ CRITICAL: Left and right channels are EXACTLY identical!")
            print("   The device is definitely outputting mono to both channels.")
        else:
            # Calculate difference
            diff = np.abs(all_left.astype(np.float32) - all_right.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"\nChannel differences:")
            print(f"  Max difference: {max_diff}")
            print(f"  Mean difference: {mean_diff:.2f}")
            
            if max_diff < 10:  # Very small difference
                print("  ⚠️  Differences are very small - likely mono duplication")
            else:
                print("  ✓ Significant differences detected - stereo is working")
        
        # Try different channel configurations
        print("\n" + "=" * 70)
        print("Testing Alternative Configurations:")
        print("=" * 70)
        
        # Try with channels=1 to see if device supports mono
        try:
            stream_mono = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE
            )
            print("✓ Device supports mono (1 channel) input")
            stream_mono.stop_stream()
            stream_mono.close()
        except Exception as e:
            print(f"✗ Device does not support mono: {e}")
        
        # Check if we can access channels separately
        print("\nRecommendation:")
        if correlation > 0.99:
            print("  The ReSpeaker Lite might be configured for mono output.")
            print("  Check:")
            print("    1. ReSpeaker firmware/configuration")
            print("    2. USB audio device mode")
            print("    3. Try using arecord to test: arecord -D hw:3,0 -f S16_LE -r 16000 -c 2 test.wav")
            print("    4. Check if ReSpeaker has a configuration tool or switch")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        p.terminate()


if __name__ == '__main__':
    check_device_channels()

