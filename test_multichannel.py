#!/usr/bin/env python3
"""
Test if ReSpeaker outputs multiple channels with different content
The USB firmware might output processed audio on some channels
and raw microphone data on others
"""

import pyaudio
import numpy as np
import time

def test_channel_count():
    """Check how many channels the device supports"""
    print("=" * 70)
    print("Testing ReSpeaker Channel Capabilities")
    print("=" * 70)
    
    p = pyaudio.PyAudio()
    
    # Find ReSpeaker
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '').lower()
        if 'respeaker' in name or 'seeed' in name:
            device_index = i
            print(f"\nFound ReSpeaker: {info['name']} at index {i}")
            print(f"  Max input channels: {info['maxInputChannels']}")
            print(f"  Default sample rate: {info['defaultSampleRate']}")
            break
    
    if device_index is None:
        print("ReSpeaker not found!")
        p.terminate()
        return
    
    max_channels = info['maxInputChannels']
    
    # Test different channel counts
    print("\n" + "=" * 70)
    print("Testing Different Channel Configurations")
    print("=" * 70)
    
    for channels in [2, 4, 6, 8]:
        if channels > max_channels:
            print(f"\nSkipping {channels} channels (max is {max_channels})")
            continue
        
        print(f"\nTesting {channels} channels...")
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=16000,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            
            print(f"  ✓ Successfully opened {channels} channel stream")
            
            # Capture a chunk
            data = stream.read(1024, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            audio = audio.reshape(-1, channels)
            
            # Check if channels are different
            print(f"  Audio shape: {audio.shape}")
            
            # Compare first two channels
            if channels >= 2:
                ch0 = audio[:, 0].astype(np.float32) / 32768.0
                ch1 = audio[:, 1].astype(np.float32) / 32768.0
                
                are_identical = np.array_equal(ch0, ch1)
                correlation = np.corrcoef(ch0, ch1)[0, 1] if len(ch0) > 1 else 1.0
                max_diff = np.abs(ch0 - ch1).max()
                
                print(f"  Channel 0 vs Channel 1:")
                print(f"    Identical: {are_identical}")
                print(f"    Correlation: {correlation:.6f}")
                print(f"    Max difference: {max_diff:.6f}")
                
                if not are_identical and correlation < 0.9:
                    print(f"  ✓ Channels 0 and 1 are DIFFERENT!")
                    print(f"    You can use these for TDOA!")
            
            # Compare other channel pairs if available
            if channels >= 4:
                ch2 = audio[:, 2].astype(np.float32) / 32768.0
                ch3 = audio[:, 3].astype(np.float32) / 32768.0
                
                are_identical_23 = np.array_equal(ch2, ch3)
                correlation_23 = np.corrcoef(ch2, ch3)[0, 1] if len(ch2) > 1 else 1.0
                
                print(f"  Channel 2 vs Channel 3:")
                print(f"    Identical: {are_identical_23}")
                print(f"    Correlation: {correlation_23:.6f}")
                
                if not are_identical_23 and correlation_23 < 0.9:
                    print(f"  ✓ Channels 2 and 3 are DIFFERENT!")
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"  ✗ Failed to open {channels} channels: {e}")
    
    p.terminate()
    
    print("\n" + "=" * 70)
    print("Recommendation:")
    print("=" * 70)
    print("If any channel pair shows different content (correlation < 0.9),")
    print("you can use those channels for TDOA instead of channels 0 and 1.")

if __name__ == '__main__':
    test_channel_count()

