#!/usr/bin/env python3
"""
Simple standalone TDOA test for ReSpeaker
Tests microphones and verifies angle estimation step-by-step
"""

import numpy as np
import pyaudio
import time

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MIC_SPACING = 0.07  # 7 cm in meters
SPEED_OF_SOUND = 343.0  # m/s

# Calculate maximum expected delay
MAX_DELAY_SAMPLES = int((MIC_SPACING / SPEED_OF_SOUND) * SAMPLE_RATE) + 1


def find_respeaker_device():
    """Find ReSpeaker device index"""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '').lower()
        if 'respeaker' in name or 'seeed' in name:
            print(f"✓ Found ReSpeaker: {info['name']} at index {i}")
            p.terminate()
            return i
    print("⚠ Warning: ReSpeaker not found, using default input device")
    p.terminate()
    return None


def test_audio_levels(stream, duration=3):
    """Test if microphones are receiving audio"""
    print("\n" + "=" * 70)
    print("TEST 1: Audio Input Test")
    print("=" * 70)
    print("Speak or make noise near the microphones...\n")
    
    max_left = 0
    max_right = 0
    chunk_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        audio = audio.reshape(-1, 2)  # Stereo
        
        left = audio[:, 0].astype(np.float32) / 32768.0
        right = audio[:, 1].astype(np.float32) / 32768.0
        
        max_left = max(max_left, np.abs(left).max())
        max_right = max(max_right, np.abs(right).max())
        chunk_count += 1
        
        if chunk_count % 10 == 0:
            left_bars = int(np.abs(left).max() * 30)
            right_bars = int(np.abs(right).max() * 30)
            
            # Check if channels are identical
            are_identical = np.array_equal(left, right)
            diff = np.abs(left - right).max() if not are_identical else 0
            
            status = "⚠ IDENTICAL" if are_identical else "✓ DIFFERENT"
            print(f"Chunk {chunk_count:3d} | "
                  f"Left: {'█' * left_bars:<30} ({np.abs(left).max():.4f}) | "
                  f"Right: {'█' * right_bars:<30} ({np.abs(right).max():.4f}) | "
                  f"{status} (diff={diff:.6f})")
    
    print(f"\nMax levels: Left={max_left:.4f}, Right={max_right:.4f}")
    
    # Check if channels are identical
    print("\n" + "-" * 70)
    print("Channel Analysis:")
    print("-" * 70)
    
    # Re-read a sample to check
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.int16)
    audio = audio.reshape(-1, 2)
    left_sample = audio[:, 0]
    right_sample = audio[:, 1]
    
    are_identical = np.array_equal(left_sample, right_sample)
    correlation = np.corrcoef(left_sample, right_sample)[0, 1] if len(left_sample) > 1 else 1.0
    
    print(f"Channels identical: {are_identical}")
    print(f"Channel correlation: {correlation:.6f}")
    
    if are_identical or correlation > 0.99:
        print("\n❌ CRITICAL PROBLEM: Channels are identical!")
        print("   This means the device is outputting MONO (same signal to both channels).")
        print("   TDOA cannot work with identical channels - angles will always be 0°.")
        print("\n   Possible causes:")
        print("   1. ReSpeaker is configured for mono output")
        print("   2. USB audio device mode issue")
        print("   3. Device needs special drivers or configuration")
        print("\n   Try:")
        print("   - Check ReSpeaker configuration/firmware")
        print("   - Run: arecord -D hw:3,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav")
        print("   - Check if device has a stereo/mono switch")
        return False
    
    if max_left < 0.01 or max_right < 0.01:
        print("⚠ WARNING: Low audio levels! Check microphone connections.")
        return False
    else:
        print("✓ Microphones are receiving audio and channels are different!")
        return True


def compute_tdoa(left, right):
    """
    Simple TDOA using basic cross-correlation
    Returns: delay in samples
    """
    # Remove DC offset
    left = left - np.mean(left)
    right = right - np.mean(right)
    
    # Basic cross-correlation
    correlation = np.correlate(left, right, mode='full')
    
    # Find peak in valid range
    center = len(correlation) // 2
    search_start = center - MAX_DELAY_SAMPLES
    search_end = center + MAX_DELAY_SAMPLES + 1
    
    correlation_window = correlation[search_start:search_end]
    peak_index = np.argmax(np.abs(correlation_window))
    
    # Convert to delay relative to center
    delay_samples = peak_index - MAX_DELAY_SAMPLES
    
    return delay_samples, correlation_window[peak_index]


def delay_to_angle(delay_samples):
    """Convert delay in samples to angle in degrees"""
    # Convert samples to time
    delay_time = delay_samples / SAMPLE_RATE
    
    # TDOA formula: sin(angle) = (delay * speed_of_sound) / mic_spacing
    sin_angle = (delay_time * SPEED_OF_SOUND) / MIC_SPACING
    
    # Clamp to valid range
    sin_angle = np.clip(sin_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = np.arcsin(sin_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Sign: negative delay means left mic heard it first (sound from left)
    if delay_samples < 0:
        angle_deg = -angle_deg
    
    return angle_deg, delay_time


def test_tdoa(stream, duration=10):
    """Test TDOA and angle estimation"""
    print("\n" + "=" * 70)
    print("TEST 2: TDOA and Angle Estimation")
    print("=" * 70)
    print("\nInstructions:")
    print("  - Speak from LEFT side → should see negative angles")
    print("  - Speak from CENTER → should see ~0°")
    print("  - Speak from RIGHT side → should see positive angles")
    print("\nStarting test...\n")
    
    angles = []
    delays = []
    zero_delay_count = 0
    chunk_count = 0
    start_time = time.time()
    last_print = start_time
    
    try:
        while time.time() - start_time < duration:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            audio = audio.reshape(-1, 2)
            
            left = audio[:, 0].astype(np.float32) / 32768.0
            right = audio[:, 1].astype(np.float32) / 32768.0
            
            # Compute TDOA
            delay_samples, corr_value = compute_tdoa(left, right)
            
            # Convert to angle
            angle_deg, delay_time_ms = delay_to_angle(delay_samples)
            
            # Track statistics
            if delay_samples == 0:
                zero_delay_count += 1
            delays.append(delay_samples)
            angles.append(angle_deg)
            chunk_count += 1
            
            # Print every 0.5 seconds
            if time.time() - last_print >= 0.5:
                left_level = np.abs(left).max()
                right_level = np.abs(right).max()
                
                print(f"Chunk {chunk_count:4d} | "
                      f"Delay: {delay_samples:4d} samples ({delay_time_ms*1000:6.2f} ms) | "
                      f"Angle: {angle_deg:7.2f}° | "
                      f"Levels: L={left_level:.3f} R={right_level:.3f}")
                last_print = time.time()
    
    except KeyboardInterrupt:
        pass
    
    # Statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"Total chunks: {chunk_count}")
    print(f"Zero delay chunks: {zero_delay_count} ({100*zero_delay_count/chunk_count:.1f}%)")
    print(f"\nDelay range: {np.min(delays)} to {np.max(delays)} samples")
    print(f"Delay mean: {np.mean(delays):.2f} samples")
    print(f"Delay std dev: {np.std(delays):.2f} samples")
    print(f"\nAngle range: {np.min(angles):.2f}° to {np.max(angles):.2f}°")
    print(f"Angle mean: {np.mean(angles):.2f}°")
    print(f"Angle std dev: {np.std(angles):.2f}°")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("Diagnosis:")
    print("=" * 70)
    
    if zero_delay_count / chunk_count > 0.8:
        print("⚠ WARNING: Most delays are zero!")
        print("   This means both mics are hearing sound at the same time.")
        print("   Possible causes:")
        print("   1. Sound is directly in front (0°) - this is normal")
        print("   2. Try speaking from the SIDE (left or right)")
        print("   3. Microphones may be too close together")
    else:
        print("✓ Delays are varying - good for direction estimation!")
    
    if np.std(angles) < 1.0:
        print("⚠ WARNING: Angles are very consistent (low variation)")
        print("   Try moving the sound source to different positions")
    else:
        print("✓ Angles are varying - estimation is working!")
    
    if abs(np.mean(angles)) > 10:
        print(f"⚠ NOTE: Average angle is {np.mean(angles):.2f}°")
        print("   This might indicate a bias. Check microphone alignment.")
    else:
        print("✓ Average angle is near 0° - good calibration!")


def show_detailed_calculation(stream):
    """Show step-by-step calculation for one chunk"""
    print("\n" + "=" * 70)
    print("TEST 3: Detailed Calculation (Single Chunk)")
    print("=" * 70)
    print("Speak near the microphones, then press Enter...")
    input()
    
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.int16)
    audio = audio.reshape(-1, 2)
    
    left = audio[:, 0].astype(np.float32) / 32768.0
    right = audio[:, 1].astype(np.float32) / 32768.0
    
    print("\nAudio Statistics:")
    print(f"  Left:  mean={np.mean(left):.6f}, std={np.std(left):.6f}, max={np.abs(left).max():.6f}")
    print(f"  Right: mean={np.mean(right):.6f}, std={np.std(right):.6f}, max={np.abs(right).max():.6f}")
    
    # TDOA
    delay_samples, corr_value = compute_tdoa(left, right)
    delay_time = delay_samples / SAMPLE_RATE
    
    print(f"\nTDOA Calculation:")
    print(f"  Delay: {delay_samples} samples")
    print(f"  Delay time: {delay_time*1000:.3f} ms")
    print(f"  Correlation peak: {corr_value:.6f}")
    
    # Angle
    sin_angle = (delay_time * SPEED_OF_SOUND) / MIC_SPACING
    sin_angle = np.clip(sin_angle, -1.0, 1.0)
    angle_rad = np.arcsin(sin_angle)
    angle_deg = np.degrees(angle_rad)
    
    if delay_samples < 0:
        angle_deg = -angle_deg
    
    print(f"\nAngle Calculation:")
    print(f"  sin(angle) = ({delay_time:.6f} s × {SPEED_OF_SOUND} m/s) / {MIC_SPACING} m")
    print(f"  sin(angle) = {sin_angle:.6f}")
    print(f"  angle = {angle_deg:.2f}°")
    
    if delay_samples == 0:
        print("\n  → Sound is directly in front (0°)")
    elif delay_samples > 0:
        print(f"\n  → Right mic heard it first → Sound from RIGHT ({angle_deg:.2f}°)")
    else:
        print(f"\n  → Left mic heard it first → Sound from LEFT ({angle_deg:.2f}°)")


def main():
    print("=" * 70)
    print("Simple TDOA Test for ReSpeaker")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_SIZE} samples")
    print(f"  Mic spacing: {MIC_SPACING*100:.1f} cm")
    print(f"  Speed of sound: {SPEED_OF_SOUND} m/s")
    print(f"  Max expected delay: ±{MAX_DELAY_SAMPLES} samples")
    
    # Initialize audio
    p = pyaudio.PyAudio()
    device_index = find_respeaker_device()
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE
    )
    
    try:
        # Test 1: Audio levels
        audio_ok = test_audio_levels(stream, duration=3)
        if not audio_ok:
            print("\n❌ Audio test failed. Fix microphone issues first.")
            return
        
        input("\nPress Enter to continue to TDOA test...")
        
        # Test 2: TDOA
        test_tdoa(stream, duration=10)
        
        input("\nPress Enter for detailed calculation...")
        
        # Test 3: Detailed calculation
        show_detailed_calculation(stream)
        
        print("\n" + "=" * 70)
        print("Testing Complete!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("\nAudio resources cleaned up.")


if __name__ == '__main__':
    main()

