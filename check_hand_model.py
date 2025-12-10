#!/usr/bin/env python3
"""
Check if pre-trained hand keypoints model exists
If not, provides instructions to train one
"""

import os
from pathlib import Path

def check_for_hand_model():
    """Check for existing hand keypoints models"""
    
    print("Checking for hand keypoints models...")
    print("=" * 70)
    
    # Check common locations
    possible_locations = [
        'runs/pose/train/weights/best.pt',
        'runs/pose/hand-keypoints-train/weights/best.pt',
        'hand-keypoints.pt',
        'yolo11n-hand-keypoints.pt',
        os.path.expanduser('~/.ultralytics/models/hand-keypoints.pt'),
    ]
    
    found_models = []
    for location in possible_locations:
        if os.path.exists(location):
            size_mb = os.path.getsize(location) / (1024 * 1024)
            found_models.append((location, size_mb))
            print(f"✓ Found: {location} ({size_mb:.1f} MB)")
    
    if not found_models:
        print("✗ No hand keypoints model found")
        print()
        print("To train a hand keypoints model:")
        print("  1. Run: bash train_hand_keypoints.sh")
        print("  2. Or manually:")
        print("     yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640")
        print()
        print("Note: Ultralytics does not provide pre-trained hand keypoints models.")
        print("      You must train on the hand-keypoints dataset.")
        return None
    else:
        print()
        print(f"Found {len(found_models)} model(s)")
        print("Recommended model:", found_models[0][0])
        return found_models[0][0]

if __name__ == '__main__':
    model_path = check_for_hand_model()
    if model_path:
        print(f"\nTo use this model:")
        print(f"  controller = HandGestureController(hand_model_path='{model_path}')")

