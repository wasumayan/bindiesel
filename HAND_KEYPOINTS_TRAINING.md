# Hand Keypoints Model Training Guide

## Overview

For best hand gesture recognition accuracy, train a YOLO model on the hand-keypoints dataset. This provides 21 keypoints per hand (vs 17 body pose keypoints), enabling more precise gesture detection.

**Reference**: https://docs.ultralytics.com/datasets/pose/hand-keypoints/

## Quick Training

```bash
# Train YOLO11n-pose model on hand-keypoints dataset
yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

# This will:
# 1. Auto-download hand-keypoints dataset (369 MB, 26,768 images)
# 2. Train for 100 epochs
# 3. Save best model as runs/pose/train/weights/best.pt
```

## Using Trained Model

```python
from hand_gesture_controller import HandGestureController

# Use your trained model
controller = HandGestureController(
    hand_model_path='runs/pose/train/weights/best.pt',  # Your trained model
    width=640,
    height=480,
    confidence=0.25
)
```

## Dataset Details

- **26,768 images** with hand keypoint annotations
- **21 keypoints per hand**: Wrist + 4 points per finger
- **Train/Val split**: 18,776 / 7,992 images
- **Format**: YOLO keypoint format (ready to use)

## Keypoint Structure

```
0: wrist
1-4: thumb (cmc, mcp, ip, tip)
5-8: index (mcp, pip, dip, tip)
9-12: middle (mcp, pip, dip, tip)
13-16: ring (mcp, pip, dip, tip)
17-20: pinky (mcp, pip, dip, tip)
```

## Training Options

```bash
# Faster training (fewer epochs)
yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=50 imgsz=640

# Higher accuracy (larger model)
yolo pose train data=hand-keypoints.yaml model=yolo11s-pose.pt epochs=100 imgsz=640

# Custom image size
yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=512
```

## Fallback Mode

If no hand keypoints model is available, `hand_gesture_controller.py` automatically falls back to using pose model (17 body keypoints). This works but is less accurate for hand gestures.

