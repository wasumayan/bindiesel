# Hand Keypoints Model Information

## Pre-trained Models

**Unfortunately, Ultralytics does NOT provide pre-trained hand keypoints models.**

The hand-keypoints dataset is available for training, but you must train the model yourself. This is because:
- Hand keypoints is a specialized task
- Models need to be trained on the specific dataset
- No official pre-trained weights are released

## Solution: Train Your Own Model

### Quick Training (Automated)

```bash
# Make script executable
chmod +x train_hand_keypoints.sh

# Run training script
bash train_hand_keypoints.sh
```

This will:
1. Auto-download hand-keypoints dataset (369 MB)
2. Train YOLO11n-pose model for 100 epochs
3. Save model to `runs/pose/hand-keypoints-train/weights/best.pt`

### Manual Training

```bash
# Activate conda environment
conda activate bindiesel

# Train model
yolo pose train \
    data=hand-keypoints.yaml \
    model=yolo11n-pose.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=cpu
```

### Training Time Estimates

- **Raspberry Pi 4**: ~12-24 hours
- **Raspberry Pi 5**: ~6-12 hours
- **Desktop CPU**: ~2-4 hours
- **GPU**: ~30-60 minutes

### Check for Existing Models

```bash
python check_hand_model.py
```

This will search for any existing trained models.

## Using Trained Model

Once trained, use it in your code:

```python
from hand_gesture_controller import HandGestureController

controller = HandGestureController(
    hand_model_path='runs/pose/hand-keypoints-train/weights/best.pt',
    width=640,
    height=480,
    confidence=0.25
)
```

## Fallback Mode

If no hand model is available, `HandGestureController` automatically falls back to using the pose model (17 body keypoints). This works but is less accurate for hand gestures.

## Alternative: Use Pose Model

For now, you can use the pose model which works reasonably well:

```python
controller = HandGestureController(
    hand_model_path=None,  # Will use pose model
    pose_model_path='yolo11n-pose.pt'
)
```

The pose model can detect basic gestures from wrist positions relative to shoulders, though it's not as precise as a dedicated hand keypoints model.

