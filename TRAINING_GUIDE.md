# Hand Keypoints Model Training Guide

## Training on MacBook

You can train the hand-keypoints model on your MacBook (much faster than on Raspberry Pi) and then push it to the repo.

### Step 1: Install Dependencies

```bash
# Make sure you have ultralytics installed
pip install ultralytics

# For GPU training (if you have Apple Silicon with Metal):
# PyTorch should already support MPS (Metal Performance Shaders)
# For Intel Macs, training will use CPU (slower but still works)
```

### Step 2: Train the Model

```bash
# Run the training script
python train_hand_keypoints.py
```

This will:
- Auto-download the hand-keypoints dataset (369 MB, 26,768 images)
- Train YOLO11n-pose for 100 epochs
- Save the model to `models/hand_keypoints/weights/best.pt` (ready for git)

**Training time estimates:**
- MacBook with M1/M2/M3 (GPU): ~30-60 minutes
- MacBook Intel (CPU): ~2-4 hours
- Raspberry Pi (CPU): ~8-12 hours (not recommended)

### Step 3: Update Config

After training, update `config.py`:

```python
YOLO_HAND_MODEL = 'models/hand_keypoints/weights/best.pt'
```

### Step 4: Commit and Push

```bash
# Add the trained model
git add models/hand_keypoints/weights/best.pt
git add config.py

# Commit
git commit -m "Add trained hand-keypoints model"

# Push to repo
git push origin main
```

### Step 5: Pull on Raspberry Pi

On your Raspberry Pi:

```bash
cd ~/Desktop/bindiesel  # or wherever your repo is
git pull origin main
```

The model will be automatically used by the hand gesture controller!

## Model File Size

The trained model (`best.pt`) is typically around **5-10 MB**, which is small enough to commit to git.

## Alternative: Use Git LFS (for larger models)

If the model file becomes too large, you can use Git LFS:

```bash
# Install git-lfs (if not already installed)
git lfs install

# Track .pt files with LFS
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add models/hand_keypoints/weights/best.pt
git commit -m "Add trained hand-keypoints model (LFS)"
git push origin main
```

## Training Options

You can customize training by editing `train_hand_keypoints.py`:

```python
# For faster training (fewer epochs, less accuracy):
epochs=50

# For better accuracy (more epochs):
epochs=200

# For larger model (more accurate, slower):
model = YOLO("yolo11s-pose.pt")  # or yolo11m-pose.pt, yolo11l-pose.pt

# For CPU training:
python train_hand_keypoints.py --cpu
```

## Verification

After pulling on the Pi, verify the model works:

```bash
python hand_gesture_controller.py
```

You should see:
```
[HandGestureController] Loading hand keypoints model: models/hand_keypoints/weights/best.pt...
[HandGestureController] Hand keypoints model loaded
```

If you see "Falling back to pose model", the model file wasn't found - check the path in `config.py`.

