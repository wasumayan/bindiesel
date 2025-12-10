#!/bin/bash
# Script to train YOLO hand keypoints model
# This will download the hand-keypoints dataset and train a model

echo "=========================================="
echo "Training YOLO Hand Keypoints Model"
echo "=========================================="
echo ""

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bindiesel

# Check if ultralytics is installed
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "ERROR: ultralytics not installed!"
    echo "Installing ultralytics..."
    pip install ultralytics
fi

echo "Starting training..."
echo "This will:"
echo "  1. Auto-download hand-keypoints dataset (369 MB, 26,768 images)"
echo "  2. Train YOLO11n-pose model for 100 epochs"
echo "  3. Save best model to runs/pose/train/weights/best.pt"
echo ""
echo "Training will take several hours depending on your hardware."
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Train the model
# The dataset will be auto-downloaded if not present
yolo pose train \
    data=hand-keypoints.yaml \
    model=yolo11n-pose.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=cpu \
    project=runs/pose \
    name=hand-keypoints-train

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: runs/pose/hand-keypoints-train/weights/best.pt"
echo ""
echo "To use the trained model:"
echo "  controller = HandGestureController("
echo "      hand_model_path='runs/pose/hand-keypoints-train/weights/best.pt'"
echo "  )"
echo ""

