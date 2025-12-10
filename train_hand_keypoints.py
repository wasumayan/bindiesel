#!/usr/bin/env python3
"""
Train YOLO hand-keypoints model
Trains a YOLO11n-pose model on the hand-keypoints dataset for gesture recognition

Reference: https://docs.ultralytics.com/datasets/pose/hand-keypoints/
"""

from ultralytics import YOLO

def train_hand_keypoints_model(output_dir="models", use_gpu=True):
    """
    Train YOLO11n-pose model on hand-keypoints dataset
    
    The dataset will be automatically downloaded (369 MB, 26,768 images)
    Training will create: models/hand_keypoints/weights/best.pt (for git)
                          and runs/pose/hand_keypoints/ (YOLO default)
    
    Args:
        output_dir: Directory to save model (default: "models" - can be committed to git)
        use_gpu: Whether to use GPU (True) or CPU (False)
    """
    import os
    import shutil
    
    print("=" * 70)
    print("YOLO Hand Keypoints Model Training")
    print("=" * 70)
    print("Dataset: hand-keypoints (26,768 images, 21 keypoints per hand)")
    print("Model: yolo11n-pose.pt (nano - fastest)")
    print("Output: models/hand_keypoints/weights/best.pt (for git)")
    print()
    print("This will:")
    print("  1. Auto-download hand-keypoints dataset (369 MB)")
    print("  2. Train for 100 epochs")
    print("  3. Save best model to models/ directory (ready for git)")
    print()
    print("=" * 70)
    print()
    
    # Load pretrained YOLO11n-pose model
    model = YOLO("yolo11n-pose.pt")
    
    # Determine device
    if use_gpu:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
            if device == 'cpu':
                print("WARNING: GPU not available, using CPU (will be slower)")
        except:
            device = 'cpu'
            print("WARNING: PyTorch not available, using CPU")
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    print()
    
    # Train the model (YOLO saves to runs/pose/hand_keypoints by default)
    results = model.train(
        data="hand-keypoints.yaml",  # Dataset YAML (auto-downloads if not found)
        epochs=100,
        imgsz=640,
        batch=16 if device != 'cpu' else 8,  # Smaller batch for CPU
        device=device,
        project="runs/pose",
        name="hand_keypoints",
        exist_ok=True
    )
    
    # Copy best model to models/ directory for git
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    models_dir = os.path.join(output_dir, "hand_keypoints", "weights")
    os.makedirs(models_dir, exist_ok=True)
    
    git_model_path = os.path.join(models_dir, "best.pt")
    shutil.copy2(best_model_path, git_model_path)
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best model saved to: {best_model_path}")
    print(f"Copied to git directory: {git_model_path}")
    print()
    print("Next steps:")
    print("  1. Update config.py:")
    print(f"     YOLO_HAND_MODEL = '{git_model_path}'")
    print("  2. Commit and push to git:")
    print("     git add models/hand_keypoints/weights/best.pt")
    print("     git add config.py")
    print("     git commit -m 'Add trained hand-keypoints model'")
    print("     git push origin main")
    print("  3. On Raspberry Pi:")
    print("     git pull origin main")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    # Check for --cpu flag
    use_gpu = '--cpu' not in sys.argv
    
    train_hand_keypoints_model(output_dir="models", use_gpu=use_gpu)

