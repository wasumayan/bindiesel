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
    
    # Determine device (prioritize: CUDA > MPS > CPU)
    if use_gpu:
        try:
            import torch
            # Check for CUDA (NVIDIA GPU) first
            if torch.cuda.is_available():
                device = 0  # CUDA device
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            # Check for MPS (Apple Silicon) second
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
                device_name = "MPS (Apple Silicon)"
            else:
                device = 'cpu'
                device_name = "CPU"
                print("WARNING: No GPU available, using CPU (will be MUCH slower)")
                print("         Consider reducing epochs or using a pre-trained model")
        except:
            device = 'cpu'
            device_name = "CPU"
            print("WARNING: PyTorch not available, using CPU")
    else:
        device = 'cpu'
        device_name = "CPU"
    
    print(f"Using device: {device_name} ({device})")
    
    # Optimize training parameters based on device
    if device == 'cpu':
        # CPU training is very slow - reduce epochs significantly
        recommended_epochs = 20  # Much faster, still gets good results
        batch_size = 8
        print()
        print("⚠️  CPU TRAINING DETECTED - Using optimized settings:")
        print(f"   Epochs: {recommended_epochs} (instead of 100)")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: ~{recommended_epochs * 78 / 60:.1f} hours")
        print()
        print("   Options:")
        print("   1. Continue with reduced epochs (recommended)")
        print("   2. Stop and use pre-trained model if available")
        print("   3. Train on a machine with GPU")
        print()
    elif device == 'mps':
        # Apple Silicon MPS - faster than CPU, but may need smaller batch
        recommended_epochs = 100  # Full training on MPS
        batch_size = 16  # Can try larger batches on MPS
        print()
        print("✅ Apple Silicon MPS detected - Using GPU acceleration!")
        print(f"   Epochs: {recommended_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: ~2-4 hours (much faster than CPU!)")
        print()
    else:
        # CUDA GPU
        recommended_epochs = 100  # Full training on GPU
        batch_size = 16
        print()
        print(f"✅ CUDA GPU detected - Using GPU acceleration!")
        print(f"   Epochs: {recommended_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: ~2-4 hours")
        print()
    
    print()
    
    # Train the model (YOLO saves to runs/pose/hand_keypoints by default)
    results = model.train(
        data="hand-keypoints.yaml",  # Dataset YAML (auto-downloads if not found)
        epochs=recommended_epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project="runs/pose",
        name="hand_keypoints",
        exist_ok=True,
        patience=10,  # Early stopping if no improvement
        workers=4 if device == 'cpu' else 8,  # Fewer workers on CPU
        cache=False,  # Don't cache images on CPU (saves memory)
        amp=(device != 'cpu' and device != 'mps')  # Mixed precision on CUDA only (MPS has issues with AMP)
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

