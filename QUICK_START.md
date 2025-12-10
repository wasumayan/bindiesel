# Quick Start Guide

## Setup (Raspberry Pi)

### 1. Install Python 3.12 (if needed)

See `PYTHON_3.12_SETUP.md` for detailed instructions.

**Quick method (Miniforge/Conda):**
```bash
cd /tmp
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
source ~/.bashrc
conda create -n bindiesel python=3.12 -y
conda activate bindiesel
```

### 2. Install Dependencies

```bash
cd ~/angleestimationbindiesel  # or your project path
pip install ultralytics opencv-contrib-python numpy
sudo apt install -y python3-picamera2
```

### 3. Test YOLO Pose Tracking

```bash
python test_yolo_pose_tracking.py --fps
```

### 4. Test Hand Gestures

```bash
python hand_gesture_controller.py
```

**Note**: For best hand gesture accuracy, train a model on hand-keypoints dataset:
```bash
yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
```

## Running Main System

```bash
python main.py
```

## Key Files

- `test_yolo_pose_tracking.py` - YOLO pose detection + tracking test
- `hand_gesture_controller.py` - Hand gesture control for manual mode
- `main.py` - Main system controller
- `config.py` - Configuration settings

## Documentation

- `PYTHON_3.12_SETUP.md` - Python 3.12 installation guide
- `YOLO_IMPROVEMENTS_REPORT.md` - Detailed YOLO capabilities and improvements
- `ARCHITECTURE.md` - System architecture overview
