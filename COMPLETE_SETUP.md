# Complete Setup Guide - Step by Step Commands

Run these commands **one by one** on your Raspberry Pi after a fresh SD card install.

## Prerequisites
- Raspberry Pi OS installed
- Internet connection
- Git repository URL (you'll need this)

---

## Step 1: Update System

```bash
sudo apt update
```

```bash
sudo apt upgrade -y
```

---

## Step 2: Install System Dependencies

```bash
sudo apt install -y git python3-pip python3-dev build-essential python3-venv
```

```bash
sudo apt install -y python3-picamera2 libcamera-dev
```

```bash
sudo apt install -y portaudio19-dev python3-pyaudio
```

**Check Python version:**
```bash
python3 --version
```
*(Should be Python 3.8+ - YOLO works with 3.8, 3.9, 3.10, 3.11)*

---

## Step 3: Clone Repository

**Navigate to your desired directory (e.g., Desktop or home):**
```bash
cd ~/Desktop
```

**OR if you prefer home directory:**
```bash
cd ~
```

**Clone the repository:**
```bash
git clone https://github.com/wasumayan/bindiesel.git bindiesel
```

**Navigate into project:**
```bash
cd bindiesel
```

---

## Step 4: Create Virtual Environment

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

**Verify you're in venv (should show (venv) in prompt):**
```bash
which python
```

**Upgrade pip:**
```bash
pip install --upgrade pip
```

---

## Step 5: Install Python Dependencies

**First, upgrade pip:**
```bash
pip install --upgrade pip
```

**Install requirements:**
```bash
pip install -r requirements.txt
```

**Note:** This may take 10-15 minutes. If you get errors about numpy/opencv conflicts, run:
```bash
pip install opencv-contrib-python
```

**Install ultralytics separately (if needed):**
```bash
pip install ultralytics
```

---

## Step 6: Create .env File

**Create the .env file:**
```bash
nano .env
```

**Add these lines (replace with your actual keys):**
```
PICOVOICE_ACCESS_KEY=your_picovoice_key_here
OPENAI_API_KEY=your_openai_key_here
```

**Save and exit:**
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter` to save

**Verify .env file exists:**
```bash
cat .env
```

---

## Step 7: Download YOLO Models (Auto-download on first use)

**The models will auto-download when you first run the code, but you can pre-download them:**

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

**This will download the pose model (~6MB).**

---

## Step 8: Verify Wake Word Model

**Check if wake word model exists:**
```bash
ls -la bin-diesel_en_raspberry-pi_v3_0_0/bin-diesel_en_raspberry-pi_v3_0_0.ppn
```

**If it doesn't exist, you'll need to download it from Picovoice console.**

---

## Step 9: Test Installation

**Test Python imports:**
```bash
python -c "import cv2; import numpy; from ultralytics import YOLO; print('All imports OK!')"
```

**Test camera:**
```bash
libcamera-hello --list-cameras
```

**Test YOLO pose tracking:**
```bash
python test_yolo_pose_tracking.py --fps
```
*(Press Ctrl+C to exit)*

---

## Step 10: Set Up Auto-Activation (Optional but Recommended)

**Add venv activation to ~/.bashrc:**
```bash
nano ~/.bashrc
```

**Add these lines at the end (replace with your actual project path):**
```bash
# Auto-activate bindiesel venv
cd ~/Desktop/bindiesel  # Change to your actual project path
source venv/bin/activate
```

**Save and exit:**
- Press `Ctrl+X`
- Press `Y`
- Press `Enter`

**Reload bashrc:**
```bash
source ~/.bashrc
```

---

## Step 11: Test Main System

**Run the main system:**
```bash
python main.py
```

**You should see:**
- System initializing messages
- "Waiting for wake word: 'bin diesel'"
- Press Ctrl+C to exit

---

## Troubleshooting

### If venv not activated:
```bash
cd ~/Desktop/bindiesel  # Your project path
source venv/bin/activate
```

### If camera not working:
```bash
sudo raspi-config
```
*Navigate to: Interface Options → Camera → Enable*

### If pip install fails:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### If ultralytics not found:
```bash
pip install ultralytics --upgrade
```

### If mediapipe errors:
```bash
pip install mediapipe --upgrade
```

### Check disk space:
```bash
df -h
```

### Clean pip cache if low on space:
```bash
pip cache purge
```

---

## Quick Reference Commands

**Activate environment:**
```bash
cd ~/Desktop/bindiesel  # Your project path
source venv/bin/activate
```

**Deactivate environment:**
```bash
deactivate
```

**Run main system:**
```bash
python main.py
```

**Test pose tracking:**
```bash
python test_yolo_pose_tracking.py --fps
```

**Test hand gestures:**
```bash
python hand_gesture_controller.py
```

---

## Next Steps

1. **Test all components** using the test files
2. **Set up auto-start** (see `AUTO_START_SETUP.md`)
3. **Train hand keypoints model** (optional, see `HAND_KEYPOINTS_TRAINING.md`)

---

## Summary Checklist

- [ ] System updated
- [ ] System dependencies installed
- [ ] Python 3.8+ verified
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Python dependencies installed
- [ ] .env file created with API keys
- [ ] YOLO models downloaded
- [ ] Wake word model present
- [ ] Camera working
- [ ] Test imports successful
- [ ] Main system runs

---

**You're all set!** 🎉

