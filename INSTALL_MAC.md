# Mac Installation Commands

## Quick Install (Copy-Paste)

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch first (for Apple Silicon support)
pip install torch torchvision

# 5. Install all other requirements
pip install -r requirements.txt
```

## Alternative: Install Everything at Once

```bash
# Create and activate venv
python3 -m venv venv && source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch then requirements
pip install torch torchvision && pip install -r requirements.txt
```

## If You Get Errors

### For Apple Silicon (M1/M2/M3) Macs:
PyTorch should auto-detect and use Metal (MPS). If you get errors, try:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### For Intel Macs:
Standard installation should work. If you get errors with PyAudio:

```bash
# Install PortAudio first (via Homebrew)
brew install portaudio

# Then install PyAudio
pip install PyAudio
```

### If ultralytics installation fails:
```bash
pip install ultralytics --no-deps
pip install numpy opencv-python torch torchvision
```

## Verify Installation

```bash
python3 -c "import torch; import ultralytics; print('✓ All installed!')"
```

