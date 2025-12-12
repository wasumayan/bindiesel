# Python Version Update Guide

## Current Versions
- **System Python**: 3.11.4
- **Conda Python**: 3.11.7
- **Latest Available**: 3.14.1

## ⚠️ Important Warnings

1. **System Python**: Updating system Python on macOS can break system tools. **NOT RECOMMENDED** unless you know what you're doing.
2. **Conda Python**: Safe to update, but test your code after updating.

## Option 1: Update Conda Python (RECOMMENDED)

### Update conda base environment:
```bash
# Update conda itself first
conda update conda

# Update Python in base environment
conda update python

# Or install specific version
conda install python=3.12
```

### Verify:
```bash
conda run python --version
```

## Option 2: Create New Conda Environment with Updated Python

**RECOMMENDED** - Keeps base environment stable:

```bash
# Create new environment with Python 3.12
conda create -n bindiesel python=3.12

# Activate it
conda activate bindiesel

# Install required packages
pip install opencv-python numpy

# Use this environment for your project
python test_apriltag_detection.py --webcam --no-control --tag-size 0.047
```

## Option 3: Update System Python (NOT RECOMMENDED)

**WARNING**: This can break macOS system tools. Only do this if you understand the risks.

```bash
# Download Python from python.org
# Or use Homebrew (if installed)
brew install python@3.12

# This will install to /usr/local/bin/python3.12
# Your system python3 will remain at 3.11.4
```

## Recommendation

**For this project**: Keep current versions (3.11.x is fine). If you need newer features, create a conda environment with Python 3.12.

**Why not update system Python?**
- macOS system tools depend on specific Python versions
- Can break system scripts
- Requires admin privileges
- Hard to rollback

