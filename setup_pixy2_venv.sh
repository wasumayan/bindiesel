#!/bin/bash
# Setup script for Pixy2 Python library in a virtual environment

echo "=" * 70
echo "Pixy2 Virtual Environment Setup"
echo "=" * 70
echo

# Check if we're in the right directory
if [ ! -d "pixy2" ]; then
    echo "ERROR: pixy2 directory not found!"
    echo "Please run this script from the directory containing the pixy2 folder"
    echo "Or clone pixy2 first:"
    echo "  git clone https://github.com/charmedlabs/pixy2"
    exit 1
fi

# Check if python3-venv is installed
if ! command -v python3 -m venv &> /dev/null; then
    echo "Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-full
fi

# Create virtual environment
echo "Creating virtual environment 'pixy2_venv'..."
python3 -m venv pixy2_venv

# Activate virtual environment
echo "Activating virtual environment..."
source pixy2_venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Navigate to pixy2 build directory
echo "Building Pixy2 library..."
cd pixy2/scripts
./build_libpixyusb2.sh

# Install Python demos
echo "Installing Pixy2 Python library..."
cd ../build/python_demos
pip install .

# Return to original directory
cd ../../..

echo
echo "=" * 70
echo "Setup complete!"
echo "=" * 70
echo
echo "To use the virtual environment:"
echo "  source pixy2_venv/bin/activate"
echo
echo "To run test scripts:"
echo "  source pixy2_venv/bin/activate"
echo "  python3 test_pixy2_native.py"
echo
echo "To deactivate:"
echo "  deactivate"
echo

