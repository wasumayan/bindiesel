#!/bin/bash
# Auto-start script for Bin Diesel system
# Activates conda environment and runs main.py
# Add this to systemd or .bashrc for auto-start on boot

# Navigate to project directory
cd ~/Desktop/angleestimationbindiesel
# or: cd ~/angleestimationbindiesel  # Adjust path as needed

# Initialize conda (if not already initialized)
source ~/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate bindiesel

# Run main system
python main.py

