"""
Helper to fix OpenCV import issues
Add this at the top of vision files if cv2 not found
"""

import sys
import os

# Add system OpenCV paths
system_paths = [
    '/usr/lib/python3/dist-packages',
    '/usr/local/lib/python3/dist-packages',
    '/usr/lib/python3.11/dist-packages',
    '/usr/lib/python3.10/dist-packages',
    '/usr/lib/python3.9/dist-packages',
]

for path in system_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Try to import cv2
try:
    import cv2
    print(f"OpenCV {cv2.__version__} loaded from: {cv2.__file__}")
except ImportError:
    print("OpenCV not found. Install with: sudo apt-get install python3-opencv")
    raise

