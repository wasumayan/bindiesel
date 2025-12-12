#!/usr/bin/env python3
"""
Pixy Camera Interface
Provides a compatible interface for Pixy2 camera to work with YOLOPoseTracker
"""

import cv2
import numpy as np
import time
import sys

try:
    from pixy import *
    from ctypes import *
    HAS_PIXY = True
except ImportError:
    HAS_PIXY = False
    print("WARNING: Pixy library not found. Install with:")
    print("  git clone https://github.com/charmedlabs/pixy2")
    print("  cd pixy2/scripts")
    print("  ./build_libpixyusb2.sh")
    print("  pip3 install ./build/python_demos")

import config


class PixyCamera:
    """
    Pixy2 camera interface that provides frame capture compatible with YOLOPoseTracker
    Note: Pixy2 is primarily for color-based object tracking, but we can use it
    to capture full frames if the firmware supports it, or use OpenCV VideoCapture
    as a fallback if Pixy2 is connected via USB.
    """
    
    def __init__(self, width=640, height=480):
        """
        Initialize Pixy camera
        
        Args:
            width: Desired frame width
            height: Desired frame height
        """
        self.width = width
        self.height = height
        self.pixy_initialized = False
        self.use_opencv_fallback = False
        self.cap = None
        
        # Try to initialize Pixy2
        if HAS_PIXY:
            try:
                status = pixy_init()
                if status == 0:
                    self.pixy_initialized = True
                    print("[PixyCamera] Pixy2 initialized successfully")
                else:
                    print(f"[PixyCamera] Warning: pixy_init() returned {status}, using OpenCV fallback")
                    self.use_opencv_fallback = True
            except Exception as e:
                print(f"[PixyCamera] Warning: Failed to initialize Pixy2: {e}, using OpenCV fallback")
                self.use_opencv_fallback = True
        else:
            print("[PixyCamera] Pixy library not available, using OpenCV fallback")
            self.use_opencv_fallback = True
        
        # Fallback to OpenCV VideoCapture (Pixy2 connected via USB appears as /dev/videoX)
        if self.use_opencv_fallback:
            # Try to find Pixy camera device (usually /dev/video0 or /dev/video1)
            for device_idx in range(4):  # Try first 4 video devices
                try:
                    self.cap = cv2.VideoCapture(device_idx)
                    if self.cap.isOpened():
                        # Test if we can read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            # Set resolution
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                            print(f"[PixyCamera] Using OpenCV VideoCapture (device {device_idx})")
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    else:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                except Exception as e:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError("Failed to initialize Pixy camera. Check USB connection.")
    
    def get_frame(self):
        """
        Get current camera frame
        
        Returns:
            Frame in RGB format (compatible with YOLOPoseTracker)
        """
        if self.use_opencv_fallback and self.cap is not None:
            # Use OpenCV VideoCapture
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                raise RuntimeError("Failed to read frame from Pixy camera")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply camera transformations
            if config.CAMERA_ROTATION == 180:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            elif config.CAMERA_ROTATION == 90:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
            elif config.CAMERA_ROTATION == 270:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            if config.CAMERA_FLIP_HORIZONTAL:
                frame_rgb = cv2.flip(frame_rgb, 1)
            if config.CAMERA_FLIP_VERTICAL:
                frame_rgb = cv2.flip(frame_rgb, 0)
            
            # Handle color swap
            if config.CAMERA_SWAP_RB:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            
            return frame_rgb
        elif self.pixy_initialized:
            # Use native Pixy2 API (if it supports full frame capture)
            # Note: Standard Pixy2 firmware doesn't support full frame capture
            # This would require custom firmware or using VideoCapture fallback
            raise NotImplementedError("Pixy2 native full-frame capture not yet implemented. Use USB VideoCapture mode.")
        else:
            raise RuntimeError("Pixy camera not initialized")
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.pixy_initialized and HAS_PIXY:
            try:
                pixy_close()
            except:
                pass
            self.pixy_initialized = False
        
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        print("[PixyCamera] Stopped")
    
    def close(self):
        """Alias for stop()"""
        self.stop()

