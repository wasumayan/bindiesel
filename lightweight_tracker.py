#!/usr/bin/env python3
"""
Lightweight Object Detection + ByteTrack Tracker
Fast person tracking using YOLO object detection (no pose) + ByteTrack
Much faster than pose detection - use after user is identified
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

import config
from logger import setup_logger, log_error, log_info

logger = setup_logger(__name__)


class LightweightTracker:
    """
    Fast person tracking using lightweight YOLO object detection + ByteTrack
    Much faster than pose detection - use after user is identified
    Uses YOLO's built-in ByteTrack for consistency with pose tracker
    """
    
    def __init__(self, 
                 model_path=None,
                 width=640, 
                 height=480, 
                 confidence=0.3,
                 device='cpu'):
        """
        Initialize lightweight tracker
        
        Args:
            model_path: Path to YOLO object detection model (not pose model)
                       Default: uses config.YOLO_MODEL
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.frame_center_x = width // 2
        self.device = device
        
        # Use regular YOLO object detection model (not pose - much faster)
        if model_path is None:
            model_path = config.YOLO_MODEL
        
        logger.info(f"[LightweightTracker] Loading YOLO object detection model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            logger.info(f"[LightweightTracker] Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"[LightweightTracker] Failed to load {model_path}: {e}")
            # Try fallback
            if config.USE_NCNN and model_path.endswith('_ncnn_model'):
                fallback_path = model_path.replace('_ncnn_model', '.pt')
                logger.info(f"[LightweightTracker] Trying PyTorch fallback: {fallback_path}...")
                try:
                    self.model = YOLO(fallback_path)
                    logger.info(f"[LightweightTracker] PyTorch model loaded")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load YOLO model: {e2}")
            else:
                raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        
        logger.info("[LightweightTracker] Initialized (lightweight mode - no pose detection)")
    
    def update(self, frame, target_track_id=None):
        """
        Update tracker with new frame and return tracking results
        Uses YOLO's built-in ByteTrack for consistency
        
        Args:
            frame: RGB frame from camera
            target_track_id: Specific track_id to track (if None, track all)
            
        Returns:
            dict with tracking results (same format as pose tracker):
            {
                'person_detected': bool,
                'person_box': (x1, y1, x2, y2) or None,
                'angle': float or None,  # Angle to steer (-45 to +45)
                'is_centered': bool,
                'track_id': int or None,
                'distance': float or None  # Estimated distance (placeholder)
            }
        """
        # Update FPS
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_frame_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time
        
        # Run YOLO object detection with tracking (lightweight - no pose)
        # Uses YOLO's built-in ByteTrack for consistency
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=[0],  # Only detect person class (class 0 in COCO)
            verbose=False,
            persist=True,  # Maintain tracking across frames
            tracker='bytetrack.yaml',  # Use same tracker as pose model
            show=False,
            imgsz=config.YOLO_INFERENCE_SIZE,
            max_det=config.YOLO_MAX_DET,
            agnostic_nms=config.YOLO_AGNOSTIC_NMS
        )
        
        if not results or len(results) == 0:
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'track_id': None,
                'distance': None
            }
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'track_id': None,
                'distance': None
            }
        
        # Find target person by track_id
        target_box = None
        target_track_id_found = None
        
        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            
            if target_track_id is not None:
                if track_id == target_track_id:
                    target_box = box
                    target_track_id_found = track_id
                    break
            else:
                # Use first person
                target_box = box
                target_track_id_found = track_id
                break
        
        if target_box is None:
            return {
                'person_detected': False,
                'person_box': None,
                'angle': None,
                'is_centered': False,
                'track_id': None,
                'distance': None
            }
        
        # Get bounding box
        x1, y1, x2, y2 = map(float, target_box.xyxy[0].cpu().numpy())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate angle (same logic as pose tracker)
        offset = center_x - self.frame_center_x
        max_offset = self.width / 2
        angle = (offset / max_offset) * 45.0
        angle = max(-45.0, min(45.0, angle))
        
        # Check if centered
        center_threshold = config.PERSON_CENTER_THRESHOLD
        is_centered = abs(offset) < center_threshold
        
        return {
            'person_detected': True,
            'person_box': (x1, y1, x2, y2),
            'angle': angle,
            'is_centered': is_centered,
            'track_id': target_track_id_found,
            'distance': None  # Could estimate from box size if needed
        }
    
    def stop(self):
        """Cleanup"""
        logger.info("[LightweightTracker] Stopped")
