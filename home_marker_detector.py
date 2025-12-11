#!/usr/bin/env python3
"""
Home Marker Detection Module
Detects a red box using YOLO object detection + OpenCV color tracking
Hardcoded for red box detection
"""

import cv2
import numpy as np
from logger import setup_logger, log_error

logger = setup_logger(__name__)


def check_color_match_red(frame, bbox):
    """
    Check if object in bounding box matches red color using OpenCV
    
    Args:
        frame: BGR frame (OpenCV format)
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        float: Percentage of pixels matching red color (0.0-1.0)
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Extract object region
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red color ranges in HSV (red wraps around 0/180)
    # Balanced ranges: catch shadowed reds but reject false positives
    # Lower red range (0-15) - moderate saturation/value to filter shadows
    lower_red1 = np.array([0, 70, 60])  # Balanced thresholds for shadowed reds
    upper_red1 = np.array([15, 255, 255])
    # Upper red range (165-180)
    lower_red2 = np.array([165, 70, 60])  # Balanced thresholds for shadowed reds
    upper_red2 = np.array([180, 255, 255])
    
    # Blue color range in HSV (for exclusion - hue around 100-130)
    # We'll use this to explicitly exclude blue objects
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for red color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # Create mask for blue color (to exclude it)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Red mask minus blue mask (exclude blue pixels from red detection)
    mask = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_blue))
    
    # Calculate percentage of pixels matching red color
    total_pixels = mask.size
    matching_pixels = np.count_nonzero(mask)
    color_match_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return color_match_ratio


def detect_red_box(yolo_model, frame, confidence_threshold=0.3, color_threshold=0.25, square_aspect_ratio_tolerance=0.55):
    """
    Detect red square object using YOLO object detection + OpenCV color tracking
    Uses YOLO to get bounding boxes (ignores labels), checks for red color and square dimensions
    
    Args:
        yolo_model: YOLO model instance
        frame: BGR frame from camera (OpenCV format) - will be converted to RGB for YOLO
        confidence_threshold: Minimum YOLO confidence (default: 0.3)
        color_threshold: Minimum color match ratio (default: 0.25 = 25%)
        square_aspect_ratio_tolerance: Tolerance for square shape (default: 0.55 = 55%)
                                      Aspect ratio must be between (1.0 - tolerance) and (1.0 + tolerance)
                                      e.g., 0.55 means aspect ratio between 0.45 and 1.55
        
    Returns:
        dict with marker info: {
            'detected': bool,
            'center_x': int,
            'center_y': int,
            'width': int,
            'height': int,
            'area': int,
            'confidence': float,
            'color_match': float,
            'aspect_ratio': float
        }
    """
    if yolo_model is None:
        return {
            'detected': False,
            'center_x': None,
            'center_y': None,
            'width': None,
            'height': None,
            'area': None,
            'confidence': None,
            'color_match': None,
            'aspect_ratio': None
        }
    
    try:
        # Convert BGR to RGB for YOLO (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO object detection (we use it only for bounding boxes, not labels)
        results = yolo_model(
            frame_rgb,
            conf=confidence_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return {
                'detected': False,
                'center_x': None,
                'center_y': None,
                'width': None,
                'height': None,
                'area': None,
                'confidence': None,
                'color_match': None,
                'aspect_ratio': None
            }
        
        result = results[0]
        
        # Check if we have any detections
        if result.boxes is None or len(result.boxes) == 0:
            return {
                'detected': False,
                'center_x': None,
                'center_y': None,
                'width': None,
                'height': None,
                'area': None,
                'confidence': None,
                'color_match': None,
                'aspect_ratio': None
            }
        
        # Look for any object that:
        # 1. Has roughly square dimensions (aspect ratio close to 1.0)
        # 2. Contains red color
        # (We ignore YOLO labels since they're inaccurate)
        best_detection = None
        best_score = 0.0  # Combined score: confidence * color_match * square_score
        
        for box in result.boxes:
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            width = x2 - x1
            height = y2 - y1
            
            # Skip if bounding box is too small
            if width < 15 or height < 15:
                continue
            
            # Calculate aspect ratio (width/height)
            # For square: aspect_ratio should be close to 1.0
            aspect_ratio = width / height if height > 0 else 0.0
            
            # Check if roughly square (aspect ratio between (1.0 - tolerance) and (1.0 + tolerance))
            min_aspect = 1.0 - square_aspect_ratio_tolerance
            max_aspect = 1.0 + square_aspect_ratio_tolerance
            is_square = min_aspect <= aspect_ratio <= max_aspect
            
            if not is_square:
                continue  # Skip non-square objects
            
            # Check color match using OpenCV (hardcoded for red)
            color_match_ratio = check_color_match_red(frame, (x1, y1, x2, y2))
            
            # Object must match red color threshold
            if color_match_ratio >= color_threshold:
                # Calculate square score (how close to perfect square, 1.0 = perfect square)
                square_score = 1.0 - abs(1.0 - aspect_ratio) / square_aspect_ratio_tolerance
                square_score = max(0.0, min(1.0, square_score))  # Clamp between 0 and 1
                
                # Combined score: confidence * color_match * square_score
                combined_score = confidence * color_match_ratio * square_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    area = width * height
                    
                    best_detection = {
                        'detected': True,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'area': area,
                        'confidence': confidence,
                        'color_match': color_match_ratio,
                        'aspect_ratio': aspect_ratio
                    }
        
        if best_detection:
            return best_detection
        else:
            return {
                'detected': False,
                'center_x': None,
                'center_y': None,
                'width': None,
                'height': None,
                'area': None,
                'confidence': None,
                'color_match': None,
                'aspect_ratio': None
            }
                
    except Exception as e:
        log_error(logger, e, "Error in red box detection")
        return {
            'detected': False,
            'center_x': None,
            'center_y': None,
            'width': None,
            'height': None,
            'area': None,
            'confidence': None,
            'color_match': None,
            'aspect_ratio': None
        }


def draw_overlay(frame, marker, yolo_result=None):
    """
    Draw detection overlay on frame showing red square detection
    
    Args:
        frame: BGR frame (OpenCV format)
        marker: Detection result dict from detect_red_box()
        yolo_result: Optional YOLO result object for default overlay
        
    Returns:
        Annotated frame in BGR format
    """
    # Frame is already in BGR format
    annotated_frame = frame.copy()
    
    # Use YOLO's built-in plot() if available for default overlays
    if yolo_result is not None:
        try:
            annotated_frame = yolo_result.plot()  # YOLO's default overlay (returns BGR)
        except Exception as e:
            logger.warning(f"YOLO plot() failed: {e}, using frame as-is")
            # Frame is already BGR, no conversion needed
    
    # Draw red box detection overlay
    if marker['detected']:
        # Draw bounding box
        x1 = marker['center_x'] - marker['width'] // 2
        y1 = marker['center_y'] - marker['height'] // 2
        x2 = marker['center_x'] + marker['width'] // 2
        y2 = marker['center_y'] + marker['height'] // 2
        
        # Green box for detected red square
        color = (0, 255, 0)  # Green in BGR
        thickness = 3
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        cv2.circle(annotated_frame, (marker['center_x'], marker['center_y']), 5, color, -1)
        
        # Draw label with detection info
        label = "RED SQUARE DETECTED"
        conf_text = f"Conf: {marker['confidence']:.2f}"
        color_text = f"Color Match: {marker['color_match']:.1%}"
        aspect_text = f"Aspect Ratio: {marker.get('aspect_ratio', 0):.2f}"
        
        # Background for text (for better visibility)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness_text = 2
        
        # Calculate text size
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness_text)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness_text)
        (color_w, color_h), _ = cv2.getTextSize(color_text, font, font_scale, thickness_text)
        (aspect_w, aspect_h), _ = cv2.getTextSize(aspect_text, font, font_scale, thickness_text)
        
        # Draw text background
        text_x = x1
        text_y = y1 - 10
        if text_y < 40:
            text_y = y2 + 40
        
        cv2.rectangle(annotated_frame, 
                     (text_x - 5, text_y - label_h - 5),
                     (text_x + max(label_w, conf_w, color_w, aspect_w) + 5, text_y + conf_h + color_h + aspect_h + 15),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (text_x, text_y),
                   font, font_scale, color, thickness_text)
        cv2.putText(annotated_frame, conf_text, (text_x, text_y + conf_h + 5),
                   font, font_scale, color, thickness_text)
        cv2.putText(annotated_frame, color_text, (text_x, text_y + conf_h + color_h + 10),
                   font, font_scale, color, thickness_text)
        cv2.putText(annotated_frame, aspect_text, (text_x, text_y + conf_h + color_h + aspect_h + 15),
                   font, font_scale, color, thickness_text)
        
        # Draw status at top
        status_text = "RED SQUARE DETECTED!"
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # No detection
        status_text = "No red square detected"
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text
    
    return annotated_frame


def main():
    """Test home marker detector with camera feed and visualization"""
    import sys
    import time
    import argparse
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        import config
        from picamera2 import Picamera2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: Missing required module: {e}")
        print("Install with: pip install ultralytics picamera2")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Test red square detection')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='YOLO confidence threshold (default: 0.3)')
    parser.add_argument('--color-threshold', type=float, default=0.3,
                       help='Color match threshold (default: 0.3 = 30%%)')
    parser.add_argument('--square-tolerance', type=float, default=0.3,
                       help='Square aspect ratio tolerance (default: 0.3 = 30%%)')
    parser.add_argument('--fps', action='store_true',
                       help='Show FPS counter')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Red Square Detection Test")
    print("=" * 70)
    print(f"Confidence threshold: {args.confidence}")
    print(f"Color match threshold: {args.color_threshold:.1%}")
    print(f"Square tolerance: {args.square_tolerance:.1%} (aspect ratio: {1.0 - args.square_tolerance:.2f} - {1.0 + args.square_tolerance:.2f})")
    print("Press 'q' to quit")
    print("=" * 70)
    print()
    
    # Initialize YOLO model - use regular YOLO model (not OBB)
    print("[TEST] Initializing YOLO model...")
    try:
        yolo_model = YOLO(config.YOLO_MODEL)
        print(f"[TEST] YOLO model loaded: {config.YOLO_MODEL}")
    except Exception as e1:
        # Fallback: if NCNN failed, try PyTorch
        if config.USE_NCNN and config.YOLO_MODEL.endswith('_ncnn_model'):
            fallback_path = config.YOLO_MODEL.replace('_ncnn_model', '.pt')
            print(f"[TEST] NCNN model not found, trying PyTorch: {fallback_path}")
            yolo_model = YOLO(fallback_path)
            print(f"[TEST] PyTorch model loaded: {fallback_path}")
        else:
            raise RuntimeError(f"Failed to load YOLO model: {e1}") from e1
    
    # Initialize camera
    print("[TEST] Initializing camera...")
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": config.CAMERA_FPS}
    )
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1.5)  # Wait for camera to initialize
    print(f"[TEST] Camera started: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print()
    
    # FPS tracking
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frame
            frame = picam2.capture_array(wait=True)  # Returns RGB
            
            # Apply camera rotation if configured
            if config.CAMERA_ROTATION == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif config.CAMERA_ROTATION == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif config.CAMERA_ROTATION == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Apply flips if configured
            if config.CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            if config.CAMERA_FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)
            
            # Fix color channel swap if needed
            if config.CAMERA_SWAP_RB:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection to get result object for overlay
            yolo_results = yolo_model(
                frame,
                conf=args.confidence,
                verbose=False
            )
            yolo_result = yolo_results[0] if yolo_results else None
            
            # Convert RGB to BGR for color detection (detect_red_box expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect red square
            marker = detect_red_box(
                yolo_model,
                frame_bgr,
                confidence_threshold=args.confidence,
                color_threshold=args.color_threshold,
                square_aspect_ratio_tolerance=args.square_tolerance
            )
            
            # Draw overlay (frame_bgr is already BGR)
            annotated_frame = draw_overlay(frame_bgr, marker, yolo_result)
            
            # Calculate FPS
            frame_count += 1
            if args.fps:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    fps_text = f'FPS: {fps:.1f}'
                    cv2.putText(annotated_frame, fps_text, 
                               (annotated_frame.shape[1] - 150, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Red Box Detection Test', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to prevent CPU spinning
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("[TEST] Cleaning up...")
        picam2.stop()
        cv2.destroyAllWindows()
        print("[TEST] Test complete")


if __name__ == '__main__':
    main()

