#!/usr/bin/env python3
"""
Real-time object detection using YOLO (Ultralytics) on Raspberry Pi
Based on: https://github.com/automaticdai/rpi-object-detection
Uses YOLO11n model for object detection with picamera2
"""

import cv2
import time
import sys
from picamera2 import Picamera2

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Try to import Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip3 install --break-system-packages ultralytics")
    sys.exit(1)

# Import config for camera settings
try:
    import config
except ImportError:
    print("WARNING: config.py not found, using default camera settings")
    # Default config values if config.py doesn't exist
    class config:
        CAMERA_ROTATION = 0
        CAMERA_FLIP_HORIZONTAL = False
        CAMERA_FLIP_VERTICAL = False
        CAMERA_SWAP_RB = False


# Configuration
# Camera Module 3 Wide: 102° horizontal FOV, supports up to 2304x1296
DISPLAY_WIDTH = 1280  # Good balance of quality and performance
DISPLAY_HEIGHT = 720
MODEL_NAME = 'yolo11n.pt'  # YOLO11 nano - fastest model
FPS_AVG_FRAME_COUNT = 10


def detect(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, model_name=MODEL_NAME, 
          conf_threshold=0.25, show_fps=True):
    """
    Continuously run YOLO inference on images acquired from the camera.
    
    Args:
        width: the width of the frame captured from the camera.
        height: the height of the frame captured from the camera.
        model_name: YOLO model to use (default: 'yolo11n.pt' for nano model)
        conf_threshold: Confidence threshold for detections (0.0-1.0, default: 0.25)
        show_fps: Whether to display FPS on screen
    """
    counter, fps = 0, 0
    fps_start_time = time.time()
    
    # Initialize picamera2 (Camera Module 3 Wide)
    print("[DEBUG] Initializing picamera2 (Camera Module 3 Wide)...")
    try:
        picam2 = Picamera2()
        # Camera Module 3 Wide supports various resolutions
        # Using main stream for object detection
        preview_config = picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        picam2.start()
        # Wait a moment for camera to stabilize
        time.sleep(0.5)
        print(f"[DEBUG] Camera Module 3 Wide started: {width}x{height}")
    except Exception as e:
        print(f"ERROR: Could not initialize camera: {e}")
        print("Make sure:")
        print("  1. Camera is enabled: sudo raspi-config → Interface Options → Camera → Enable")
        print("  2. picamera2 is installed: sudo apt install python3-picamera2")
        return
    
    # Load YOLO model (will auto-download on first run)
    print(f"[DEBUG] Loading YOLO model: {model_name}...")
    print("Note: Model will be downloaded automatically on first run (~6MB)")
    try:
        model = YOLO(model_name)
        print(f"[DEBUG] Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        picam2.stop()
        picam2.close()
        return
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\n" + "=" * 70)
    print("YOLO Object Detection Running")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Confidence Threshold: {conf_threshold}")
    print("=" * 70)
    print("Press 'q' or ESC to quit")
    print("=" * 70)
    print()
    
    try:
        while True:
            # Capture frame from picamera2 (returns RGB)
            array = picam2.capture_array()
            
            # Apply camera rotation if configured
            if config.CAMERA_ROTATION == 180:
                array = cv2.rotate(array, cv2.ROTATE_180)
            elif config.CAMERA_ROTATION == 90:
                array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
            elif config.CAMERA_ROTATION == 270:
                array = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Apply flips if configured
            if config.CAMERA_FLIP_HORIZONTAL:
                array = cv2.flip(array, 1)  # Horizontal flip
            if config.CAMERA_FLIP_VERTICAL:
                array = cv2.flip(array, 0)  # Vertical flip
            
            # Fix color channel swap (red/blue)
            if config.CAMERA_SWAP_RB:
                # Swap red and blue channels: RGB -> BGR -> RGB (swaps R and B)
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            
            counter += 1
            
            # Run YOLO inference on the frame
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Draw results on the frame (YOLO handles visualization)
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            if counter % FPS_AVG_FRAME_COUNT == 0:
                fps_end_time = time.time()
                fps = FPS_AVG_FRAME_COUNT / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                
                # Print detection count to terminal
                num_detections = len(results[0].boxes)
                if num_detections > 0:
                    print(f"[Frame {counter}] Detected {num_detections} object(s) | FPS: {fps:.1f}")
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        print(f"  - {class_name}: {confidence:.2f}")
            
            # Show FPS on screen
            if show_fps:
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('YOLO Object Detection (Press q to quit)', annotated_frame)
            
            # Stop the program if the 'Q' key or ESC is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("[DEBUG] Stopping camera...")
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        elapsed = time.time() - fps_start_time
        avg_fps = counter / elapsed if elapsed > 0 else 0
        print(f"\nTotal frames: {counter}, Average FPS: {avg_fps:.2f}")
        print("Camera closed.")


def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time object detection using YOLO (Ultralytics) on Raspberry Pi')
    parser.add_argument('--width', type=int, default=DISPLAY_WIDTH,
                       help=f'Camera width (default: {DISPLAY_WIDTH})')
    parser.add_argument('--height', type=int, default=DISPLAY_HEIGHT,
                       help=f'Camera height (default: {DISPLAY_HEIGHT})')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help=f'YOLO model to use (default: {MODEL_NAME})')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (0.0-1.0, default: 0.25)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter on screen')
    
    args = parser.parse_args()
    
    # Available YOLO models (from fastest to most accurate):
    # - yolo11n.pt (nano) - fastest, least accurate
    # - yolo11s.pt (small)
    # - yolo11m.pt (medium)
    # - yolo11l.pt (large)
    # - yolo11x.pt (extra large) - slowest, most accurate
    
    detect(
        width=args.width,
        height=args.height,
        model_name=args.model,
        conf_threshold=args.conf,
        show_fps=not args.no_fps
    )


if __name__ == '__main__':
    main()
