#!/usr/bin/env python3
"""
Real-time object detection using TensorFlow Lite on Raspberry Pi
Based on: https://github.com/BashMocha/object-detection-on-raspberry-pi
Uses EfficientDet-Lite0 model for object detection with picamera2
"""

import time
import sys
import os
import cv2
import numpy as np
from picamera2 import Picamera2

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow Lite support
try:
    from tflite_support.task import core
    from tflite_support.task import processor
    from tflite_support.task import vision
except ImportError:
    print("ERROR: tflite-support not installed!")
    print("Install with: pip3 install --break-system-packages tflite-support")
    print("Note: tflite-support requires Python 3.7-3.9")
    sys.exit(1)


# Configuration
# Camera Module 3 Wide: 102° horizontal FOV, supports up to 2304x1296
THREAD_NUM = 4
DISPLAY_WIDTH = 1280  # Good balance of quality and performance
DISPLAY_HEIGHT = 720
DEFAULT_MODEL = 'efficientdet_lite0.tflite'
CORAL_MODEL = 'efficientdet_lite0_edgetpu.tflite'
FPS_POS = (20, 60)
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_HEIGHT = 1.5
FPS_WEIGHT = 3
FPS_COLOR = (255, 0, 0)
FPS_AVG_FRAME_COUNT = 10

# Visualization settings
_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def download_model(model_name, model_url):
    """
    Download TFLite model if it doesn't exist
    
    Args:
        model_name: Name of the model file
        model_url: URL to download the model from
        
    Returns:
        Path to model file
    """
    if os.path.exists(model_name):
        print(f"[DEBUG] Model {model_name} already exists")
        return model_name
    
    print(f"[DEBUG] Downloading model {model_name}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_name)
        print(f"[DEBUG] Model downloaded successfully")
        return model_name
    except Exception as e:
        print(f"ERROR: Could not download model: {e}")
        return None


def visualize(image, detection_result):
    """
    Draws bounding boxes on the input image and return it.
    
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
      
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
        
        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    
    return image


def detect(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, num_threads=THREAD_NUM, 
          enable_edgetpu=False, score_threshold=0.3, max_results=4):
    """
    Continuously run inference on images acquired from the camera.
    
    Args:
        width: the width of the frame captured from the camera.
        height: the height of the frame captured from the camera.
        num_threads: the number of CPU threads to run the model.
        enable_edgetpu: True/False whether the model is a EdgeTPU model.
        score_threshold: Minimum confidence score for detections (0.0-1.0).
        max_results: Maximum number of detections to return.
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
    
    # Determine model to use
    if enable_edgetpu:
        model_name = CORAL_MODEL
        model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/efficientdet_lite0_edgetpu_metadata.tflite'
    else:
        model_name = DEFAULT_MODEL
        model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
    
    # Download model if needed
    model_path = download_model(model_name, model_url)
    if model_path is None:
        print(f"ERROR: Could not get model file: {model_name}")
        picam2.stop()
        picam2.close()
        return
    
    # Initialize the object detection model
    print("[DEBUG] Initializing TensorFlow Lite model...")
    try:
        base_options = core.BaseOptions(
            file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(
            max_results=max_results, score_threshold=score_threshold)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)
        print("[DEBUG] Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not initialize model: {e}")
        picam2.stop()
        picam2.close()
        return
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\n" + "=" * 70)
    print("Object Detection Running")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Score Threshold: {score_threshold}")
    print(f"Max Results: {max_results}")
    print(f"Threads: {num_threads}")
    print("=" * 70)
    print("Press 'q' to quit")
    print("=" * 70)
    print()
    
    try:
        while True:
            # Capture frame from picamera2
            image = picam2.capture_array()
            
            # Flip image (optional, depends on camera orientation)
            # image = cv2.flip(image, -1)
            
            counter += 1
            
            # Convert the image from RGB to RGB (picamera2 already gives RGB)
            # But TensorImage expects RGB format
            image_RGB = image.copy()
            
            # Create a TensorImage object from the RGB image
            image_tensor = vision.TensorImage.create_from_array(image_RGB)
            
            # Run object detection estimation using the model
            detections = detector.detect(image_tensor)
            
            # Draw bounding boxes and labels on input image
            image = visualize(image, detections)
            
            # Calculate the FPS
            if counter % FPS_AVG_FRAME_COUNT == 0:
                fps_end_time = time.time()
                fps = FPS_AVG_FRAME_COUNT / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                
                # Print detection count to terminal
                num_detections = len(detections.detections)
                if num_detections > 0:
                    print(f"[Frame {counter}] Detected {num_detections} object(s) | FPS: {fps:.1f}")
                    for i, detection in enumerate(detections.detections):
                        category = detection.categories[0]
                        print(f"  [{i+1}] {category.category_name}: {category.score:.2f}")
            
            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            cv2.putText(image, fps_text,
                        FPS_POS, FPS_FONT, FPS_HEIGHT, FPS_COLOR, FPS_WEIGHT)
            
            # Convert RGB to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow('Object Detection (Press q to quit)', image_bgr)
            
            # Stop the program if the 'Q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
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
        description='Real-time object detection using TensorFlow Lite on Raspberry Pi')
    parser.add_argument('--width', type=int, default=DISPLAY_WIDTH,
                       help=f'Camera width (default: {DISPLAY_WIDTH})')
    parser.add_argument('--height', type=int, default=DISPLAY_HEIGHT,
                       help=f'Camera height (default: {DISPLAY_HEIGHT})')
    parser.add_argument('--threads', type=int, default=THREAD_NUM,
                       help=f'Number of CPU threads (default: {THREAD_NUM})')
    parser.add_argument('--edgetpu', action='store_true',
                       help='Use EdgeTPU model (requires Coral USB Accelerator)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Score threshold for detections (0.0-1.0, default: 0.3)')
    parser.add_argument('--max-results', type=int, default=4,
                       help='Maximum number of detections (default: 4)')
    
    args = parser.parse_args()
    
    detect(
        width=args.width,
        height=args.height,
        num_threads=args.threads,
        enable_edgetpu=args.edgetpu,
        score_threshold=args.threshold,
        max_results=args.max_results
    )


if __name__ == '__main__':
    main()

