#!/usr/bin/env python3
"""
YOLO Oriented Bounding Boxes (OBB) Detection Test
Tests YOLO11-OBB models for object detection with rotation
Useful for detecting trash at various angles (in hands, on floor, etc.)

Reference: https://docs.ultralytics.com/tasks/obb/
"""

import cv2
import numpy as np
import time
import argparse
import sys
from picamera2 import Picamera2
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

import config
from logger import setup_logger

logger = setup_logger(__name__)


class YOLOOBBDetector:
    """
    YOLO Oriented Bounding Boxes detector
    Detects objects with rotation-aware bounding boxes
    Better for objects at various angles (trash, containers, etc.)
    """
    
    def __init__(self, 
                 model_path='yolo11n-obb.pt',  # OBB model (nano for speed)
                 width=640, 
                 height=480, 
                 confidence=0.25,
                 device='cpu'):
        """
        Initialize YOLO OBB detector
        
        Args:
            model_path: Path to YOLO OBB model (yolo11n-obb.pt, yolo11s-obb.pt, etc.)
            width: Camera width
            height: Camera height
            confidence: Detection confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.width = width
        self.height = height
        self.confidence = confidence
        self.frame_center_x = width // 2
        
        # Initialize YOLO OBB model
        logger.info(f"Loading YOLO OBB model: {model_path}...")
        try:
            # For OBB models, specify task='obb' during initialization
            self.model = YOLO(model_path, task='obb')
            logger.info(f"Model loaded: {model_path}")
            logger.info(f"Model task: {self.model.task}")
            logger.info(f"Model classes: {len(self.model.names)} classes")
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}, trying default...")
            # Try to download if not found
            try:
                self.model = YOLO('yolo11n-obb.pt', task='obb')  # Will auto-download
                logger.info("Default OBB model loaded")
                logger.info(f"Model task: {self.model.task}")
            except Exception as e2:
                logger.error(f"Failed to load default OBB model: {e2}")
                raise RuntimeError(f"Could not load OBB model: {e2}")
        
        # Initialize camera
        logger.info("Initializing camera...")
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(0.5)  # Let camera stabilize
            logger.info(f"Camera started: {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
        
        # Stats
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.detection_count = 0
    
    def get_frame(self):
        """
        Get current camera frame with rotation and color correction
        
        Returns:
            Frame in RGB format
        """
        array = self.picam2.capture_array()  # Returns RGB
        
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
        
        return array
    
    def detect(self, frame):
        """
        Run YOLO OBB detection
        
        Args:
            frame: RGB frame
            
        Returns:
            dict with detection results:
            {
                'objects': list of detected objects,
                'fps': float,
                'trash_detected': bool,
                'trash_locations': list of trash locations
            }
        """
        results = {
            'objects': [],
            'fps': self.fps,
            'trash_detected': False,
            'trash_locations': []
        }
        
        # Run YOLO OBB inference
        # Note: task='obb' should match model task, but can be omitted if model was initialized with task
        yolo_results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )
        
        # Process results
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            
            # Debug: Check what we got
            logger.debug(f"YOLO result type: {type(result)}")
            logger.debug(f"Has obb: {hasattr(result, 'obb')}")
            logger.debug(f"Has boxes: {hasattr(result, 'boxes')}")
            
            # Get oriented bounding boxes
            if hasattr(result, 'obb') and result.obb is not None:
                obbs = result.obb
                
                for i, obb in enumerate(obbs):
                    # Get class and confidence
                    class_id = int(obb.cls[0])
                    confidence = float(obb.conf[0])
                    class_name = self.model.names[class_id]
                    
                    # Get oriented bounding box
                    # OBB uses xywhr format internally: center (x, y), width, height, rotation
                    # Reference: https://docs.ultralytics.com/tasks/obb/
                    try:
                        # Try to get xywhr format (standard for OBB)
                        xywhr = obb.xywhr.cpu().numpy()[0]  # [cx, cy, w, h, r]
                        cx, cy, w, h, r = xywhr[0], xywhr[1], xywhr[2], xywhr[3], xywhr[4]
                        
                        # Convert normalized coordinates to pixels
                        cx_px = int(cx * self.width)
                        cy_px = int(cy * self.height)
                        w_px = int(w * self.width)
                        h_px = int(h * self.height)
                        rotation = float(r) * 180 / np.pi  # Convert radians to degrees
                        
                        # Calculate 4 corners from center, width, height, and rotation
                        # Create unrotated corners first
                        corners_unrotated = np.array([
                            [-w_px/2, -h_px/2],
                            [w_px/2, -h_px/2],
                            [w_px/2, h_px/2],
                            [-w_px/2, h_px/2]
                        ])
                        
                        # Apply rotation
                        cos_r = np.cos(r)
                        sin_r = np.sin(r)
                        rotation_matrix = np.array([
                            [cos_r, -sin_r],
                            [sin_r, cos_r]
                        ])
                        
                        corners_rotated = corners_unrotated @ rotation_matrix.T
                        corners_rotated[:, 0] += cx_px
                        corners_rotated[:, 1] += cy_px
                        
                        corners = [(int(c[0]), int(c[1])) for c in corners_rotated]
                        
                        center_x = cx_px
                        center_y = cy_px
                        
                    except (AttributeError, IndexError) as e:
                        # Fallback: try xyxyxyxy format (4 corner points)
                        try:
                            obb_data = obb.xyxyxyxy.cpu().numpy()[0]  # Get 4 corner points
                            corners = []
                            for j in range(4):
                                x = int(obb_data[j*2] * self.width)
                                y = int(obb_data[j*2 + 1] * self.height)
                                corners.append((x, y))
                            
                            xs = [c[0] for c in corners]
                            ys = [c[1] for c in corners]
                            center_x = int(np.mean(xs))
                            center_y = int(np.mean(ys))
                            rotation = 0.0  # Can't determine rotation from corners alone
                        except:
                            logger.warning(f"Could not extract OBB data: {e}")
                            continue
                    
                    # Fix: xs and ys might not be defined if we used xywhr format
                    if 'xs' not in locals():
                        xs = [c[0] for c in corners]
                        ys = [c[1] for c in corners]
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'corners': corners,  # 4 corner points
                        'center': (center_x, center_y),
                        'rotation': rotation,
                        'bbox': (min(xs), min(ys), max(xs), max(ys))  # Axis-aligned bbox
                    }
                    
                    results['objects'].append(detection)
                    
                    # Check if it's trash-related
                    trash_keywords = ['bottle', 'can', 'cup', 'paper', 'bag', 'trash', 'garbage', 
                                    'waste', 'container', 'box', 'package', 'wrapper']
                    if any(keyword in class_name.lower() for keyword in trash_keywords):
                        results['trash_detected'] = True
                        results['trash_locations'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'center': (center_x, center_y),
                            'rotation': rotation,
                            'corners': corners
                        })
                        logger.debug(f"Trash detected: {class_name} (conf: {confidence:.2f})")
            else:
                # OBB attribute doesn't exist - might be using regular detection
                logger.warning("No OBB results found. Model might not be OBB type or no detections.")
                # Try regular boxes as fallback
                if hasattr(result, 'boxes') and result.boxes is not None:
                    logger.info(f"Found {len(result.boxes)} regular detections (not OBB)")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        logger.info(f"  - {class_name}: {confidence:.2f}")
        else:
            logger.debug("No YOLO results returned")
        
        # Calculate FPS
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            dt = current_time - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)  # Exponential moving average
        self.last_frame_time = current_time
        results['fps'] = self.fps
        
        return results, yolo_results[0]
    
    def draw_detections(self, frame, results, yolo_result):
        """
        Draw OBB detections on frame
        
        Args:
            frame: RGB frame
            results: Detection results dict
            yolo_result: YOLO result object (for default overlay)
        
        Returns:
            Annotated frame in BGR format
        """
        # Use YOLO's built-in plot() method for default overlays
        # This should show all detections with bounding boxes
        try:
            annotated_frame = yolo_result.plot()  # YOLO's default overlay (returns BGR)
        except Exception as e:
            logger.warning(f"YOLO plot() failed: {e}, using frame as-is")
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add custom information
        y_offset = 30
        font_scale = 0.6
        thickness = 2
        
        # Show trash detection status
        if results['trash_detected']:
            text = f"TRASH DETECTED: {len(results['trash_locations'])} item(s)"
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            y_offset += 30
            
            # Show trash details
            for i, trash in enumerate(results['trash_locations']):
                text = f"  - {trash['class']}: {trash['confidence']:.2f} (rot: {trash['rotation']:.0f}°)"
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 25
        
        # Draw FPS
        fps_text = f'FPS: {results["fps"]:.1f}'
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detection count and status
        det_text = f"Detections: {len(results['objects'])}"
        color = (0, 255, 0) if len(results['objects']) > 0 else (0, 0, 255)  # Green if detections, red if none
        cv2.putText(annotated_frame, det_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show confidence threshold
        conf_text = f"Conf: {self.confidence:.2f}"
        cv2.putText(annotated_frame, conf_text, (10, annotated_frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # If no detections, show hint
        if len(results['objects']) == 0:
            hint_text = "No detections - try --conf 0.05 or point at objects"
            cv2.putText(annotated_frame, hint_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)  # Orange
        
        return annotated_frame
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        logger.info("YOLO OBB detector stopped")


def compare_with_regular_detection(obb_results, regular_model_path='yolo11n.pt'):
    """
    Compare OBB detection with regular YOLO detection
    This helps evaluate if OBB is more accurate
    
    Args:
        obb_results: Results from OBB detection
        regular_model_path: Path to regular YOLO model
    
    Returns:
        Comparison dict
    """
    try:
        regular_model = YOLO(regular_model_path)
        logger.info("Regular YOLO model loaded for comparison")
        return True
    except Exception as e:
        logger.warning(f"Could not load regular model for comparison: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test YOLO OBB detection')
    parser.add_argument('--model', type=str, default='yolo11n-obb.pt', 
                       help='YOLO OBB model (yolo11n-obb.pt, yolo11s-obb.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.1, 
                       help='Confidence threshold (default: 0.1 for better detection)')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare with regular YOLO detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("YOLO Oriented Bounding Boxes (OBB) Detection Test")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  - Oriented bounding boxes (rotation-aware)")
    logger.info("  - Better detection of objects at various angles")
    logger.info("  - Trash detection (bottles, cans, containers, etc.)")
    logger.info("  - Useful for detecting trash in hands or on floor")
    logger.info("")
    logger.info("Controls:")
    logger.info("  - Press 'q' to quit")
    logger.info("  - Press 's' to save current frame")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Initialize detector
        detector = YOLOOBBDetector(
            model_path=args.model,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=args.conf
        )
        
        logger.info("YOLO OBB detector initialized")
        logger.info("Starting camera feed...")
        logger.info("")
        
        # Start OpenCV window
        cv2.startWindowThread()
        
        frame_count = 0
        trash_detection_count = 0
        
        while True:
            # Get frame
            frame = detector.get_frame()
            
            # Run detection
            results, yolo_result = detector.detect(frame)
            
            # Draw detections
            frame_bgr = detector.draw_detections(frame, results, yolo_result)
            
            # Display frame
            cv2.imshow('YOLO OBB Detection - Press q to quit', frame_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # 's' to save
                filename = f"obb_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_bgr)
                logger.info(f"Frame saved: {filename}")
            
            # Print to terminal periodically
            frame_count += 1
            if results['trash_detected']:
                trash_detection_count += 1
            
            if frame_count % 30 == 0:  # Print every 30 frames
                output = f"[Frame {frame_count}] Detections: {len(results['objects'])}"
                if len(results['objects']) > 0:
                    # Show all detected objects
                    for obj in results['objects'][:3]:  # Show first 3
                        output += f" | {obj['class_name']} ({obj['confidence']:.2f})"
                if results['trash_detected']:
                    output += f" | TRASH: {len(results['trash_locations'])} item(s)"
                    for trash in results['trash_locations']:
                        output += f" ({trash['class']}: {trash['confidence']:.2f})"
                output += f" | FPS: {results['fps']:.1f}"
                logger.info(output)
            
            # Debug output every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"[DEBUG] Total objects detected: {len(results['objects'])}")
                if len(results['objects']) == 0:
                    logger.info("[DEBUG] No detections - try lowering confidence threshold with --conf 0.1")
        
        logger.info(f"\nTotal frames: {frame_count}")
        logger.info(f"Frames with trash: {trash_detection_count}")
        logger.info(f"Trash detection rate: {trash_detection_count/frame_count*100:.1f}%")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.stop()
        cv2.destroyAllWindows()
        logger.info("Test complete")


if __name__ == '__main__':
    main()

