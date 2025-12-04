#!/usr/bin/env python3
"""
Robust colored flag detection using picamera2
Detects colored flags and calculates angle relative to center
Uses enhanced filtering and morphological operations for robust detection
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Suppress ALSA warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')


class ColorFlagDetector:
    """Robust colored flag detector with enhanced filtering"""
    
    def __init__(self, color='red', horizontal_fov=102.0, min_area=500):
        """
        Initialize color flag detector
        
        Args:
            color: Color to detect ('red', 'green', 'blue', 'yellow')
            horizontal_fov: Horizontal field of view in degrees (default 102 for Camera Module 3 Wide)
            min_area: Minimum contour area to consider as flag (default 500)
        """
        self.color = color.lower()
        self.horizontal_fov = horizontal_fov
        self.min_area = min_area
        
        # Enhanced HSV color ranges - more tolerant for varying lighting
        # Lower saturation/value thresholds to catch colors in different lighting
        self.color_ranges = {
            'red': [
                # Lower red range - more tolerant
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                # Upper red range - more tolerant
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'green': [
                # Wider green range for better detection
                (np.array([35, 40, 40]), np.array([85, 255, 255]))
            ],
            'blue': [
                # Wider blue range
                (np.array([95, 40, 40]), np.array([135, 255, 255]))
            ],
            'yellow': [
                # Wider yellow range
                (np.array([15, 50, 50]), np.array([35, 255, 255]))
            ]
        }
    
    def detect_flag(self, frame):
        """
        Robustly detect colored flag in frame with enhanced filtering
        
        Args:
            frame: Camera frame (BGR format from OpenCV)
            
        Returns:
            (center_x, center_y, area, bbox) of flag or None if not detected, mask
        """
        if frame is None:
            return None, None
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Get color range for selected color
        if self.color not in self.color_ranges:
            color_ranges = self.color_ranges['red']
        else:
            color_ranges = self.color_ranges[self.color]
        
        # Create mask for color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask += cv2.inRange(hsv, lower, upper)
        
        # Enhanced morphological operations for robust detection
        # Use larger kernel for better noise removal
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        # Erosion to remove small noise
        mask = cv2.erode(mask, kernel_small, iterations=1)
        
        # Dilation to fill gaps
        mask = cv2.dilate(mask, kernel_medium, iterations=2)
        
        # Opening to remove small objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium)
        
        # Closing to fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Final dilation to ensure connected regions
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, mask
        
        # Filter contours by area and other properties
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Filter by aspect ratio (flags are usually somewhat rectangular)
            # Allow wide range: 0.3 to 3.0 (very flexible)
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Filter by solidity (flags should be reasonably solid)
            if solidity < 0.3:
                continue
            
            valid_contours.append((contour, area, (x, y, w, h)))
        
        if len(valid_contours) == 0:
            return None, mask
        
        # Select the largest valid contour (assumed to be the flag)
        largest = max(valid_contours, key=lambda x: x[1])
        largest_contour, area, bbox = largest
        
        # Calculate center of flag using moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, mask
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y, area, bbox), mask
    
    def calculate_angle(self, flag_center, frame_width):
        """
        Calculate angle of flag relative to car center
        
        Args:
            flag_center: (x, y) tuple of flag center
            frame_width: Width of camera frame
            
        Returns:
            Angle in degrees (-90 to +90, negative = left, positive = right)
        """
        if flag_center is None:
            return None
        
        center_x, center_y = flag_center
        frame_center_x = frame_width / 2
        
        # Calculate horizontal offset
        offset = center_x - frame_center_x
        
        # Convert pixel offset to angle using camera's horizontal FOV
        angle = (offset / frame_width) * self.horizontal_fov
        
        return angle


def draw_detection(frame, flag_data, angle, color_name, frame_width, mask):
    """
    Draw detection overlay on frame with mask display
    
    Args:
        frame: Camera frame
        flag_data: (center_x, center_y, area, bbox) or None
        angle: Calculated angle in degrees or None
        color_name: Name of color being detected
        frame_width: Width of frame
        mask: Detection mask
    """
    h, w = frame.shape[:2]
    
    # Draw center line (car center)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.putText(frame, "CENTER", (w // 2 + 5, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if flag_data is not None:
        center_x, center_y, area, bbox = flag_data
        x, y, bw, bh = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
        
        # Draw flag center point
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        
        # Draw line from center to flag
        cv2.line(frame, (w // 2, center_y), (center_x, center_y), (0, 255, 255), 2)
        
        # Draw angle text
        if angle is not None:
            angle_text = f"Angle: {angle:.1f}°"
            direction = "LEFT" if angle < 0 else "RIGHT" if angle > 0 else "CENTER"
            cv2.putText(frame, angle_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, direction, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw position info
        pos_text = f"Pos: ({center_x}, {center_y})"
        area_text = f"Area: {int(area)}"
        cv2.putText(frame, pos_text, (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, area_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw color being detected
    color_text = f"Detecting: {color_name.upper()}"
    cv2.putText(frame, color_text, (w - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw status
    status_text = "NO FLAG DETECTED" if flag_data is None else "FLAG DETECTED"
    status_color = (0, 0, 255) if flag_data is None else (0, 255, 0)
    cv2.putText(frame, status_text, (w - 200, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust colored flag detection using picamera2')
    parser.add_argument('--color', type=str, default='red',
                       choices=['red', 'green', 'blue', 'yellow'],
                       help='Color to detect (default: red)')
    parser.add_argument('--fov', type=float, default=102.0,
                       help='Horizontal field of view in degrees (default: 102 for Camera Module 3 Wide)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height (default: 480)')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum contour area to detect as flag (default: 500)')
    parser.add_argument('--no-mask', action='store_true',
                       help='Hide color detection mask (mask shown by default)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Robust Colored Flag Detection")
    print("=" * 70)
    print(f"Color: {args.color}")
    print(f"Horizontal FOV: {args.fov}°")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Min Area: {args.min_area} pixels")
    print("=" * 70)
    print("Press 'q' to quit")
    print("Press 'r', 'g', 'b', 'y' to switch colors (red, green, blue, yellow)")
    print("Press '+'/'-' to adjust min area threshold")
    print("=" * 70)
    print()
    
    # Initialize detector
    detector = ColorFlagDetector(color=args.color, horizontal_fov=args.fov, min_area=args.min_area)
    
    # Initialize picamera2
    print("[DEBUG] Initializing picamera2...")
    try:
        picam2 = Picamera2()
    except Exception as e:
        print(f"ERROR: Could not initialize picamera2: {e}")
        return
    
    # Configure camera
    print("[DEBUG] Configuring camera...")
    try:
        preview_config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (args.width, args.height)}
        )
        picam2.configure(preview_config)
        print(f"[DEBUG] Camera configured: {args.width}x{args.height}")
    except Exception as e:
        print(f"ERROR: Could not configure camera: {e}")
        return
    
    # Start camera
    print("[DEBUG] Starting camera...")
    try:
        picam2.start()
        print("[DEBUG] Camera started successfully")
    except Exception as e:
        print(f"ERROR: Could not start camera: {e}")
        return
    
    # Start OpenCV window thread
    cv2.startWindowThread()
    
    print("\nDisplaying camera stream with flag detection...")
    print("Mask display: ON (use --no-mask to hide)\n")
    
    frame_count = 0
    start_time = time.time()
    show_mask = not args.no_mask
    
    try:
        while True:
            # Capture frame from picamera2
            array = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            frame_width = frame.shape[1]
            
            # Detect flag
            flag_data, mask = detector.detect_flag(frame)
            
            # Calculate angle if flag detected
            angle = None
            if flag_data is not None:
                center_x, center_y, _, _ = flag_data
                angle = detector.calculate_angle((center_x, center_y), frame_width)
            
            # Draw detection overlay
            draw_detection(frame, flag_data, angle, detector.color, frame_width, mask)
            
            # Print to terminal periodically
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                if flag_data is not None:
                    print(f"[Frame {frame_count}] FLAG DETECTED | "
                          f"Center: ({flag_data[0]:4d}, {flag_data[1]:4d}) | "
                          f"Area: {int(flag_data[2]):6d} | "
                          f"Angle: {angle:.2f}° | FPS: {fps:.1f}")
                else:
                    print(f"[Frame {frame_count}] No flag detected | FPS: {fps:.1f}")
            
            # Show video with mask by default
            if show_mask:
                # Show mask alongside frame
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # Resize mask to match frame height if needed
                if mask_colored.shape[0] != frame.shape[0]:
                    mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))
                combined = np.hstack([frame, mask_colored])
                cv2.imshow('Flag Detection - Original | Mask (Press q to quit)', combined)
            else:
                cv2.imshow('Flag Detection (Press q to quit)', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.color = 'red'
                print(f"\nSwitched to detecting: RED")
            elif key == ord('g'):
                detector.color = 'green'
                print(f"\nSwitched to detecting: GREEN")
            elif key == ord('b'):
                detector.color = 'blue'
                print(f"\nSwitched to detecting: BLUE")
            elif key == ord('y'):
                detector.color = 'yellow'
                print(f"\nSwitched to detecting: YELLOW")
            elif key == ord('+') or key == ord('='):
                detector.min_area = min(detector.min_area + 100, 10000)
                print(f"\nMin area increased to: {detector.min_area}")
            elif key == ord('-') or key == ord('_'):
                detector.min_area = max(detector.min_area - 100, 100)
                print(f"\nMin area decreased to: {detector.min_area}")
            elif key == ord('m'):
                show_mask = not show_mask
                print(f"\nMask display: {'ON' if show_mask else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        print("[DEBUG] Stopping camera...")
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nTotal frames: {frame_count}, Average FPS: {avg_fps:.2f}")
        print("Camera closed.")


if __name__ == "__main__":
    main()
