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
        
        # Centroid smoothing for stable detection
        self.last_centroid = None
        self.centroid_history = []
        self.history_size = 5  # Average over last 5 detections
        
        # HSV color ranges for color detection
        # OpenCV HSV: H=[0,179], S=[0,255], V=[0,255]
        # These ranges are based on standard color detection practices
        self.color_ranges = {
            'red': [
                # Red wraps around 0/180, so we need two ranges
                # Lower red (0-10 degrees)
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                # Upper red (170-180 degrees)
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'green': [
                # Green is around 60 degrees in HSV
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ],
            'blue': [
                # Blue is around 120 degrees in HSV
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            'yellow': [
                # Yellow is around 30 degrees in HSV
                (np.array([20, 50, 50]), np.array([30, 255, 255]))
            ]
        }
    
    def smooth_centroid(self, new_centroid):
        """
        Smooth centroid using moving average to reduce jitter
        
        Args:
            new_centroid: (x, y) tuple of new centroid
            
        Returns:
            Smoothed (x, y) centroid
        """
        if new_centroid is None:
            return self.last_centroid
        
        self.centroid_history.append(new_centroid)
        if len(self.centroid_history) > self.history_size:
            self.centroid_history.pop(0)
        
        # Calculate average
        if len(self.centroid_history) > 0:
            avg_x = int(np.mean([c[0] for c in self.centroid_history]))
            avg_y = int(np.mean([c[1] for c in self.centroid_history]))
            smoothed = (avg_x, avg_y)
            self.last_centroid = smoothed
            return smoothed
        
        self.last_centroid = new_centroid
        return new_centroid
    
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
        
        # Apply blur to reduce noise (like rpi-object-detection does)
        # Use simple blur first, then Gaussian for better performance
        blurred = cv2.blur(frame, (3, 3))
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        
        # Convert BGR to HSV
        # CRITICAL: frame must be in BGR format for this to work correctly
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
        
        # Additional validation: check if mask has enough pixels
        mask_pixel_count = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_percentage = (mask_pixel_count / total_pixels) * 100
        
        # If mask is too sparse (< 0.1% of image), likely false detection
        if mask_percentage < 0.1:
            return None, mask
        
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
            # Allow wide range: 0.2 to 5.0 (reasonable for flags)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Filter by minimum size (both width and height should be reasonable)
            if w < 20 or h < 20:  # Reject very small detections
                continue
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Filter by solidity (flags should be reasonably solid)
            if solidity < 0.3:
                continue
            
            # Additional validation: check if contour fills reasonable portion of bounding box
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.4:  # Contour should fill at least 40% of bounding box (stricter)
                continue
            
            # Additional check: reject if contour is too fragmented (many small pieces)
            # Count how many separate regions are in the bounding box
            # This helps reject text/logos on objects
            
            # Calculate centroid using moments for accuracy
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            # Check if centroid is within bounding box (sanity check)
            if not (x <= centroid_x <= x + w and y <= centroid_y <= y + h):
                continue
            
            valid_contours.append((contour, area, (x, y, w, h), (centroid_x, centroid_y)))
        
        if len(valid_contours) == 0:
            return None, mask
        
        # Select the largest valid contour (assumed to be the flag)
        largest = max(valid_contours, key=lambda x: x[1])
        largest_contour, area, bbox, centroid = largest
        
        # Use centroid from moments (more accurate than bounding box center)
        center_x, center_y = centroid
        
        # Smooth the centroid to reduce jitter
        smoothed_centroid = self.smooth_centroid((center_x, center_y))
        if smoothed_centroid:
            center_x, center_y = smoothed_centroid
        
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
        frame: Camera frame (BGR format for OpenCV drawing functions)
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
    print("Press 'm' to toggle mask display")
    print("Click on image to see HSV values at that point (for debugging)")
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
    
    # Debug: Mouse callback to show HSV values at clicked point
    mouse_pos = None
    hsv_values = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, hsv_values
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = (x, y)
            # Get HSV value at clicked point
            if 'frame_hsv' in param:
                hsv_val = param['frame_hsv'][y, x]
                hsv_values = tuple(hsv_val)
                print(f"\n[DEBUG] Clicked at ({x}, {y})")
                print(f"  HSV: H={hsv_val[0]}, S={hsv_val[1]}, V={hsv_val[2]}")
                print(f"  Current color range for {detector.color}:")
                if detector.color in detector.color_ranges:
                    for i, (lower, upper) in enumerate(detector.color_ranges[detector.color]):
                        print(f"    Range {i+1}: H[{lower[0]}-{upper[0]}], S[{lower[1]}-{upper[1]}], V[{lower[2]}-{upper[2]}]")
    
    # Create window and set mouse callback
    window_name = 'Flag Detection - Original | Mask (Press q to quit)'
    cv2.namedWindow(window_name)
    
    try:
        while True:
            # Capture frame from picamera2
            # Note: picamera2.capture_array() returns RGB888 format
            array = picam2.capture_array()
            
            # IMPORTANT: picamera2 returns RGB, OpenCV expects BGR
            # Convert RGB to BGR for all OpenCV operations
            frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            frame_width = frame_bgr.shape[1]
            
            # Keep RGB version for display (so colors look correct when shown)
            frame_rgb = array.copy()
            
            # Convert BGR to HSV for color detection (must use BGR version!)
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # Set up mouse callback with HSV frame
            cv2.setMouseCallback(window_name, mouse_callback, {'frame_hsv': frame_hsv})
            
            # Detect flag (use BGR for processing)
            flag_data, mask = detector.detect_flag(frame_bgr)
            
            # Calculate angle if flag detected
            angle = None
            if flag_data is not None:
                center_x, center_y, _, _ = flag_data
                angle = detector.calculate_angle((center_x, center_y), frame_width)
            
            # Draw detection overlay on RGB frame (for correct color display)
            # Convert RGB to BGR temporarily for drawing, then convert back
            frame_display = frame_rgb.copy()
            # OpenCV drawing functions expect BGR, so convert temporarily
            frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            draw_detection(frame_display_bgr, flag_data, angle, detector.color, frame_width, mask)
            # Convert back to RGB for display
            frame_display = cv2.cvtColor(frame_display_bgr, cv2.COLOR_BGR2RGB)
            
            # Debug: Show HSV values at mouse position
            if mouse_pos and hsv_values:
                x, y = mouse_pos
                h, s, v = hsv_values
                debug_text = f"HSV at ({x},{y}): H={h}, S={s}, V={v}"
                # Convert to BGR for drawing, then back to RGB
                frame_temp = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_temp, debug_text, (10, frame_display.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # Draw crosshair at clicked point
                cv2.circle(frame_temp, (x, y), 5, (255, 255, 0), 2)
                cv2.line(frame_temp, (x-10, y), (x+10, y), (255, 255, 0), 1)
                cv2.line(frame_temp, (x, y-10), (x, y+10), (255, 255, 0), 1)
                frame_display = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)
            
            # Debug: Show mask statistics
            if mask is not None:
                mask_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_percent = (mask_pixels / total_pixels) * 100
                mask_stats = f"Mask: {mask_pixels}/{total_pixels} pixels ({mask_percent:.1f}%)"
                # Convert to BGR for drawing, then back to RGB
                frame_temp = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_temp, mask_stats, (10, frame_display.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                frame_display = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)
            
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
                    # Debug: Show why no flag detected
                    if mask is not None:
                        mask_pixels = np.sum(mask > 0)
                        total_pixels = mask.shape[0] * mask.shape[1]
                        mask_percent = (mask_pixels / total_pixels) * 100
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) > 0:
                            largest_area = max([cv2.contourArea(c) for c in contours])
                            print(f"[Frame {frame_count}] No flag detected | "
                                  f"Mask: {mask_percent:.1f}% | "
                                  f"Largest contour: {int(largest_area)} (min: {detector.min_area}) | "
                                  f"FPS: {fps:.1f}")
                        else:
                            print(f"[Frame {frame_count}] No flag detected | "
                                  f"Mask: {mask_percent:.1f}% | "
                                  f"No contours found | "
                                  f"FPS: {fps:.1f}")
                    else:
                        print(f"[Frame {frame_count}] No flag detected | FPS: {fps:.1f}")
            
            # Show video with mask by default
            if show_mask:
                # Show mask alongside frame
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # Resize mask to match frame height if needed
                if mask_colored.shape[0] != frame_display.shape[0]:
                    mask_colored = cv2.resize(mask_colored, (frame_display.shape[1], frame_display.shape[0]))
                
                # Also show HSV visualization for debugging
                # Create a visualization of the HSV space
                # Method: Show the original frame with HSV color space overlay
                # This helps see what the camera sees in HSV
                hsv_vis = frame_hsv.copy()
                # Convert HSV back to BGR, then to RGB for display
                hsv_bgr = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR)
                hsv_display = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display
                
                # Combine: Original (RGB) | Mask | HSV (RGB)
                combined = np.hstack([frame_display, mask_colored, hsv_display])
                cv2.imshow(window_name, combined)
            else:
                cv2.imshow('Flag Detection (Press q to quit)', frame_display)
            
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
