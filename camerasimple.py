#!/usr/bin/env python3
"""
Simple camera test for color flag detection
Shows video stream with color detection and provides terminal updates
"""

# Fix OpenCV import if needed
try:
    import cv2
except ImportError:
    import sys
    import os
    system_paths = [
        '/usr/lib/python3/dist-packages',
        '/usr/local/lib/python3/dist-packages',
    ]
    for path in system_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not found!")
        print("Install with: sudo apt-get install python3-opencv")
        sys.exit(1)

import numpy as np
import time
import argparse
import os

# Suppress GStreamer warnings (they're harmless but noisy)
os.environ['GST_DEBUG'] = '0'
os.environ['GST_DEBUG_NO_COLOR'] = '1'


class ColorFlagDetector:
    """Detects colored flag in camera frame"""
    
    def __init__(self, color='red', horizontal_fov=102.0):
        """
        Initialize color flag detector
        
        Args:
            color: Color to detect ('red', 'green', 'blue', 'yellow')
            horizontal_fov: Horizontal field of view in degrees (default 102 for Camera Module 3 Wide)
        """
        self.color = color.lower()
        self.horizontal_fov = horizontal_fov
        
        # Define HSV color ranges for different colors
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Lower red
                (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Upper red
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ],
            'blue': [
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))
            ]
        }
    
    def detect_flag(self, frame):
        """
        Detect colored flag in frame
        
        Args:
            frame: Camera frame (BGR format)
            
        Returns:
            (center_x, center_y, area) of flag or None if not detected
        """
        if frame is None:
            return None
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range for selected color
        if self.color not in self.color_ranges:
            color_ranges = self.color_ranges['red']
        else:
            color_ranges = self.color_ranges[self.color]
        
        # Create mask for color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask += cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, mask
        
        # Find largest contour (assumed to be the flag)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area to reduce noise
        area = cv2.contourArea(largest_contour)
        if area < 500:
            return None, mask
        
        # Calculate center of flag
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, mask
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y, area), mask
    
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
        # Camera Module 3 Wide has 102° horizontal FOV
        angle = (offset / frame_width) * self.horizontal_fov
        
        return angle


def draw_detection(frame, flag_data, angle, color_name):
    """
    Draw detection overlay on frame
    
    Args:
        frame: Camera frame
        flag_data: (center_x, center_y, area) or None
        angle: Calculated angle in degrees or None
        color_name: Name of color being detected
    """
    h, w = frame.shape[:2]
    
    # Draw center line (car center)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.putText(frame, "CENTER", (w // 2 + 5, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if flag_data is not None:
        center_x, center_y, area = flag_data
        
        # Draw flag center point
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        
        # Draw line from center to flag
        cv2.line(frame, (w // 2, center_y), (center_x, center_y), (0, 255, 255), 2)
        
        # Draw angle text
        if angle is not None:
            angle_text = f"Angle: {angle:.1f}°"
            direction = "LEFT" if angle < 0 else "RIGHT"
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


def print_terminal_update(flag_data, angle, frame_count, fps):
    """
    Print detection info to terminal
    
    Args:
        flag_data: (center_x, center_y, area) or None
        angle: Calculated angle in degrees or None
        frame_count: Current frame count
        fps: Current FPS
    """
    # Clear previous line (simple approach)
    print("\r" + " " * 80, end="", flush=True)
    
    if flag_data is not None:
        center_x, center_y, area = flag_data
        angle_str = f"{angle:.2f}°" if angle is not None else "N/A"
        direction = "LEFT" if angle and angle < 0 else "RIGHT" if angle and angle > 0 else "CENTER"
        
        print(f"\r[Frame {frame_count:05d} | FPS: {fps:.1f}] "
              f"FLAG DETECTED | Center: ({center_x:4d}, {center_y:4d}) | "
              f"Area: {int(area):6d} | Angle: {angle_str:>7s} ({direction})", end="", flush=True)
    else:
        print(f"\r[Frame {frame_count:05d} | FPS: {fps:.1f}] "
              f"NO FLAG DETECTED - Looking for color...", end="", flush=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Simple camera test for color flag detection'
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--color', type=str, default='red',
                       choices=['red', 'green', 'blue', 'yellow'],
                       help='Color to detect (default: red)')
    parser.add_argument('--fov', type=float, default=102.0,
                       help='Horizontal field of view in degrees (default: 102 for Camera Module 3 Wide)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height (default: 480)')
    parser.add_argument('--show-mask', action='store_true',
                       help='Show color detection mask')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Simple Camera Color Flag Detection")
    print("=" * 70)
    print(f"Camera: {args.camera}")
    print(f"Color: {args.color}")
    print(f"Horizontal FOV: {args.fov}°")
    print(f"Resolution: {args.width}x{args.height}")
    print("=" * 70)
    print("Press 'q' to quit")
    print("Press 'r', 'g', 'b', 'y' to switch colors (red, green, blue, yellow)")
    print("=" * 70)
    print()
    
    # Initialize detector
    detector = ColorFlagDetector(color=args.color, horizontal_fov=args.fov)
    
    # Initialize camera - try different backends
    print("Opening camera...")
    cap = None
    
    # Try V4L2 backend first (more reliable on Linux/Raspberry Pi)
    # Check if CAP_V4L2 is available (it might not be on all systems)
    backends_to_try = []
    if hasattr(cv2, 'CAP_V4L2'):
        backends_to_try.append((cv2.CAP_V4L2, "V4L2"))
    backends_to_try.append((cv2.CAP_ANY, "ANY (auto-detect)"))
    
    for backend_id, backend_name in backends_to_try:
        print(f"Trying {backend_name} backend...")
        try:
            cap = cv2.VideoCapture(args.camera, backend_id)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✓ Camera opened successfully with {backend_name} backend!")
                    break
                else:
                    print(f"  Camera opened but cannot read frames, trying next backend...")
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"  Error with {backend_name}: {e}")
            if cap:
                cap.release()
                cap = None
    
    if cap is None or not cap.isOpened():
        print(f"\nTrying default backend as last resort...")
        cap = cv2.VideoCapture(args.camera)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✓ Camera opened with default backend")
            else:
                cap.release()
                cap = None
    
    if cap is None or not cap.isOpened():
        print(f"\nError: Could not open camera {args.camera} with any backend")
        print("Trying to list available cameras...")
        # Try to find available camera
        for i in range(5):
            # Try V4L2 if available, otherwise default
            backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
            test_cap = cv2.VideoCapture(i, backend)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    print(f"Found working camera at index {i}")
                test_cap.release()
        return
    
    print("Camera opened successfully!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Test reading a frame
    print("Testing frame capture...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("ERROR: Cannot read frames from camera!")
        print("This might indicate:")
        print("  - Camera is being used by another application")
        print("  - Camera hardware issue")
        print("  - Driver/permission problem")
        cap.release()
        return
    print(f"Frame test successful! Frame shape: {test_frame.shape}")
    
    # Check if display is available
    try:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test')
        print("Display available - video window will be shown")
    except:
        print("WARNING: No display available - video window cannot be shown")
        print("Running in headless mode (terminal updates only)")
    
    print("Starting video stream...\n")
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    # Detection tracking for terminal updates (update every N frames)
    last_terminal_update = -1  # Start at -1 so first frame always updates
    terminal_update_interval = 5  # Update terminal every 5 frames (more frequent)
    
    print("Entering main loop...")
    print("(If you don't see updates, the loop may not be running)\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nError: Failed to read frame from camera")
                print("This might indicate:")
                print("  - Camera is being used by another application")
                print("  - Camera permissions issue")
                print("  - Camera hardware issue")
                break
            
            if frame is None or frame.size == 0:
                print("\nError: Received empty frame")
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
            
            # Detect flag
            flag_data, mask = detector.detect_flag(frame)
            
            # Calculate angle if flag detected
            angle = None
            if flag_data is not None:
                center_x, center_y, _ = flag_data
                angle = detector.calculate_angle((center_x, center_y), frame.shape[1])
            
            # Update terminal periodically
            if frame_count - last_terminal_update >= terminal_update_interval:
                print_terminal_update(flag_data, angle, frame_count, fps)
                last_terminal_update = frame_count
            
            # Draw detection overlay
            draw_detection(frame, flag_data, angle, detector.color)
            
            # Show video
            try:
                if args.show_mask:
                    # Show mask alongside frame
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    combined = np.hstack([frame, mask_colored])
                    cv2.imshow('Camera - Color Detection (Press q to quit)', combined)
                else:
                    cv2.imshow('Camera - Color Detection (Press q to quit)', frame)
            except Exception as e:
                print(f"\nError displaying video: {e}")
                print("Make sure you have a display available (X11 forwarding if SSH)")
                break
            
            # Handle keyboard input (waitKey is required for imshow to work)
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
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 70)
        print("Camera test complete")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        print("=" * 70)


if __name__ == "__main__":
    main()

