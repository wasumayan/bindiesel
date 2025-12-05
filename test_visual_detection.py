#!/usr/bin/env python3
"""
Test script for visual detection (person + arm raising)
Shows camera feed with detection overlays
"""

import cv2
import time
import argparse
import sys
from visual_detector import VisualDetector
import config

def main():
    parser = argparse.ArgumentParser(description='Test visual detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--fps', action='store_true', help='Show FPS counter')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Visual Detection Test")
    print("=" * 70)
    print("This will show camera feed with detection overlays")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'd' to toggle debug mode")
    print("=" * 70)
    print()
    
    debug_mode = args.debug
    
    try:
        # Initialize visual detector
        detector = VisualDetector(
            model_path=config.YOLO_MODEL,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            confidence=config.YOLO_CONFIDENCE
        )
        
        print("[TEST] Visual detector initialized")
        print("[TEST] Starting camera feed...")
        print()
        
        # FPS tracking
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        # Start OpenCV window
        cv2.startWindowThread()
        
        while True:
            # Get detection results
            result = detector.update()
            
            # Get frame for display
            frame = detector.get_frame()
            
            # Draw detection overlays
            if result['person_detected']:
                x1, y1, x2, y2 = result['person_box']
                
                # Draw bounding box
                color = (0, 255, 0) if result['arm_raised'] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Draw angle line
                frame_center_x = frame.shape[1] // 2
                cv2.line(frame, (frame_center_x, center_y), (center_x, center_y), color, 2)
                
                # Draw text
                label = f"Person (conf: {result.get('confidence', 0):.2f})"
                if result['arm_raised']:
                    label += " - ARM RAISED!"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw angle info
                if result['angle'] is not None:
                    angle_text = f"Angle: {result['angle']:.1f}°"
                    direction = "LEFT" if result['angle'] < 0 else "RIGHT" if result['angle'] > 0 else "CENTER"
                    cv2.putText(frame, angle_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, direction, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw centered status
                if result['is_centered']:
                    cv2.putText(frame, "CENTERED", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw center line
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
            
            # Draw FPS
            if args.fps:
                fps_counter += 1
                if fps_counter % 10 == 0:
                    fps_end_time = time.time()
                    fps = 10 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(frame, fps_text, (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Debug mode output
            if debug_mode:
                debug_y = 120
                cv2.putText(frame, f"DEBUG MODE", (10, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                debug_y += 20
                
                if result['person_detected']:
                    cv2.putText(frame, f"Person box: {result['person_box']}", (10, debug_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    debug_y += 15
                    cv2.putText(frame, f"Arm raised: {result['arm_raised']}", (10, debug_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    debug_y += 15
                    cv2.putText(frame, f"Arm confidence: {result['arm_confidence']:.2f}", (10, debug_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    debug_y += 15
                    cv2.putText(frame, f"Is centered: {result['is_centered']}", (10, debug_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Display frame
            cv2.imshow('Visual Detection Test - Press q to quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[TEST] Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            # Print to terminal periodically
            if result['person_detected']:
                print(f"[TEST] Person detected: angle={result['angle']:.1f}°, "
                      f"centered={result['is_centered']}, "
                      f"arm_raised={result['arm_raised']}")
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'detector' in locals():
            detector.stop()
        cv2.destroyAllWindows()
        print("[TEST] Test complete")


if __name__ == '__main__':
    main()

