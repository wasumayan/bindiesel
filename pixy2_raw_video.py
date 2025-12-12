#!/usr/bin/env python3
"""
Pixy2 Raw Video Frame Access
Uses libpixyusb2 to get raw video frames from Pixy2 camera
Compatible with existing YOLO/ArUco code
"""

import sys
import numpy as np
import cv2
import time

# Try to import Pixy2 library
try:
    from pixy import *
    from ctypes import *
    HAS_PIXY = True
except ImportError:
    HAS_PIXY = False
    print("ERROR: Pixy2 library not found!")
    print("Please install libpixyusb2 first:")
    print("  git clone https://github.com/charmedlabs/pixy2")
    print("  cd pixy2/scripts")
    print("  ./build_libpixyusb2.sh")
    print("  cd ../build/python_demos")
    print("  pip install .  (or use venv)")
    sys.exit(1)

import config


class Pixy2RawVideo:
    """
    Access raw video frames from Pixy2 using libpixyusb2
    Provides interface compatible with existing camera code
    """
    
    def __init__(self, width=320, height=200):
        """
        Initialize Pixy2 for raw video capture
        
        Args:
            width: Desired frame width (Pixy2 native: 320x200)
            height: Desired frame height
        """
        self.width = width
        self.height = height
        
        # Pixy2 native resolution
        self.pixy_width = 320
        self.pixy_height = 200
        
        print("[Pixy2RawVideo] Initializing Pixy2...")
        
        # Initialize Pixy2
        status = pixy_init()
        if status != 0:
            raise RuntimeError(f"Failed to initialize Pixy2: error code {status}\n"
                             "Make sure Pixy2 is connected via USB and you have permissions")
        
        print("[Pixy2RawVideo] Pixy2 initialized successfully")
        
        # Set camera to video mode (if supported)
        # Note: Some Pixy2 firmware versions may not support raw video mode
        try:
            # Try to set video mode
            # This may not be available in all firmware versions
            pixy_cam_set_mode(PIXY_CAM_DEFAULT_MODE)
        except:
            print("[Pixy2RawVideo] Warning: Could not set video mode, using default")
        
        # Frame buffer for raw frame data
        self.frame_buffer = None
        
        print(f"[Pixy2RawVideo] Ready - Native resolution: {self.pixy_width}x{self.pixy_height}")
    
    def get_frame(self):
        """
        Get current raw video frame from Pixy2
        
        Returns:
            Frame as numpy array in RGB format (compatible with existing code)
        """
        try:
            # Get raw frame from Pixy2
            # Note: This uses the getFrame API from libpixyusb2
            # The exact API may vary - check pixy2 documentation
            
            # Method 1: Try using pixy_get_frame() if available
            try:
                # Allocate frame buffer if needed
                if self.frame_buffer is None:
                    # Frame size: width * height * 3 (RGB)
                    frame_size = self.pixy_width * self.pixy_height * 3
                    self.frame_buffer = (c_uint8 * frame_size)()
                
                # Get frame (this is a placeholder - actual API may differ)
                # Check pixy2/src/host/libpixyusb2/include/pixy.h for exact function
                status = pixy_get_frame(self.pixy_width, self.pixy_height, 
                                       self.frame_buffer, self.pixy_width * self.pixy_height * 3)
                
                if status == 0:
                    # Convert to numpy array
                    frame = np.frombuffer(self.frame_buffer, dtype=np.uint8)
                    frame = frame.reshape((self.pixy_height, self.pixy_width, 3))
                    
                    # Resize if needed
                    if self.width != self.pixy_width or self.height != self.pixy_height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Apply camera transformations (from config)
                    frame = self._apply_transformations(frame)
                    
                    return frame
                else:
                    raise RuntimeError(f"Failed to get frame: error code {status}")
                    
            except AttributeError:
                # pixy_get_frame() not available - try alternative method
                # Some Pixy2 versions may use different API
                raise NotImplementedError(
                    "Raw video frame capture not available in this Pixy2 firmware.\n"
                    "Try using USB video mode instead: python3 test_pixy2_video.py"
                )
                
        except Exception as e:
            raise RuntimeError(f"Error getting frame from Pixy2: {e}")
    
    def _apply_transformations(self, frame):
        """Apply camera transformations from config (rotation, flips, etc.)"""
        # Apply rotation
        if config.CAMERA_ROTATION == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif config.CAMERA_ROTATION == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif config.CAMERA_ROTATION == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply flips
        if config.CAMERA_FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        if config.CAMERA_FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)
        
        # Color channel swap
        if config.CAMERA_SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def stop(self):
        """Stop camera and cleanup"""
        try:
            pixy_close()
            print("[Pixy2RawVideo] Stopped")
        except:
            pass
    
    def close(self):
        """Alias for stop()"""
        self.stop()


# Alternative: If raw video API is not available, check the actual API
def check_pixy2_api():
    """Check what Pixy2 API functions are available"""
    print("Checking available Pixy2 API functions...")
    print()
    
    try:
        import pixy
        
        # List all available functions
        print("Available functions in pixy module:")
        for attr in dir(pixy):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        print()
        
        # Check for frame-related functions
        frame_funcs = [attr for attr in dir(pixy) if 'frame' in attr.lower() or 'video' in attr.lower()]
        if frame_funcs:
            print("Frame/video related functions:")
            for func in frame_funcs:
                print(f"  - {func}")
        else:
            print("No frame/video functions found in pixy module")
            print("You may need to use USB video mode instead")
        
    except Exception as e:
        print(f"Error checking API: {e}")


if __name__ == '__main__':
    # Test raw video access
    print("=" * 70)
    print("Pixy2 Raw Video Frame Test")
    print("=" * 70)
    print()
    
    # First, check what API is available
    check_pixy2_api()
    print()
    
    try:
        # Initialize Pixy2
        pixy_video = Pixy2RawVideo(width=640, height=480)
        
        print("=" * 70)
        print("Starting video capture...")
        print("Press 'q' to quit")
        print("=" * 70)
        print()
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                frame = pixy_video.get_frame()
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add FPS text
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Pixy2 Raw Video", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except NotImplementedError as e:
                print(f"ERROR: {e}")
                print()
                print("Raw video mode not available. Try USB video mode instead:")
                print("  python3 test_pixy2_video.py")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        pixy_video.stop()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

