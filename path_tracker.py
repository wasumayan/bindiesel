"""
Path Tracker Module
Tracks the car's movement path for reverse navigation
"""

import time
from collections import deque


class PathTracker:
    """Tracks movement path for reverse navigation"""
    
    def __init__(self, max_path_length=1000):
        """
        Initialize path tracker
        
        Args:
            max_path_length: Maximum number of path segments to store
        """
        self.path = deque(maxlen=max_path_length)
        self.is_tracking = False
        print("[PathTracker] Initialized")
    
    def start_tracking(self):
        """Start tracking path"""
        self.path.clear()
        self.is_tracking = True
        print("[PathTracker] Started tracking path")
    
    def stop_tracking(self):
        """Stop tracking path"""
        self.is_tracking = False
        print(f"[PathTracker] Stopped tracking (recorded {len(self.path)} segments)")
    
    def add_segment(self, motor_speed, servo_position, duration):
        """
        Add a path segment
        
        Args:
            motor_speed: Motor speed (0.0 to 1.0)
            servo_position: Servo position (-1.0 to 1.0, 0.0 = center)
            duration: Duration of this segment in seconds
        """
        if not self.is_tracking:
            return
        
        segment = {
            'motor_speed': motor_speed,
            'servo_position': servo_position,
            'duration': duration,
            'timestamp': time.time()
        }
        
        self.path.append(segment)
    
    def get_reverse_path(self):
        """
        Get reversed path for return navigation
        
        Returns:
            List of path segments in reverse order with reversed commands
        """
        if len(self.path) == 0:
            return []
        
        # Reverse the path and invert servo positions
        reverse_path = []
        for segment in reversed(self.path):
            reverse_segment = {
                'motor_speed': segment['motor_speed'],  # Same speed
                'servo_position': -segment['servo_position'],  # Invert steering
                'duration': segment['duration']  # Same duration
            }
            reverse_path.append(reverse_segment)
        
        return reverse_path
    
    def clear(self):
        """Clear path history"""
        self.path.clear()
        print("[PathTracker] Path cleared")
    
    def get_path_length(self):
        """Get number of path segments"""
        return len(self.path)


if __name__ == '__main__':
    # Test path tracker
    print("Testing path tracker...")
    
    tracker = PathTracker()
    tracker.start_tracking()
    
    # Simulate some movement
    tracker.add_segment(motor_speed=0.5, servo_position=0.0, duration=2.0)  # Forward straight
    tracker.add_segment(motor_speed=0.5, servo_position=0.3, duration=1.0)  # Forward right
    tracker.add_segment(motor_speed=0.5, servo_position=-0.2, duration=1.5)  # Forward left
    tracker.add_segment(motor_speed=0.3, servo_position=0.0, duration=1.0)  # Forward slow
    
    print(f"Path length: {tracker.get_path_length()}")
    
    # Get reverse path
    reverse = tracker.get_reverse_path()
    print(f"Reverse path has {len(reverse)} segments")
    
    for i, segment in enumerate(reverse):
        print(f"  Segment {i+1}: speed={segment['motor_speed']:.2f}, "
              f"servo={segment['servo_position']:.2f}, duration={segment['duration']:.1f}s")
    
    print("Test complete!")

