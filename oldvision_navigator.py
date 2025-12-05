"""
Vision-based navigation controller
Combines person tracking, obstacle avoidance, and PSoC communication
"""

import time
from typing import Optional, Tuple
from vision_person_tracker import PersonTracker
from obstacle_detector import ObstacleDetector
from psoc_communicator import PSoCCommunicator
import numpy as np


class VisionNavigator:
    """Main navigation controller using vision"""
    
    def __init__(self, psoc_port: str = '/dev/ttyUSB0', 
                 psoc_baudrate: int = 115200,
                 camera_index: int = 0):
        """
        Initialize vision-based navigation system
        
        Args:
            psoc_port: Serial port for PSoC
            psoc_baudrate: Serial baud rate
            camera_index: Camera device index
        """
        # Initialize components
        self.person_tracker = PersonTracker(camera_index=camera_index)
        self.obstacle_detector = ObstacleDetector()
        self.psoc_communicator = PSoCCommunicator(port=psoc_port, baudrate=psoc_baudrate)
        
        # Navigation state
        self.mode = 'idle'  # 'idle', 'following', 'stopped'
        self.target_angle = 0.0
        self.target_speed = 0.0
        self.last_update_time = 0
        self.update_rate = 10.0  # Hz
        
        # Control parameters
        self.max_speed = 0.5  # Maximum speed (0-1.0)
        self.max_angle = 45.0  # Maximum steering angle (degrees)
        self.follow_distance = 1.0  # Desired distance to person (meters)
        
    def start(self):
        """Start navigation system"""
        print("Starting Vision Navigation System...")
        
        # Start camera
        self.person_tracker.start_camera()
        time.sleep(0.5)
        
        # Connect to PSoC
        if not self.psoc_communicator.connect():
            print("Warning: Could not connect to PSoC. Continuing without car control.")
        
        print("System ready!")
    
    def update(self):
        """Update navigation (call in main loop)"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < (1.0 / self.update_rate):
            return
        
        self.last_update_time = current_time
        
        # Update person tracking
        detection = self.person_tracker.update()
        
        if self.mode == 'following' and detection:
            self._navigate_to_person()
        elif self.mode == 'stopped':
            self._send_stop_command()
        else:
            self._send_stop_command()
    
    def _navigate_to_person(self):
        """Navigate toward detected person while avoiding obstacles"""
        frame = self.person_tracker.get_frame()
        if frame is None:
            return
        
        # Get person position and direction
        person_angle = self.person_tracker.calculate_direction(frame.shape[1])
        person_distance = self.person_tracker.estimate_distance(frame.shape[0])
        
        if person_angle is None:
            self._send_stop_command()
            return
        
        # Check for obstacles
        safe_angle = self.obstacle_detector.find_safe_direction(
            frame, person_angle
        )
        
        if safe_angle is None:
            # No safe path found
            print("Warning: No safe path to person")
            self._send_stop_command()
            return
        
        # Calculate speed based on distance
        if person_distance:
            if person_distance > self.follow_distance * 1.5:
                # Far away - move faster
                speed = self.max_speed
            elif person_distance < self.follow_distance * 0.8:
                # Too close - slow down or stop
                speed = 0.0
            else:
                # Good distance - moderate speed
                speed = self.max_speed * 0.5
        else:
            speed = self.max_speed * 0.5
        
        # Calculate steering angle
        angle = np.clip(safe_angle, -self.max_angle, self.max_angle)
        
        # Send commands to PSoC
        self._send_navigation_command(angle, speed)
        
        print(f"Navigating: angle={angle:.1f}°, speed={speed:.2f}, distance={person_distance:.2f}m" if person_distance else f"Navigating: angle={angle:.1f}°, speed={speed:.2f}")
    
    def _send_navigation_command(self, angle: float, speed: float):
        """Send navigation command to PSoC"""
        # Format: "NAV:ANGLE:XX.XX:SPEED:XX.XX\n"
        message = f"NAV:ANGLE:{angle:.2f}:SPEED:{speed:.2f}\n"
        self.psoc_communicator.serial_connection.write(message.encode('ascii'))
        self.psoc_communicator.serial_connection.flush()
    
    def _send_stop_command(self):
        """Send stop command to PSoC"""
        message = "NAV:STOP\n"
        if self.psoc_communicator.serial_connection and self.psoc_communicator.serial_connection.is_open:
            self.psoc_communicator.serial_connection.write(message.encode('ascii'))
            self.psoc_communicator.serial_connection.flush()
    
    def start_following(self):
        """Start following mode"""
        self.mode = 'following'
        print("Following mode activated")
    
    def stop(self):
        """Stop navigation"""
        self.mode = 'stopped'
        self._send_stop_command()
        print("Stopped")
    
    def set_idle(self):
        """Set to idle mode"""
        self.mode = 'idle'
        self._send_stop_command()
        print("Idle mode")
    
    def get_frame_with_overlay(self):
        """Get camera frame with detection overlays"""
        frame = self.person_tracker.get_frame()
        if frame is None:
            return None
        
        # Draw person detection
        frame = self.person_tracker.draw_detection(frame)
        
        # Draw obstacles
        frame = self.obstacle_detector.draw_obstacles(frame)
        
        # Draw status
        status_text = f"Mode: {self.mode.upper()}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        self.person_tracker.cleanup()
        self.psoc_communicator.disconnect()

