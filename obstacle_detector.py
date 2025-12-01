"""
Obstacle detection using camera for navigation
Uses depth estimation or object detection to avoid obstacles
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class ObstacleDetector:
    """Detects obstacles in camera view for navigation"""
    
    def __init__(self, method: str = 'depth'):
        """
        Initialize obstacle detector
        
        Args:
            method: 'depth' (monocular depth estimation) or 'object' (object detection)
        """
        self.method = method
        
        if method == 'depth':
            # For monocular depth estimation, we can use:
            # - Stereo vision (if available)
            # - Monocular depth estimation models
            # - Simple edge detection + size estimation
            self.use_depth = True
        else:
            self.use_depth = False
        
        # Obstacle detection parameters
        self.obstacle_threshold = 0.3  # meters (minimum safe distance)
        self.detection_zone_height = 0.6  # Use bottom 60% of frame for obstacles
        
    def detect_obstacles_depth(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect obstacles using depth estimation
        
        Args:
            frame: Camera frame
            
        Returns:
            List of (x, y, estimated_distance) for obstacles
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use bottom portion of frame (where obstacles would be)
        h, w = frame.shape[:2]
        roi_y = int(h * (1 - self.detection_zone_height))
        roi = gray[roi_y:, :]
        
        # Edge detection to find potential obstacles
        edges = cv2.Canny(roi, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Adjust y coordinate for ROI offset
                y += roi_y
                
                # Estimate distance based on size (larger = closer)
                # This is a rough approximation
                size = max(w_box, h_box)
                estimated_distance = 2.0 / (size / 100.0 + 0.1)  # Rough calibration
                
                # Center of obstacle
                center_x = x + w_box // 2
                center_y = y + h_box // 2
                
                obstacles.append((center_x, center_y, estimated_distance))
        
        return obstacles
    
    def detect_obstacles_object(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect obstacles using object detection (furniture, walls, etc.)
        
        Args:
            frame: Camera frame
            
        Returns:
            List of (x, y, estimated_distance) for obstacles
        """
        # This would use YOLO or similar to detect objects
        # For now, return empty list (implement if needed)
        # You could use YOLOv8 or similar for object detection
        
        obstacles = []
        # TODO: Implement object detection if needed
        
        return obstacles
    
    def detect_obstacles(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect obstacles in frame
        
        Args:
            frame: Camera frame
            
        Returns:
            List of (x, y, estimated_distance) for obstacles
        """
        if self.use_depth:
            return self.detect_obstacles_depth(frame)
        else:
            return self.detect_obstacles_object(frame)
    
    def get_obstacle_map(self, frame: np.ndarray, grid_size: int = 5) -> np.ndarray:
        """
        Create obstacle map divided into grid cells
        
        Args:
            frame: Camera frame
            grid_size: Number of horizontal grid cells
            
        Returns:
            2D array where True = obstacle detected, False = clear
        """
        obstacles = self.detect_obstacles(frame)
        
        h, w = frame.shape[:2]
        cell_width = w // grid_size
        
        # Create grid
        obstacle_map = np.zeros(grid_size, dtype=bool)
        
        for x, y, distance in obstacles:
            if distance < self.obstacle_threshold:
                # Map x coordinate to grid cell
                cell = min(int(x / cell_width), grid_size - 1)
                obstacle_map[cell] = True
        
        return obstacle_map
    
    def find_safe_direction(self, frame: np.ndarray, target_angle: float, 
                           grid_size: int = 5) -> Optional[float]:
        """
        Find safe direction considering obstacles
        
        Args:
            frame: Camera frame
            target_angle: Desired direction angle
            grid_size: Number of horizontal grid cells
            
        Returns:
            Safe direction angle, or None if no safe path
        """
        obstacle_map = self.get_obstacle_map(frame, grid_size)
        
        # Convert target angle to grid cell
        h, w = frame.shape[:2]
        fov = 60.0  # Field of view
        target_cell = int((target_angle / fov + 0.5) * grid_size)
        target_cell = np.clip(target_cell, 0, grid_size - 1)
        
        # Check if target direction is clear
        if not obstacle_map[target_cell]:
            return target_angle  # Safe to go directly
        
        # Find nearest clear cell
        for offset in range(1, grid_size):
            # Try left
            left_cell = target_cell - offset
            if left_cell >= 0 and not obstacle_map[left_cell]:
                angle = ((left_cell / grid_size) - 0.5) * fov
                return angle
            
            # Try right
            right_cell = target_cell + offset
            if right_cell < grid_size and not obstacle_map[right_cell]:
                angle = ((right_cell / grid_size) - 0.5) * fov
                return angle
        
        # No clear path found
        return None
    
    def draw_obstacles(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected obstacles on frame"""
        obstacles = self.detect_obstacles(frame)
        
        for x, y, distance in obstacles:
            # Color based on distance (red = close, yellow = medium, green = far)
            if distance < self.obstacle_threshold:
                color = (0, 0, 255)  # Red - too close!
            elif distance < self.obstacle_threshold * 2:
                color = (0, 255, 255)  # Yellow - warning
            else:
                color = (0, 255, 0)  # Green - safe
            
            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.putText(frame, f"{distance:.1f}m", (x-20, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

