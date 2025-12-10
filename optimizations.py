"""
Performance optimization utilities
Provides caching, frame management, and performance monitoring
"""

import time
from functools import lru_cache, wraps
from collections import deque
import numpy as np


class FrameCache:
    """Cache frames to avoid redundant captures"""
    
    def __init__(self, max_age=0.05):  # 50ms max age
        self.frame = None
        self.timestamp = 0
        self.max_age = max_age
    
    def get(self, get_frame_func):
        """Get cached frame or capture new one"""
        current_time = time.time()
        if self.frame is None or (current_time - self.timestamp) > self.max_age:
            self.frame = get_frame_func()
            self.timestamp = current_time
        return self.frame
    
    def invalidate(self):
        """Invalidate cache"""
        self.frame = None
        self.timestamp = 0


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        """Update performance metrics"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
        self.last_time = current_time
    
    def get_fps(self):
        """Get average FPS"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.fps_history:
            return {'fps': 0.0, 'fps_min': 0.0, 'fps_max': 0.0}
        
        fps_list = list(self.fps_history)
        return {
            'fps': sum(fps_list) / len(fps_list),
            'fps_min': min(fps_list),
            'fps_max': max(fps_list),
            'fps_std': np.std(fps_list) if len(fps_list) > 1 else 0.0
        }


def conditional_log(logger, level, message, condition=True, *args, **kwargs):
    """
    Conditional logging - only log if condition is True
    Reduces logging overhead in hot paths
    """
    if condition:
        if level == 'debug':
            logger.debug(message, *args, **kwargs)
        elif level == 'info':
            logger.info(message, *args, **kwargs)
        elif level == 'warning':
            logger.warning(message, *args, **kwargs)
        elif level == 'error':
            logger.error(message, *args, **kwargs)


def skip_frames(frame_count, skip_interval):
    """
    Determine if frame should be skipped
    Returns True if frame should be processed, False if skipped
    """
    return frame_count % skip_interval == 0


def memoize_with_ttl(ttl=1.0):
    """
    Memoize function results with time-to-live
    Useful for caching expensive computations
    """
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Check if cached result is still valid
            if key in cache:
                if current_time - cache_times[key] < ttl:
                    return cache[key]
                else:
                    # Expired, remove from cache
                    del cache[key]
                    del cache_times[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time
            return result
        
        return wrapper
    return decorator


class FrameProcessor:
    """Optimized frame processor with caching and reuse"""
    
    def __init__(self):
        self.current_frame = None
        self.frame_timestamp = 0
        self.frame_cache_age = 0.05  # 50ms
    
    def get_frame(self, capture_func):
        """Get frame with caching"""
        current_time = time.time()
        if (self.current_frame is None or 
            (current_time - self.frame_timestamp) > self.frame_cache_age):
            self.current_frame = capture_func()
            self.frame_timestamp = current_time
        return self.current_frame
    
    def invalidate(self):
        """Invalidate frame cache"""
        self.current_frame = None
        self.frame_timestamp = 0

