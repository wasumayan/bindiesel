"""
State Machine Module
Manages system states and transitions for Bin Diesel workflow
"""

import time
from enum import Enum


class State(Enum):
    """System states"""
    IDLE = "idle"  # Waiting for wake word
    ACTIVE = "active"  # Awake, waiting for mode selection
    TRACKING_USER = "tracking_user"  # Detecting and tracking user
    FOLLOWING_USER = "following_user"  # Moving toward user
    STOPPED = "stopped"  # At target distance, waiting
    RETURNING_TO_START = "returning_to_start"  # Navigating back
    MANUAL_MODE = "manual_mode"  # Waiting for voice commands
    EXECUTING_COMMAND = "executing_command"  # Executing manual command


class StateMachine:
    """Manages system state and transitions"""
    
    def __init__(self, tracking_timeout=30.0):
        """
        Initialize state machine
        
        Args:
            tracking_timeout: Seconds before returning to idle if no user detected
        """
        self.state = State.IDLE
        self.tracking_timeout = tracking_timeout
        self.state_start_time = time.time()
        self.start_position = None  # Store starting position for return navigation
        
        print(f"[StateMachine] Initialized, starting in {self.state.value}")
    
    def transition_to(self, new_state):
        """
        Transition to new state
        
        Args:
            new_state: New State enum value
        """
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.state_start_time = time.time()
            print(f"[StateMachine] Transition: {old_state.value} → {new_state.value}")
    
    def get_state(self):
        """Get current state"""
        return self.state
    
    def get_time_in_state(self):
        """Get time spent in current state (seconds)"""
        return time.time() - self.state_start_time
    
    def is_timeout(self):
        """Check if tracking timeout has been reached"""
        if self.state in [State.TRACKING_USER, State.FOLLOWING_USER]:
            return self.get_time_in_state() > self.tracking_timeout
        return False
    
    def set_start_position(self, position):
        """
        Set starting position for return navigation
        
        Args:
            position: Starting position (can be any format, e.g., (x, y) or None)
        """
        self.start_position = position
        print(f"[StateMachine] Start position set: {position}")
    
    def get_start_position(self):
        """Get starting position"""
        return self.start_position


if __name__ == '__main__':
    # Test state machine
    print("Testing state machine...")
    
    sm = StateMachine(tracking_timeout=30.0)
    
    # Simulate workflow
    print(f"Current state: {sm.get_state().value}")
    
    sm.transition_to(State.ACTIVE)
    time.sleep(0.1)
    print(f"Time in state: {sm.get_time_in_state():.2f}s")
    
    sm.transition_to(State.TRACKING_USER)
    time.sleep(0.1)
    print(f"Time in state: {sm.get_time_in_state():.2f}s")
    
    sm.transition_to(State.FOLLOWING_USER)
    time.sleep(0.1)
    print(f"Time in state: {sm.get_time_in_state():.2f}s")
    
    print("State machine test complete!")

