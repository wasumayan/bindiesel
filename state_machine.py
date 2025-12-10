"""
State Machine Module
Manages system states and transitions for Bin Diesel workflow
"""

import time
from enum import Enum, auto
import config


class State(Enum):
    """System states"""
    IDLE = auto()  # Wake word only - no voice recognizer (exclusive mic access)
    ACTIVE = auto()  # Post-wake-word: voice commands and visual detection for mode selection
    TRACKING_USER = auto()
    FOLLOWING_USER = auto()
    STOPPED = auto()  # At target distance, waiting for trash collection
    HOME = auto()  # Returning to home marker (turn 180, detect marker, drive to it)
    MANUAL_MODE = auto()
    RADD_MODE = auto()  # Drive towards users not wearing full pants or closed-toe shoes
    # Legacy states (for compatibility)
    DORMANT = auto()  # Deprecated - use IDLE instead
    RETURNING_TO_START = auto()  # Deprecated - use HOME instead
    DRIVING_TO_USER = auto()
    STOPPED_AT_USER = auto()
    RETURNING = auto()


class StateMachine:
    def __init__(self, tracking_timeout=30.0):
        self.state = State.IDLE  # Start in IDLE state (wake word only)
        self.state_enter_time = time.time()
        self.tracking_timeout = tracking_timeout

        self.forward_start_time = None
        self.forward_elapsed_time = 0.0 

        if config.DEBUG_STATE:
            print(f"[SM] initial state: {self.state.name}")
    

    def get_state(self):
        return self.state
    
    def get_time_in_state(self):
        return time.time() - self.state_enter_time
    
    def transition_to(self, new_state: State):
        if config.DEBUG_STATE:
            print(f"[SM] {self.state.name} -> {new_state.name}")
       
        self.state = new_state
        self.state_enter_time = time.time()
    
    def is_timeout(self):
        """Check if tracking timeout has been exceeded"""
        return self.get_time_in_state() > self.tracking_timeout
    
    def set_start_position(self, position):
        self.start_position = position
        print(f"[StateMachine] Start position set: {position}")
    
    def get_start_position(self):
        return getattr(self, 'start_position', None)


if __name__ == '__main__':
    print("Testing state machine...")

    sm = StateMachine()

    print(f"Current state: {sm.get_state().name}")
    time.sleep(0.5)

    sm.transition_to(State.DRIVING_TO_USER)
    time.sleep(0.5)
    print(f"State: {sm.get_state().name}, time in state: {sm.get_time_in_state():.2f}s")

    sm.transition_to(State.STOPPED_AT_USER)
    time.sleep(0.5)
    print(f"State: {sm.get_state().name}, time in state: {sm.get_time_in_state():.2f}s")

    print("State machine test complete!")

