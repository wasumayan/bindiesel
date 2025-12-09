"""
State Machine Module
Manages system states and transitions for Bin Diesel workflow
"""

import time
from enum import Enum, auto
import config


class State(Enum):
    """System states"""
    IDLE = auto() 
    DRIVING_TO_USER = auto()
    STOPPED_AT_USER = auto()
    RETURNING = auto()


class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self.state_enter_time = time.time()

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
    
    def set_start_position(self, position):
        self.start_position = position
        print(f"[StateMachine] Start position set: {position}")
    
    def get_start_position(self):
        return self.start_position


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

