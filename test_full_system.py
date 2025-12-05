#!/usr/bin/env python3
"""
Test script for full system (simulation mode)
Tests state machine and workflow without hardware
"""

import time
import sys
from state_machine import StateMachine, State
import config

def simulate_visual_detection():
    """Simulate visual detection results"""
    # Simulate different scenarios
    scenarios = [
        {'person_detected': False, 'arm_raised': False},
        {'person_detected': True, 'arm_raised': False, 'angle': -15.0, 'is_centered': False},
        {'person_detected': True, 'arm_raised': True, 'angle': 10.0, 'is_centered': True},
        {'person_detected': True, 'arm_raised': True, 'angle': 0.0, 'is_centered': True},
    ]
    
    import random
    return random.choice(scenarios)

def simulate_tof_sensor():
    """Simulate TOF sensor readings"""
    import random
    # Simulate distance between 5cm and 50cm
    return random.uniform(50, 500)  # mm

def main():
    print("=" * 70)
    print("Full System Test (Simulation Mode)")
    print("=" * 70)
    print("This simulates the full workflow without hardware")
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    try:
        # Initialize state machine
        sm = StateMachine(tracking_timeout=config.TRACKING_TIMEOUT)
        
        print("[TEST] State machine initialized")
        print("[TEST] Starting simulation...")
        print()
        
        step = 0
        
        while True:
            step += 1
            current_state = sm.get_state()
            time_in_state = sm.get_time_in_state()
            
            print(f"[TEST Step {step}] State: {current_state.value} (time: {time_in_state:.1f}s)")
            
            # Simulate state transitions based on current state
            if current_state == State.IDLE:
                # Simulate wake word detection
                if step > 2:
                    print("  → Wake word detected!")
                    sm.transition_to(State.ACTIVE)
            
            elif current_state == State.ACTIVE:
                # Simulate visual detection
                visual = simulate_visual_detection()
                if visual['person_detected'] and visual.get('arm_raised'):
                    print(f"  → Person detected with raised arm (angle: {visual.get('angle', 0):.1f}°)")
                    sm.transition_to(State.TRACKING_USER)
                    sm.set_start_position("origin")
            
            elif current_state == State.TRACKING_USER:
                # Simulate tracking
                visual = simulate_visual_detection()
                if visual.get('arm_raised'):
                    print("  → User tracking confirmed")
                    sm.transition_to(State.FOLLOWING_USER)
                elif time_in_state > 5:
                    print("  → Timeout, returning to idle")
                    sm.transition_to(State.IDLE)
            
            elif current_state == State.FOLLOWING_USER:
                # Simulate following
                visual = simulate_visual_detection()
                tof_distance = simulate_tof_sensor()
                
                if tof_distance <= config.TOF_STOP_DISTANCE_MM:
                    print(f"  → TOF sensor: {tof_distance/10:.1f}cm (STOP threshold reached)")
                    sm.transition_to(State.STOPPED)
                elif not visual['person_detected']:
                    print("  → Person lost")
                    if time_in_state > 5:
                        sm.transition_to(State.IDLE)
                else:
                    print(f"  → Following (angle: {visual.get('angle', 0):.1f}°, distance: {tof_distance/10:.1f}cm)")
            
            elif current_state == State.STOPPED:
                # Simulate waiting for trash
                if time_in_state > 3:
                    print("  → Trash collection complete, returning to start")
                    sm.transition_to(State.RETURNING_TO_START)
            
            elif current_state == State.RETURNING_TO_START:
                # Simulate return
                if time_in_state > 2:
                    print("  → Returned to start position")
                    sm.transition_to(State.IDLE)
            
            elif current_state == State.MANUAL_MODE:
                # Simulate manual mode (not implemented in this test)
                print("  → Manual mode (not simulated)")
                if time_in_state > 2:
                    sm.transition_to(State.IDLE)
            
            time.sleep(1)  # Wait 1 second between steps
            
            # Reset after completing a full cycle
            if current_state == State.IDLE and step > 20:
                print()
                print("[TEST] Full cycle completed, restarting...")
                print()
                step = 0
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[TEST] Test complete")


if __name__ == '__main__':
    main()

