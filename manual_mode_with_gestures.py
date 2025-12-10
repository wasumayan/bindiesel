#!/usr/bin/env python3
"""
Manual Mode with Hand Gestures Integration Example
Shows how to integrate hand gesture control alongside voice commands
Add this to main.py's handle_manual_mode_state() method
"""

# This is an example showing how to integrate hand gestures into manual mode
# Add these imports to main.py:
# from hand_gesture_controller import HandGestureController, get_gesture_command

"""
Example integration in BinDieselSystem.__init__():

# Initialize hand gesture controller (optional)
print("\n[Main] Initializing hand gesture controller...")
try:
    self.gesture_controller = HandGestureController(
        model_path='yolo11n-pose.pt',
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        confidence=config.YOLO_CONFIDENCE,
        gesture_hold_time=0.5  # Hold gesture for 0.5s before executing
    )
    print("[Main] Hand gesture controller initialized")
except Exception as e:
    print(f"[Main] WARNING: Failed to initialize gesture controller: {e}")
    self.gesture_controller = None
"""

"""
Example integration in handle_manual_mode_state():

def handle_manual_mode_state(self):
    \"\"\"Handle MANUAL_MODE state - voice commands OR hand gestures\"\"\"
    
    # Priority 1: Check for voice command (existing functionality)
    if self.voice:
        command = self.voice.recognize_command(timeout=0.5)
        if command:
            if command == 'AUTOMATIC_MODE':
                print("[Main] Returning to automatic mode")
                self.current_manual_command = None
                self.motor.stop()
                self.servo.center()
                self.state_machine.transition_to(State.ACTIVE)
                return
            elif command == 'STOP':
                print("[Main] Stopping current command")
                self.current_manual_command = None
                self.motor.stop()
                self.servo.center()
            else:
                print(f"[Main] Voice command: {command}")
                self.current_manual_command = command
                self.last_command_time = time.time()
    
    # Priority 2: Check for hand gesture (if voice didn't provide command)
    if self.gesture_controller and not self.current_manual_command:
        gesture_command = get_gesture_command(self.gesture_controller)
        if gesture_command:
            print(f"[Main] Gesture command: {gesture_command}")
            self.current_manual_command = gesture_command
            self.last_command_time = time.time()
    
    # Execute current command (from either voice or gesture)
    if self.current_manual_command:
        self.execute_manual_command_continuous(self.current_manual_command)
"""

"""
Benefits of this approach:
1. Voice commands take priority (more reliable)
2. Hand gestures work as backup/alternative input
3. Both can be used simultaneously
4. Non-blocking gesture detection (doesn't slow down voice)
5. Gesture hold time prevents accidental commands
"""

