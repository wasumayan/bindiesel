#!/usr/bin/env python3
"""
Bin Diesel Main Control System
Main entry point that coordinates all modules
"""

import sys
import time
import signal
import os
from pathlib import Path

# Add parent directory to path for wake word model
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
import config
from state_machine import StateMachine, State
from wake_word_detector import WakeWordDetector
from visual_detector import VisualDetector
from motor_controller import MotorController
from servo_controller import ServoController
from tof_sensor import TOFSensor
from voice_recognizer import VoiceRecognizer


class BinDieselSystem:
    """Main system controller"""
    
    def __init__(self):
        """Initialize all system components"""
        print("=" * 70)
        print("Bin Diesel System Initializing...")
        print("=" * 70)
        
        # Initialize state machine
        self.state_machine = StateMachine(
            tracking_timeout=config.TRACKING_TIMEOUT
        )
        
        # Initialize wake word detector
        print("\n[Main] Initializing wake word detector...")
        try:
            wake_word_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'bin-diesel_en_raspberry-pi_v3_0_0',
                'bin-diesel_en_raspberry-pi_v3_0_0.ppn'
            )
            self.wake_word = WakeWordDetector(
                model_path=wake_word_path,
                access_key=config.WAKE_WORD_ACCESS_KEY
            )
            self.wake_word.start_listening()
        except Exception as e:
            print(f"[Main] ERROR: Failed to initialize wake word detector: {e}")
            sys.exit(1)
        
        # Initialize visual detector
        print("\n[Main] Initializing visual detector...")
        try:
            self.visual = VisualDetector(
                model_path=config.YOLO_MODEL,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                confidence=config.YOLO_CONFIDENCE
            )
        except Exception as e:
            print(f"[Main] ERROR: Failed to initialize visual detector: {e}")
            self.cleanup()
            sys.exit(1)
        
        # Initialize motor controller
        print("\n[Main] Initializing motor controller...")
        try:
            self.motor = MotorController(
                pwm_pin=config.MOTOR_PWM_PIN,
                frequency=config.PWM_FREQUENCY
            )
        except Exception as e:
            print(f"[Main] ERROR: Failed to initialize motor controller: {e}")
            self.cleanup()
            sys.exit(1)
        
        # Initialize servo controller
        print("\n[Main] Initializing servo controller...")
        try:
            self.servo = ServoController(
                pwm_pin=config.SERVO_PWM_PIN,
                frequency=config.PWM_FREQUENCY,
                center_duty=config.SERVO_CENTER,
                left_max_duty=config.SERVO_LEFT_MAX,
                right_max_duty=config.SERVO_RIGHT_MAX
            )
        except Exception as e:
            print(f"[Main] ERROR: Failed to initialize servo controller: {e}")
            self.cleanup()
            sys.exit(1)
        
        # Initialize TOF sensor
        print("\n[Main] Initializing TOF sensor...")
        try:
            self.tof = TOFSensor(
                stop_distance_mm=config.TOF_STOP_DISTANCE_MM,
                emergency_distance_mm=config.TOF_EMERGENCY_DISTANCE_MM
            )
        except Exception as e:
            print(f"[Main] WARNING: Failed to initialize TOF sensor: {e}")
            print("[Main] Continuing without TOF sensor (safety feature disabled)")
            self.tof = None
        
        # Initialize voice recognizer (for manual mode)
        print("\n[Main] Initializing voice recognizer...")
        try:
            self.voice = VoiceRecognizer(
                api_key=config.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"[Main] WARNING: Failed to initialize voice recognizer: {e}")
            print("[Main] Manual mode will not be available")
            self.voice = None
        
        # Control flags
        self.running = True
        self.last_visual_update = 0
        self.visual_update_interval = 0.1  # Update visual detection every 100ms
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("\n" + "=" * 70)
        print("System Ready!")
        print("=" * 70)
        print("Waiting for wake word: 'bin diesel'")
        print("Press Ctrl+C to exit")
        print("=" * 70)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n[Main] Shutdown signal received, cleaning up...")
        self.running = False
    
    def handle_idle_state(self):
        """Handle IDLE state - waiting for wake word"""
        if self.wake_word.detect():
            self.state_machine.transition_to(State.ACTIVE)
            print("[Main] System activated!")
    
    def handle_active_state(self):
        """Handle ACTIVE state - waiting for mode selection"""
        # Check for wake word again (for "manual mode" command)
        # For now, we'll check visual detection to see if user raises arm
        # Or wait for voice command "manual mode"
        
        # Check visual detection
        current_time = time.time()
        if current_time - self.last_visual_update > self.visual_update_interval:
            result = self.visual.update()
            self.last_visual_update = current_time
            
            if result['person_detected'] and result['arm_raised']:
                # User raised arm - enter autonomous mode
                self.state_machine.transition_to(State.TRACKING_USER)
                self.state_machine.set_start_position("origin")  # Store starting position
                print("[Main] Autonomous mode: User detected with raised arm")
        
        # TODO: Check for "manual mode" voice command
        # For now, we'll skip manual mode detection in this basic version
    
    def handle_tracking_user_state(self):
        """Handle TRACKING_USER state - detecting and tracking user"""
        # Update visual detection
        result = self.visual.update()
        
        if not result['person_detected']:
            # User lost - check timeout
            if self.state_machine.is_timeout():
                print("[Main] User tracking timeout, returning to idle")
                self.state_machine.transition_to(State.IDLE)
                self.motor.stop()
                self.servo.center()
            return
        
        if result['arm_raised']:
            # User has arm raised - start following
            self.state_machine.transition_to(State.FOLLOWING_USER)
            print("[Main] User tracking confirmed, starting to follow")
        else:
            # User detected but arm not raised - wait
            pass
    
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""
        # Check TOF sensor for emergency stop
        if self.tof and config.EMERGENCY_STOP_ENABLED:
            if self.tof.is_emergency_stop():
                print("[Main] EMERGENCY STOP: Object too close!")
                self.motor.stop()
                self.servo.center()
                return
        
        # Update visual detection
        result = self.visual.update()
        
        if not result['person_detected']:
            # User lost
            if self.state_machine.is_timeout():
                print("[Main] User lost, returning to idle")
                self.state_machine.transition_to(State.IDLE)
                self.motor.stop()
                self.servo.center()
            return
        
        # Check if user is too close (TOF sensor)
        if self.tof and self.tof.is_too_close():
            print("[Main] User reached (TOF sensor), stopping")
            self.motor.stop()
            self.servo.center()
            self.state_machine.transition_to(State.STOPPED)
            return
        
        # Calculate steering based on angle
        if result['angle'] is not None:
            angle = result['angle']
            
            # Convert angle to steering position
            # Use configurable gain to adjust sensitivity
            steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
            steering_position = max(-1.0, min(1.0, steering_position))
            
            self.servo.set_angle(angle)
            
            # Adjust speed based on how centered user is
            if result['is_centered']:
                # User is centered - move forward
                self.motor.forward(config.FOLLOW_SPEED)
            else:
                # User not centered - slow down while turning
                self.motor.forward(config.FOLLOW_SPEED * 0.7)
        else:
            # No angle data - stop
            self.motor.stop()
            self.servo.center()
    
    def handle_stopped_state(self):
        """Handle STOPPED state - at target distance, waiting"""
        # Wait for user to place trash
        # For now, we'll wait a fixed time, then return to start
        # In a real system, you might wait for a signal or button press
        
        wait_time = 10.0  # Wait 10 seconds for trash placement
        if self.state_machine.get_time_in_state() > wait_time:
            print("[Main] Trash collection complete, returning to start")
            self.state_machine.transition_to(State.RETURNING_TO_START)
    
    def handle_returning_to_start_state(self):
        """Handle RETURNING_TO_START state - navigating back"""
        # Simple return: move backward for a fixed time/distance
        # In a real system, you'd use odometry or visual navigation
        
        return_time = 5.0  # Return for 5 seconds (adjust based on distance)
        if self.state_machine.get_time_in_state() > return_time:
            print("[Main] Returned to start position")
            self.motor.stop()
            self.servo.center()
            self.state_machine.transition_to(State.IDLE)
        else:
            # Move backward
            self.motor.backward(config.RETURN_SPEED)
            self.servo.center()  # Keep straight
    
    def handle_manual_mode_state(self):
        """Handle MANUAL_MODE state - waiting for voice commands"""
        if not self.voice:
            print("[Main] Voice recognizer not available, returning to idle")
            self.state_machine.transition_to(State.IDLE)
            return
        
        # Get voice command
        command = self.voice.recognize_command()
        
        if command:
            self.state_machine.transition_to(State.EXECUTING_COMMAND)
            self.execute_manual_command(command)
    
    def execute_manual_command(self, command):
        """Execute manual voice command"""
        print(f"[Main] Executing command: {command}")
        
        if command == 'FORWARD':
            self.motor.forward(config.MOTOR_MEDIUM)
            self.servo.center()
            time.sleep(2)  # Move for 2 seconds
            self.motor.stop()
        
        elif command == 'LEFT':
            self.motor.forward(config.MOTOR_SLOW)
            self.servo.turn_left(0.5)
            time.sleep(1)  # Turn for 1 second
            self.motor.stop()
            self.servo.center()
        
        elif command == 'RIGHT':
            self.motor.forward(config.MOTOR_SLOW)
            self.servo.turn_right(0.5)
            time.sleep(1)  # Turn for 1 second
            self.motor.stop()
            self.servo.center()
        
        elif command == 'STOP':
            self.motor.stop()
            self.servo.center()
        
        elif command == 'TURN_AROUND':
            self.motor.stop()
            self.servo.turn_left(1.0)
            time.sleep(2)  # Turn around
            self.servo.center()
        
        # Return to manual mode
        self.state_machine.transition_to(State.MANUAL_MODE)
    
    def run(self):
        """Main control loop"""
        try:
            while self.running:
                state = self.state_machine.get_state()
                
                # Route to appropriate handler based on state
                if state == State.IDLE:
                    self.handle_idle_state()
                
                elif state == State.ACTIVE:
                    self.handle_active_state()
                
                elif state == State.TRACKING_USER:
                    self.handle_tracking_user_state()
                
                elif state == State.FOLLOWING_USER:
                    self.handle_following_user_state()
                
                elif state == State.STOPPED:
                    self.handle_stopped_state()
                
                elif state == State.RETURNING_TO_START:
                    self.handle_returning_to_start_state()
                
                elif state == State.MANUAL_MODE:
                    self.handle_manual_mode_state()
                
                elif state == State.EXECUTING_COMMAND:
                    # Commands execute synchronously, so we don't loop here
                    pass
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user")
        except Exception as e:
            print(f"\n[Main] ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\n[Main] Cleaning up...")
        
        # Stop all movement
        if hasattr(self, 'motor'):
            self.motor.stop()
        if hasattr(self, 'servo'):
            self.servo.center()
        
        # Stop all components
        if hasattr(self, 'wake_word'):
            self.wake_word.stop()
        if hasattr(self, 'visual'):
            self.visual.stop()
        if hasattr(self, 'motor'):
            self.motor.cleanup()
        if hasattr(self, 'servo'):
            self.servo.cleanup()
        if hasattr(self, 'voice'):
            self.voice.cleanup()
        
        print("[Main] Cleanup complete")


if __name__ == '__main__':
    # Check for required environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('PICOVOICE_ACCESS_KEY'):
        print("ERROR: PICOVOICE_ACCESS_KEY not set in environment!")
        print("Create a .env file with: PICOVOICE_ACCESS_KEY=your_key")
        sys.exit(1)
    
    # Create and run system
    system = BinDieselSystem()
    system.run()

