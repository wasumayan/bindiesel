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
# Use YOLO pose tracker instead of basic visual detector
from test_yolo_pose_tracking import YOLOPoseTracker
from motor_controller import MotorController
from servo_controller import ServoController
from tof_sensor import ToFSensor
from voice_recognizer import VoiceRecognizer
from path_tracker import PathTracker
from hand_gesture_controller import HandGestureController, get_gesture_command
from radd_detector import RADDDetector
from logger import setup_logger, log_error, log_warning, log_info, log_debug
from optimizations import FrameCache, PerformanceMonitor, conditional_log, skip_frames


class BinDieselSystem:
    def __init__(self):
        """Initialize all system components"""

        self.logger = setup_logger(__name__)
        
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "Bin Diesel System Initializing...")
        log_info(self.logger, "=" * 70)
        
        # Initialize state machine
        # Initialize state machine
        self.sm = StateMachine(
            tracking_timeout=config.TRACKING_TIMEOUT
        )

        self.debug_mode = config.DEBUG_MODE
        
        # Initialize path tracker
        self.path_tracker = PathTracker()
        
        # Initialize wake word detector
        log_info(self.logger, "Initializing wake word detector...")
        try:
            wake_word_path = os.path.join(
                os.path.dirname(__file__),
                'bin-diesel_en_raspberry-pi_v3_0_0',
                'bin-diesel_en_raspberry-pi_v3_0_0.ppn'
            )
            self.wake_word = WakeWordDetector(
                model_path=wake_word_path,
                access_key=config.WAKE_WORD_ACCESS_KEY
            )
            self.wake_word.start_listening()
            log_info(self.logger, "Wake word detector initialized successfully")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize wake word detector")
            self.cleanup()
            sys.exit(1)
        
        # Initialize YOLO pose tracker (replaces visual detector)
        log_info(self.logger, "Initializing YOLO pose tracker...")
        try:
            self.visual = YOLOPoseTracker(
                model_path=config.YOLO_POSE_MODEL,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                confidence=config.YOLO_CONFIDENCE,
                tracker='bytetrack.yaml',
                device='cpu'
            )
            log_info(self.logger, "YOLO pose tracker initialized with tracking enabled")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize YOLO pose tracker")
            self.cleanup()
            sys.exit(1)
        
        # Initialize motor controller
        log_info(self.logger, "Initializing motor controller...")
    
        self.motor = MotorController(
            pwm_pin=config.MOTOR_PWM_PIN,
            frequency=config.PWM_FREQUENCY
        )
        self.motor.stop()
        log_info(self.logger, "Motor controller initialized successfully")

        
        # Initialize servo controller
        log_info(self.logger, "Initializing servo controller...")
        self.servo = ServoController(
            pwm_pin=config.SERVO_PWM_PIN,
            frequency=config.PWM_FREQUENCY,
            center_duty=config.SERVO_CENTER,
            left_max_duty=config.SERVO_LEFT_MAX,
            right_max_duty=config.SERVO_RIGHT_MAX
        )
        self.servo.center()
        log_info(self.logger, "Servo controller initialized successfully")

        
        # Initialize TOF sensor
        log_info(self.logger, "Initializing TOF sensor...")
        self.tof = ToFSensor()
        log_info(self.logger, "TOF sensor initialized successfully")


        # Control flags
        self.running = True
        self.last_visual_update = 0
        self.visual_update_interval = 0.1  # Update visual detection every 100ms
        
        # Performance optimizations
        self.frame_cache = FrameCache(max_age=0.05)  # Cache frames for 50ms
        self.performance_monitor = PerformanceMonitor()
        self.frame_count = 0
        self.cached_visual_result = None  # Cache visual detection results
        self.cached_visual_timestamp = 0
     
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "System Ready!")
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "Waiting for wake word: 'bin diesel'")
        log_info(self.logger, "Available modes: autonomous, manual, radd")
        log_info(self.logger, "Press Ctrl+C to exit")
        log_info(self.logger, "=" * 70)
    
    #################################################################
    #################################################################
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        log_info(self.logger, "Shutdown signal received, cleaning up...")
        self.running = False
    
    #################################################################
    #################################################################
    def handle_idle_state(self):
        """Handle IDLE state - waiting for wake word"""
        if self.wake_word.detect():
            self.sm.transition_to(State.ACTIVE)
            print("[Main] System activated!")
            if self.debug_mode:
                print("[Main] DEBUG: Wake word detected, transitioning to ACTIVE")
  
    #################################################################
    #################################################################
    def handle_active_state(self):
        """Handle ACTIVE state - waiting for mode selection"""

        # Check visual detection for autonomous mode (with caching)
        current_time = time.time()
        if current_time - self.last_visual_update > self.visual_update_interval:
            try:
                # Use cached result if available and fresh (< 100ms old)
                if (self.cached_visual_result and 
                    (current_time - self.cached_visual_timestamp) < 0.1):
                    result = self.cached_visual_result
                else:
                    result = self.visual.update()
                    self.cached_visual_result = result
                    self.cached_visual_timestamp = current_time
                
                self.last_visual_update = current_time
                
                if result['person_detected'] and result['arm_raised']:
                    # User raised arm - enter autonomous mode
                    self.sm.transition_to(State.TRACKING_USER)
                    self.sm.set_start_position("origin")  # Store starting position
                    self.path_tracker.start_tracking()  # Start tracking path

            except Exception as e:
                log_error(self.logger, e, "Error in visual detection update")
    
    #################################################################
    #################################################################
    def handle_tracking_user_state(self):
        """Handle TRACKING_USER state - detecting and tracking user"""
        # Update visual detection (use cached if available)
        current_time = time.time()
        if (self.cached_visual_result and 
            (current_time - self.cached_visual_timestamp) < 0.1):
            result = self.cached_visual_result
        else:
            result = self.visual.update()
            self.cached_visual_result = result
            self.cached_visual_timestamp = current_time
        
        if not result['person_detected']:
            # User lost - check timeout
            if self.sm.is_timeout():
                conditional_log(self.logger, 'info', 
                              "User tracking timeout, returning to idle",
                              config.DEBUG_MODE)
                self.sm.transition_to(State.IDLE)
                self.motor.stop()
                self.servo.center()
            return
        
        if result['arm_raised']:
            # User has arm raised - start following
            self.sm.transition_to(State.FOLLOWING_USER)
            conditional_log(self.logger, 'info',
                          "User tracking confirmed, starting to follow",
                          config.DEBUG_MODE)
    
    #################################################################
    #################################################################
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""

        # Update visual detection (use cached if available)
        current_time = time.time()
        if (self.cached_visual_result and 
            (current_time - self.cached_visual_timestamp) < 0.1):
            result = self.cached_visual_result
        else:
            result = self.visual.update()
            self.cached_visual_result = result
            self.cached_visual_timestamp = current_time
        
        if not result['person_detected']:
            # User lost
            if self.sm.is_timeout():
                conditional_log(self.logger, 'info',
                              "User lost, returning to idle",
                              config.DEBUG_MODE)
                self.sm.transition_to(State.IDLE)
                self.motor.stop()
                self.servo.center()
            return
        
        # Check if user is too close (TOF sensor)
        if self.tof.detect():
            print("[Main] User reached (TOF sensor), stopping")
            self.motor.stop()
            self.servo.center()
            self.sm.transition_to(State.STOPPED)
            return
        
        # Calculate steering based on angle
        if result['angle'] is not None:
            angle = result['angle']
            
            conditional_log(self.logger, 'debug',
                          f"Person angle: {angle:.1f}°, centered: {result['is_centered']}",
                          self.debug_mode and config.DEBUG_VISUAL)
            
            # Convert angle to steering position
            # Use configurable gain to adjust sensitivity
            steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
            steering_position = max(-1.0, min(1.0, steering_position))
            
            conditional_log(self.logger, 'debug',
                          f"Setting servo angle: {angle:.1f}° (position: {steering_position:.2f})",
                          self.debug_mode and config.DEBUG_SERVO)
            
            self.servo.set_angle(angle)
            
            # Adjust speed based on how centered user is
            if result['is_centered']:
                # User is centered - move forward
                speed = config.FOLLOW_SPEED
                conditional_log(self.logger, 'debug',
                              f"User centered, moving forward at {speed*100:.0f}%",
                              self.debug_mode and config.DEBUG_MOTOR)
                self.motor.forward(speed)
            else:
                # User not centered - slow down while turning
                speed = config.FOLLOW_SPEED * 0.6
                conditional_log(self.logger, 'debug',
                              f"User not centered, moving forward at {speed*100:.0f}% while turning",
                              self.debug_mode and config.DEBUG_MOTOR)
                self.motor.forward(speed)
            
            # Track path segment
            segment_duration = time.time() - self.last_command_time if self.last_command_time > 0 else 0.1
            self.path_tracker.add_segment(speed, steering_position, segment_duration)
            self.last_command_time = time.time()
        else:
            # No angle data - stop
            self.motor.stop()
            self.servo.center()
            self.sm.transition_to(State.IDLE)
            print("[Main] No angle data - returning to IDLE")
    
    #################################################################
    #################################################################
    def handle_stopped_state(self):
        """Handle STOPPED state - at target distance, waiting"""
        # Wait for user to place trash

        wait_time = 10.0  # Wait 10 seconds for trash placement
        if self.state_machine.get_time_in_state() > wait_time:
            print("[Main] Trash collection complete, returning to start")
            self.state_machine.transition_to(State.RETURNING_TO_START)
    #################################################################
    #################################################################
    def handle_returning_to_start_state(self):
        """Handle RETURNING_TO_START state - navigating back using reverse path"""
        # Get reverse path
        reverse_path = self.path_tracker.get_reverse_path()
        
        if len(reverse_path) == 0:
            print("[Main] No path recorded, cannot return to start")
            self.motor.stop()
            self.servo.center()
            self.path_tracker.stop_tracking()
            self.state_machine.transition_to(State.IDLE)
            return
        
        # Execute reverse path segments
        # For simplicity, we'll execute them sequentially
        # In a more sophisticated system, you'd track which segment you're on
        
        # Get the first segment (last movement in original path)
        if not hasattr(self, 'return_path_index'):
            self.return_path_index = 0
            self.return_segment_start_time = time.time()
        
        if self.return_path_index < len(reverse_path):
            segment = reverse_path[self.return_path_index]
            
            # Execute current segment
            self.motor.forward(segment['motor_speed'])
            self.servo.set_position(segment['servo_position'])
            
            # Check if segment duration elapsed
            if time.time() - self.return_segment_start_time >= segment['duration']:
                self.return_path_index += 1
                self.return_segment_start_time = time.time()
                
                if self.debug_mode:
                    print(f"[Main] DEBUG: Completed return segment {self.return_path_index}/{len(reverse_path)}")
        else:
            # All segments completed
            print("[Main] Returned to start position")
            self.motor.stop()
            self.servo.center()
            self.path_tracker.stop_tracking()
            self.path_tracker.clear()
            delattr(self, 'return_path_index')
            self.sm.transition_to(State.IDLE)
    
   
    def run(self):
        """Main control loop"""
        try:
            while self.running:
                # Update performance monitor
                self.performance_monitor.update()
                self.frame_count += 1
                
                state = self.sm.get_state()
                
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
                

                
                # Log performance stats periodically (every 5 seconds)
                if self.frame_count % 500 == 0:  # ~10 FPS * 50 = 5 seconds
                    stats = self.performance_monitor.get_stats()
                    conditional_log(self.logger, 'debug',
                                  f"Performance: FPS={stats['fps']:.1f} "
                                  f"(min={stats['fps_min']:.1f}, max={stats['fps_max']:.1f})",
                                  config.DEBUG_MODE)
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            log_info(self.logger, "Interrupted by user")
        except Exception as e:
            log_error(self.logger, e, "Fatal error in main loop")
            import traceback
            if config.DEBUG_MODE:
                traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        log_info(self.logger, "Cleaning up...")
        
        # Stop all movement
        try:
            if hasattr(self, 'motor'):
                self.motor.stop()
            if hasattr(self, 'servo'):
                self.servo.center()
        except Exception as e:
            log_error(self.logger, e, "Error stopping motors during cleanup")
        
        # Stop all components
        try:
            if hasattr(self, 'wake_word'):
                self.wake_word.stop()
            if hasattr(self, 'visual'):
                self.visual.stop()
            if hasattr(self, 'gesture_controller') and self.gesture_controller:
                # Only stop if it has its own camera (shouldn't if we're sharing)
                if hasattr(self.gesture_controller, 'picam2') and self.gesture_controller.picam2:
                    self.gesture_controller.stop()
            if hasattr(self, 'motor'):
                self.motor.cleanup()
            if hasattr(self, 'servo'):
                self.servo.cleanup()
            if hasattr(self, 'voice') and self.voice:
                self.voice.cleanup()
        except Exception as e:
            log_error(self.logger, e, "Error during component cleanup")
        
        log_info(self.logger, "Cleanup complete")


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

