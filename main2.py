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
import cv2
from state_machine import StateMachine, State
from wake_word_detector import WakeWordDetector
from test_yolo_pose_tracking import YOLOPoseTracker
from motor_controller import MotorController
from servo_controller import ServoController
from tof_sensor import ToFSensor
from logger import setup_logger, log_error, log_warning, log_info, log_debug
from optimizations import FrameCache, PerformanceMonitor, conditional_log, skip_frames
from test_apriltag_detection import ArUcoDetector


class BinDieselSystem:
    """Main system controller"""
    
    ################################################################################################################################# __init__
    #################################################################################################################################
    def __init__(self):
        """Initialize all system components"""
        self.logger = setup_logger(__name__)
    

        self.sm = StateMachine(
            tracking_timeout=config.TRACKING_TIMEOUT
        )
        
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "Bin Diesel System Initializing...")
        log_info(self.logger, "=" * 70)

        # Control flags 
        self.last_visual_update = 0
        self.visual_update_interval = config.VISUAL_UPDATE_INTERVAL  # Use configurable update interval
        self.running = True
        self._wake_word_stopped = False  # Track if wake word detector has been stopped

        
        # Initialize servo controller
        log_info(self.logger, "Initializing servo controller...")
        try:
            self.servo = ServoController(
                pwm_pin=config.SERVO_PWM_PIN,
                frequency=config.PWM_FREQUENCY_SERVO,
                center_duty=config.SERVO_CENTER,
                left_max_duty=config.SERVO_LEFT_MAX,
                right_max_duty=config.SERVO_RIGHT_MAX
            )
            
            log_info(self.logger, "Servo controller initialized successfully")

            self.servo.last_servo_angle = 0.0  # Track that servo is centered

        except Exception as e:
            log_error(self.logger, e, "Failed to initialize servo controller")
            self.cleanup()
            sys.exit(1)


            # Initialize motor controller
        log_info(self.logger, "Initializing motor controller...")
        try:
            self.motor = MotorController(
                pwm_pin=config.MOTOR_PWM_PIN,
                frequency=config.PWM_FREQUENCY
            )
            self.motor.stop()  
            log_info(self.logger, "Motor controller initialized successfully")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize motor controller")
            self.cleanup()
            sys.exit(1)
        
        self.servo.center()
        self.last_servo_angle = 0.0  # Track that servo is centered
        time.sleep(3.0)
        
        # Initialize TOF sensor
        log_info(self.logger, "Initializing TOF sensor...")
        try:
            self.tof = ToFSensor()
            log_info(self.logger, "TOF sensor initialized successfully")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize TOF sensor: {e}", "Continuing without TOF sensor (safety feature disabled)")
            self.tof = None    
       
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
        
        # Initialize YOLO pose tracker (for initial user identification)
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
        
        # Initialize lightweight tracker (for fast tracking after user identified)
        log_info(self.logger, "Initializing lightweight tracker...")
        try:
            from lightweight_tracker import LightweightTracker
            self.lightweight_tracker = LightweightTracker(
                model_path=config.YOLO_MODEL,  # Use regular object detection model (not pose)
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                confidence=config.YOLO_CONFIDENCE,
                device='cpu'
            )
            log_info(self.logger, "Lightweight tracker initialized (for fast tracking after user identified)")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize lightweight tracker: {e}", "Will use pose tracker for all states")
            self.lightweight_tracker = None
        
        # Initialize ArUco marker detector for home marker detection
        log_info(self.logger, "Initializing ArUco marker detector for home marker...")
        try:
            # Use tag size from config (default 0.047m = 47mm)
            # You can add ARUCO_TAG_SIZE_M to config.py if needed
            tag_size_m = getattr(config, 'ARUCO_TAG_SIZE_M', 0.047)  # Default 47mm
            self.aruco_detector = ArUcoDetector(tag_size_m=tag_size_m)
            log_info(self.logger, f"ArUco marker detector initialized (tag size: {tag_size_m}m)")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize ArUco detector: {e}", "Home marker detection will not work")
            self.aruco_detector = None
        
        
        # Tracking state - store target track_id to ensure we follow the same person
        self.target_track_id = None  # Track ID of the person we're following
        self.use_lightweight_tracker = False  # Flag to switch to lightweight tracker after user identified
    
        self.last_error_angle = 0.0  # Last error angle for lost user recovery
        
        # Performance optimizations
        self.frame_cache = FrameCache(max_age=0.05)  # Cache frames for 50ms
        self.performance_monitor = PerformanceMonitor()
        self.frame_count = 0
        self.cached_visual_result = None  # Cache visual detection results
        self.cached_visual_timestamp = 0
        self.frame_skip_counter = 0  # Counter for frame skipping
        # self.current_manual_command = None  # Current active manual command

        self.sleeptimer = 0.3 # for re-finding user 
        
        # Debug mode
        self.debug_mode = config.DEBUG_MODE
        if self.debug_mode:
            log_info(self.logger, "DEBUG MODE ENABLED")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "System Ready!")
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "State: IDLE - Waiting for wake word: 'bin diesel'")
        log_info(self.logger, "After wake word: ACTIVE state will look for user with arm raised")
        log_info(self.logger, "Press Ctrl+C to exit")
        log_info(self.logger, "=" * 70)

    ############################################################################################################################# signal_handler
    ##############################################################################################################################
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        log_info(self.logger, "Shutdown signal received, cleaning up...")
        self.running = False

    ##############################################################################################################################_transition_to
    #################################################################################################################################
    def _transition_to(self, new_state):
        """Transition to a new state with highlighted logging"""
        current_state = self.sm.get_state()
        if current_state != new_state:
            log_info(self.logger, "*" * 70)
            log_info(self.logger, "*" * 70)
            log_info(self.logger, f"STATE TRANSITION: {current_state.name} -> {new_state.name}")
            log_info(self.logger, "*" * 70)
            log_info(self.logger, "*" * 70)
        self.sm.transition_to(new_state)

    def safe_center_servo(self):
        """Center servo only if it's not already centered"""
        if self.last_servo_angle != 0.0:
            self.servo.center()
            self.last_servo_angle = 0.0
        
    ########################################################################################################################## handle_idle_state
    ##############################################################################################################################
    def handle_idle_state(self):
        """Handle IDLE state - wake word only, no voice recognizer (exclusive mic access)"""
  
        # Ensure wake word detector is running
        if self._wake_word_stopped: 
            try:
                self.wake_word.start_listening()
                self._wake_word_stopped = False
                log_info(self.logger, "Wake word detector started in IDLE state")
            except Exception as e:
                log_warning(self.logger, f"Failed to start wake word detector: {e}", "IDLE state")
                time.sleep(0.5)
                return
        
        # Check for wake word - when detected, transition to ACTIVE
        if self.wake_word.detect():
            print("[Main] Wake word detected!")
            if self.debug_mode:
                print("[Main] DEBUG: Wake word detected, transitioning to ACTIVE")
            
            # Stop wake word detector to prevent overlap with voice recognizer
            try:
                self.wake_word.stop()
                self._wake_word_stopped = True
            except Exception as e:
                log_warning(self.logger, f"Error stopping wake word detector: {e}", "State transition")
            
            # Transition to FOLLOWING_USER state for post-wake-word functionality
            self._transition_to(State.FOLLOWING_USER)
    
    
    ################################################################################################################# handle_tracking_user_state
    ############################################################################################################################################
    
    def handle_tracking_user_state(self):
        """Handle TRACKING_USER state - detecting and tracking user"""
        # Update visual detection (use cached if available)
        
        current_time = time.time()
        
        # Frame skipping: only process every Nth frame for better performance
        self.frame_skip_counter += 1
        should_update = (self.frame_skip_counter % config.FRAME_SKIP_INTERVAL == 0)
        
        if (self.cached_visual_result and 
            (current_time - self.cached_visual_timestamp) < 0.1):
            result = self.cached_visual_result
        elif should_update:
            # Pass target_track_id to ensure we only follow the specific person
            result = self.visual.update(target_track_id=self.target_track_id)
            self.cached_visual_result = result
            self.cached_visual_timestamp = current_time
        else:
            # Use cached result if skipping this frame
            result = self.cached_visual_result if self.cached_visual_result else {
                'person_detected': False,
                'arm_raised': False,
                'angle': None,
                'is_centered': False,
                'track_id': None
            }
        
        if result['arm_raised']:
            # User has arm raised - store their track_id and start following
            self.target_track_id = result.get('track_id')  # Store the track_id of the person who raised their arm
            log_info(self.logger, f"Arm raised detected! Track ID: {result.get('track_id', 'N/A')}, "
                                 f"Angle: {result.get('angle', 'N/A'):.1f}°")
            conditional_log(self.logger, 'info',
                          f"User tracking confirmed (Track ID: {self.target_track_id}), switching to lightweight tracker",
                          config.DEBUG_MODE)
            # Switch to lightweight tracker for faster performance
            self.use_lightweight_tracker = True
            self._transition_to(State.FOLLOWING_USER)
        
    ################################################################################################################ handle_following_user_state
    ############################################################################################################################################
    
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""

        if not self.sm.old_state == self.sm.state:
            self.motor.forward(config.MOTOR_MEDIUM) 
            self.sm.old_state = self.sm.state
            conditional_log(self.logger, 'info', f"Motor forward start at speed {config.MOTOR_FAST}", config.DEBUG_MODE)
        
        # Update visual detection (use cached if available)
        current_time = time.time()
        
        # Use lightweight tracker if available and user has been identified
        if self.use_lightweight_tracker and self.lightweight_tracker is not None:
            # Get frame from camera
            frame = self.visual.get_frame()
            # Use lightweight tracker (much faster - no pose detection)
            result = self.lightweight_tracker.update(frame, target_track_id=self.target_track_id)
            # Add arm_raised field for compatibility (always False in lightweight mode)
            result['arm_raised'] = False
        else:
            # Fallback to pose tracker (slower but works)
            # Frame skipping: only process every Nth frame for better performance
            self.frame_skip_counter += 1
            should_update = (self.frame_skip_counter % config.FRAME_SKIP_INTERVAL == 0)
            
            if (self.cached_visual_result and 
                (current_time - self.cached_visual_timestamp) < 0.1):
                result = self.cached_visual_result
            elif should_update:
                # Pass target_track_id to ensure we only follow the specific person
                result = self.visual.update(target_track_id=self.target_track_id)
                self.cached_visual_result = result
                self.cached_visual_timestamp = current_time
            else:
                # Use cached result if skipping this frame
                result = self.cached_visual_result if self.cached_visual_result else {
                    'person_detected': False,
                    'arm_raised': False,
                    'angle': None,
                    'is_centered': False,
                    'track_id': None
                }
        
        # Check if we're still tracking the same person (by track_id)
        if self.target_track_id is not None:
            # Only follow if the detected person matches our target track_id
            if result.get('track_id') != self.target_track_id:
                # Different person detected - treat as person lost
                result['person_detected'] = False  # Treat as person lost
        
        if not result['person_detected']:
            # User lost - stop car and revert to tracking state to search for user
            log_info(self.logger, "User lost during following, going other way...")
            self.motor.forward(config.MOTOR_SLOW)
            # steer opposite of last known error to search
            self.servo.set_angle(self.last_error_angle * -2)
            self.last_error_angle = self.last_error_angle * -1  # Flip for next time
            self.target_track_id = None  # Clear target track_id
            self.use_lightweight_tracker = False  # Switch back to pose tracker for re-identification
            time.sleep(self.sleeptimer)
            if self.sleeptimer < 2.0:
                self.sleeptimer += 0.1

            return
        
        # (TOF check now happens at top of run() for immediate emergency response)
        
        # Calculate steering based on angle 
        if result['angle'] is not None:
            self.sleeptimer = config.SLEEP_TIMER  # reset sleep timer
            angle = result['angle']    
            conditional_log(self.logger, 'debug', f"Person angle: {angle:.1f}°, centered: {result['is_centered']}",
                          self.debug_mode and config.DEBUG_VISUAL)
            
            
            # Direct angle steering 
            steering_angle = max(-45.0, min(45.0, angle))  # Clamp to servo range
            self.servo.set_angle(steering_angle)
            self.last_error_angle = steering_angle 
            time.sleep(0.25)
            self.servo.center()
            
            # Adjust speed based on how centered user is
            if result['is_centered']:
                # User is centered - move forward
                speed = config.MOTOR_FAST
                conditional_log(self.logger, 'info',
                              f"User CENTERED, moving forward at {speed*100:.0f}%", config.DEBUG_MODE)
                self.motor.forward(speed)
                time.sleep(0.5)
            else:
                # User not centered - slow down while turning
                speed = config.MOTOR_MEDIUM 
                conditional_log(self.logger, 'info',
                              f"User not centered, moving forward at {speed*100:.0f}% while turning", config.DEBUG_MODE)
                self.motor.forward(speed)
                
        else:
            # No angle data, approaching user? 
            conditional_log(self.logger, 'info', "No angle data, approaching user? Moving slow", config.DEBUG_MODE)
            self.motor.forward(config.MOTOR_SLOW)
            self.safe_center_servo()
            # self._transition_to(State.TRACKING_USER)
            log_info(self.logger, "No angle data")
    
    ####################################################################################################################### handle_stopped_state
    ############################################################################################################################################
    def handle_stopped_state(self):
        """Handle STOPPED state - at target distance, waiting for trash collection"""
        # Wait for fixed amount of time for trash placement, then go to HOME
        conditional_log(self.logger, 'info', "STOPPED: Waiting for trash collection", config.DEBUG_MODE)

        wait_time = 10.0  # Wait 10 seconds for trash placement
        if self.sm.get_time_in_state() > wait_time:
            log_info(self.logger, "Trash collection complete, returning to home")
            self._transition_to(State.HOME)
    
    ########################################################################################################################## handle_home_state
    ############################################################################################################################################
    def handle_home_state(self):
        """Handle HOME state - simplified: turn 180°, find ArUco marker, drive to it"""
        # Step 1: Turn 180 degrees (only once when entering this state)
        if not hasattr(self, 'return_turn_complete'):
            log_info(self.logger, "Returning to home: Turning 180 degrees...")
            self.motor.forward(config.MOTOR_STOP)  # Stop before turning
            self.servo.turn_left(1.0)  # Max left turn
            self.motor.forward(config.MOTOR_TURN)
            time.sleep(config.TURN_180_DURATION)  # Turn for specified duration
            self.servo.center()  # Center steering
            self.motor.stop()
            self.return_turn_complete = True
            log_info(self.logger, "Turn complete, scanning for ArUco marker...")
            return  # Exit early to allow turn to complete
        
        # Step 2: Scan for home marker using ArUco marker detection
        if self.aruco_detector is None:
            log_warning(self.logger, "ArUco detector not available", "Cannot return to home")
            self.motor.stop()
            self.servo.center()
            if hasattr(self, 'return_turn_complete'):
                delattr(self, 'return_turn_complete')
            self._transition_to(State.IDLE)
            return
        
        try:
            frame = self.visual.get_frame()
            # Convert RGB to BGR for ArUco detection (ArUco expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           
            # Detect ArUco marker
            detection = self.aruco_detector.detect_tag(frame_bgr)
            
            if detection['detected']:
                # Found ArUco marker!
                center_x = detection['center_x']
                frame_center_x = config.CAMERA_WIDTH // 2
                offset = center_x - frame_center_x
                distance_m = detection.get('distance_m')
                tag_id = detection.get('tag_id', 'N/A')
                is_centered = detection.get('is_centered', False)
                
                conditional_log(self.logger, 'info',
                              f"ArUco marker detected! ID: {tag_id}, Distance: {distance_m:.2f}m, Center: {center_x}, Centered: {is_centered}",
                              self.debug_mode)
                
                # Check if close enough to stop (using distance in meters)
                stop_distance_m = getattr(config, 'HOME_MARKER_STOP_DISTANCE_M', 0.3)  # Default 30cm
                if distance_m and distance_m < stop_distance_m:
                    # Close enough - stop!
                    log_info(self.logger, f"Reached home marker! Distance: {distance_m:.2f}m < {stop_distance_m}m. Stopping.")

                    self.motor.stop()  # Stop before turning
                    self.servo.turn_left(1.0)  # Max left turn
                    self.motor.forward(config.MOTOR_TURN)
                    time.sleep(config.TURN_180_DURATION)  # Turn for specified duration
                    self.servo.center()  # Center steering
                    self.motor.stop()

                    if hasattr(self, 'return_turn_complete'):
                        delattr(self, 'return_turn_complete')
                    self._transition_to(State.IDLE)
                    return
                
                # Drive towards marker
                # Use the angle calculated by ArUco detector (already in -45 to +45 range)
                angle = detection['angle']
                if angle is None:
                    # Fallback: calculate angle from center offset
                    angle = (offset / config.CAMERA_WIDTH) * 90.0
                    angle = max(-45.0, min(45.0, angle))
                
                # Set steering towards marker
                self.servo.set_angle(angle)
                
                # Adjust speed based on centering (similar to user following logic)
                if is_centered:
                    speed = config.MOTOR_MEDIUM  # Faster when centered
                else:
                    speed = config.MOTOR_SLOW  # Slower when turning
                
                self.motor.forward(speed)
                
                conditional_log(self.logger, 'debug',
                              f"Driving towards home marker: angle={angle:.1f}°, distance={distance_m:.2f}m, centered={is_centered}",
                              self.debug_mode)
            else:
                # Marker not found - search by turning slowly
                log_info(self.logger, "ArUco marker not found, searching...")
                # Turn slowly while searching
                self.servo.turn_left(0.3)  # Small left turn
                self.motor.forward(config.MOTOR_SLOW)  # Very slow forward
                
        except Exception as e:
            log_error(self.logger, e, "Error in return to home detection")
            # On error, just stop
            self.motor.stop()
            self.servo.center()
            if hasattr(self, 'return_turn_complete'):
                delattr(self, 'return_turn_complete')
            self._transition_to(State.IDLE)
    
    ################################################################################################################################### run(self)
    #############################################################################################################################################
    def run(self):
        """Main control loop"""
        try:
            while self.running:

                state = self.sm.get_state()

                # SAFETY: Check TOF sensor FIRST before any other processing
                # This ensures immediate emergency stop response
                if self.tof and self.tof.detect() and state != State.IDLE and state != State.STOPPED:   
                    if state == State.HOME: 
                        log_info(self.logger, "TRYING TO TURN, PLEASE MOVE AWAY FROM BIN DIESEL")
                        continue  # Skip all other processing this frame
                    log_info(self.logger, "=" * 70)
                    log_info(self.logger, "EMERGENCY STOP: TOF sensor triggered!")
                    log_info(self.logger, "=" * 70)
                    self.motor.stop()
                    self.servo.center()

                    # Transition to STOPPED state if currently in a movement state
                    if state in (State.FOLLOWING_USER, State.TRACKING_USER):
                        self._transition_to(State.STOPPED)

                    else: 
                        state = State.IDLE
                    time.sleep(0.05)  # Small delay to allow motor to stop
                    continue  # Skip all other processing this frame
                
                # Update performance monitor
                self.performance_monitor.update()
                self.frame_count += 1
                
                
            
                # Route to appropriate handler based on state
                if state == State.IDLE:
                    self.handle_idle_state()
                
                elif state == State.TRACKING_USER:
                    self.handle_tracking_user_state()
                
                elif state == State.FOLLOWING_USER:
                    self.handle_following_user_state()
                
                elif state == State.STOPPED:
                    self.handle_stopped_state()
                
                elif state == State.HOME:
                    self.handle_home_state()
                

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
        
        # Stop all components (with individual error handling to prevent one failure from stopping cleanup)
        if hasattr(self, 'wake_word'):
            try:
                self.wake_word.stop()
            except Exception as e:
                log_warning(self.logger, f"Error stopping wake word detector: {e}", "Cleanup")
        
        if hasattr(self, 'visual'):
            try:
                self.visual.stop()
            except Exception as e:
                log_warning(self.logger, f"Error stopping visual detector: {e}", "Cleanup")
        
        if hasattr(self, 'motor'):
            try:
                self.motor.cleanup()
            except Exception as e:
                log_warning(self.logger, f"Error cleaning up motor: {e}", "Cleanup")
        
        if hasattr(self, 'servo'):
            try:
                self.servo.cleanup()
            except Exception as e:
                log_warning(self.logger, f"Error cleaning up servo: {e}", "Cleanup")
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

