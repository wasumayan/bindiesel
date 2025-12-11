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
from test_yolo_pose_tracking import YOLOPoseTracker
from motor_controller import MotorController
from servo_controller import ServoController
from tof_sensor import ToFSensor
from logger import setup_logger, log_error, log_warning, log_info, log_debug
from optimizations import FrameCache, PerformanceMonitor, conditional_log, skip_frames
from home_marker_detector import detect_red_box


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
        
        # Initialize YOLO pose tracker 
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
        
        # Initialize YOLO object detection model for home marker detection
        log_info(self.logger, "Initializing YOLO object detection for home marker...")
        try:
            from ultralytics import YOLO
            try:
                self.home_marker_model = YOLO(config.YOLO_MODEL)  # Try NCNN or PyTorch from config
                log_info(self.logger, f"YOLO object detection model initialized: {config.YOLO_MODEL}")
            except Exception as e1:
                # Fallback: if NCNN failed, try PyTorch
                if config.USE_NCNN and config.YOLO_MODEL.endswith('_ncnn_model'):
                    fallback_path = config.YOLO_MODEL.replace('_ncnn_model', '.pt')
                    log_info(self.logger, f"NCNN model not found, trying PyTorch: {fallback_path}")
                    self.home_marker_model = YOLO(fallback_path)
                    log_info(self.logger, f"PyTorch model loaded: {fallback_path}")
                else:
                    raise e1
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize YOLO object detection: {e}", "Home marker detection will not work")
            self.home_marker_model = None
        
        # Initialize motor controller
        log_info(self.logger, "Initializing motor controller...")
        try:
            self.motor = MotorController(
                pwm_pin=config.MOTOR_PWM_PIN,
                frequency=config.PWM_FREQUENCY
            )
            log_info(self.logger, "Motor controller initialized successfully")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize motor controller")
            self.cleanup()
            sys.exit(1)
        
        # Initialize servo controller
        log_info(self.logger, "Initializing servo controller...")
        try:
            self.servo = ServoController(
                pwm_pin=config.SERVO_PWM_PIN,
                frequency=config.PWM_FREQUENCY,
                center_duty=config.SERVO_CENTER,
                left_max_duty=config.SERVO_LEFT_MAX,
                right_max_duty=config.SERVO_RIGHT_MAX
            )
            log_info(self.logger, "Servo controller initialized successfully")
        except Exception as e:
            log_error(self.logger, e, "Failed to initialize servo controller")
            self.cleanup()
            sys.exit(1)
        
        # Initialize TOF sensor
        log_info(self.logger, "Initializing TOF sensor...")
        try:
            self.tof = ToFSensor()
            log_info(self.logger, "TOF sensor initialized successfully")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize TOF sensor: {e}", "Continuing without TOF sensor (safety feature disabled)")
            self.tof = None
        
        # Tracking state - store target track_id to ensure we follow the same person
        self.target_track_id = None  # Track ID of the person we're following
        
        # Performance optimizations
        self.frame_cache = FrameCache(max_age=0.05)  # Cache frames for 50ms
        self.performance_monitor = PerformanceMonitor()
        self.frame_count = 0
        self.cached_visual_result = None  # Cache visual detection results
        self.cached_visual_timestamp = 0
        self.frame_skip_counter = 0  # Counter for frame skipping
        # self.current_manual_command = None  # Current active manual command
        
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
            log_info(self.logger, "=" * 70)
            log_info(self.logger, "=" * 70)
            log_info(self.logger, f"STATE TRANSITION: {current_state.name} -> {new_state.name}")
            log_info(self.logger, "=" * 70)
            log_info(self.logger, "=" * 70)
        self.sm.transition_to(new_state)
        
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
                          f"User tracking confirmed (Track ID: {self.target_track_id}), starting to follow",
                          config.DEBUG_MODE)
            self._transition_to(State.FOLLOWING_USER)
            return
        
    ################################################################################################################ handle_following_user_state
    ############################################################################################################################################
    
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""

        if not self.sm.old_state == self.sm.state:
            self.motor.forward(config.MOTOR_FAST) 
            self.sm.transition_to(State.TRACKING_USER)
            conditional_log(self.logger, 'info', f"Motor forward start at speed {config.MOTOR_FAST}", config.DEBUG_MODE)
        
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
        
        # Check if we're still tracking the same person (by track_id)
        if self.target_track_id is not None:
            # Only follow if the detected person matches our target track_id
            if result.get('track_id') != self.target_track_id:
                # Different person detected - treat as person lost
                result['person_detected'] = False  # Treat as person lost
        
        if not result['person_detected']:
            # User lost - stop car and revert to tracking state to search for user
            log_info(self.logger, "User lost during following, stopping and searching...")
            self.motor.stop()
            self.servo.center()
            self.target_track_id = None  # Clear target track_id
            # Revert to TRACKING_USER state to search for user again
            self._transition_to(State.TRACKING_USER)
            return
        
        if self.tof.detect():
            log_info(self.logger, "=" * 70)
            log_info(self.logger, "=" * 70)
            print("[Main] User reached (TOF sensor), stopping")
            conditional_log(self.logger, 'debug', f"ToF = {self.tof.detect()}", config.DEBUG_MODE)
            log_info(self.logger, "=" * 70)
            log_info(self.logger, "=" * 70)
            self.motor.stop()
            self.servo.center()
            self._transition_to(State.STOPPED)
            return
        
        # Calculate steering based on angle
        if result['angle'] is not None:
            angle = result['angle']
            
            conditional_log(self.logger, 'debug',
                          f"Person angle: {angle:.1f}°, centered: {result['is_centered']}",
                          self.debug_mode and config.DEBUG_VISUAL)

            steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
            steering_position = max(-1.0, min(1.0, steering_position))
            
            conditional_log(self.logger, 'debug',
                          f"Setting servo angle: {angle:.1f}° (position: {steering_position:.2f})",
                          self.debug_mode and config.DEBUG_SERVO)
            
            self.servo.set_angle(angle)
            
            # Adjust speed based on how centered user is
            if result['is_centered']:
                # User is centered - move forward
                speed = config.MOTOR_FAST
                conditional_log(self.logger, 'debug',
                              f"User centered, moving forward at {speed*100:.0f}%",
                              self.debug_mode and config.DEBUG_MOTOR)
                self.motor.forward(speed)
            else:
                # User not centered - slow down while turning
                speed = config.MOTOR_MEDIUM 
                conditional_log(self.logger, 'debug',
                              f"User not centered, moving forward at {speed*100:.0f}% while turning",
                              self.debug_mode and config.DEBUG_MOTOR)
                self.motor.forward(speed)
                
        else:
            # No angle data, approaching use? 
            self.motor.forward(config.MOTOR_SLOW)
            self.servo.center()
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
        """Handle HOME state - simplified: turn 180°, find red square, drive to it"""
        # Step 1: Turn 180 degrees (only once when entering this state)
        if not hasattr(self, 'return_turn_complete'):
            log_info(self.logger, "Returning to home: Turning 180 degrees...")
            self.motor.stop()  # Stop before turning
            # Turn left max (or right max - you can change this)
            self.servo.turn_left(1.0)  # Max left turn
            time.sleep(config.TURN_180_DURATION)  # Turn for specified duration
            self.servo.center()  # Center steering
            self.return_turn_complete = True
            log_info(self.logger, "Turn complete, scanning for red square object...")
            return  # Exit early to allow turn to complete
        
        # Step 2: Scan for home marker using YOLO object detection + OpenCV color tracking (hardcoded for red square)
        try:
            frame = self.visual.get_frame()
            # Use home_marker_detector module (hardcoded for red square)
            marker = detect_red_box(
                self.home_marker_model,
                frame,
                confidence_threshold=config.HOME_MARKER_CONFIDENCE,
                color_threshold=config.HOME_MARKER_COLOR_THRESHOLD,
                square_aspect_ratio_tolerance=0.3  # 30% tolerance for square shape
            )
            
            if marker['detected']:
                # Found home marker!
                center_x = marker['center_x']
                frame_center_x = config.CAMERA_WIDTH // 2
                offset = center_x - frame_center_x
                marker_width = marker['width']
                
                color_match = marker.get('color_match', 0.0)
                aspect_ratio = marker.get('aspect_ratio', 0.0)
                confidence = marker.get('confidence', 0.0)
                conditional_log(self.logger, 'info',
                              f"Red square detected! Confidence: {confidence:.2f}, Color match: {color_match:.1%}, Aspect ratio: {aspect_ratio:.2f}, Center: {center_x}, Width: {marker_width}px",
                              self.debug_mode)
                
                # Check if close enough to stop
                if marker_width >= config.HOME_MARKER_STOP_DISTANCE:
                    # Close enough - stop!
                    log_info(self.logger, "Reached home marker! Stopping.")
                    self.motor.stop()
                    self.servo.center()
                    # Clean up return state
                    if hasattr(self, 'return_turn_complete'):
                        delattr(self, 'return_turn_complete')
                    self._transition_to(State.IDLE)
                    return
                
                # Drive towards marker
                # Calculate steering angle based on marker position
                angle = (offset / config.CAMERA_WIDTH) * 90.0  # Convert to angle
                angle = max(-45.0, min(45.0, angle))  # Clamp to servo range
                
                # Set steering towards marker
                self.servo.set_angle(angle)
                
                # Move forward at slow speed
                self.motor.forward(config.MOTOR_SLOW)
                
                conditional_log(self.logger, 'debug',
                              f"Driving towards home marker: angle={angle:.1f}°, width={marker_width}px",
                              self.debug_mode)
            else:
                # Marker not found - search by turning slowly
                log_info(self.logger, "Home marker not found, searching...")
                # Turn slowly while searching
                self.servo.turn_left(0.3)  # Small left turn
                self.motor.forward(config.MOTOR_SLOW * 0.5)  # Very slow forward
                
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
                
                elif state == State.HOME:
                    self.handle_home_state()
                
                elif state == State.RETURNING_TO_START:  # Legacy - redirects to HOME
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

