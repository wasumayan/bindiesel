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
# from voice_recognizer import VoiceRecognizer  # COMMENTED OUT - no voice commands
# from hand_gesture_controller import HandGestureController, get_gesture_command
from logger import setup_logger, log_error, log_warning, log_info, log_debug
from optimizations import FrameCache, PerformanceMonitor, conditional_log, skip_frames


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
        
        # Initialize voice recognizer - COMMENTED OUT (no voice commands)
        # NOTE: Defer initialization until needed to avoid audio device conflicts with wake word detector
        # The wake word detector needs exclusive access to the microphone
        # log_info(self.logger, "Voice recognizer will be initialized on-demand (to avoid audio conflicts)")
        # self.voice = None
        # self._voice_initialized = False
        
        # Initialize hand gesture controller (for manual mode) - COMMENTED OUT
        # NOTE: Defer initialization until manual mode is activated to avoid unnecessary initialization
        # The gesture controller will use frames from YOLOPoseTracker (shared camera)
        # log_info(self.logger, "Hand gesture controller will be initialized on-demand (when entering manual mode)")
        # self.gesture_controller = None
        # self._gesture_controller_initialized = False
        
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
            log_info(self.logger, f"STATE TRANSITION: {current_state.name} -> {new_state.name}")
            log_info(self.logger, "=" * 70)
        self.sm.transition_to(new_state)
        
      
    ################################################################################################################## _ensure_voice_initialized
    ##############################################################################################################################
    # def _ensure_voice_initialized(self):
    #     """Lazy initialization of voice recognizer to avoid audio conflicts - COMMENTED OUT"""
    #     if self._voice_initialized:
    #         return self.voice is not None
    #     
    #     self._voice_initialized = True
    #     log_info(self.logger, "Initializing voice recognizer on-demand...")
    #     try:
    #         # Get the same microphone device index as wake word detector
    #         device_index = None
    #         if hasattr(self, 'wake_word') and hasattr(self.wake_word, 'input_device_index'):
    #             device_index = self.wake_word.input_device_index
    #             log_info(self.logger, f"Using same microphone device as wake word detector (index {device_index})")
    #         
    #         self.voice = VoiceRecognizer(
    #             api_key=config.OPENAI_API_KEY,
    #             model=config.OPENAI_MODEL,
    #             device_index=device_index  # Use same device as wake word detector
    #         )
    #         log_info(self.logger, "Voice recognizer initialized successfully")
    #         return True
    #     except Exception as e:
    #         log_warning(self.logger, f"Failed to initialize voice recognizer: {e}", "Manual mode voice commands will not be available")
    #         self.voice = None
    #         return False
    
    # def _ensure_gesture_controller_initialized(self):
    #     """Lazy initialization of hand gesture controller (only when entering manual mode) - COMMENTED OUT"""
    #     if self._gesture_controller_initialized:
    #         return self.gesture_controller is not None
    #     
    #     self._gesture_controller_initialized = True
    #     log_info(self.logger, "Initializing hand gesture controller on-demand...")
    #     try:
    #         self.gesture_controller = HandGestureController(
    #             hand_model_path=config.YOLO_HAND_MODEL,  # Use trained hand-keypoints model if available
    #             pose_model_path=config.YOLO_POSE_MODEL,  # Fallback to pose model
    #             width=config.CAMERA_WIDTH,
    #             height=config.CAMERA_HEIGHT,
    #             confidence=config.YOLO_CONFIDENCE,
    #             gesture_hold_time=config.HAND_GESTURE_HOLD_TIME,
    #             skip_camera=True  # Skip camera initialization - we'll use YOLOPoseTracker's camera
    #         )
    #         log_info(self.logger, "Hand gesture controller initialized successfully (using shared camera frame)")
    #         return True
    #     except Exception as e:
    #         log_warning(self.logger, f"Failed to initialize hand gesture controller: {e}", "Manual mode hand gestures will not be available")
    #         self.gesture_controller = None
    #         return False

    ########################################################################################################################## handle_idle_state
    ##############################################################################################################################
    def handle_idle_state(self):
        """Handle IDLE state - wake word only, no voice recognizer (exclusive mic access)"""
        # Clean up voice recognizer - COMMENTED OUT (no voice commands)
        # if self._voice_initialized and self.voice:
        #     try:
        #         self.voice.cleanup()
        #         log_info(self.logger, "Voice recognizer cleaned up for IDLE state")
        #     except Exception as e:
        #         log_warning(self.logger, f"Error cleaning up voice recognizer: {e}", "IDLE state")
        #     self.voice = None
        #     self._voice_initialized = False
        
        # Ensure wake word detector is running (exclusive mic access)
        if self._wake_word_stopped: # hasattr(self, '_wake_word_stopped') and  ########################################## see if this breaks
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
            
            # Stop wake word detector - no longer needed after wake word detected
            try:
                self.wake_word.stop()
                self._wake_word_stopped = True
                log_info(self.logger, "Wake word detector stopped (wake word detected)")
            except Exception as e:
                log_warning(self.logger, f"Error stopping wake word detector: {e}", "State transition")
            
            # Transition to ACTIVE state for post-wake-word functionality
            self._transition_to(State.ACTIVE)
    
    ######################################################################################################################## handle_active_state
    ########################################################################################################################## 
    def handle_active_state(self):
        """Handle ACTIVE state - post-wake-word: look for user hitting the pose (arm raised)"""
        # Voice commands - COMMENTED OUT
        # Initialize voice recognizer if not already initialized (now that wake word is done)
        # if not self._voice_initialized:
        #     self._ensure_voice_initialized()
        
        # Get time since entering ACTIVE state
        # time_in_active = self.sm.get_time_in_state()
        # VOICE_LISTENING_TIMEOUT = 5.0  # Listen for voice commands for 5 seconds
        
        # For first 5 seconds: only check voice commands - COMMENTED OUT
        # if time_in_active < VOICE_LISTENING_TIMEOUT:
        #     if self.voice:
        #         try:
        #             # Listen for voice commands with remaining time
        #             remaining_time = VOICE_LISTENING_TIMEOUT - time_in_active
        #             # Use a reasonable timeout (at least 0.5s, but not more than remaining time)
        #             listen_timeout = max(0.5, min(remaining_time, 2.0))
        #             
        #             command = self.voice.recognize_command(timeout=listen_timeout)
        #             if command:
        #                 command_upper = command.upper()
        #                 # MANUAL_MODE and RADD_MODE commands - COMMENTED OUT
        #                 # if command_upper == 'MANUAL_MODE':
        #                 #     log_info(self.logger, "Manual mode activated via voice")
        #                 #     # Initialize gesture controller when entering manual mode
        #                 #     if not self._gesture_controller_initialized:
        #                 #         self._ensure_gesture_controller_initialized()
        #                 #     self.sm.transition_to(State.MANUAL_MODE)
        #                 #     self.current_manual_command = None
        #                 #     return
        #                 # elif command_upper == 'RADD_MODE':
        #                 #     log_info(self.logger, "RADD mode activated via voice")
        #                 #     self.sm.transition_to(State.RADD_MODE)
        #                 #     return
        #         except Exception as e:
        #             # Don't log timeout errors - they're expected when no speech
        #             if 'timeout' not in str(e).lower() and 'WaitTimeoutError' not in str(type(e).__name__):
        #                 log_error(self.logger, e, "Error in voice recognition")
        # else:
        #     # After 5 seconds: check visual detection for autonomous mode
        
        # Check visual detection for user with arm raised
        current_time = time.time()
        if current_time - self.last_visual_update > self.visual_update_interval: # update camera at intervals 
            try:
                # Use cached result if available and fresh (< 100ms old)
                if (self.cached_visual_result and 
                    (current_time - self.cached_visual_timestamp) < 0.1):
                    result = self.cached_visual_result
                else:
                    # In ACTIVE state, no target_track_id yet - will prioritize person with arm raised
                    result = self.visual.update(target_track_id=None)
                    self.cached_visual_result = result
                    self.cached_visual_timestamp = current_time
                
                self.last_visual_update = current_time
                
                # If both person and gesture detected, transition to TRACKING_USER
                if result['person_detected'] and result['arm_raised']:
                    # User raised arm - enter autonomous mode
                    log_info(self.logger, f"Person detected with arm raised! Track ID: {result.get('track_id', 'N/A')}, "
                                         f"Angle: {result.get('angle', 'N/A'):.1f}°")
                    self._transition_to(State.TRACKING_USER)
                    return  
                elif result['person_detected']:
                    conditional_log(self.logger, 'debug', 
                                  f"Person detected (no arm raised). Track ID: {result.get('track_id', 'N/A')}",
                                  self.debug_mode)
            except Exception as e:
                log_error(self.logger, e, "Error in visual detection update")
    
    ################################################################################################################# handle_tracking_user_state
    ########################################################################################################################## 
    
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
            self._transition_to(State.FOLLOWING_USER)
            conditional_log(self.logger, 'info',
                          f"User tracking confirmed (Track ID: {self.target_track_id}), starting to follow",
                          config.DEBUG_MODE)
        
        if not result['person_detected']: # IF USER IS LOST - STOP AND CONTINUE MONITORING
            conditional_log(self.logger, 'info', 
                          "User lost, stopping and searching...",
                          config.DEBUG_MODE)
            self.servo.center()
    
    ################################################################################################################ handle_following_user_state
    ########################################################################################################################## 
    
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""

        if not self.sm.old_state == self.sm.state:
            self.motor.forward(config.MOTOR_FAST) 
            self.sm.transition_to(State.TRACKING_USER)
        
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
                conditional_log(self.logger, 'info',
                              f"Target person (Track ID: {self.target_track_id}) not found, "
                              f"detected person has Track ID: {result.get('track_id', 'N/A')}",
                              config.DEBUG_MODE)
                result['person_detected'] = False  # Treat as person lost
        
        if not result['person_detected']:
            # User lost - stop car and revert to tracking state to search for user
            conditional_log(self.logger, 'info',
                          f"User lost during following (Track ID: {self.target_track_id}), stopping and searching...",
                          config.DEBUG_MODE)
            self.motor.stop()
            self.servo.center()
            self.target_track_id = None  # Clear target track_id
            # Revert to TRACKING_USER state to search for user again
            self._transition_to(State.TRACKING_USER)
            return
        
        if self.tof.detect():
            print("[Main] User reached (TOF sensor), stopping -------------------")
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
            # No angle data - stop
            self.motor.stop()
            self.servo.center()
            self._transition_to(State.TRACKING_USER)
            log_info(self.logger, "No angle data - returning to TRACKING_USER state")
    
    def handle_stopped_state(self):
        """Handle STOPPED state - at target distance, waiting for trash collection"""
        # Wait for fixed amount of time for trash placement, then go to HOME
        wait_time = 10.0  # Wait 10 seconds for trash placement
        if self.sm.get_time_in_state() > wait_time:
            log_info(self.logger, "Trash collection complete, returning to home")
            self._transition_to(State.HOME)
    
    def _check_color_match(self, frame, bbox, target_color):
        """
        Check if object in bounding box matches target color
        
        Args:
            frame: RGB frame
            bbox: Bounding box (x1, y1, x2, y2)
            target_color: Color name ('red', 'blue', 'green', etc.)
            
        Returns:
            float: Percentage of pixels matching color (0.0-1.0)
        """
        import cv2
        import numpy as np
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract object region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        # Convert RGB to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 50, 50]), np.array([30, 255, 255]))
            ],
            'orange': [
                (np.array([10, 50, 50]), np.array([20, 255, 255]))
            ],
            'purple': [
                (np.array([130, 50, 50]), np.array([160, 255, 255]))
            ],
            'pink': [
                (np.array([160, 50, 50]), np.array([170, 255, 255]))
            ]
        }
        
        target_color_lower = target_color.lower()
        if target_color_lower not in color_ranges:
            # Unknown color - return 0 (no match)
            return 0.0
        
        # Create mask for target color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges[target_color_lower]:
            mask += cv2.inRange(hsv, lower, upper)
        
        # Calculate percentage of pixels matching color
        total_pixels = mask.size
        matching_pixels = np.count_nonzero(mask)
        color_match_ratio = matching_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return color_match_ratio
    
    def _detect_home_marker(self, frame):
        """
        Detect home marker using YOLO object detection + color tracking
        Combines object class detection with color matching for robust detection
        
        Args:
            frame: RGB frame from camera
            
        Returns:
            dict with marker info: {'detected': bool, 'center_x': int, 'center_y': int, 'width': int, 'area': int, 'confidence': float, 'color_match': float}
        """
        if self.home_marker_model is None:
            return {'detected': False, 'center_x': None, 'center_y': None, 'width': None, 'area': None, 'confidence': None, 'color_match': None}
        
        try:
            # Run YOLO object detection
            results = self.home_marker_model(
                frame,
                conf=config.HOME_MARKER_CONFIDENCE,
                verbose=False,
                imgsz=config.YOLO_INFERENCE_SIZE,
                max_det=config.YOLO_MAX_DET
            )
            
            if not results or len(results) == 0:
                return {'detected': False, 'center_x': None, 'center_y': None, 'width': None, 'area': None, 'confidence': None, 'color_match': None}
            
            result = results[0]
            
            # Check if we have any detections
            if result.boxes is None or len(result.boxes) == 0:
                return {'detected': False, 'center_x': None, 'center_y': None, 'width': None, 'area': None, 'confidence': None, 'color_match': None}
            
            # Look for the specified object class (e.g., 'box') that also matches the target color
            target_class = config.HOME_MARKER_OBJECT_CLASS.lower()
            target_color = config.HOME_MARKER_COLOR.lower()
            best_detection = None
            best_score = 0.0  # Combined score: confidence * color_match
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.home_marker_model.names[class_id].lower()
                
                # Check if this is the target class
                if target_class in class_name or class_name in target_class:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Check color match
                    color_match_ratio = self._check_color_match(frame, (x1, y1, x2, y2), target_color)
                    
                    # Object must match color threshold
                    if color_match_ratio >= config.HOME_MARKER_COLOR_THRESHOLD:
                        # Combined score: confidence weighted by color match
                        combined_score = confidence * color_match_ratio
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            width = x2 - x1
                            height = y2 - y1
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            area = width * height
                            
                            best_detection = {
                                'detected': True,
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': width,
                                'height': height,
                                'area': area,
                                'confidence': confidence,
                                'color_match': color_match_ratio,
                                'class_name': class_name
                            }
            
            if best_detection:
                return best_detection
            else:
                return {'detected': False, 'center_x': None, 'center_y': None, 'width': None, 'area': None, 'confidence': None, 'color_match': None}
                
        except Exception as e:
            log_error(self.logger, e, "Error in home marker detection")
            return {'detected': False, 'center_x': None, 'center_y': None, 'width': None, 'area': None, 'confidence': None, 'color_match': None}
    
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
            log_info(self.logger, f"Turn complete, scanning for home marker (object: {config.HOME_MARKER_OBJECT_CLASS})...")
            return  # Exit early to allow turn to complete
        
        # Step 2: Scan for home marker using YOLO object detection
        try:
            frame = self.visual.get_frame()
            marker = self._detect_home_marker(frame)
            
            if marker['detected']:
                # Found home marker!
                center_x = marker['center_x']
                frame_center_x = config.CAMERA_WIDTH // 2
                offset = center_x - frame_center_x
                marker_width = marker['width']
                
                color_match = marker.get('color_match', 0.0)
                class_name = marker.get('class_name', 'unknown')
                confidence = marker.get('confidence', 0.0)
                conditional_log(self.logger, 'info',
                              f"Home marker detected! Class: {class_name}, Confidence: {confidence:.2f}, Color match: {color_match:.1%}, Center: {center_x}, Width: {marker_width}px",
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
    
    # def handle_manual_mode_state(self):
    #     """Handle MANUAL_MODE state - waiting for voice commands and hand gestures - COMMENTED OUT"""
    #     # Initialize gesture controller on-demand when entering manual mode
    #     if not self._gesture_controller_initialized:
    #         self._ensure_gesture_controller_initialized()
    #     
    #     # Voice recognizer should already be initialized after wake word detection
    #     # (initialized in handle_idle_state when wake word is detected)
    #     voice_command = None
    #     if self.voice:
    #         voice_command = self.voice.recognize_command(timeout=0.1)  # Quick check
    #     
    #     # Check for hand gesture command (if available)
    #     gesture_command = None
    #     if self.gesture_controller:
    #         try:
    #             # Get frame from pose tracker (shared camera, use cached frame)
    #             frame = self.frame_cache.get(self.visual.get_frame)
    #             gesture_command = self.gesture_controller.detect_command(frame)
    #         except Exception as e:
    #             conditional_log(self.logger, 'debug',
    #                           f"Gesture detection error: {e}",
    #                           self.debug_mode)
    #     
    #     # Process commands (voice takes priority, then gesture)
    #     command = voice_command or gesture_command
    #     
    #     if command:
    #         if command == 'AUTOMATIC_MODE':
    #             # Return to automatic mode
    #             print("[Main] Returning to automatic mode")
    #             self.current_manual_command = None
    #             self.motor.stop()
    #             self.servo.center()
    #             self._transition_to(State.IDLE)
    #         elif command == 'STOP':
    #             # Stop current command
    #             print(f"[Main] Stopping current command (from {'voice' if voice_command else 'gesture'})")
    #             self.current_manual_command = None
    #             self.motor.stop()
    #             self.servo.center()
    #         else:
    #             # New command received
    #             source = 'voice' if voice_command else 'gesture'
    #             print(f"[Main] New command received: {command} (from {source})")
    #             self.current_manual_command = command
    #     
    #     # If no voice or gesture controller available, return to idle
    #     if not self.voice and not self.gesture_controller:
    #         log_info(self.logger, "No input method available, returning to IDLE state")
    #         self.sm.transition_to(State.IDLE)
    #         return
    #     
    #     # Execute current command continuously until new command/stop
    #     if self.current_manual_command:
    #         self.execute_manual_command_continuous(self.current_manual_command)
    # 
    # def execute_manual_command_continuous(self, command):
    #     """Execute manual command continuously until stopped - COMMENTED OUT"""
    #     if command == 'FORWARD':
    #         self.motor.forward(config.MOTOR_MEDIUM)
    #         self.servo.center()
    #     
    #     elif command == 'LEFT':
    #         self.motor.forward(config.MOTOR_SLOW)
    #         self.servo.turn_left(0.5)
    #     
    #     elif command == 'RIGHT':
    #         self.motor.forward(config.MOTOR_SLOW)
    #         self.servo.turn_right(0.5)
    #     
    #     elif command == 'TURN_AROUND':
    #         # Turn around is a one-time action
    #         if not hasattr(self, 'turn_around_complete'):
    #             self.motor.stop()
    #             self.servo.turn_left(1.0)
    #             time.sleep(2)  # Turn around
    #             self.servo.center()
    #             self.turn_around_complete = True
    #             # After turn around, continue forward
    #             self.current_manual_command = 'FORWARD'
    #             self.motor.forward(config.MOTOR_MEDIUM)
    #     
    #     # Reset turn around flag if command changed
    #     if command != 'TURN_AROUND' and hasattr(self, 'turn_around_complete'):
    #         delattr(self, 'turn_around_complete')
    
    # def handle_radd_mode_state(self):
    #     """Handle RADD_MODE state - drive towards users violating dress code - COMMENTED OUT"""
    #     if not self.radd_detector:
    #         log_warning(self.logger, "RADD detector not available, returning to IDLE state")
    #         self.sm.transition_to(State.IDLE)
    #         return
    #     
    #     # Get pose detection results (use cached frame if available)
    #     try:
    #         # Use frame cache to avoid redundant captures
    #         frame = self.frame_cache.get(self.visual.get_frame)
    #         results, yolo_result = self.visual.detect(frame)
    #     except Exception as e:
    #         log_error(self.logger, e, "Error in visual detection for RADD mode")
    #         return
    #     
    #     # Get tracked persons from results
    #     tracked_persons = {}
    #     for pose in results.get('poses', []):
    #         track_id = pose.get('track_id')
    #         if track_id is not None:
    #             tracked_persons[track_id] = {
    #                 'box': pose.get('box'),
    #                 'keypoints': pose.get('keypoints'),
    #                 'angle': pose.get('angle'),
    #                 'is_centered': pose.get('is_centered')
    #             }
    #     
    #     # Detect violations for all tracked persons and maintain state
    #     violation_info = self.radd_detector.detect_violations_for_tracked_persons(
    #         frame, 
    #         tracked_persons
    #     )
    #     
    #     active_violators = violation_info['active_violators']
    #     tracked_violators = violation_info['tracked_violators']
    #     
    #     if not active_violators:
    #         # No active violators - stop and wait
    #         if self.sm.get_time_in_state() > 5.0:  # Wait 5 seconds
    #             conditional_log(self.logger, 'info',
    #                           "No RADD violations detected, returning to idle",
    #                           config.DEBUG_MODE)
    #             self.motor.stop()
    #             self.servo.center()
    #             self._transition_to(State.IDLE)
    #         return
    #     
    #     # Select target violator (prioritize most recent or closest)
    #     target_violator_id = None
    #     target_violator_info = None
    #     
    #     # Strategy: Follow the violator we've been tracking longest (most persistent)
    #     # Or the one currently in frame
    #     for violator_id in active_violators:
    #         violator_info = tracked_violators[violator_id]
    #         # Check if this violator is currently in frame
    #         if violator_id in tracked_persons:
    #             target_violator_id = violator_id
    #             target_violator_info = violator_info
    #             # Get current position from tracked persons
    #             person_data = tracked_persons[violator_id]
    #             target_violator_info['current_box'] = person_data['box']
    #             target_violator_info['current_angle'] = person_data.get('angle')
    #             target_violator_info['is_centered'] = person_data.get('is_centered', False)
    #             break
    #     
    #     # If no violator in current frame, use most recent one
    #     if target_violator_id is None and tracked_violators:
    #         # Get most recently seen violator
    #         most_recent = max(tracked_violators.items(), 
    #                         key=lambda x: x[1]['last_seen'])
    #         target_violator_id, target_violator_info = most_recent
    #         conditional_log(self.logger, 'debug',
    #                       f"Following violator {target_violator_id} (not in current frame)",
    #                       config.DEBUG_MODE)
    #     
    #     # Drive towards the target violator
    #     if target_violator_info:
    #         # Check TOF sensor for emergency stop
    #         if self.tof and config.EMERGENCY_STOP_ENABLED:
    #             distance = self.tof.read_distance()
    #             if self.tof.is_emergency_stop():
    #                 conditional_log(self.logger, 'info',
    #                               "EMERGENCY STOP: Object too close in RADD mode!",
    #                               config.DEBUG_MODE)
    #                 self.motor.stop()
    #                 self.servo.center()
    #                 return
    #             
    #             if self.tof.is_too_close():
    #                 conditional_log(self.logger, 'info',
    #                               f"Target violator {target_violator_id} reached, stopping",
    #                               config.DEBUG_MODE)
    #                 self.motor.stop()
    #                 self.servo.center()
    #                 
    #                 # "Yell" at violator - display violation message prominently
    #                 violation_type = []
    #                 if target_violator_info['no_full_pants']:
    #                     violation_type.append("SHORTS/NO PANTS")
    #                 if target_violator_info['no_closed_toe_shoes']:
    #                     violation_type.append("NON-CLOSED-TOE SHOES")
    #                 
    #                 violation_text = " AND ".join(violation_type) if violation_type else "DRESS CODE VIOLATION"
    #                 
    #                 # Log prominently
    #                 print("\n" + "=" * 70)
    #                 print(f"⚠️  RADD VIOLATION DETECTED ⚠️")
    #                 print("=" * 70)
    #                 print(f"Person ID: {target_violator_id}")
    #                 print(f"Violation: {violation_text}")
    #                 print(f"Confidence: {target_violator_info['confidence']:.2f}")
    #                 print(f"First Detected: {target_violator_info.get('first_detected', 'N/A')}")
    #                 print("=" * 70)
    #                 print("🚨 DRESS CODE VIOLATION - PLEASE COMPLY 🚨")
    #                 print("=" * 70 + "\n")
    #                 
    #                 log_info(self.logger, 
    #                        f"RADD VIOLATION: Person {target_violator_id} - {violation_text}")
    #                 
    #                 # TODO: Add audio "yelling" here (TTS or pre-recorded audio)
    #                 # Example: self.audio_player.play("dress_code_violation.wav")
    #                 
    #                 return
    #         
    #         # Calculate steering based on violator position
    #         person_box = target_violator_info.get('current_box') or target_violator_info.get('person_box')
    #         if person_box:
    #             x1, y1, x2, y2 = person_box
    #             person_center_x = (x1 + x2) / 2
    #             frame_center_x = config.CAMERA_WIDTH / 2
    #             offset = person_center_x - frame_center_x
    #             angle = (offset / config.CAMERA_WIDTH) * 102.0  # Camera FOV
    #             
    #             # Convert angle to steering
    #             steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
    #             steering_position = max(-1.0, min(1.0, steering_position))
    #             
    #             # Move towards target violator
    #             if abs(offset) < config.PERSON_CENTER_THRESHOLD:
    #                 # Violator is centered - move forward
    #                 speed = config.FOLLOW_SPEED
    #                 self.motor.forward(speed)
    #                 self.servo.center()
    #                 conditional_log(self.logger, 'debug',
    #                               f"RADD: Violator {target_violator_id} centered, moving forward",
    #                               config.DEBUG_MODE)
    #             else:
    #                 # Violator not centered - turn towards them
    #                 speed = config.FOLLOW_SPEED * 0.7
    #                 self.motor.forward(speed)
    #                 self.servo.set_angle(angle)
    #                 conditional_log(self.logger, 'debug',
    #                               f"RADD: Turning towards violator {target_violator_id} (angle: {angle:.1f}°)",
    #                               config.DEBUG_MODE)
    
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
                
                # elif state == State.MANUAL_MODE:  # COMMENTED OUT
                #     self.handle_manual_mode_state()
                # 
                # elif state == State.RADD_MODE:  # COMMENTED OUT
                #     self.handle_radd_mode_state()
                
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
        
        # if hasattr(self, 'gesture_controller') and self.gesture_controller:  # COMMENTED OUT
        #     try:
        #         # Only stop if it has its own camera (shouldn't if we're sharing)
        #         if hasattr(self.gesture_controller, 'picam2') and self.gesture_controller.picam2:
        #             self.gesture_controller.stop()
        #     except Exception as e:
        #         log_warning(self.logger, f"Error stopping gesture controller: {e}", "Cleanup")
        
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
        
        # Voice recognizer cleanup - COMMENTED OUT (no voice commands)
        # if hasattr(self, 'voice') and self.voice:
        #     try:
        #         self.voice.cleanup()
        #     except Exception as e:
        #         log_warning(self.logger, f"Error cleaning up voice recognizer: {e}", "Cleanup")
        
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

