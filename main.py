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
# Use YOLO pose tracker instead of basic visual detector
from test_yolo_pose_tracking import YOLOPoseTracker
from motor_controller import MotorController
from servo_controller import ServoController
from tof_sensor import TOFSensor
from voice_recognizer import VoiceRecognizer
from path_tracker import PathTracker
from hand_gesture_controller import HandGestureController, get_gesture_command
from radd_detector import RADDDetector
from logger import setup_logger, log_error, log_warning, log_info, log_debug
from optimizations import FrameCache, PerformanceMonitor, conditional_log, skip_frames


class BinDieselSystem:
    """Main system controller"""
    
    def __init__(self):
        """Initialize all system components"""
        # Set up logging
        self.logger = setup_logger(__name__)
        
        log_info(self.logger, "=" * 70)
        log_info(self.logger, "Bin Diesel System Initializing...")
        log_info(self.logger, "=" * 70)
        
        # Initialize state machine
        self.state_machine = StateMachine(
            tracking_timeout=config.TRACKING_TIMEOUT
        )
        
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
            self.tof = TOFSensor(
                stop_distance_mm=config.TOF_STOP_DISTANCE_MM,
                emergency_distance_mm=config.TOF_EMERGENCY_DISTANCE_MM
            )
            log_info(self.logger, "TOF sensor initialized successfully")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize TOF sensor: {e}", "Continuing without TOF sensor (safety feature disabled)")
            self.tof = None
        
        # Initialize voice recognizer (for manual mode)
        log_info(self.logger, "Initializing voice recognizer...")
        try:
            self.voice = VoiceRecognizer(
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_MODEL
            )
            log_info(self.logger, "Voice recognizer initialized successfully")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize voice recognizer: {e}", "Manual mode voice commands will not be available")
            self.voice = None
        
        # Initialize hand gesture controller (for manual mode)
        # Note: We'll share the camera frame from pose tracker to avoid duplicate cameras
        log_info(self.logger, "Initializing hand gesture controller...")
        try:
            self.gesture_controller = HandGestureController(
                hand_model_path=config.YOLO_HAND_MODEL,  # Use trained hand-keypoints model if available
                pose_model_path=config.YOLO_POSE_MODEL,  # Fallback to pose model
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                confidence=config.YOLO_CONFIDENCE,
                gesture_hold_time=config.HAND_GESTURE_HOLD_TIME
            )
            # Stop the gesture controller's camera since we'll use pose tracker's frame
            # This avoids having two cameras open simultaneously
            if hasattr(self.gesture_controller, 'picam2') and self.gesture_controller.picam2:
                self.gesture_controller.picam2.stop()
                self.gesture_controller.picam2.close()
                self.gesture_controller.picam2 = None
            log_info(self.logger, "Hand gesture controller initialized (using shared camera frame)")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize hand gesture controller: {e}", "Manual mode hand gestures will not be available")
            self.gesture_controller = None
        
        # Initialize RADD detector (for RADD mode)
        log_info(self.logger, "Initializing RADD detector...")
        try:
            self.radd_detector = RADDDetector(
                model_path=config.YOLO_CLOTHING_MODEL,
                confidence=config.YOLO_CONFIDENCE
            )
            log_info(self.logger, "RADD detector initialized successfully")
        except Exception as e:
            log_warning(self.logger, f"Failed to initialize RADD detector: {e}", "RADD mode will not be available")
            self.radd_detector = None
        
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
        
        # Manual mode state
        self.current_manual_command = None  # Current active manual command
        self.last_command_time = 0  # Time of last command execution
        
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
        log_info(self.logger, "Waiting for wake word: 'bin diesel'")
        log_info(self.logger, "Available modes: autonomous, manual, radd")
        log_info(self.logger, "Press Ctrl+C to exit")
        log_info(self.logger, "=" * 70)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        log_info(self.logger, "Shutdown signal received, cleaning up...")
        self.running = False
    
    def handle_idle_state(self):
        """Handle IDLE state - waiting for wake word"""
        if self.wake_word.detect():
            self.state_machine.transition_to(State.ACTIVE)
            print("[Main] System activated!")
            if self.debug_mode:
                print("[Main] DEBUG: Wake word detected, transitioning to ACTIVE")
    
    def handle_active_state(self):
        """Handle ACTIVE state - waiting for mode selection"""
        # Check for voice commands (if voice available)
        if self.voice:
            try:
                command = self.voice.recognize_command(timeout=0.1)  # Quick check
                if command:
                    command_upper = command.upper()
                    if command_upper == 'MANUAL_MODE':
                        log_info(self.logger, "Manual mode activated via voice")
                        self.state_machine.transition_to(State.MANUAL_MODE)
                        self.current_manual_command = None
                        return
                    elif command_upper in ['RADD MODE', 'RADD', 'RAD MODE']:
                        log_info(self.logger, "RADD mode activated via voice")
                        self.state_machine.transition_to(State.RADD_MODE)
                        self.state_machine.set_start_position("origin")
                        self.path_tracker.start_tracking()
                        return
            except Exception as e:
                log_error(self.logger, e, "Error in voice recognition")
        
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
                    self.state_machine.transition_to(State.TRACKING_USER)
                    self.state_machine.set_start_position("origin")  # Store starting position
                    self.path_tracker.start_tracking()  # Start tracking path
                    conditional_log(self.logger, 'info', 
                                  "Autonomous mode: User detected with raised arm",
                                  config.DEBUG_MODE)
            except Exception as e:
                log_error(self.logger, e, "Error in visual detection update")
    
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
            if self.state_machine.is_timeout():
                conditional_log(self.logger, 'info', 
                              "User tracking timeout, returning to idle",
                              config.DEBUG_MODE)
                self.state_machine.transition_to(State.IDLE)
                self.motor.stop()
                self.servo.center()
            return
        
        if result['arm_raised']:
            # User has arm raised - start following
            self.state_machine.transition_to(State.FOLLOWING_USER)
            conditional_log(self.logger, 'info',
                          "User tracking confirmed, starting to follow",
                          config.DEBUG_MODE)
    
    def handle_following_user_state(self):
        """Handle FOLLOWING_USER state - moving toward user"""
        # Check TOF sensor for emergency stop
        if self.tof and config.EMERGENCY_STOP_ENABLED:
            distance = self.tof.read_distance()
            if self.debug_mode and config.DEBUG_TOF:
                print(f"[Main] DEBUG: TOF distance: {distance/10:.1f}cm" if distance else "[Main] DEBUG: TOF read error")
            
            if self.tof.is_emergency_stop():
                print("[Main] EMERGENCY STOP: Object too close!")
                if self.debug_mode:
                    print(f"[Main] DEBUG: Emergency stop triggered at {distance/10:.1f}cm")
                self.motor.stop()
                self.servo.center()
                return
        
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
            if self.state_machine.is_timeout():
                conditional_log(self.logger, 'info',
                              "User lost, returning to idle",
                              config.DEBUG_MODE)
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
                speed = config.FOLLOW_SPEED * 0.7
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
            self.state_machine.transition_to(State.IDLE)
    
    def handle_manual_mode_state(self):
        """Handle MANUAL_MODE state - waiting for voice commands and hand gestures"""
        # Check for voice command (if available)
        voice_command = None
        if self.voice:
            voice_command = self.voice.recognize_command(timeout=0.1)  # Quick check
        
        # Check for hand gesture command (if available)
        gesture_command = None
        if self.gesture_controller:
            try:
                # Get frame from pose tracker (shared camera, use cached frame)
                frame = self.frame_cache.get(self.visual.get_frame)
                gesture_command = self.gesture_controller.detect_command(frame)
            except Exception as e:
                conditional_log(self.logger, 'debug',
                              f"Gesture detection error: {e}",
                              self.debug_mode)
        
        # Process commands (voice takes priority, then gesture)
        command = voice_command or gesture_command
        
        if command:
            if command == 'AUTOMATIC_MODE':
                # Return to automatic mode
                print("[Main] Returning to automatic mode")
                self.current_manual_command = None
                self.motor.stop()
                self.servo.center()
                self.state_machine.transition_to(State.ACTIVE)
            elif command == 'STOP':
                # Stop current command
                print(f"[Main] Stopping current command (from {'voice' if voice_command else 'gesture'})")
                self.current_manual_command = None
                self.motor.stop()
                self.servo.center()
            else:
                # New command received
                source = 'voice' if voice_command else 'gesture'
                print(f"[Main] New command received: {command} (from {source})")
                self.current_manual_command = command
                self.last_command_time = time.time()
        
        # If no voice or gesture controller available, return to idle
        if not self.voice and not self.gesture_controller:
            print("[Main] No input method available, returning to idle")
            self.state_machine.transition_to(State.IDLE)
            return
        
        # Execute current command continuously until new command/stop
        if self.current_manual_command:
            self.execute_manual_command_continuous(self.current_manual_command)
    
    def execute_manual_command_continuous(self, command):
        """Execute manual command continuously until stopped"""
        if command == 'FORWARD':
            self.motor.forward(config.MOTOR_MEDIUM)
            self.servo.center()
        
        elif command == 'LEFT':
            self.motor.forward(config.MOTOR_SLOW)
            self.servo.turn_left(0.5)
        
        elif command == 'RIGHT':
            self.motor.forward(config.MOTOR_SLOW)
            self.servo.turn_right(0.5)
        
        elif command == 'TURN_AROUND':
            # Turn around is a one-time action
            if not hasattr(self, 'turn_around_complete'):
                self.motor.stop()
                self.servo.turn_left(1.0)
                time.sleep(2)  # Turn around
                self.servo.center()
                self.turn_around_complete = True
                # After turn around, continue forward
                self.current_manual_command = 'FORWARD'
                self.motor.forward(config.MOTOR_MEDIUM)
        
        # Reset turn around flag if command changed
        if command != 'TURN_AROUND' and hasattr(self, 'turn_around_complete'):
            delattr(self, 'turn_around_complete')
    
    def handle_radd_mode_state(self):
        """Handle RADD_MODE state - drive towards users violating dress code"""
        if not self.radd_detector:
            log_warning(self.logger, "RADD detector not available, returning to idle")
            self.state_machine.transition_to(State.IDLE)
            return
        
        # Get pose detection results (use cached frame if available)
        try:
            # Use frame cache to avoid redundant captures
            frame = self.frame_cache.get(self.visual.get_frame)
            results, yolo_result = self.visual.detect(frame)
        except Exception as e:
            log_error(self.logger, e, "Error in visual detection for RADD mode")
            return
        
        # Get tracked persons from results
        tracked_persons = {}
        for pose in results.get('poses', []):
            track_id = pose.get('track_id')
            if track_id is not None:
                tracked_persons[track_id] = {
                    'box': pose.get('box'),
                    'keypoints': pose.get('keypoints'),
                    'angle': pose.get('angle'),
                    'is_centered': pose.get('is_centered')
                }
        
        # Detect violations for all tracked persons and maintain state
        violation_info = self.radd_detector.detect_violations_for_tracked_persons(
            frame, 
            tracked_persons
        )
        
        active_violators = violation_info['active_violators']
        tracked_violators = violation_info['tracked_violators']
        
        if not active_violators:
            # No active violators - stop and wait
            if self.state_machine.get_time_in_state() > 5.0:  # Wait 5 seconds
                conditional_log(self.logger, 'info',
                              "No RADD violations detected, returning to idle",
                              config.DEBUG_MODE)
                self.motor.stop()
                self.servo.center()
                self.state_machine.transition_to(State.IDLE)
            return
        
        # Select target violator (prioritize most recent or closest)
        target_violator_id = None
        target_violator_info = None
        
        # Strategy: Follow the violator we've been tracking longest (most persistent)
        # Or the one currently in frame
        for violator_id in active_violators:
            violator_info = tracked_violators[violator_id]
            # Check if this violator is currently in frame
            if violator_id in tracked_persons:
                target_violator_id = violator_id
                target_violator_info = violator_info
                # Get current position from tracked persons
                person_data = tracked_persons[violator_id]
                target_violator_info['current_box'] = person_data['box']
                target_violator_info['current_angle'] = person_data.get('angle')
                target_violator_info['is_centered'] = person_data.get('is_centered', False)
                break
        
        # If no violator in current frame, use most recent one
        if target_violator_id is None and tracked_violators:
            # Get most recently seen violator
            most_recent = max(tracked_violators.items(), 
                            key=lambda x: x[1]['last_seen'])
            target_violator_id, target_violator_info = most_recent
            conditional_log(self.logger, 'debug',
                          f"Following violator {target_violator_id} (not in current frame)",
                          config.DEBUG_MODE)
        
        # Drive towards the target violator
        if target_violator_info:
            # Check TOF sensor for emergency stop
            if self.tof and config.EMERGENCY_STOP_ENABLED:
                distance = self.tof.read_distance()
                if self.tof.is_emergency_stop():
                    conditional_log(self.logger, 'info',
                                  "EMERGENCY STOP: Object too close in RADD mode!",
                                  config.DEBUG_MODE)
                    self.motor.stop()
                    self.servo.center()
                    return
                
                if self.tof.is_too_close():
                    conditional_log(self.logger, 'info',
                                  f"Target violator {target_violator_id} reached, stopping",
                                  config.DEBUG_MODE)
                    self.motor.stop()
                    self.servo.center()
                    
                    # "Yell" at violator - display violation message prominently
                    violation_type = []
                    if target_violator_info['no_full_pants']:
                        violation_type.append("SHORTS/NO PANTS")
                    if target_violator_info['no_closed_toe_shoes']:
                        violation_type.append("NON-CLOSED-TOE SHOES")
                    
                    violation_text = " AND ".join(violation_type) if violation_type else "DRESS CODE VIOLATION"
                    
                    # Log prominently
                    print("\n" + "=" * 70)
                    print(f"⚠️  RADD VIOLATION DETECTED ⚠️")
                    print("=" * 70)
                    print(f"Person ID: {target_violator_id}")
                    print(f"Violation: {violation_text}")
                    print(f"Confidence: {target_violator_info['confidence']:.2f}")
                    print(f"First Detected: {target_violator_info.get('first_detected', 'N/A')}")
                    print("=" * 70)
                    print("🚨 DRESS CODE VIOLATION - PLEASE COMPLY 🚨")
                    print("=" * 70 + "\n")
                    
                    log_info(self.logger, 
                           f"RADD VIOLATION: Person {target_violator_id} - {violation_text}")
                    
                    # TODO: Add audio "yelling" here (TTS or pre-recorded audio)
                    # Example: self.audio_player.play("dress_code_violation.wav")
                    
                    return
            
            # Calculate steering based on violator position
            person_box = target_violator_info.get('current_box') or target_violator_info.get('person_box')
            if person_box:
                x1, y1, x2, y2 = person_box
                person_center_x = (x1 + x2) / 2
                frame_center_x = config.CAMERA_WIDTH / 2
                offset = person_center_x - frame_center_x
                angle = (offset / config.CAMERA_WIDTH) * 102.0  # Camera FOV
                
                # Convert angle to steering
                steering_position = (angle / 45.0) * config.ANGLE_TO_STEERING_GAIN
                steering_position = max(-1.0, min(1.0, steering_position))
                
                # Move towards target violator
                if abs(offset) < config.PERSON_CENTER_THRESHOLD:
                    # Violator is centered - move forward
                    speed = config.FOLLOW_SPEED
                    self.motor.forward(speed)
                    self.servo.center()
                    conditional_log(self.logger, 'debug',
                                  f"RADD: Violator {target_violator_id} centered, moving forward",
                                  config.DEBUG_MODE)
                else:
                    # Violator not centered - turn towards them
                    speed = config.FOLLOW_SPEED * 0.7
                    self.motor.forward(speed)
                    self.servo.set_angle(angle)
                    conditional_log(self.logger, 'debug',
                                  f"RADD: Turning towards violator {target_violator_id} (angle: {angle:.1f}°)",
                                  config.DEBUG_MODE)
                
                # Track path
                segment_duration = time.time() - self.last_command_time if self.last_command_time > 0 else 0.1
                self.path_tracker.add_segment(speed, steering_position, segment_duration)
                self.last_command_time = time.time()
    
    def run(self):
        """Main control loop"""
        try:
            while self.running:
                # Update performance monitor
                self.performance_monitor.update()
                self.frame_count += 1
                
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
                
                elif state == State.RADD_MODE:
                    self.handle_radd_mode_state()
                
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

