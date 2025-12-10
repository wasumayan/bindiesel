"""
RADD Mode Detector
Detects users not wearing full pants or closed-toe shoes
Uses YOLO clothing detection model for accurate detection

Reference: https://github.com/kesimeg/YOLO-Clothing-Detection
"""

import numpy as np
import cv2
import config
from logger import setup_logger

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    YOLO = None

logger = setup_logger(__name__)


class RADDDetector:
    """
    Detects users violating dress code using YOLO clothing detection:
    - Not wearing full pants (shorts, skirt, or no pants)
    - Not wearing closed-toe shoes (sandals, flip-flops, barefoot)
    
    Uses trained YOLO model for clothing detection (clothing, shoes, bags, accessories)
    Reference: https://github.com/kesimeg/YOLO-Clothing-Detection
    """
    
    def __init__(self, model_path=None, confidence=0.25):
        """
        Initialize RADD detector with clothing detection model
        
        Args:
            model_path: Path to trained clothing detection model (default: from config)
            confidence: Detection confidence threshold
        """
        self.logger = logger
        self.confidence = confidence
        
        # Get model path from config or use default
        if model_path is None:
            model_path = getattr(config, 'YOLO_CLOTHING_MODEL', None)
        
        # Initialize clothing detection model
        if YOLO is None:
            self.logger.error("Ultralytics not available, RADD detector will use fallback heuristics")
            self.model = None
            self.use_heuristics = True
        else:
            try:
                if model_path:
                    self.logger.info(f"Loading clothing detection model: {model_path}")
                    self.model = YOLO(model_path)
                    self.use_heuristics = False
                    self.logger.info("Clothing detection model loaded successfully")
                else:
                    self.logger.warning("No clothing model path specified, using heuristics fallback")
                    self.logger.warning("To use clothing detection, set YOLO_CLOTHING_MODEL in config.py")
                    self.model = None
                    self.use_heuristics = True
            except Exception as e:
                self.logger.error(f"Failed to load clothing model: {e}, using heuristics fallback")
                self.model = None
                self.use_heuristics = True
        
        # Class mappings for clothing detection model
        # Based on: https://github.com/kesimeg/YOLO-Clothing-Detection
        # Categories: Clothing, Shoes, Bags, Accessories
        self.clothing_classes = {
            'clothing': 0,  # Full clothing (pants, shirts, etc.)
            'shoes': 1,     # Shoes
            'bags': 2,      # Bags
            'accessories': 3  # Accessories
        }
        
        # Person tracking and violation state
        # Track violations per person ID to maintain state across frames
        self.tracked_violators = {}  # track_id -> violation_info
        self.violation_timeout = getattr(config, 'RADD_VIOLATION_TIMEOUT', 2.0)  # Seconds before removing from tracked violators if not seen
        
        # For determining violations:
        # - Full pants = "clothing" detected in lower body region
        # - Closed-toe shoes = "shoes" detected (we'll need to classify shoe type)
        # For now, we'll assume any "shoes" detection means closed-toe (can be improved)
        
        self.logger.info(f"RADD detector initialized (mode: {'clothing_model' if not self.use_heuristics else 'heuristics'})")
    
    def detect_violations_for_tracked_persons(self, frame, tracked_persons):
        """
        Detect violations for multiple tracked persons and maintain violation state
        
        Args:
            frame: RGB frame
            tracked_persons: dict of {track_id: person_data} from pose tracker
                           person_data should have 'box' and optionally 'keypoints'
            
        Returns:
            dict with:
            {
                'violators': list of violating person IDs,
                'violations': dict of {track_id: violation_info},
                'active_violators': list of currently active violator IDs
            }
        """
        import time
        current_time = time.time()
        
        # Clean up old violators (not seen recently)
        active_violator_ids = []
        for track_id, violator_info in list(self.tracked_violators.items()):
            if current_time - violator_info['last_seen'] < self.violation_timeout:
                active_violator_ids.append(track_id)
            else:
                del self.tracked_violators[track_id]
                self.logger.debug(f"Removed violator {track_id} (timeout)")
        
        # Check each tracked person for violations
        current_violators = []
        violations = {}
        
        for track_id, person_data in tracked_persons.items():
            person_box = person_data.get('box')
            keypoints = person_data.get('keypoints')
            
            if person_box is None:
                continue
            
            # Detect violation for this person
            violation_result = self.detect_violation(
                frame, 
                person_box=person_box, 
                keypoints=keypoints
            )
            
            if violation_result['violation_detected']:
                # Update or add to tracked violators
                self.tracked_violators[track_id] = {
                    'track_id': track_id,
                    'person_box': person_box,
                    'violation_result': violation_result,
                    'first_detected': self.tracked_violators.get(track_id, {}).get('first_detected', current_time),
                    'last_seen': current_time,
                    'no_full_pants': violation_result['no_full_pants'],
                    'no_closed_toe_shoes': violation_result['no_closed_toe_shoes'],
                    'confidence': violation_result['confidence']
                }
                
                current_violators.append(track_id)
                violations[track_id] = violation_result
                
                # Log new violator prominently
                if track_id not in active_violator_ids:
                    violation_details = []
                    if violation_result['no_full_pants']:
                        violation_details.append("SHORTS/NO PANTS")
                    if violation_result['no_closed_toe_shoes']:
                        violation_details.append("NON-CLOSED-TOE SHOES")
                    
                    violation_text = " + ".join(violation_details)
                    self.logger.info(f"🚨 NEW RADD VIOLATOR: Person ID={track_id}")
                    self.logger.info(f"   Violations: {violation_text}")
                    self.logger.info(f"   Confidence: {violation_result['confidence']:.2f}")
                    print(f"\n🚨 RADD VIOLATOR DETECTED: Person {track_id}")
                    print(f"   Violations: {violation_text}\n")
            else:
                # Person is compliant - remove from violators if was previously violating
                if track_id in self.tracked_violators:
                    self.logger.info(f"Person {track_id} is now compliant, removing from violators")
                    del self.tracked_violators[track_id]
        
        return {
            'violators': current_violators,
            'violations': violations,
            'active_violators': list(self.tracked_violators.keys()),  # All tracked violators
            'tracked_violators': self.tracked_violators  # Full violator info
        }
    
    def detect_violation(self, frame, person_box=None, keypoints=None):
        """
        Detect if person violates dress code using clothing detection model
        
        Args:
            frame: RGB frame (can be full frame or person ROI)
            person_box: Optional bounding box (x1, y1, x2, y2) to extract person region
            keypoints: Optional YOLO pose keypoints (for fallback heuristics)
            
        Returns:
            dict with:
            {
                'violation_detected': bool,
                'no_full_pants': bool,
                'no_closed_toe_shoes': bool,
                'confidence': float (0.0-1.0),
                'details': dict with detection details
            }
        """
        # Use clothing detection model if available
        if not self.use_heuristics and self.model is not None:
            return self._detect_violation_with_model(frame, person_box)
        else:
            # Fallback to heuristics if model not available
            if keypoints is None:
                return {
                    'violation_detected': False,
                    'no_full_pants': False,
                    'no_closed_toe_shoes': False,
                    'confidence': 0.0,
                    'details': {'error': 'No keypoints available for heuristics'}
                }
            return self._detect_violation_with_heuristics(keypoints, person_box)
    
    def get_tracked_violator(self, track_id):
        """
        Get information about a tracked violator
        
        Args:
            track_id: Person tracking ID
            
        Returns:
            dict with violator info or None if not found
        """
        return self.tracked_violators.get(track_id)
    
    def get_all_violators(self):
        """
        Get all currently tracked violators
        
        Returns:
            dict of {track_id: violator_info}
        """
        return self.tracked_violators.copy()
    
    def _detect_violation_with_model(self, frame, person_box):
        """
        Detect violations using YOLO clothing detection model
        
        Args:
            frame: RGB frame
            person_box: Optional bounding box to extract person region
            
        Returns:
            dict with violation detection results
        """
        # Extract person region if box provided
        if person_box:
            x1, y1, x2, y2 = person_box
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                person_roi = frame[y1:y2, x1:x2]
            else:
                person_roi = frame
        else:
            person_roi = frame
        
        # Run clothing detection
        try:
            results = self.model(
                person_roi,
                conf=self.confidence,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Clothing detection error: {e}")
            return {
                'violation_detected': False,
                'no_full_pants': False,
                'no_closed_toe_shoes': False,
                'confidence': 0.0,
                'details': {'error': f'Detection failed: {e}'}
            }
        
        if not results or len(results) == 0:
            return {
                'violation_detected': False,
                'no_full_pants': False,
                'no_closed_toe_shoes': False,
                'confidence': 0.0,
                'details': {'detections': 0}
            }
        
        result = results[0]
        detections = {
            'clothing': [],
            'shoes': [],
            'bags': [],
            'accessories': []
        }
        
        # Process detections
        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id].lower()
                
                # Map to our categories
                if 'cloth' in class_name:
                    detections['clothing'].append({
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })
                elif 'shoe' in class_name or 'footwear' in class_name:
                    detections['shoes'].append({
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })
                elif 'bag' in class_name:
                    detections['bags'].append({
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })
                elif 'accessor' in class_name:
                    detections['accessories'].append({
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })
        
        # Determine violations
        # Strategy: Analyze clothing and shoe detections to identify violations
        
        clothing_detected = len(detections['clothing']) > 0
        shoes_detected = len(detections['shoes']) > 0
        
        # For pants detection:
        # - If clothing is detected in lower body region, likely full pants
        # - If no clothing detected OR clothing only in upper body, likely shorts/skirt
        # We'll check if clothing is detected in lower half of person box
        no_full_pants = True  # Default to violation unless proven otherwise
        
        if clothing_detected and person_box:
            # Check if clothing is in lower body region (bottom 60% of person box)
            x1, y1, x2, y2 = person_box
            person_height = y2 - y1
            lower_body_y_start = y1 + int(person_height * 0.4)  # Lower 60% of body
            
            for clothing in detections['clothing']:
                box = clothing['box']
                # Check if clothing box overlaps with lower body region
                clothing_y_center = (box[1] + box[3]) / 2  # y center of clothing box
                if clothing_y_center > lower_body_y_start:
                    # Clothing detected in lower body - likely full pants
                    no_full_pants = False
                    break
        elif not clothing_detected:
            # No clothing detected at all - likely shorts/skirt or no pants
            no_full_pants = True
        
        # For shoes detection:
        # - If shoes detected, assume closed-toe (can be improved with better model)
        # - If no shoes detected, assume violation (barefoot, sandals, flip-flops)
        # Note: To distinguish closed-toe vs open-toe, would need model trained on specific shoe types
        no_closed_toe_shoes = not shoes_detected
        
        violation_detected = no_full_pants or no_closed_toe_shoes
        
        # Calculate overall confidence
        all_confidences = []
        for category in detections.values():
            for det in category:
                all_confidences.append(det['confidence'])
        
        overall_confidence = max(all_confidences) if all_confidences else 0.0
        
        details = {
            'clothing_detected': clothing_detected,
            'shoes_detected': shoes_detected,
            'detections': detections,
            'method': 'clothing_model'
        }
        
        if violation_detected and config.DEBUG_MODE:
            self.logger.debug(f"RADD violation (model): pants={no_full_pants}, shoes={no_closed_toe_shoes}, "
                            f"conf={overall_confidence:.2f}, clothing={clothing_detected}, shoes={shoes_detected}")
        
        return {
            'violation_detected': violation_detected,
            'no_full_pants': no_full_pants,
            'no_closed_toe_shoes': no_closed_toe_shoes,
            'confidence': overall_confidence,
            'details': details
        }
    
    def _detect_violation_with_heuristics(self, keypoints, person_box):
        """
        Fallback: Detect violations using pose keypoint heuristics
        (Original implementation)
        """
        # YOLO pose keypoint indices
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        # 15: left_ankle, 16: right_ankle
        
        # Check keypoint confidence threshold
        min_confidence = 0.3
        
        # Detect "no full pants" (shorts/skirt/bare legs)
        no_full_pants = self._detect_no_full_pants(keypoints, min_confidence)
        
        # Detect "no closed-toe shoes" (sandals/flip-flops/barefoot)
        no_closed_toe_shoes = self._detect_no_closed_toe_shoes(keypoints, min_confidence)
        
        violation_detected = no_full_pants or no_closed_toe_shoes
        
        # Calculate confidence based on keypoint visibility
        confidence = self._calculate_confidence(keypoints, min_confidence)
        
        details = {
            'no_full_pants': no_full_pants,
            'no_closed_toe_shoes': no_closed_toe_shoes,
            'keypoint_confidence': confidence,
            'method': 'heuristics'
        }
        
        if violation_detected and config.DEBUG_MODE:
            self.logger.debug(f"RADD violation (heuristics): pants={no_full_pants}, shoes={no_closed_toe_shoes}, conf={confidence:.2f}")
        
        return {
            'violation_detected': violation_detected,
            'no_full_pants': no_full_pants,
            'no_closed_toe_shoes': no_closed_toe_shoes,
            'confidence': confidence,
            'details': details
        }
    
    def _detect_no_full_pants(self, keypoints, min_confidence):
        """
        Detect if person is not wearing full pants
        Heuristic: If leg is visible (hip->knee->ankle visible) and 
        there's a large gap or angle suggesting shorts/skirt
        
        Args:
            keypoints: YOLO pose keypoints
            min_confidence: Minimum keypoint confidence
            
        Returns:
            bool: True if no full pants detected
        """
        # Keypoint indices
        left_hip_idx = 11
        right_hip_idx = 12
        left_knee_idx = 13
        right_knee_idx = 14
        left_ankle_idx = 15
        right_ankle_idx = 16
        
        # Check both legs
        left_leg_visible = (
            keypoints[left_hip_idx][2] >= min_confidence and
            keypoints[left_knee_idx][2] >= min_confidence and
            keypoints[left_ankle_idx][2] >= min_confidence
        )
        
        right_leg_visible = (
            keypoints[right_hip_idx][2] >= min_confidence and
            keypoints[right_knee_idx][2] >= min_confidence and
            keypoints[right_ankle_idx][2] >= min_confidence
        )
        
        if not (left_leg_visible or right_leg_visible):
            return False  # Can't determine without visible legs
        
        violations = []
        
        # Check left leg
        if left_leg_visible:
            hip = np.array([keypoints[left_hip_idx][0], keypoints[left_hip_idx][1]])
            knee = np.array([keypoints[left_knee_idx][0], keypoints[left_knee_idx][1]])
            ankle = np.array([keypoints[left_ankle_idx][0], keypoints[left_ankle_idx][1]])
            
            # Calculate leg segment lengths
            upper_leg = np.linalg.norm(knee - hip)
            lower_leg = np.linalg.norm(ankle - knee)
            total_leg = np.linalg.norm(ankle - hip)
            
            # Heuristic: If upper leg is relatively short compared to total leg,
            # it might indicate shorts (more of leg is visible below knee)
            # Also check if knee is high relative to hip (suggesting shorts)
            if total_leg > 0:
                upper_leg_ratio = upper_leg / total_leg
                # For full pants, upper leg should be ~50-60% of total
                # For shorts, upper leg might be shorter relative to visible leg
                if upper_leg_ratio < 0.45:  # More leg visible below knee
                    violations.append(True)
                
                # Check vertical position: if knee is close to hip vertically,
                # might indicate shorts (more leg visible)
                vertical_gap = abs(knee[1] - hip[1])
                if vertical_gap < total_leg * 0.3:  # Knee close to hip
                    violations.append(True)
        
        # Check right leg
        if right_leg_visible:
            hip = np.array([keypoints[right_hip_idx][0], keypoints[right_hip_idx][1]])
            knee = np.array([keypoints[right_knee_idx][0], keypoints[right_knee_idx][1]])
            ankle = np.array([keypoints[right_ankle_idx][0], keypoints[right_ankle_idx][1]])
            
            upper_leg = np.linalg.norm(knee - hip)
            lower_leg = np.linalg.norm(ankle - knee)
            total_leg = np.linalg.norm(ankle - hip)
            
            if total_leg > 0:
                upper_leg_ratio = upper_leg / total_leg
                if upper_leg_ratio < 0.45:
                    violations.append(True)
                
                vertical_gap = abs(knee[1] - hip[1])
                if vertical_gap < total_leg * 0.3:
                    violations.append(True)
        
        # If at least one leg shows violation, return True
        return len(violations) > 0
    
    def _detect_no_closed_toe_shoes(self, keypoints, min_confidence):
        """
        Detect if person is not wearing closed-toe shoes
        Heuristic: Check ankle position and angle relative to ground
        If ankle is at extreme angle or foot appears "open", likely sandals/flip-flops
        
        Args:
            keypoints: YOLO pose keypoints
            min_confidence: Minimum keypoint confidence
            
        Returns:
            bool: True if no closed-toe shoes detected
        """
        left_ankle_idx = 15
        right_ankle_idx = 16
        left_knee_idx = 13
        right_knee_idx = 14
        
        violations = []
        
        # Check left foot
        if (keypoints[left_ankle_idx][2] >= min_confidence and
            keypoints[left_knee_idx][2] >= min_confidence):
            
            ankle = np.array([keypoints[left_ankle_idx][0], keypoints[left_ankle_idx][1]])
            knee = np.array([keypoints[left_knee_idx][0], keypoints[left_knee_idx][1]])
            
            # Calculate angle of lower leg from vertical
            leg_vector = ankle - knee
            if np.linalg.norm(leg_vector) > 0:
                # Angle from vertical (0° = straight down, 90° = horizontal)
                angle_from_vertical = np.arctan2(abs(leg_vector[0]), abs(leg_vector[1])) * 180 / np.pi
                
                # If foot is at extreme angle (splayed out), might indicate open footwear
                # Closed-toe shoes typically keep foot more aligned
                if angle_from_vertical > 25:  # Foot splayed out significantly
                    violations.append(True)
        
        # Check right foot
        if (keypoints[right_ankle_idx][2] >= min_confidence and
            keypoints[right_knee_idx][2] >= min_confidence):
            
            ankle = np.array([keypoints[right_ankle_idx][0], keypoints[right_ankle_idx][1]])
            knee = np.array([keypoints[right_knee_idx][0], keypoints[right_knee_idx][1]])
            
            leg_vector = ankle - knee
            if np.linalg.norm(leg_vector) > 0:
                angle_from_vertical = np.arctan2(abs(leg_vector[0]), abs(leg_vector[1])) * 180 / np.pi
                
                if angle_from_vertical > 25:
                    violations.append(True)
        
        # If at least one foot shows violation, return True
        return len(violations) > 0
    
    def _calculate_confidence(self, keypoints, min_confidence):
        """
        Calculate overall confidence based on keypoint visibility
        
        Args:
            keypoints: YOLO pose keypoints
            min_confidence: Minimum confidence threshold
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Key leg/foot keypoints
        relevant_indices = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
        
        confidences = []
        for idx in relevant_indices:
            if keypoints[idx][2] >= min_confidence:
                confidences.append(keypoints[idx][2])
        
        if not confidences:
            return 0.0
        
        # Average confidence of visible keypoints
        return sum(confidences) / len(confidences)

