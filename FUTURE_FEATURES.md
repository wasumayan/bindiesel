# Future Features Ideas

This document outlines potential fun features to add to Bin Diesel.

## 1. Shorts Detection & "Attack" Mode

**Concept**: Detect if a person is wearing shorts instead of pants, and "attack" them (playfully drive toward them).

### Implementation Ideas:

```python
# In visual_detector.py
def detect_clothing(self, person_box, frame):
    """
    Detect if person is wearing shorts vs pants
    Uses YOLO to detect person, then analyzes lower body region
    """
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    
    # Extract lower body region (bottom 40% of person)
    lower_body = frame[int(y1 + h*0.6):y2, x1:x2]
    
    # Analyze color/texture patterns
    # Shorts typically show more skin (legs visible)
    # Pants cover legs completely
    
    # Method 1: Color analysis
    # Look for skin tones in lower body region
    skin_mask = detect_skin_color(lower_body)
    skin_percentage = np.sum(skin_mask) / lower_body.size
    
    # Method 2: Edge detection
    # Shorts show more edges (legs visible)
    edges = cv2.Canny(lower_body, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Threshold: if skin percentage > 20% or edge density > 15%, likely shorts
    is_shorts = (skin_percentage > 0.20) or (edge_density > 0.15)
    
    return is_shorts
```

### State Machine Addition:
- New state: `ATTACKING_SHORTS_WEARER`
- Trigger: Person detected + shorts detected
- Behavior: Drive toward person at medium speed, stop at safe distance

---

## 2. Ball Detection & Goal Scoring

**Concept**: Detect a moving ball, track it, and drive into it to "score" toward a goal.

### Implementation Ideas:

```python
# In visual_detector.py
def detect_ball(self, frame):
    """
    Detect ball using YOLO (sports ball class) or color detection
    """
    # Method 1: YOLO detection
    results = self.yolo_model(frame, conf=0.3, verbose=False)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = self.yolo_model.names[class_id]
        if 'ball' in class_name.lower() or class_name == 'sports ball':
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            return (x1, y1, x2, y2)
    
    # Method 2: Color-based detection (if YOLO fails)
    # Detect common ball colors (white, orange, etc.)
    return detect_colored_ball(frame, color='white')
```

```python
# Ball tracking
class BallTracker:
    """Track ball position and predict movement"""
    
    def __init__(self):
        self.ball_history = deque(maxlen=10)
        self.ball_velocity = (0, 0)
    
    def update(self, ball_position):
        """Update ball tracking"""
        if ball_position:
            self.ball_history.append(ball_position)
            if len(self.ball_history) > 1:
                # Calculate velocity
                prev_pos = self.ball_history[-2]
                curr_pos = self.ball_history[-1]
                self.ball_velocity = (
                    curr_pos[0] - prev_pos[0],
                    curr_pos[1] - prev_pos[1]
                )
    
    def predict_intercept_point(self):
        """Predict where to intercept the ball"""
        if len(self.ball_history) < 2:
            return None
        
        # Simple linear prediction
        last_pos = self.ball_history[-1]
        intercept_x = last_pos[0] + self.ball_velocity[0] * 5  # Predict 5 frames ahead
        intercept_y = last_pos[1] + self.ball_velocity[1] * 5
        
        return (intercept_x, intercept_y)
```

### State Machine Addition:
- New state: `BALL_TRACKING`
- New state: `INTERCEPTING_BALL`
- New state: `SCORING_GOAL`
- Behavior:
  1. Detect ball
  2. Track ball movement
  3. Calculate intercept point
  4. Drive toward intercept point
  5. When close, drive into ball
  6. Celebrate (blink lights, play sound, etc.)

---

## 3. Obstacle Course Navigation

**Concept**: Navigate through an obstacle course using TOF sensors and visual detection.

### Implementation Ideas:

```python
# Multiple TOF sensors for 360° detection
class MultiTOFSensor:
    """Multiple TOF sensors for obstacle avoidance"""
    
    def __init__(self):
        self.front_sensor = TOFSensor()  # Front
        self.left_sensor = TOFSensor()   # Left side
        self.right_sensor = TOFSensor()  # Right side
    
    def get_obstacle_map(self):
        """Get obstacle distances in all directions"""
        return {
            'front': self.front_sensor.read_distance(),
            'left': self.left_sensor.read_distance(),
            'right': self.right_sensor.read_distance()
        }
    
    def find_best_direction(self):
        """Find direction with most clearance"""
        distances = self.get_obstacle_map()
        max_distance = max(distances.values())
        best_direction = max(distances, key=distances.get)
        return best_direction, max_distance
```

---

## 4. Voice-Controlled Behaviors

**Concept**: Add more voice commands for fun behaviors.

### New Commands:
- "DANCE" - Perform a dance routine (spin, move back/forth)
- "FOLLOW ME" - Enter follow mode
- "STAY" - Stop and wait
- "COME HERE" - Drive toward user
- "SPIN" - Rotate in place
- "REVERSE" - Move backward

---

## 5. Multi-Object Tracking

**Concept**: Track multiple people/objects simultaneously.

### Implementation:
```python
# Enhanced tracking for multiple objects
class MultiObjectTracker:
    """Track multiple objects simultaneously"""
    
    def __init__(self):
        self.trackers = {}  # {object_id: SimpleTracker}
    
    def update(self, detections):
        """Update tracking for all objects"""
        # Match detections to existing trackers
        # Create new trackers for unmatched detections
        # Remove old trackers
        pass
```

---

## 6. Gesture-Based Control

**Concept**: More gestures beyond arm raising.

### New Gestures:
- **Two arms raised**: "Come here"
- **Pointing left/right**: Turn in that direction
- **Waving**: Acknowledge/hello
- **Thumbs up**: Good job/continue
- **Stop sign (hand up)**: Stop immediately

### Implementation:
```python
def detect_gesture(self, person_box, frame):
    """Detect various gestures"""
    # Use MediaPipe or custom detection
    # Analyze hand positions, arm angles, etc.
    gestures = {
        'two_arms_raised': detect_two_arms_raised(),
        'pointing_left': detect_pointing_left(),
        'pointing_right': detect_pointing_right(),
        'waving': detect_waving(),
        'thumbs_up': detect_thumbs_up(),
        'stop_sign': detect_stop_sign()
    }
    return gestures
```

---

## 7. Sound Effects & Feedback

**Concept**: Add audio feedback for different actions.

### Implementation:
```python
# Audio feedback system
class AudioFeedback:
    """Play sounds for different events"""
    
    def __init__(self):
        self.sounds = {
            'wake': 'wake_sound.wav',
            'person_detected': 'person_detected.wav',
            'arm_raised': 'arm_raised.wav',
            'too_close': 'warning.wav',
            'goal_scored': 'celebration.wav',
            'shorts_detected': 'attack_sound.wav'
        }
    
    def play(self, sound_name):
        """Play sound effect"""
        if sound_name in self.sounds:
            os.system(f"aplay {self.sounds[sound_name]}")
```

---

## 8. LED Status Indicators

**Concept**: Visual feedback using LEDs.

### Implementation:
```python
# LED control
class LEDController:
    """Control status LEDs"""
    
    def __init__(self, pin):
        self.pin = pin
        GPIO.setup(pin, GPIO.OUT)
    
    def set_color(self, color):
        """Set LED color (if RGB LED)"""
        # Red, Green, Blue, Yellow, etc.
        pass
    
    def blink(self, times=3, interval=0.5):
        """Blink LED"""
        for _ in range(times):
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(interval)
```

---

## Implementation Priority

1. **High Priority** (Easy to implement):
   - Sound effects
   - LED indicators
   - More voice commands

2. **Medium Priority** (Moderate complexity):
   - Ball detection (using YOLO)
   - Multi-object tracking
   - More gestures

3. **Low Priority** (Complex):
   - Shorts detection (requires custom ML model)
   - Obstacle course navigation (needs multiple sensors)
   - Advanced gesture recognition (needs MediaPipe/OpenPose)

---

## Notes

- All features should have safety checks (TOF sensor, emergency stop)
- Features can be enabled/disabled via config.py
- Test each feature individually before integrating
- Keep code modular for easy addition/removal

