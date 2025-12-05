"""
Servo Controller Module
Controls steering servo using PWM signals via GPIO
"""

import time
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("WARNING: RPi.GPIO not available (not on Raspberry Pi?)")
    GPIO_AVAILABLE = False
    # Create mock GPIO for testing
    class MockGPIO:
        BCM = 'BCM'
        OUT = 'OUT'
        @staticmethod
        def setmode(mode):
            pass
        @staticmethod
        def setup(pin, mode):
            pass
        @staticmethod
        def PWM(pin, freq):
            class MockPWM:
                def start(self, duty):
                    print(f"[MockGPIO] Servo PWM started at {duty*100:.1f}%")
                def ChangeDutyCycle(self, duty):
                    print(f"[MockGPIO] Servo PWM changed to {duty*100:.1f}%")
                def stop(self):
                    print("[MockGPIO] Servo PWM stopped")
            return MockPWM()
        @staticmethod
        def cleanup():
            pass
    GPIO = MockGPIO


class ServoController:
    """Controls steering servo using PWM"""
    
    def __init__(self, pwm_pin, frequency=50, center_duty=0.075, 
                 left_max_duty=0.05, right_max_duty=0.10):
        """
        Initialize servo controller
        
        Args:
            pwm_pin: GPIO pin number (BCM) for servo PWM
            frequency: PWM frequency in Hz (default 50Hz for servos)
            center_duty: Duty cycle for center position (default 7.5%)
            left_max_duty: Duty cycle for full left (default 5%)
            right_max_duty: Duty cycle for full right (default 10%)
        """
        self.pwm_pin = pwm_pin
        self.frequency = frequency
        self.center_duty = center_duty
        self.left_max_duty = left_max_duty
        self.right_max_duty = right_max_duty
        self.pwm = None
        self.current_position = 0.0  # -1.0 (left) to 1.0 (right), 0.0 = center
        
        if GPIO_AVAILABLE:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pwm_pin, GPIO.OUT)
            
            # Create PWM instance
            self.pwm = GPIO.PWM(pwm_pin, frequency)
            self.pwm.start(center_duty * 100)  # Start at center
            print(f"[ServoController] Initialized on GPIO {pwm_pin} at {frequency}Hz")
            print(f"[ServoController] Center: {center_duty*100:.1f}%, "
                  f"Left: {left_max_duty*100:.1f}%, Right: {right_max_duty*100:.1f}%")
        else:
            print(f"[ServoController] Mock mode - GPIO {pwm_pin} would be used")
    
    def set_position(self, position):
        """
        Set servo position
        
        Args:
            position: Position from -1.0 (full left) to 1.0 (full right)
                     0.0 = center
        """
        # Clamp position to valid range
        position = max(-1.0, min(1.0, position))
        
        # Calculate duty cycle
        if position < 0:  # Left
            duty_cycle = self.center_duty + (position * (self.center_duty - self.left_max_duty))
        else:  # Right or center
            duty_cycle = self.center_duty + (position * (self.right_max_duty - self.center_duty))
        
        # Convert to percentage
        duty_percent = duty_cycle * 100
        
        if self.pwm:
            self.pwm.ChangeDutyCycle(duty_percent)
        
        self.current_position = position
        
        # Debug output
        if abs(position) > 0.01:  # Only print if not centered
            direction = "LEFT" if position < 0 else "RIGHT"
            print(f"[ServoController] Position: {direction} {abs(position)*100:.1f}% "
                  f"(duty cycle: {duty_percent:.1f}%)")
    
    def center(self):
        """Center the servo"""
        print("Center")
        self.set_position(0.0)
    
    def turn_left(self, amount=1.0):
        """
        Turn left
        
        Args:
            amount: Turn amount from 0.0 (no turn) to 1.0 (full left)
        """
        print("Turning left")
        self.set_position(-amount)
    
    def turn_right(self, amount=1.0):
        """
        Turn right
        
        Args:
            amount: Turn amount from 0.0 (no turn) to 1.0 (full right)
        """
        print("Turning right")
        self.set_position(amount)
    
    def set_angle(self, angle_degrees):
        """
        Set steering angle based on visual detection angle
        
        Args:
            angle_degrees: Angle in degrees (-90 to +90)
                          Negative = left, Positive = right
        """
        # Convert angle to position (-1.0 to 1.0)
        # Assuming max steering angle is ±45 degrees
        max_angle = 45.0
        position = max(-1.0, min(1.0, angle_degrees / max_angle))
        self.set_position(position)
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.pwm:
            self.pwm.stop()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("[ServoController] Cleaned up")


if __name__ == '__main__':
    # Test servo controller
    import config
    
    print("Testing servo controller...")
    print("This will sweep left to right")
    print("WARNING: Make sure servo is properly connected!")
    
    try:
        servo = ServoController(
            pwm_pin=config.SERVO_PWM_PIN,
            frequency=config.PWM_FREQUENCY,
            center_duty=config.SERVO_CENTER,
            left_max_duty=config.SERVO_LEFT_MAX,
            right_max_duty=config.SERVO_RIGHT_MAX
        )
        
        # Test sequence
        print("Centering...")
        servo.center()
        time.sleep(1)
        
        print("Turning left...")
        servo.turn_left(0.5)
        time.sleep(1)
        
        print("Turning full left...")
        servo.turn_left(1.0)
        time.sleep(1)
        
        print("Centering...")
        servo.center()
        time.sleep(1)
        
        print("Turning right...")
        servo.turn_right(0.5)
        time.sleep(1)
        
        print("Turning full right...")
        servo.turn_right(1.0)
        time.sleep(1)
        
        print("Centering...")
        servo.center()
        time.sleep(1)
        
        print("Test complete!")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if 'servo' in locals():
            servo.cleanup()

