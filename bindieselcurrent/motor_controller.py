"""
Motor Controller Module
Controls motor speed using PWM signals via GPIO
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
                    print(f"[MockGPIO] Motor PWM started at {duty*100:.1f}%")
                def ChangeDutyCycle(self, duty):
                    print(f"[MockGPIO] Motor PWM changed to {duty*100:.1f}%")
                def stop(self):
                    print("[MockGPIO] Motor PWM stopped")
            return MockPWM()
        @staticmethod
        def cleanup():
            pass
    GPIO = MockGPIO


class MotorController:
    """Controls motor speed using PWM"""
    
    def __init__(self, pwm_pin, frequency=50):
        """
        Initialize motor controller
        
        Args:
            pwm_pin: GPIO pin number (BCM) for motor PWM
            frequency: PWM frequency in Hz (default 50Hz)
        """
        self.pwm_pin = pwm_pin
        self.frequency = frequency
        self.pwm = None
        self.current_speed = 0.0
        
        if GPIO_AVAILABLE:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pwm_pin, GPIO.OUT)
            
            # Create PWM instance
            self.pwm = GPIO.PWM(pwm_pin, frequency)
            self.pwm.start(0)  # Start with 0% duty cycle (stopped)
            print(f"[MotorController] Initialized on GPIO {pwm_pin} at {frequency}Hz")
        else:
            print(f"[MotorController] Mock mode - GPIO {pwm_pin} would be used")
    
    def set_speed(self, speed_percent):
        """
        Set motor speed
        
        Args:
            speed_percent: Speed as percentage (0.0 to 1.0)
                         0.0 = stopped, 1.0 = maximum speed
        """
        # Clamp speed to valid range
        speed_percent = max(0.0, min(1.0, speed_percent))
        
        # Convert to duty cycle (0-100)
        duty_cycle = speed_percent * 100
        
        if self.pwm:
            self.pwm.ChangeDutyCycle(duty_cycle)
        
        self.current_speed = speed_percent
        
        # Debug output
        if abs(speed_percent) > 0.01:  # Only print if moving
            print(f"[MotorController] Speed set to {speed_percent*100:.1f}% (duty cycle: {duty_cycle:.1f}%)")
    
    def stop(self):
        """Stop motor immediately"""
        self.set_speed(0.0)
        print("[MotorController] Motor stopped")
    
    def forward(self, speed_percent):
        """
        Move forward at specified speed
        
        Args:
            speed_percent: Speed as percentage (0.0 to 1.0)
        """
        self.set_speed(speed_percent)
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.pwm:
            self.pwm.stop()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("[MotorController] Cleaned up")


if __name__ == '__main__':
    # Test motor controller
    import config
    
    print("Testing motor controller...")
    print("This will run for 5 seconds")
    print("WARNING: Make sure motor is properly connected!")
    
    try:
        motor = MotorController(
            pwm_pin=config.MOTOR_PWM_PIN,
            frequency=config.PWM_FREQUENCY
        )
        
        # Test sequence
        print("Starting motor at 30% speed...")
        motor.forward(0.3)
        time.sleep(2)
        
        print("Increasing to 50% speed...")
        motor.forward(0.5)
        time.sleep(2)
        
        print("Stopping...")
        motor.stop()
        time.sleep(1)
        
        print("Test complete!")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if 'motor' in locals():
            motor.cleanup()

