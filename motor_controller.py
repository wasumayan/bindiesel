"""
Motor Controller Module
Controls motor speed using PWM signals via GPIO
"""

import time
import config
from model_GPIO import ModelGPIO

if config.USE_GPIO:
    import RPi.GPIO as GPIO

else: GPIO = ModelGPIO()


class MotorController:
    def __init__(self, pwm_pin, frequency): # what is self, exactly?
        self.pwm_pin = pwm_pin
        self.frequency = frequency        # set correct f?
        self.pwm = None 

        if config.USE_GPIO:
            GPIO.setmode(GPIO.BCM)          
            GPIO.setup(self.pwm_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pwm_pin, frequency)  #39HZ -> 255 period in PSoC 
            self.pwm.start(0); # % dutcy cycle

        if config.DEBUG_MOTOR:
            print(f"[Motor] Initialized on pin {self.pwm_pin} at {self.frequency} Hz")

    def forward(self, speed: float):        # check calculation
        duty = max(0.0, min(1.0, speed)) * 100.0 

        if config.USE_GPIO:
            self.pwm.ChangeDutyCycle(duty)

        if config.DEBUG_MOTOR:
            print(f"[Motor] forward speed = {speed:.2f} (duty = {duty:.1f}%)")

    def stop(self):
        if config.USE_GPIO and self.pwm:
            self.pwm.ChangeDutyCycle(0.0)

        if config.DEBUG_MOTOR:
            print("[Motor] stop()")

    def cleanup(self):                  # what is cleanup? 
        if config.USE_GPIO:
            if self.pwm:
                self.pwm.stop()
            GPIO.cleanup(self.pwm_pin)

        if config.DEBUG_MOTOR:
            print("[Motor] cleanup()")