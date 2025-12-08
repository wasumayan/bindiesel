"""
Servo Controller Module
Controls steering servo using PWM signals via GPIO
"""

import time
import config
from model_GPIO import ModelGPIO

if config.USE_GPIO:
    import RPi.GPIO as GPIO
else:
    GPIO = ModelGPIO




class ServoController:
    def __init__(self, pwm_pin, frequency, center_duty, left_max_duty, right_max_duty):
        self.pwm_pin = pwm_pin
        self.frequency = frequency
        self.center_duty = center_duty
        self.left_max_duty = left_max_duty
        self.right_max_duty = right_max_duty
        self.pwm = None

        if config.USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pwm_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pwm_pin, self.frequency) # 45 Hz matching
            self.pwm.start(self.center_duty)

        if config.DEBUG_SERVO:
            print(f"[Servo] Initialized on pin {self.pwm_pin} at {self.frequency} Hz")
            print(
                f"[Servo] center={self.center_duty:.2f}%, "
                f"left_max={self.left_max_duty:.2f}%, "
                f"right_max={self.right_max_duty:.2f}%"
            )

    def _set_duty(self, duty):
        duty = max(min(duty, self.left_max_duty), self.right_max_duty)

        if duty < self.left_max_duty:
            duty = self.left_max_duty

        if duty > self.right_max_duty:
            duty = self.right_max_duty

        if config.USE_GPIO and self.pwm:
            self.pwm.ChangeDutyCycle(duty)

        if config.DEBUG_SERVO:
            print(f"[Servo] duty = {duty:.2f}%")

    def center(self):
        if config.DEBUG_SERVO:
            print("[Servo] center()")
        self._set_duty(self.center_duty)

    def set_angle(self, angle_deg: float):
        if angle_deg < -45.0:
            angle_deg = -45.0
        if angle_deg > 45.0:
            angle_deg = 45.0 

        if angle_deg >= 0:
            duty = self.center_duty + (self.right_max_duty - self.center_duty) * (angle_deg / 45.0)
        else:
            duty = self.center_duty + (self.left_max_duty - self.center_duty) * (angle_deg / -45.0)

        if config.DEBUG_SERVO:
            print(f"[Servo] set_angle({angle_deg:.1f}) degrees -> duty = {duty:.2f}%")

        self._set_duty(duty)

    def cleanup(self):
        if config.USE_GPIO:
            if self.pwm:
                self.pwm.stop()
            GPIO.cleanup(self.pwm_pin)

        if config.DEBUG_SERVO:
            print("[Servo] cleanup()")

