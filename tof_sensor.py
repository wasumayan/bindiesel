"""
Time-of-Flight (TOF) Sensor Module
VL53L0X distance sensor for obstacle detection and user proximity
"""

import config
from model_GPIO import ModelGPIO

if config.USE_GPIO:
    import RPi.GPIO as GPIO

else: GPIO = ModelGPIO(); 

class ToFSensor:
    def __init__(self):

        if config.USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.ToF_DIGITAL_PIN, GPIO.IN)

        if config.DEBUG_TOF:
            print(f"[ToF] Initialized digital input on pin {config.ToF_DIGITAL_PIN}")

    def state(self) -> bool:
        if config.USE_GPIO:    
           val = bool(GPIO.input(config.ToF_DIGITAL_PIN))
           return val 
        
        if config.DEBUG_TOF:
            print(f"{GPIO.input(config.ToF_DIGITAL_PIN)}"); 
    
    def detect(self) -> bool: 
        state = self.state()

        if config.DEBUG_TOF:
            print(f"[TOF] detect -> {state}")

        return state 