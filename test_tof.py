#!/usr/bin/env python3
"""
Test script for TOF sensor
Tests VL53L0X distance measurement
"""

import time
from tof_sensor import ToFSensor
import config

def main():
    print("\n === Testing Tof Sensor ===")
    print(f"USE_GPIO = {config.USE_GPIO}")
    print("Reading sensor for 10 seconds... \n")
    
    tof = ToFSensor()

    start = time.time()
    while time.time() - start < 10.0:
        state = tof.state()
        print(f"ToF state: {state}")
        time.sleep(0.1)

    print("\nDone :p")


if __name__ == '__main__':
    main()

