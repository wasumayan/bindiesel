from motor_controller import MotorController
from tof_sensor import ToFSensor
from servo_controller import ServoController
import config
import time


def main():
    print("[ToF Test] Starting Time-of-Flight Sensor Test")
    tof = ToFSensor()
    servo = ServoController(
                pwm_pin=config.SERVO_PWM_PIN,
                frequency=config.PWM_FREQUENCY_SERVO,
                center_duty=config.SERVO_CENTER,
                left_max_duty=config.SERVO_LEFT_MAX,
                right_max_duty=config.SERVO_RIGHT_MAX
            )
            
    motor = MotorController(
                pwm_pin=config.MOTOR_PWM_PIN,
                frequency=config.PWM_FREQUENCY
            )
    motor.stop() 
    servo.center()

    time.sleep(1)  # Allow time for initialization
    motor.forward(config.MOTOR_MAX)
    print("[ToF Test] Motor started at max speed. Move an object close to the ToF sensor to test detection.")
    try:
        while True:
            if tof.detect():
                print("[ToF Test] Obstacle detected! Stopping motor.")
                motor.stop()

            time.sleep(0.1)  # Check every 0.5 seconds
    except KeyboardInterrupt:
        print("[ToF Test] Stopping test.")
    finally:
        motor.cleanup()
        servo.cleanup()