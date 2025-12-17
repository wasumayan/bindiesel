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
    flag = 0

    time.sleep(1)  # Allow time for initialization

    motor.forward(config.MOTOR_FAST)
    print("[ToF Test] Motor started at max speed. Move an object close to the ToF sensor to test detection.")
    try:
        while True:
            servo.set_angle(0.0)  # Keep servo centered
            if tof.detect() and flag == 0:
                start = time.time()
                print("[ToF Test] Obstacle detected! Stopping motor.")
                motor.stop()
                flag = 1; 




            time.sleep(0.1)  # Check every 0.5 seconds
    except KeyboardInterrupt:
        print("[ToF Test] Stopping test.")
        end = time.time() - start
        print(f"[ToF Test] Obstacle detected after {end:.2f} seconds.")
    finally:
        motor.cleanup()
        servo.cleanup()

if __name__ == "__main__":
    main()