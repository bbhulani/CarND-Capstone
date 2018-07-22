import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base,
                 steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.480
        ki = 0.03
        kd = 0.4
        mn = 0.0  # Minimum throttle value
        mx = 0.5  # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5  # 1/(2pi*tau) = cutoff frequency
        ts = 0.02  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

        # Adding schmitt trigter to reset throttle PID after sustained repeated values
        # The integral portion takes time to fully decay, in order to increase the rate of decay,
        # the integral factor woudl also need increased, but this isn't suitable to the needs here
        self.throttle_reset_threshold = 25          # Half sample frequency
        self.throttle_reset_count = 0
        self.throttle_reset_last_vel = 0
        self.throttle_reset_hysteresis = 0.001

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        current_vel = self.vel_lpf.filt(current_vel)

        # Schmitt trigger for resetting PID integral factor over periods of sustained
        # repeated values
        if current_vel <= (self.throttle_reset_last_vel + self.throttle_reset_hysteresis) and \
           current_vel >= (self.throttle_reset_last_vel - self.throttle_reset_hysteresis):
            self.throttle_reset_count = self.throttle_reset_count + 1
            if self.throttle_reset_count >= self.throttle_reset_threshold:
                self.throttle_controller.reset();
                self.throttle_reset_count = 0
        else:
            self.throttle_reset_last_vel = current_vel
            self.throttle_reset_count = 0

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0
            brake = 700  # N*m - to hold car in place if we are stopped at a light. Acceleration ~ 1m/s^2
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius  # Torque N*m

        return throttle, brake, steering
