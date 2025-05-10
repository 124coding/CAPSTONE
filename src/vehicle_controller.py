import airsim, time, math, numpy as np

class VehicleController:
    def __init__(self, Kp_steer=1.0, Kp_v=0.5, Ki_v=0.1):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.car_controls = airsim.CarControls()
        # speed PID internal
        self.Kp_v, self.Ki_v = Kp_v, Ki_v
        self.speed_err_sum = 0.0
        self.dt = 0.1
        self.Kp_steer = Kp_steer

    def compute_steering(self, tx, ty):
        state = self.client.getCarState()
        px, py = state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        dx, dy = tx - px, ty - py
        target_yaw = math.atan2(dy, dx)
        err = (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi
        return float(np.clip(err * self.Kp_steer, -1, 1))

    def compute_throttle(self, target_speed):
        current_speed = self.client.getCarState().speed
        err = target_speed - current_speed
        self.speed_err_sum += err * self.dt
        throttle = self.Kp_v*err + self.Ki_v*self.speed_err_sum
        return float(np.clip(throttle, 0, 1))

    def set_controls(self, steer, throttle, brake=False):
        self.car_controls.steering = steer
        self.car_controls.throttle = throttle
        self.car_controls.brake = 1.0 if brake else 0.0
        self.client.setCarControls(self.car_controls)

    def drive_along_path(self, path, origin, res, avoidance_policy=None):
        for gx, gy in path:
            # 월드 좌표 변환
            wx = (gx - origin[0]) * res
            wy = (gy - origin[1]) * res

            # 장애물 감지 & 회피
            if avoidance_policy and avoidance_policy.detect_obstacle():
                steer, thr = avoidance_policy.predict_action(...)
            else:
                steer = self.compute_steering(wx, wy)
                thr = self.compute_throttle(target_speed=5.0)

            self.set_controls(steer, thr)
            time.sleep(self.dt)

        # 도착 후 정지
        self.set_controls(0.0, 0.0, brake=True)
        print("경로 주행 완료")
