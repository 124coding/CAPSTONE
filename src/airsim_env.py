import gym
import airsim
import numpy as np
import time
import random

class AirSimStraightEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, client_ip="127.0.0.1", client_port=41451):
        super().__init__()
        self.client = airsim.CarClient(ip=client_ip, port=client_port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.car_controls = airsim.CarControls()

        self.dt = 0.3
        self.total_distance = 220.0
        self.target_speed = 5.0

        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_y = self.start_y + self.total_distance

        self.prev_pos = None
        self.no_movement_counter = 0
        self.stuck_timeout = 10
        self.min_movement_threshold = 0.1

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        time.sleep(1.0)

        self.start_x = random.uniform(-80.0, 80.0)
        self.start_y = 0.0
        self.goal_y = self.start_y + self.total_distance

        self.car_controls = airsim.CarControls()
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = 1
        self.car_controls.gear_immediate = True
        self.car_controls.steering = 0.0
        self.car_controls.throttle = 0.0
        self.car_controls.brake = 1.0
        self.client.setCarControls(self.car_controls)

        pose = airsim.Pose(
            airsim.Vector3r(self.start_x, self.start_y, 0),
            airsim.to_quaternion(0, 0, -3 * np.pi / 2)
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        self.prev_pos = (self.start_x, self.start_y)
        self.no_movement_counter = 0

        return self._get_obs(), {}

    def step(self, action):
        self.car_controls.steering = float(action[0])
        self.car_controls.throttle = float(action[1])
        self.car_controls.brake = 0.0
        self.client.setCarControls(self.car_controls)

        time.sleep(self.dt)
        obs = self._get_obs()
        pos = self.client.getCarState().kinematics_estimated.position
        distance_traveled = pos.y_val - self.start_y

        # 움직임 감지
        dx = pos.x_val - self.prev_pos[0]
        dy = pos.y_val - self.prev_pos[1]
        delta = np.hypot(dx, dy)

        if delta < self.min_movement_threshold:
            self.no_movement_counter += 1
        else:
            self.no_movement_counter = 0

        self.prev_pos = (pos.x_val, pos.y_val)

        if self.no_movement_counter > self.stuck_timeout:
            return obs, -10.0, True, False, {}

        reward, done = self._get_reward_done(obs)

        if distance_traveled >= self.total_distance:
            done = True
            lateral_deviation = abs(pos.x_val - self.start_x)
            reward -= lateral_deviation * 0.05

        return obs, reward, done, False, {}

    def _get_obs(self):
        def get_distance(name):
            try:
                d = self.client.getDistanceSensorData(name).distance
                return np.clip(d if d > 0 else 10.0, 0, 20)
            except:
                return 20.0

        front = get_distance("DistanceSensorFront")
        left45 = get_distance("DistanceSensorLeft45")
        right45 = get_distance("DistanceSensorRight45")
        left22 = get_distance("DistanceSensorLeft22")
        right22 = get_distance("DistanceSensorRight22")

        speed = self.client.getCarState().speed
        return np.array([speed, front, left45, right45, left22, right22], dtype=np.float32)

    def _get_reward_done(self, obs):
        speed, front, left45, right45, left22, right22 = obs
        reward = 0.0

        if min(front, left45, right45, left22, right22) < 0.5:
            reward -= 10.0
            return reward, True

        avoidance_triggered = any(sensor < 5.0 for sensor in [front, left45, right45, left22, right22])

        if avoidance_triggered:
            reward += 0.5
            reward -= abs(self.car_controls.steering) * speed * 0.02

            def balance_reward(d1, d2):
                diff = abs(d1 - d2)
                return max(0.0, 0.3 - diff * 0.05)

            reward += balance_reward(left45, right45)
            reward += balance_reward(left22, right22)

        return reward, False

    def render(self, mode='human'):
        pass

    def close(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
