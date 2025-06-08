import gym
import numpy as np
import airsim
import time
import os
import random

class AirSimObstacleAvoidanceEnv(gym.Env):
    def __init__(self, client_ip="127.0.0.1", client_port=41451, path_dir="maps/precomputed_paths", min_wp_gap=500):
        super().__init__()
        self.client = airsim.CarClient(ip=client_ip, port=client_port)
        self.port = client_port
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.car_controls = airsim.CarControls()

        self.path_dir = path_dir
        self.path = self._load_random_path(self.path_dir)
        self.current_wp_idx = 0
        self.min_wp_gap = min_wp_gap

        self.max_steps = 100000
        self.step_count = 0
        self.stuck_counter = 0
        self.same_wp_counter = 0
        self.last_wp_idx = -1
        self.off_path_counter = 0
        self.prev_position = None
        self.max_distance = 100.0

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-np.pi, 0.0, 0.0, 0.0, 0.0]), high=np.array([np.pi, 100.0, 100.0, 100.0, 100.0]), dtype=np.float32)

        self.reset()

    def _load_random_path(self, path_dir):
        path_files = [os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith(".npy")]
        raw_path = np.load(random.choice(path_files))
        shift_x, shift_y = -400.0, -400.0
        return np.array([[x + shift_x, y + shift_y] for x, y in raw_path])

    def _visualize_path(self):
        points = [airsim.Vector3r(x, y, -2.0) for x, y in self.path]
        self.client.simFlushPersistentMarkers()
        self.client.simPlotPoints(points, color_rgba=[1.0, 0.0, 0.0, 1.0], size=15.0, duration=0.0, is_persistent=True)

    def reset(self):
        self.client.reset()
        time.sleep(0.5)
        self.current_wp_idx = 0
        self.last_dist_to_path = 0.0
        self.same_wp_counter = 0
        self.last_wp_idx = -1
        pose = airsim.Pose(airsim.Vector3r(0.0, 0.0, -2.0), airsim.to_quaternion(0, 0, np.pi / 2))
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.3)
        self.step_count = 0
        self.stuck_counter = 0
        self.off_path_counter = 0
        self.prev_position = None
        self._visualize_path()
        time.sleep(0.01)
        return self._get_obs()

    def _get_obs(self):
        state = self.client.getCarState()
        speed = abs(state.speed)
        orientation = state.kinematics_estimated.orientation
        _, _, angle = airsim.to_eularian_angles(orientation)
        front = self.client.getDistanceSensorData("DistanceSensorFront").distance
        left = self.client.getDistanceSensorData("DistanceSensorLeft22").distance
        right = self.client.getDistanceSensorData("DistanceSensorRight22").distance
        left45 = self.client.getDistanceSensorData("DistanceSensorLeft45").distance
        right45 = self.client.getDistanceSensorData("DistanceSensorRight45").distance
        def clip(d): return np.clip(d if d > 0 else self.max_distance, 0, self.max_distance)
        self.last_left45 = clip(left45)
        self.last_right45 = clip(right45)
        return np.array([angle, clip(front), clip(left), clip(right), speed], dtype=np.float32)

    def get_forward_wp_idx(self, x, y, yaw):
        min_score = float('inf')
        best_idx = self.current_wp_idx
        allow_jump_range = 300
        max_jump_idx = 800
        idx_range = range(self.current_wp_idx, min(len(self.path), self.current_wp_idx + (max_jump_idx if self.off_path_counter > 50 else allow_jump_range)))
        for i in idx_range:
            px, py = self.path[i]
            dx, dy = px - x, py - y
            dist = np.hypot(dx, dy)
            if dist > 60.0:
                continue
            angle = np.arctan2(dy, dx)
            angle_diff = abs((angle - yaw + np.pi) % (2 * np.pi) - np.pi)
            if angle_diff > np.pi / 2:
                continue
            idx_gap = i - self.current_wp_idx
            score = dist + 0.05 * idx_gap
            if score < min_score:
                best_idx = i
                min_score = score
        return best_idx

    def is_stuck(self, pos, speed, threshold=0.2):
        if self.prev_position is None:
            return False
        dx, dy = pos.x_val - self.prev_position[0], pos.y_val - self.prev_position[1]
        return np.hypot(dx, dy) < threshold and speed < 1.0

    def step(self, action):
        self.step_count += 1
        obs = self._get_obs()
        angle, front, left, right, speed = obs
        steer, throttle = float(action[0]), float(action[1])
        reward = 0.0
        done = False

        pos = self.client.getCarState().kinematics_estimated.position
        yaw = angle

        # 경로 타겟 설정
        target_idx = self.get_forward_wp_idx(pos.x_val, pos.y_val, yaw)
        target_x, target_y = self.path[target_idx]
        dist_to_path = np.hypot(pos.x_val - target_x, pos.y_val - target_y)

        # 경로 중심 유지 보상
        reward += np.exp(-dist_to_path * 0.3) * 0.3

        # waypoint 통과 보상
        if target_idx > self.current_wp_idx:
            reward += 0.2 * (target_idx - self.current_wp_idx)
            self.current_wp_idx = target_idx
            self.same_wp_counter = 0
        else:
            if target_idx == self.last_wp_idx:
                self.same_wp_counter += 1
            else:
                self.same_wp_counter = 0
            self.last_wp_idx = target_idx

        # 경로 이탈 감점
        if dist_to_path > 3.0:
            reward -= 0.1
        if dist_to_path > 6.0:
            reward -= 0.2

        # 일정 시간 경로 이탈 시 종료
        if dist_to_path > 5.0:
            self.off_path_counter += 1
            if self.off_path_counter > 300:
                reward -= 1.0
                done = True
        else:
            self.off_path_counter = 0

        # 속도 보상
        target_speed = 5.0
        reward += np.clip(speed / target_speed, 0.0, 1.0) * 0.05

        # 조향/가속 제어 안정성 패널티
        reward -= abs(steer) * speed * 0.002
        reward -= abs(throttle - 0.5) * 0.01

        # 같은 waypoint에 오래 머물면 종료
        if self.same_wp_counter > 40:
            reward -= 0.5
            done = True

        # 충돌 감지 및 종료
        if front < 1.0 or left < 0.5 or right < 0.5:
            reward -= 2.0
            done = True

        # 차량 제어 적용
        self.car_controls.steering = steer
        self.car_controls.throttle = abs(throttle)
        self.car_controls.brake = 0.0
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = 1 if throttle >= 0 else -1
        self.car_controls.gear_immediate = True
        self.client.setCarControls(self.car_controls)

        # stuck 감지
        if self.is_stuck(pos, speed):
            self.stuck_counter += 1
            if self.step_count > 1000:
                reward -= 1.0
                done = True
        else:
            self.stuck_counter = 0

        # 최대 스텝 종료
        if self.step_count >= self.max_steps:
            done = True

        self.prev_position = (pos.x_val, pos.y_val)
        info = {
            "current_wp_idx": self.current_wp_idx,
            "distance_to_path": dist_to_path
        }

        time.sleep(0.03)
        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)
