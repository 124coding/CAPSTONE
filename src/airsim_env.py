import gym
import airsim
import numpy as np
import time
import cv2
from keras.models import load_model

class AirSimEnv(gym.Env):
    """
    AirSim 기반 장애물 회피 학습 환경.
    상태: [CNN 예비 조향각, 현재 속도, 전방 거리]
    액션: [steering (-1~1), throttle (0~1)]
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 client_ip: str = "127.0.0.1",
                 camera_name: str = "0",
                 dist_sensor_name: str = "DistanceSensorFront",
                 cnn_model_path: str = "models/cnn_angle_predictor.h5",
                 target_speed: float = 5.0,
                 dt: float = 0.1):
        super().__init__()
        # AirSim 연결
        self.client = airsim.CarClient(ip=client_ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # 차량 제어 객체
        self.car_controls = airsim.CarControls()

        # 센서/카메라 이름
        self.camera_name = camera_name
        self.dist_name = dist_sensor_name
        # CNN 모델 로드
        self.cnn_model = load_model(cnn_model_path)

        # 목표 속도 및 타임스텝
        self.target_speed = target_speed
        self.dt = dt

        # Observation/Action Space 정의
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 50.0, 20.0]),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self):
        # 시뮬레이터 초기화
        self.client.reset()
        time.sleep(1)
        # 초기 관측값 반환
        return self._get_obs()

    def step(self, action):
        # 액션 적용
        steer, throttle = np.clip(action, self.action_space.low, self.action_space.high)
        self.car_controls.steering = float(steer)
        self.car_controls.throttle = float(throttle)
        self.car_controls.brake = 0.0
        self.client.setCarControls(self.car_controls)
        # 시간 경과
        time.sleep(self.dt)
        # 다음 관측 및 보상
        obs = self._get_obs()
        reward, done = self._get_reward_done(obs)
        return obs, reward, done, {}

    def _get_obs(self):
        # 1) 카메라 ROI
        img_resp = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
        ])[0]
        img1d = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(img_resp.height, img_resp.width, 3)
        roi = img[76:135, 0:255, :]  # 학습한 CNN ROI
        # CNN 예비 조향값
        norm = roi.astype(np.float32) / 255.0
        angle = self.cnn_model.predict_on_batch(np.expand_dims(norm, axis=0))[0][0]

        # 2) 속도
        speed = self.client.getCarState().speed

        # 3) 거리센서
        dist_data = self.client.getDistanceSensorData(self.dist_name)
        distance = dist_data.distance if dist_data.distance > 0 else 20.0

        return np.array([angle, speed, distance], dtype=np.float32)

    def _get_reward_done(self, obs):
        angle, speed, distance = obs
        # 보상: 안정 주행 유지
        steer_penalty = -abs(angle)
        speed_penalty = -abs(speed - self.target_speed)
        dist_penalty = 0 if distance > 2.0 else -10
        reward = steer_penalty + speed_penalty + dist_penalty
        # 종료: 충돌(거리 매우 가까움)
        done = distance < 0.1
        return float(reward), bool(done)

    def render(self, mode='human'):
        pass

    def close(self):
        # API 제어 해제
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
