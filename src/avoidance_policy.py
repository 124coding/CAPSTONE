from stable_baselines3 import PPO
import torch
import numpy as np
from keras.models import load_model
import cv2

class AvoidancePolicy:
    """
    CNN으로 조향각을 예측하고, PPO 정책으로 최종 조향/스로틀 명령을 반환하는 클래스
    """
    def __init__(self, ppo_path: str = "models/ppo_policy.zip", cnn_path: str = "models/cnn_angle_predictor.h5"):
        # PPO 모델 로드
        try:
            self.ppo_model = PPO.load(ppo_path)
        except Exception as e:
            raise RuntimeError(f"PPO 모델 로드 실패: {e}")
        # CNN 모델 로드
        try:
            self.cnn_model = load_model(cnn_path)
        except Exception as e:
            raise RuntimeError(f"CNN 모델 로드 실패: {e}")

    def detect_obstacle(self, distance: float, threshold: float = 2.0) -> bool:
        """
        DistanceSensor 데이터를 기반으로 장애물 여부 판단
        :param distance: 센서로부터 측정된 거리 (미터)
        :param threshold: 임계 거리 (미터)
        :return: True if obstacle detected
        """
        return 0 < distance < threshold

    def preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        CNN 입력용 전처리 수행
        :param image_rgb: 원본 RGB 이미지 (H, W, 3)
        :return: 정규화 및 리사이즈된 이미지
        """
        # 모델 학습 시 사용한 ROI 및 크롭을 동일하게 적용해야 할 수 있음
        img = cv2.resize(image_rgb, (255, 59))
        return img.astype(np.float32) / 255.0

    def predict_action(self, image_rgb: np.ndarray, speed: float, distance: float) -> tuple:
        """
        CNN 예측 조향각 + PPO 정책으로 조향/스로틀 결정
        :param image_rgb: 카메라 RGB 이미지 (H, W, 3)
        :param speed: 현재 차량 속도 (m/s)
        :param distance: DistanceSensor 측정 거리 (m)
        :return: (steer, throttle)
        """
        # 1) CNN으로 예비 조향각 예측
        img = self.preprocess_image(image_rgb)
        angle = self.cnn_model.predict_on_batch(np.expand_dims(img, axis=0))[0][0]

        # 2) PPO 정책 입력 벡터 구성
        obs = np.array([angle, speed, distance], dtype=np.float32)

        # 3) PPO로 최종 액션 예측 (steer, throttle)
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        steer, throttle = action
        return float(steer), float(throttle)
