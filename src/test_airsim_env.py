# test_airsim_env.py
import time
import numpy as np
import os
from pathlib import Path
from airsim_env import AirSimEnv

def main():
    # 1) 경로 설정: 프로젝트 루트의 models 폴더 내 CNN 모델
    BASE_DIR = Path(__file__).resolve().parent        # …/src
    PROJECT_ROOT = BASE_DIR.parent                    # project_root
    cnn_model_path = PROJECT_ROOT / "models" / "cnn_angle_predictor.h5"

    # 2) 환경 생성 (AirSim 서버가 실행 중이어야 함)
    env = AirSimEnv(
        client_ip="127.0.0.1",
        camera_name="0",
        dist_sensor_name="DistanceSensorFront",
        cnn_model_path=str(cnn_model_path),
        target_speed=5.0,
        dt=0.1
    )

    try:
        # 3) reset 및 초기 관측 확인
        obs = env.reset()
        print("초기 관측값:", obs)
        assert env.observation_space.contains(obs), "reset() 반환 obs가 observation_space를 벗어남"

        # 4) 몇 단계 step() 호출
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {step+1}: obs={obs}, reward={reward:.3f}, done={done}")
            assert env.observation_space.contains(obs), "step() 반환 obs가 observation_space를 벗어남"
            if done:
                print("→ done=True, 환경 재시작")
                obs = env.reset()

        print("✅ AirSimEnv 기본 스모크 테스트 통과!")
    except Exception as e:
        print("❌ AirSimEnv 스모크 테스트 실패:", e)
    finally:
        # 5) 환경 종료
        env.close()

if __name__ == "__main__":
    main()
