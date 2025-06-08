import os
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from airsim_env import AirSimObstacleAvoidanceEnv

# ✅ PPO 환경 생성 함수
def make_env(rank):
    def _init():
        port = 41451 + rank
        env = AirSimObstacleAvoidanceEnv(client_ip="127.0.0.1", client_port=port)
        return env
    return _init

if __name__ == '__main__':
    # 🔹 1. 환경 설정
    n_envs = 2
    raw_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    monitor_env = VecMonitor(raw_env)

    eval_env = SubprocVecEnv([make_env(100)])
    eval_monitor_env = VecMonitor(eval_env)

    # 🔹 2. 정책 구조
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    # 🔹 3. 콜백 설정
    eval_callback = EvalCallback(
        eval_monitor_env,
        best_model_save_path="models/best_ppo",
        log_path="logs/eval",
        eval_freq=100000,
        deterministic=True,
        render=False,
        callback_after_eval=StopTrainingOnRewardThreshold(
            reward_threshold=1000.0,
            verbose=1
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="models/checkpoints",
        name_prefix="ppo"
    )

    os.makedirs("models", exist_ok=True)

    # 🔹 4. PPO 모델 로드 또는 초기화
    try:
        print("[INFO] 기존 모델 로드 시도")
        model = PPO.load("models/ppo_policy.zip", env=monitor_env, custom_objects={"learning_rate": 1e-4})
        print("[INFO] 모델 로드 완료")
    except Exception as e:
        print(f"[WARN] 모델 로드 실패, 새로 생성합니다: {e}")
        model = PPO(
            "MlpPolicy",
            env=monitor_env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,          # rollout 길이는 유지
            batch_size=256,        # 🔧 update 빈도 줄임
            n_epochs=5,            # 🔧 안정성 확보
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/ppo_avoidance"
        )

    # 🔹 5. 학습 시작
    print("[INFO] PPO 학습 시작")
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, checkpoint_callback]
    )

    model.save("models/ppo_policy.zip")
    print("[INFO] PPO 정책 저장 완료")
