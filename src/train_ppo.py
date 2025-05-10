import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback
)
from torch import nn

# AirSimEnv를 import
from airsim_env import AirSimEnv  

if __name__ == '__main__':
    # 1. AirSimEnv 병렬 환경 생성
    env = make_vec_env(AirSimEnv, n_envs=4, seed=0)
    eval_env = make_vec_env(AirSimEnv, n_envs=1, seed=42)

    # 2. 정책 네트워크 설정
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )

    # 3. 콜백 설정
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_ppo",
        log_path="logs/eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints",
        name_prefix="ppo"
    )
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=100,
        verbose=1
    )

    # 4. PPO 모델 생성 및 학습
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="logs/ppo_avoidance"
    )
    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, checkpoint_callback, stop_callback]
    )

    # 5. 모델 저장
    model.save("models/ppo_policy.zip")
    print("[INFO] PPO 정책 학습 완료, 저장됨: models/ppo_policy.zip")
