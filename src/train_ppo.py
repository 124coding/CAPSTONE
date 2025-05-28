import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from torch import nn
import os

from airsim_env import AirSimStraightEnv

def make_env(rank):
    def _init():
        port = 41451 + rank
        return AirSimStraightEnv(client_ip="127.0.0.1", client_port=port)
    return _init

if __name__ == '__main__':
    n_envs = 2
    raw_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    monitor_env = VecMonitor(raw_env)

    eval_env = SubprocVecEnv([make_env(100)])
    eval_monitor_env = VecMonitor(eval_env)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    eval_callback = EvalCallback(
        eval_monitor_env,
        best_model_save_path="models/best_ppo",
        log_path="logs/eval",
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_after_eval=StopTrainingOnRewardThreshold(
            reward_threshold=1000,
            verbose=1
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints",
        name_prefix="ppo"
    )

    try:
        print("[INFO] 모델 로딩 시도: models/ppo_policy.zip")
        model = PPO.load("models/ppo_policy.zip", env=monitor_env, custom_objects={"learning_rate": 1e-4})
        print("[INFO] 기존 모델 로드 및 환경 설정 완료")
    except Exception as e:
        print(f"[WARN] 모델 로드 실패, 새로 초기화함: {e}")
        model = PPO(
            "MlpPolicy",
            env=monitor_env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/ppo_avoidance"
        )

    print("[INFO] 학습 시작")
    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, checkpoint_callback]
    )

    model.save("models/ppo_policy.zip")
    print("[INFO] PPO 정책 저장 완료")
