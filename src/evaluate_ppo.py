import time
import numpy as np
from stable_baselines3 import PPO
from airsim_env import AirSimStraightEnv

model_path = "models/ppo_policy.zip"
num_trials = 10
max_steps = 1000

env = AirSimStraightEnv(random_start=True)
model = PPO.load(model_path)

successes = 0
collisions = 0
rewards = []

for trial in range(num_trials):
    obs = env.reset()
    time.sleep(1.0)
    total_reward = 0.0
    collided = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # 센서 충돌 검사 (예외 처리)
        try:
            front = env.client.getDistanceSensorData("DistanceSensorFront").distance
            if 0 < front < 1.0:
                collided = True
                print(f"[Trial {trial+1}] 충돌 발생 (step {step})")
                collisions += 1
                break
        except:
            pass

        if done:
            break

    if not collided:
        successes += 1
        print(f"[Trial {trial+1}] 성공, 총 보상: {total_reward:.2f}")

    rewards.append(total_reward)

# 요약 출력
print("\n=== 평가 결과 ===")
print(f"성공: {successes}/{num_trials}")
print(f"충돌: {collisions}")
print(f"평균 보상: {np.mean(rewards):.2f}")
print(f"최고 보상: {np.max(rewards):.2f}")
print(f"최저 보상: {np.min(rewards):.2f}")
