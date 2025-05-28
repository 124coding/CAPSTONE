from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from airsim_env import AirSimStraightEnv
import time

def make_env():
    return AirSimStraightEnv(client_ip="127.0.0.1", client_port=41451)

env = DummyVecEnv([make_env])
env = VecNormalize.load("models/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/ppo_policy.zip", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(0.05)
