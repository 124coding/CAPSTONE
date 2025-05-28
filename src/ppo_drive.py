# 개선된 ppo_drive.py
import airsim
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from airsim_env import AirSimEnv
from convert_path_to_world import convert_path_to_world
from global_planner import load_grid_map, generate_priority_path

# 기본 설정
grid_map = load_grid_map("maps/grid_map.npy")
origin_x, origin_y = 0, 0
resolution = 0.1
start = (400, 400)  # grid 좌표 기준 (예시)
shelters = [(150, 100), (300, 350), (200, 400)]  # grid 좌표 기준 (예시)

# A* 경로 생성
path_grid = generate_priority_path(grid_map, start, shelters)
if not path_grid:
    raise ValueError("[ERROR] A* 경로가 비어 있습니다. 대피소 좌표 또는 grid_map을 확인하세요.")

path_world = convert_path_to_world(path_grid, origin_x, origin_y, resolution)

# PPO 로드
def make_env():
    return AirSimEnv(client_ip="127.0.0.1", client_port=41451)

env = DummyVecEnv([make_env])
env = VecNormalize.load("models/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/ppo_policy.zip", env=env)
obs = env.reset()

# 주행 루프
client = airsim.CarClient()
client.confirmConnection()

waypoint_threshold = 3.0  # 도달 판단 거리
curr_wp_idx = 0
max_steps = 5000

print(f"[INFO] 전체 경로 길이: {len(path_world)}개 웨이포인트")

for step in range(max_steps):
    if curr_wp_idx >= len(path_world):
        print("[SUCCESS] 모든 waypoint 도달 완료")
        break

    pos = client.getCarState().kinematics_estimated.position
    x, y = pos.x_val, pos.y_val

    # 현재 목표 지점까지 거리 계산
    goal_x, goal_y = path_world[curr_wp_idx]
    dist = np.hypot(x - goal_x, y - goal_y)

    # 도달했으면 다음으로
    if dist < waypoint_threshold:
        print(f"[INFO] Waypoint {curr_wp_idx} 도달 → 다음으로 이동")
        curr_wp_idx += 1
        continue

    # 장애물 감지
    front_dist = client.getDistanceSensorData("DistanceSensorFront").distance

    if front_dist < 4.0:
        # PPO로 회피
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
    else:
        # A* 경로 추적 (간단한 proportional 조향)
        dx = goal_x - x
        dy = goal_y - y
        heading = np.arctan2(dy, dx)
        car_heading = client.getCarState().kinematics_estimated.orientation
        yaw = airsim.to_eularian_angles(car_heading)[2]
        angle_diff = heading - yaw

        steer = np.clip(angle_diff, -1.0, 1.0)
        throttle = 0.5

        controls = airsim.CarControls()
        controls.steering = float(steer)
        controls.throttle = float(throttle)
        controls.brake = 0.0
        controls.is_manual_gear = True
        controls.manual_gear = 1
        controls.gear_immediate = True
        client.setCarControls(controls)

        time.sleep(0.1)

print("[DONE] PPO + A* 기반 주행 완료")
