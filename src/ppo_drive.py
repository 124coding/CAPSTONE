import airsim
import time
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from airsim_env import AirSimStraightEnv
from convert_path_to_world import convert_path_to_world
from global_planner import load_grid_map, generate_priority_path, inflate_obstacles
import scipy.interpolate as si

# ==============================
# 경로 및 유틸 함수
# ==============================
def get_forward_nearest_wp_idx_dynamic(x, y, yaw, path, start_idx, max_distance=50.0):
    best_idx = start_idx
    best_dist = float('inf')
    for i in range(start_idx, len(path)):
        px, py = path[i]
        dx, dy = px - x, py - y
        dist = np.hypot(dx, dy)
        if dist > max_distance:
            continue
        angle = np.arctan2(dy, dx)
        angle_diff = (angle - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_diff) < (3 * np.pi / 4) and dist < best_dist:
            best_idx = i
            best_dist = dist
    return best_idx

def smooth_path(path, smooth_factor=3):
    if len(path) < 4:
        return path
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    t = range(len(path))
    t_new = np.linspace(0, len(path) - 1, len(path) * smooth_factor)
    spl_x = si.make_interp_spline(t, x, k=3)
    spl_y = si.make_interp_spline(t, y, k=3)
    return list(zip(spl_x(t_new), spl_y(t_new)))

def draw_waypoints_on_airsim(client, path):
    for (x, y) in path:
        client.simPlotPoints([airsim.Vector3r(x, y, -2.5)], color_rgba=[1.0, 0.0, 0.0, 1.0], size=10.0, is_persistent=True)

def world_to_grid(wx, wy, origin_x, origin_y, resolution):
    return (int((wx - origin_x) / resolution), int((wy - origin_y) / resolution))

def downsample_path(path, step=5):
    return path[::step] + [path[-1]] if path else []

# ==============================
# 설정 및 초기화
# ==============================
resolution = 0.1
grid_cells = 8000
origin_x = -(grid_cells * resolution) / 2
origin_y = -(grid_cells * resolution) / 2

client = airsim.CarClient()
client.confirmConnection()
time.sleep(1.0)

car_state = client.getCarState()
start = (car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val)

shelters = [(310, 7), (45, -290), (-215, -67), (45, 260)]
random.shuffle(shelters)

start_grid = world_to_grid(*start, origin_x, origin_y, resolution)
shelters_grid = [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in shelters]
grid_map = load_grid_map("maps/grid_map.npy")
grid_map = inflate_obstacles(grid_map, inflation_radius=20)

path_grid = generate_priority_path(grid_map, start_grid, shelters_grid)
path_grid = downsample_path(path_grid)
path_world = convert_path_to_world(path_grid, origin_x, origin_y, resolution)
path_world = smooth_path(path_world)

draw_waypoints_on_airsim(client, path_world)

# PPO 로딩
env = DummyVecEnv([lambda: AirSimStraightEnv(client_ip="127.0.0.1", client_port=41451, random_start=False)])
model = PPO.load("models/ppo_policy.zip", env=env)
obs = env.reset()

waypoint_threshold = 10.0
curr_wp_idx = 0
max_steps = 20000
safe_margin = 10 * resolution
max_speed_mps = 6.0
is_avoiding = False
avoidance_start_time = None
min_avoid_time = 2.0
min_recovery_idx = 0

goal_indices = []
for gx, gy in shelters:
    closest_idx = np.argmin([np.hypot(wx - gx, wy - gy) for wx, wy in path_world])
    goal_indices.append(closest_idx)
goal_indices.sort()
next_goal_idx = 0

print(f"[INFO] 전체 경로 길이: {len(path_world)}개 웨이포인트")

# ==============================
# 주행 루프
# ==============================
for step in range(max_steps):
    if next_goal_idx >= len(goal_indices):
        print("[SUCCESS] 모든 목적지 도달 완료")
        break

    pos = client.getCarState().kinematics_estimated.position
    x, y = pos.x_val, pos.y_val
    current_speed = client.getCarState().speed

    goal_x, goal_y = path_world[curr_wp_idx]
    dist = np.hypot(x - goal_x, y - goal_y)

    if dist < waypoint_threshold:
        print(f"[INFO] Waypoint {curr_wp_idx} 도달 → 다음으로 이동")
        curr_wp_idx += 1
        if next_goal_idx < len(goal_indices) and curr_wp_idx > goal_indices[next_goal_idx]:
            print(f"[GOAL] {next_goal_idx+1}번째 목적지 도달 완료")
            next_goal_idx += 1
        continue

    # 장애물 센서 값
    front_dist = client.getDistanceSensorData("DistanceSensorFront").distance
    left22_dist = client.getDistanceSensorData("DistanceSensorLeft22").distance
    right22_dist = client.getDistanceSensorData("DistanceSensorRight22").distance
    left45_dist = client.getDistanceSensorData("DistanceSensorLeft45").distance
    right45_dist = client.getDistanceSensorData("DistanceSensorRight45").distance

    controls = airsim.CarControls()
    dynamic_margin = safe_margin + max(1.0, current_speed * 1.5)

    # 장애물 회피 모드 진입
    if not is_avoiding and (
        front_dist < dynamic_margin or
        left22_dist < safe_margin or
        right22_dist < safe_margin or
        left45_dist < safe_margin * 0.8 or
        right45_dist < safe_margin * 0.8
    ):
        is_avoiding = True
        avoidance_start_time = time.time()
        print("[PPO] 장애물 회피 모드 진입")

    if is_avoiding:
        # PPO 예측 수행 후, action이 스칼라인지 배열인지 확인
        action, _ = model.predict(obs, deterministic=True)

        # action이 배열일 경우 첫 번째 값 사용, 스칼라일 경우 그대로 사용
        if isinstance(action, np.ndarray):  # action이 배열인 경우
            action = action[0]  # 첫 번째 값만 사용
        elif isinstance(action, (float, int)):  # action이 스칼라일 경우 그대로 사용
            action = np.array([action, 0.5])  # 스칼라인 경우 throttle은 기본값 0.5로 설정

        # steering과 throttle 값을 각각 처리
        steer = float(action[0])  # action이 배열일 경우 첫 번째 값만 사용
        throttle = float(action[1]) if len(action) > 1 else 0.5  # 배열일 경우 두 번째 값이 있다면 사용, 아니면 기본값 0.5

        # 차량 제어
        controls.steering = steer
        controls.throttle = 0.0 if current_speed >= max_speed_mps else max(throttle, 0.3)
        controls.brake = 0.2 if current_speed >= max_speed_mps else 0.0
        controls.is_manual_gear = True
        controls.manual_gear = 1
        controls.gear_immediate = True
        client.setCarControls(controls)

        # PPO 환경 스텝 수행
        obs, _, _, _ = env.step([action])
        obs = obs[0]

        front_dist = client.getDistanceSensorData("DistanceSensorFront").distance
        left22_dist = client.getDistanceSensorData("DistanceSensorLeft22").distance
        right22_dist = client.getDistanceSensorData("DistanceSensorRight22").distance

        safe = (
            front_dist > safe_margin * 3.0 and
            left22_dist > safe_margin * 1.5 and
            right22_dist > safe_margin * 1.5
        )

        if safe and avoidance_duration >= min_avoid_time:
            # 회피 종료 후 경로 복귀 시도
            yaw = airsim.to_eularian_angles(client.getCarState().kinematics_estimated.orientation)[2]
            recovery_idx = get_forward_nearest_wp_idx_dynamic(x, y, yaw, path_world, curr_wp_idx, max_distance=80.0)
            max_allowed_jump = 100
            recovery_idx = min(len(path_world) - 1, max(curr_wp_idx, recovery_idx))
            if recovery_idx - curr_wp_idx > max_allowed_jump:
                recovery_idx = curr_wp_idx + max_allowed_jump

            curr_wp_idx = recovery_idx
            print(f"[PPO] 회피 종료 → A* 경로 복귀: wp#{curr_wp_idx}")

            is_avoiding = False
            avoidance_start_time = None

    # 회피 이후 경로 추적
    goal_x, goal_y = path_world[curr_wp_idx]
    heading = np.arctan2(goal_y - y, goal_x - x)
    yaw = airsim.to_eularian_angles(client.getCarState().kinematics_estimated.orientation)[2]
    angle_diff = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
    steer = np.clip(angle_diff * 0.5, -1.0, 1.0)

    # 속도 조절
    throttle = 1.0 - min(abs(angle_diff) * 1.5, 0.9)
    if current_speed >= max_speed_mps:
        throttle = 0.0
        controls.brake = 0.2
    elif current_speed < 2.0:
        throttle = max(throttle, 0.6)
        controls.brake = 0.0
    else:
        controls.brake = 0.0

    controls.steering = float(steer)
    controls.throttle = float(throttle)
    controls.is_manual_gear = True
    controls.manual_gear = 1
    controls.gear_immediate = True
    client.setCarControls(controls)
    time.sleep(0.1)

print("[DONE] PPO + A* 안정 주행 완료")
