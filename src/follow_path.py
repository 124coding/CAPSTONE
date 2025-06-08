import airsim
import numpy as np
import time
import os
import random
import math
from global_planner import AStarPlanner, load_grid_map

# === 경로 및 맵 설정 ===
path_dir = "maps/precomputed_paths"
path_files = [f for f in os.listdir(path_dir) if f.endswith(".npy")]
selected_path = np.load(os.path.join(path_dir, random.choice(path_files)))
shifted_path = selected_path + np.array([-400.0, -400.0])

# 필터링 없이 바로 사용
inflated_path = shifted_path

grid_map = load_grid_map("maps/grid_map.npy")

planner = AStarPlanner(grid_map)

# === AirSim 연결 ===
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# === 차량 초기 위치 설정 ===
start_x, start_y = inflated_path[0]
pose = airsim.Pose(airsim.Vector3r(start_x, start_y, -2.0), airsim.to_quaternion(0, 0, np.pi / 2))
client.simSetVehiclePose(pose, ignore_collision=True)

# === 초기 제어 입력 ===
controls = airsim.CarControls()
controls.steering = 0.0
controls.throttle = 0.0
controls.brake = 1.0
client.setCarControls(controls)
time.sleep(0.1)

# === 경로 시각화 ===
points = [airsim.Vector3r(x, y, -2.0) for x, y in inflated_path]
client.simFlushPersistentMarkers()
client.simPlotPoints(points, color_rgba=[1, 0, 0, 1], size=15.0, duration=0.0, is_persistent=True)

# === 유틸 함수 ===
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def steer_from_error(error, k=0.6, max_steer=0.7):
    steer = k * error
    return np.clip(steer, -max_steer, max_steer)

def compute_cornering_steer(angle_diff):
    steer = np.tanh(2.5 * angle_diff)
    return np.clip(steer, -1.0, 1.0)

# === 제어 변수 초기화 ===
original_path = inflated_path
current_wp_idx = 0
reach_thresh = 6.0

valid_sensor_labels = [
    'DistanceSensorL90', 'DistanceSensorL67', 'DistanceSensorL45', 'DistanceSensorL33',
    'DistanceSensorL22', 'DistanceSensorL11', 'DistanceSensorFront',
    'DistanceSensorR11', 'DistanceSensorR22', 'DistanceSensorR33', 'DistanceSensorR45',
    'DistanceSensorR67', 'DistanceSensorR90'
]

sensor_weights = {
    'DistanceSensorL90': -0.65, 'DistanceSensorL67': -0.6, 'DistanceSensorL45': -0.55,
    'DistanceSensorL33': -0.5, 'DistanceSensorL22': -0.45, 'DistanceSensorL11': -0.4,
    'DistanceSensorR11': 0.4, 'DistanceSensorR22': 0.45, 'DistanceSensorR33': 0.5,
    'DistanceSensorR45': 0.55, 'DistanceSensorR67': 0.6, 'DistanceSensorR90': 0.65,
    'DistanceSensorFront': 0.0
}

target_distance = 1.8
emergency_brake_distance = 1.0
front_critical = 6.0
recover_duration = 10

in_avoid_mode = False
avoid_counter = 0
avoid_duration = 10
avoid_direction = 0
recover_mode = False
recover_counter = 0
emergency_mode = False
last_steer = 0.0
avoid_decision_timer = 0
avoid_decision_delay = 1

# 실시간 주행 중 웨이포인트 도달 확인 및 점프 허용 (300 범위 내)
def has_reached_waypoint(car_x, car_y, wp_x, wp_y, thresh=6.0):
    return np.hypot(car_x - wp_x, car_y - wp_y) < thresh

def update_waypoint_idx(car_x, car_y, path, current_idx):
    for offset in range(300):
        idx = current_idx + offset
        if idx >= len(path):
            break
        wp_x, wp_y = path[idx]
        if has_reached_waypoint(car_x, car_y, wp_x, wp_y):
            return idx + 1
    return current_idx

while True:
    state = client.getCarState()
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    yaw = airsim.to_eularian_angles(ori)[2]
    car_x, car_y = pos.x_val, pos.y_val

    distances = {}
    for label in valid_sensor_labels:
        try:
            distances[label] = client.getDistanceSensorData(label).distance
        except:
            distances[label] = float('inf')

    left_vals = [distances.get(f"DistanceSensorL{i}", float('inf')) for i in [11, 22, 33, 45, 67, 90]]
    right_vals = [distances.get(f"DistanceSensorR{i}", float('inf')) for i in [11, 22, 33, 45, 67, 90]]
    left_min = min(left_vals)
    right_min = min(right_vals)
    front_min = distances.get("DistanceSensorFront", float('inf'))

    steer = 0.0
    steer_adjust = 0.0

    if min(left_min, right_min, front_min) < emergency_brake_distance:
        emergency_mode = True

    if emergency_mode:
        print("[EMERGENCY BRAKE] Obstacle too close!")
        max_clear_label = max(sensor_weights, key=lambda k: distances.get(k, 0))
        escape_direction = sensor_weights[max_clear_label]
        controls.throttle = -0.1
        controls.brake = 0.0
        controls.steering = np.clip(escape_direction * 1.5, -1.0, 1.0)
        client.setCarControls(controls)
        time.sleep(0.05)
        if min(left_min, right_min, front_min) > target_distance + 0.5:
            emergency_mode = False
        continue

    if front_min < front_critical:
        if not in_avoid_mode and avoid_decision_timer <= 0:
            in_avoid_mode = True
            avoid_counter = avoid_duration
            avoid_decision_timer = avoid_decision_delay
            if right_min < left_min - 0.2:
                avoid_direction = -1
            elif left_min < right_min - 0.2:
                avoid_direction = 1
            else:
                avoid_direction = 1 if random.random() < 0.5 else -1

        steer_adjust = 4.5 * avoid_direction
        for label, dist in distances.items():
            if 0.2 < dist < target_distance * 2.5:
                error = target_distance - dist
                weight = sensor_weights.get(label, 0.0)
                steer_adjust += 3.5 * weight * error / target_distance

    elif in_avoid_mode:
        steer_adjust = 1.4 * avoid_direction
        avoid_counter -= 1
        if avoid_counter <= 0:
            in_avoid_mode = False
            recover_mode = True
            recover_counter = recover_duration

    elif recover_mode:
        recover_counter -= 1
        steer_adjust = steer_adjust * (recover_counter / recover_duration)
        if recover_counter <= 0:
            recover_mode = False

    else:
        for label, dist in distances.items():
            if 0.2 < dist < target_distance * 2.5:
                error = target_distance - dist
                weight = sensor_weights.get(label, 0.0)
                steer_adjust += 2.6 * weight * error / target_distance

    steer = steer_from_error(steer_adjust)
    steer = 0.7 * last_steer + 0.3 * steer
    last_steer = steer
    avoid_decision_timer = max(0, avoid_decision_timer - 1)

    current_wp_idx = update_waypoint_idx(car_x, car_y, original_path, current_wp_idx)

    if current_wp_idx >= len(original_path):
        print("[DONE] Path complete.")
        break

    if not in_avoid_mode and not recover_mode and abs(steer_adjust) < 0.05:
        lookahead = 10
        lookahead_idx = min(current_wp_idx + lookahead, len(original_path) - 1)
        target = original_path[lookahead_idx]
        dx, dy = target[0] - car_x, target[1] - car_y
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = normalize_angle(angle_to_target - yaw)
        steer = compute_cornering_steer(angle_diff)

        min_side = min(left_min, right_min, front_min)
        if abs(angle_diff) > np.pi / 3 and min_side < 4.0:
            controls.throttle = 0.15
            controls.brake = 0.0
            print(f"[CORNERING] Slow turn with nearby obstacle: steer={steer:.2f}")
        else:
            controls.throttle = 0.4 if abs(steer) < 0.9 else 0.25
            controls.brake = 0.0

        print(f"[DRIVE] WP={current_wp_idx} steer={steer:.2f}")
    else:
        print(f"[AVOID] steer={steer:.2f} (adjust={steer_adjust:.2f})")
        controls.throttle = 0.35 if abs(steer) < 0.9 else 0.2
        controls.brake = 0.0

    controls.steering = np.clip(steer, -1.2, 1.2)
    client.setCarControls(controls)
    time.sleep(0.003)
