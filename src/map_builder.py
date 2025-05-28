import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import json

# --- 저장 경로 설정 ---
grid_npy_path = os.path.join('maps', 'grid_map.npy')
grid_img_path = os.path.join('maps', 'grid_map.png')
grid_meta_path = os.path.join('maps', 'grid_map_meta.json')
os.makedirs('maps', exist_ok=True)

# --- LiDAR 및 맵 파라미터 초기값 ---
default_resolution = 0.1
default_size_m = 800
default_cells = int(default_size_m / default_resolution)
origin_x = 0
origin_y = 0

# --- 기존 맵이 있으면 불러오기 ---
if os.path.exists(grid_npy_path) and os.path.exists(grid_meta_path):
    print("[INFO] 기존 grid_map 불러오는 중...")
    grid_map = np.load(grid_npy_path)
    with open(grid_meta_path, 'r') as f:
        meta = json.load(f)
        grid_resolution = meta["grid_resolution"]
        grid_size_m = meta["grid_size_m"]
        grid_cells = meta["grid_cells"]
        origin_x = meta["origin_x"]
        origin_y = meta["origin_y"]
else:
    print("[INFO] 새 grid_map 생성")
    grid_resolution = default_resolution
    grid_size_m = default_size_m
    grid_cells = default_cells
    grid_map = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

# --- AirSim 연결 ---
client = airsim.CarClient()
try:
    client.confirmConnection()
    client.enableApiControl(False)
    print("[INFO] AirSim 연결 성공")
except Exception as e:
    print(f"[ERROR] AirSim 연결 실패: {e}")
    exit(1)

# --- LiDAR 및 차량 팽창 설정 ---
lidar_name = "Lidar"
z_min = -10
z_max = 0
vehicle_length = 4.0
vehicle_width = 2.0
safety_margin = 0.05
inflation_radius = (max(vehicle_length, vehicle_width) / 2.0) + safety_margin
inflation_cells = math.ceil(inflation_radius / grid_resolution)
print(f"[INFO] 팽창 반경: {inflation_radius:.2f}m ({inflation_cells} cells)")

# --- 좌표 변환 함수 ---
def world_to_grid(x, y):
    gx = int((x - origin_x) / grid_resolution + grid_cells // 2)
    gy = int((y - origin_y) / grid_resolution + grid_cells // 2)
    return gx, gy

def grid_to_display(gx, gy):
    dx = (gx - grid_cells // 2) * grid_resolution + origin_x
    dy = (gy - grid_cells // 2) * grid_resolution + origin_y
    return dx, dy

# --- 시각화 초기화 ---
plt.ion()
fig, ax = plt.subplots()
extent = [
    -grid_cells // 2 * grid_resolution + origin_x,
     grid_cells // 2 * grid_resolution + origin_x,
    -grid_cells // 2 * grid_resolution + origin_y,
     grid_cells // 2 * grid_resolution + origin_y
]
img_plot = ax.imshow(grid_map, cmap='gray', vmin=0, vmax=1, origin='lower', extent=extent)
scatter = ax.scatter([], [], c='red', s=20, label='Car')
plt.legend()
plt.colorbar(img_plot, label='Occupied (1) / Free (0)')
ax.set_title("2D Occupancy Grid from LiDAR")

# --- 저장 함수들 ---
def save_grid_map_image(grid, filename=grid_img_path):
    arr = (grid * 255).astype(np.uint8)
    arr = np.flipud(arr)
    Image.fromarray(arr, 'L').save(filename)
    print(f"[INFO] 이미지 저장 완료: {filename}")

def save_grid_map_npy(grid, filename=grid_npy_path):
    np.save(filename, grid)
    print(f"[INFO] NumPy 맵 저장 완료: {filename}")

def save_metadata():
    meta = {
        "grid_resolution": grid_resolution,
        "grid_size_m": grid_size_m,
        "grid_cells": grid_cells,
        "origin_x": origin_x,
        "origin_y": origin_y
    }
    with open(grid_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] 메타데이터 저장 완료: {grid_meta_path}")

# --- 메인 루프 ---
try:
    print("[INFO] LiDAR 데이터 수집 시작...")
    while True:
        pos = client.getCarState().kinematics_estimated.position
        car_world_x = pos.y_val
        car_world_y = pos.x_val

        data = client.getLidarData(lidar_name)
        points = np.array(data.point_cloud, dtype=np.float32)

        if points.size and points.size % 3 == 0:
            pts = points.reshape(-1, 3)
            mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            pts = pts[mask]

            for pt in pts:
                gx, gy = world_to_grid(pt[1], pt[0])
                if 0 <= gx < grid_cells and 0 <= gy < grid_cells:
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        for dx in range(-inflation_cells, inflation_cells + 1):
                            iy, ix = gy + dy, gx + dx
                            if 0 <= iy < grid_cells and 0 <= ix < grid_cells:
                                grid_map[iy, ix] = 1

        gx, gy = world_to_grid(car_world_x, car_world_y)
        dx, dy = grid_to_display(gx, gy)
        scatter.set_offsets([[dx, dy]])
        img_plot.set_data(grid_map)
        fig.canvas.flush_events()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n[INFO] 수집 중단 → 저장 중...")

except Exception as e:
    print(f"[ERROR] 런타임 오류: {e}")

finally:
    scatter.set_offsets(np.empty((0, 2)))
    fig.canvas.draw()
    save_grid_map_npy(grid_map)
    save_grid_map_image(grid_map)
    save_metadata()
    plt.close(fig)
    print("[INFO] 완료.")
