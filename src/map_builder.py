#!/usr/bin/env python3
"""
Standalone script to generate a 2D occupancy grid map from AirSim LiDAR data.
Run this once before running main_pipeline to produce `maps/grid_map.npy` and `maps/grid_map.png`.
"""
import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

# Output paths
grid_npy_path = os.path.join('maps', 'grid_map.npy')
grid_img_path = os.path.join('maps', 'grid_map.png')

# Ensure output directory exists
os.makedirs('maps', exist_ok=True)

# AirSim client setup
client = airsim.CarClient()
try:
    client.confirmConnection()
    client.enableApiControl(False)
    print("[INFO] AirSim 연결 성공")
except Exception as e:
    print(f"[ERROR] AirSim 연결 실패: {e}")
    exit(1)

# Grid map parameters
grid_resolution = 0.2  # meters per cell
grid_size_m = 300      # total size in meters
grid_cells = int(grid_size_m / grid_resolution)
center_x = grid_cells // 2
center_y = grid_cells // 2

# Initialize grid map
grid_map = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

# LiDAR settings
lidar_name = "Lidar"
z_min = -300
z_max = 0

# Vehicle dimensions and safety margin
vehicle_length = 4.5
vehicle_width = 2.0
safety_margin = 0.2

# Compute inflation radius in cells
inflation_radius = (max(vehicle_length, vehicle_width) / 2.0) + safety_margin
inflation_cells = math.ceil(inflation_radius / grid_resolution)
print(f"[INFO] 팽창 반경: {inflation_radius:.2f}m ({inflation_cells} cells)")

# Setup matplotlib for real-time display
plt.ion()
fig, ax = plt.subplots()
img_plot = ax.imshow(
    grid_map, cmap='gray', vmin=0, vmax=1, origin='lower',
    extent=[-grid_size_m/2, grid_size_m/2, -grid_size_m/2, grid_size_m/2]
)
plt.colorbar(img_plot, label='Occupied (1) / Free (0)')
ax.set_title('2D Occupancy Grid from LiDAR')

# Save functions
def save_grid_map_image(grid, filename=grid_img_path):
    arr = (grid * 255).astype(np.uint8)
    Image.fromarray(arr, 'L').save(filename)
    print(f"[INFO] 그리드맵 이미지 저장: {filename}")

def save_grid_map_npy(grid, filename=grid_npy_path):
    np.save(filename, grid)
    print(f"[INFO] 그리드맵 NumPy 저장: {filename}")

# Main loop
try:
    print("[INFO] LiDAR 데이터 수집 시작...")
    while True:
        # Fetch LiDAR data
        data = client.getLidarData(lidar_name)
        points = np.array(data.point_cloud, dtype=np.float32)
        if points.size and points.size % 3 == 0:
            pts = points.reshape(-1, 3)
            # Filter by height
            mask = (pts[:,2] >= z_min) & (pts[:,2] <= z_max)
            pts = pts[mask]
            if pts.size:
                # Convert to grid indices
                xs = (pts[:,1] / grid_resolution + center_x).astype(int)
                ys = (pts[:,0] / grid_resolution + center_y).astype(int)
                valid = (xs>=0)&(xs<grid_cells)&(ys>=0)&(ys<grid_cells)
                xs, ys = xs[valid], ys[valid]
                grid_map[ys, xs] = 1
                # Inflation
                for (gy, gx) in zip(ys, xs):
                    for dy in range(-inflation_cells, inflation_cells+1):
                        for dx in range(-inflation_cells, inflation_cells+1):
                            iy, ix = gy+dy, gx+dx
                            if 0 <= iy < grid_cells and 0 <= ix < grid_cells:
                                grid_map[iy, ix] = 1
        # Update plot
        img_plot.set_data(grid_map)
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] 수집 중단, 맵 저장 중...")
    save_grid_map_npy(grid_map)
    save_grid_map_image(grid_map)

except Exception as e:
    print(f"[ERROR] 런타임 오류: {e}")
    save_grid_map_npy(grid_map)
    save_grid_map_image(grid_map)

finally:
    plt.close(fig)
    print("[INFO] 완료.")
