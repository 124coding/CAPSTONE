import numpy as np
import matplotlib.pyplot as plt

# 1. grid_map 불러오기 (경로는 직접 지정)
grid_map = np.load("maps/reachable_goal_mask.npy")  # 예시 경로

# 2. 시각화
plt.figure(figsize=(10, 10))
plt.imshow(grid_map, cmap='Greys', origin='lower')
plt.title("Grid Map (0 = Free, 1 = Obstacle)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.colorbar(label='Grid Value')
plt.show()
