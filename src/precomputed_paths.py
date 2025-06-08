import os
import json
import numpy as np
import itertools

from global_planner import load_grid_map, inflate_obstacles, generate_priority_path
from convert_path_to_world import convert_path_to_world

# AirSim 기준 고정된 목적지 (Grid 좌표 직접 사용)
grid_waypoints = [
    (7000, 4000),  # (300, 6)
    (4450, 1200),  # (47, -280)
    (2000, 3300),  # (-200, 67)
    (4444, 6450)   # (44, 245)
]

def generate_all_fixed_priority_paths(grid_map_path="maps/grid_map.npy",
                                      meta_path="maps/grid_map_meta.json",
                                      save_dir="maps/precomputed_paths",
                                      waypoints=grid_waypoints):
    os.makedirs(save_dir, exist_ok=True)

    grid_map = load_grid_map(grid_map_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    resolution = meta["grid_resolution"]
    origin_x = meta["origin_x"]
    origin_y = meta["origin_y"]

    def is_valid_cell(row, col):
        return (0 <= row < grid_map.shape[0] and
                0 <= col < grid_map.shape[1] and
                grid_map[row, col] == 0)

    start = (4000, 4000)  # 그리드 맵 중앙
    permutations = list(itertools.permutations(waypoints, 4))

    for i, perm in enumerate(permutations):
        inflated = inflate_obstacles(grid_map, inflation_radius=0)

        if not is_valid_cell(*start):
            print(f"[경고] 시작 지점 {start}이 유효하지 않습니다.")
            continue

        if not all(is_valid_cell(*g) for g in perm):
            print(f"[경고] 일부 목적지가 유효하지 않음: {perm}")
            continue

        path = generate_priority_path(inflated, start, perm)
        if path is None or len(path) == 0:
            print(f"[경고] 경로 생성 실패 (순열 {i}): {perm}")
            continue

        world_path = convert_path_to_world(path, origin_x, origin_y, resolution)
        save_path = os.path.join(save_dir, f"path_{i}.npy")
        np.save(save_path, world_path)
        print(f"[INFO] 경로 저장 완료: {save_path}")

if __name__ == '__main__':
    generate_all_fixed_priority_paths()
    print("[INFO] 모든 목적지 조합에 대한 A* 경로 저장 완료")