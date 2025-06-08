# global_planner.py - 차량 크기 고려한 A* 경로 생성 및 순차 경유
import numpy as np
import heapq
import math

def load_grid_map(path='maps/grid_map.npy'):
    return np.load(path)

def inflate_obstacles(grid_map, inflation_radius=10):
    from scipy.ndimage import grey_dilation
    kernel_size = 2 * inflation_radius + 1
    structure = np.ones((kernel_size, kernel_size))
    inflated = grey_dilation(grid_map, footprint=structure)
    return (inflated > 0).astype(np.uint8)

class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid_map[nx, ny] != 1:
                    move_cost = math.hypot(dx, dy)
                    cell_cost = 1 if self.grid_map[nx, ny] == 0 else self.grid_map[nx, ny]
                    neighbors.append(((nx, ny), move_cost * cell_cost))
        return neighbors

    def plan(self, start, goal):
        open_set = [(self.heuristic(start, goal), 0, start, [])]
        visited = set()

        while open_set:
            est_total, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]

            if current == goal:
                return path

            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                new_cost = cost + move_cost
                est_total = new_cost + self.heuristic(neighbor, goal)
                heapq.heappush(open_set, (est_total, new_cost, neighbor, path))

        print(f"[경고] {start}에서 {goal}로 가는 경로를 찾을 수 없습니다.")
        return []

def generate_priority_path(grid_map, start, waypoint_list):
    planner = AStarPlanner(grid_map)
    full_path = []
    current = start

    for wp in waypoint_list:
        segment = planner.plan(current, wp)
        if not segment:
            print(f"[경고] 대피소 {wp}로 가는 경로를 찾을 수 없습니다.")
            continue
        if full_path:
            segment = segment[1:]  # 중복 제거
        full_path.extend(segment)
        current = wp

    return full_path
