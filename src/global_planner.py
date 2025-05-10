import numpy as np
import heapq
import math
import matplotlib.pyplot as plt

class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape

    def heuristic(self, a, b):
        # 대각선 허용 시 유클리드 거리
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8방향
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid_map[nx, ny] != 1:  # 장애물 제외
                    move_cost = math.hypot(dx, dy)
                    cell_cost = self.grid_map[nx, ny] if self.grid_map[nx, ny] > 0 else 1
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

def load_grid_map(path='maps/grid_map.npy'):
    return np.load(path)

def generate_priority_path(grid_map, start, shelter_priority_list):
    planner = AStarPlanner(grid_map)
    full_path = []
    current = start

    for shelter in shelter_priority_list:
        segment = planner.plan(current, shelter)
        if not segment:
            print(f"[경고] 대피소 {shelter}로 가는 경로를 찾을 수 없습니다.")
            continue
        if full_path:
            segment = segment[1:]  # 이전 경로의 마지막과 중복 제거
        full_path.extend(segment)
        current = shelter

    return full_path

def visualize_path(grid_map, path):
    vis_map = np.copy(grid_map)
    for x, y in path:
        vis_map[x, y] = 0.5
    plt.imshow(vis_map, cmap='gray')
    plt.title("전역 경로 시각화")
    plt.show()