# 개선된 ppo_drive.py - 수동 좌표 확인용 코드 포함
import airsim
import time

resolution = 0.1
grid_cells = 8000  # 예: grid_map.npy의 크기가 8000 x 8000
origin_x = - (grid_cells * resolution) / 2  # = -400.0
origin_y = - (grid_cells * resolution) / 2  # = -400.0

# 수동 조작 좌표 확인 루프
client = airsim.CarClient()
client.confirmConnection()

print("[INFO] 수동 조작 중: 차량 위치와 grid 좌표를 실시간 출력합니다. 종료하려면 Ctrl+C")

try:
    while True:
        pos = client.getCarState().kinematics_estimated.position
        x, y = pos.x_val, pos.y_val

        # 현재 위치 기준의 grid 좌표 출력
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        print(f"[AirSim 좌표] x: {x:.2f}, y: {y:.2f} | [Grid 좌표] x: {grid_x}, y: {grid_y}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n[INFO] 수동 조작 좌표 출력 종료됨.")
