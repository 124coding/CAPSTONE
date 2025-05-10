import airsim
import numpy as np
import cv2
import os
import time

# --- 설정 ---
data_dir = "ppo_data"
os.makedirs(data_dir, exist_ok=True)

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

print("[INFO] 랜덤+자동 탐색 기반 데이터 수집 시작...")

# 초기 조향 및 타이머
current_steer = np.random.uniform(-1.0, 1.0)
change_interval = 10.0  # 조향 변경 주기 (초)
last_change_time = time.time()
data_count = 0

# 차량 정지 감지 타이머
stuck_start_time = None
stuck_duration_limit = 5.0  # 차량 정지 시 허용 시간 (초)

try:
    while True:
        now = time.time()

        # 차량 제어: 조기 적용
        car_controls.steering = float(current_steer)
        car_controls.throttle = 0.4
        car_controls.brake = 0.0
        client.setCarControls(car_controls)

        # 일정 주기마다 랜덤 조향 변경
        if now - last_change_time > change_interval:
            current_steer = np.random.uniform(-1.0, 1.0)
            print(f"[INFO] 새 조향 설정: {current_steer:.2f}")
            last_change_time = now

        # 차량 상태
        car_state = client.getCarState()
        speed = car_state.speed
        dist_data = client.getDistanceSensorData("DistanceSensorFront")
        distance = dist_data.distance if dist_data.distance > 0 else 20.0

        # 정지 시간 누적 검사
        if speed < 0.2:
            if stuck_start_time is None:
                stuck_start_time = now
            elif now - stuck_start_time > stuck_duration_limit:
                print("[ERROR] 차량이 너무 오래 정지 상태입니다. 수집 종료.")
                break
            print("[SKIP] 차량 거의 정지")
            time.sleep(0.1)
            continue
        else:
            stuck_start_time = None  # 정지 해제되면 초기화

        # 품질 조건 필터링
        if distance < 0.3 or distance > 20.0:
            print("[SKIP] 유효 거리 아님")
            time.sleep(0.1)
            continue

        # 이미지 수집
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        roi = img_rgb[76:135, 0:255, :]

        # 밝기 조건 필터링
        if np.mean(roi) < 25:
            print("[SKIP] 너무 어두운 이미지")
            time.sleep(0.1)
            continue

        # 저장
        cv2.imwrite(f"{data_dir}/img_{data_count:04d}.png", roi)
        with open(f"{data_dir}/label_{data_count:04d}.txt", "w") as f:
            f.write(str(current_steer))

        print(f"[SAVE] #{data_count:04d} 조향: {current_steer:.2f}, 속도: {speed:.2f}, 거리: {distance:.2f}m")
        data_count += 1

        time.sleep(0.3)  # 수집 주기

except KeyboardInterrupt:
    print("[INFO] 수집 중단 by 사용자")

finally:
    client.enableApiControl(False)
    print("[INFO] AirSim 제어 해제 및 종료")
