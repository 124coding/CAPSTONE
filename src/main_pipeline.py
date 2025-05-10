import airsim
import numpy as np
import time
import cv2
from global_planner import load_grid_map, generate_priority_path
from vehicle_controller import VehicleController
from avoidance_policy import AvoidancePolicy


def main():
    # 전역 설정 및 경로 생성
    grid_map = load_grid_map("maps/grid_map.npy")
    resolution = 0.2
    origin = (grid_map.shape[0] // 2, grid_map.shape[1] // 2)
    start = (10, 10)
    shelters = [(80, 60), (60, 20), (90, 90), (20, 90)]
    path = generate_priority_path(grid_map, start, shelters)
    print(f"[INFO] 전체 경로 {len(path)}개 지점 생성 완료")

    # 차량 및 회피 정책 초기화
    controller = VehicleController()
    policy = AvoidancePolicy()
    target_speed = 5.0  # 목표 속도 (m/s)

    print("[INFO] 주행 시작")
    for idx, (gx, gy) in enumerate(path, 1):
        # 그리드→월드 좌표 변환
        wx = (gx - origin[0]) * resolution
        wy = (gy - origin[1]) * resolution
        print(f"[{idx}/{len(path)}] 목표 지점: ({gx},{gy}) → 월드: ({wx:.2f}, {wy:.2f})")

        # 1) 카메라 이미지 수집
        response = controller.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        roi = img_rgb[76:135, 0:255, :]

        # 2) 센서 데이터
        speed = controller.client.getCarState().speed
        dist_data = controller.client.getDistanceSensorData("DistanceSensorFront")
        distance = dist_data.distance if dist_data.distance > 0 else np.inf

        # 3) 장애물 감지 및 제어 계산
        if policy.detect_obstacle(distance):
            print(f"[WARNING] 장애물 감지 (거리 {distance:.2f}m) → 회피 정책 적용")
            steer, throttle = policy.predict_action(roi, speed, distance)
        else:
            steer = controller.compute_steering(wx, wy)
            throttle = controller.compute_throttle(target_speed)

        # 4) 제어 입력 전송
        controller.set_controls(steer, throttle, brake=False)
        time.sleep(controller.dt)

    # 최종 정지
    controller.stop()
    print("[INFO] 경로 주행 완료")


if __name__ == "__main__":
    main()
