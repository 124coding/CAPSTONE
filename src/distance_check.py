import airsim
import time

# 센서 이름 리스트
sensor_names = [
    "DistanceSensorFront",
    "DistanceSensorLeft22",
    "DistanceSensorRight22",
    "DistanceSensorLeft45",
    "DistanceSensorRight45"
]

# 바닥 인식 제거용 필터
def filter_distance(d):
    return d if d > 0.05 else float('inf')

# AirSim 연결
client = airsim.CarClient()
client.confirmConnection()

print("📡 DistanceSensor 데이터 확인 시작...")
print("중단하려면 Ctrl + C 를 누르세요.\n")

try:
    while True:
        readings = {}
        for name in sensor_names:
            raw = client.getDistanceSensorData(name).distance
            filtered = filter_distance(raw)
            readings[name] = filtered

        # 출력
        print(" | ".join(f"{k}: {v:.2f}m" if v != float('inf') else f"{k}: ∞" for k, v in readings.items()))
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n📴 거리 센서 모니터링 종료됨.")
