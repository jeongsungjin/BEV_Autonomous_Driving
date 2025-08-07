# Global Path Pure Pursuit 자율주행 시스템

이 시스템은 `/home/carla/.ros/global_path_1.txt` 파일에 저장된 경로를 로드하여 Pure Pursuit 알고리즘을 사용한 자율주행을 수행합니다.

## 파일 구조

- `global_path_pure_pursuit.py`: Pure Pursuit 컨트롤러
- `spawn_vehicle_at_first_waypoint.py`: 첫 번째 waypoint 위치에 차량 스폰
- `global_path_autonomous_driving.launch`: 전체 시스템 launch 파일

## 경로 파일 형식

경로 파일은 다음과 같은 형식으로 저장되어야 합니다:
```
# CARLA Global Path - Map Coordinates
# Format: x, y, z, qx, qy, qz, qw
27.142294, 66.283257, 0.001711, 0.000000, 0.000000, -1.000000, 0.000639
27.139757, 66.283257, 0.001754, 0.000032, 0.000001, -1.000000, 0.000638
...
```

## 사용법

### 1. 개별 스크립트 실행

#### 차량 스폰
```bash
rosrun carla_camera_publisher spawn_vehicle_at_first_waypoint.py
```

#### Pure Pursuit 컨트롤러 실행
```bash
rosrun carla_camera_publisher global_path_pure_pursuit.py
```

### 2. Launch 파일을 사용한 전체 시스템 실행

```bash
roslaunch carla_camera_publisher global_path_autonomous_driving.launch
```

## 파라미터 설정

### Pure Pursuit 컨트롤러 파라미터
- `lookahead_distance`: 전방 주시 거리 (기본값: 3.0m)
- `max_steer`: 최대 스티어링 각도 (기본값: 0.6rad)
- `target_speed`: 목표 속도 (기본값: 5.0m/s)
- `wheelbase`: 차량 휠베이스 (기본값: 2.8m)
- `path_file`: 경로 파일 경로 (기본값: /home/carla/.ros/global_path_1.txt)

### 차량 스포너 파라미터
- `vehicle_model`: 차량 모델 (기본값: vehicle.tesla.model3)
- `path_file`: 경로 파일 경로

## 동작 원리

1. **경로 로드**: `global_path_1.txt` 파일에서 waypoints를 로드합니다.
2. **차량 스폰**: 첫 번째 waypoint 위치에 차량을 스폰합니다.
3. **Pure Pursuit 제어**: 
   - 현재 위치에서 `lookahead_distance`만큼 앞의 waypoint를 타겟으로 설정
   - Pure Pursuit 공식을 사용하여 스티어링 각도 계산
   - 곡선 구간에서는 속도 자동 조정

## 디버그 정보

컨트롤러는 다음 정보를 실시간으로 출력합니다:
- 타겟 포인트 좌표
- 스티어링 각도
- 현재 속도
- 경로까지의 거리

## 주의사항

1. CARLA 서버가 실행 중이어야 합니다.
2. 경로 파일이 올바른 형식으로 저장되어 있어야 합니다.
3. 첫 번째 waypoint 위치에 차량을 스폰할 수 있는 충분한 공간이 있어야 합니다.
4. 경로가 너무 급격한 곡선을 포함하지 않도록 주의하세요.

## 문제 해결

### 차량이 경로를 벗어나는 경우
- `lookahead_distance` 값을 줄여보세요
- `target_speed` 값을 줄여보세요

### 차량이 진동하는 경우
- `lookahead_distance` 값을 늘려보세요
- `max_steer` 값을 줄여보세요

### 차량이 경로를 따라가지 못하는 경우
- `target_speed` 값을 줄여보세요
- 경로의 곡선이 너무 급격하지 않은지 확인하세요 