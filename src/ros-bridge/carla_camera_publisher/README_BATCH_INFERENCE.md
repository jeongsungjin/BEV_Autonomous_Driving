# Multi-Vehicle YOLOP Batch Inference System

3대의 자율주행 차량에서 각각 BEV(Bird's Eye View) 이미지를 수집하고, 시간 동기화를 통한 배치 추론을 수행하는 YOLOP 시스템입니다.

## 개요

이 시스템은 다음과 같은 기능을 제공합니다:
- 3대의 CARLA 차량에서 각각 BEV 이미지 발행
- 시간 동기화를 통한 배치 추론
- 각 차량별 개별 추론 결과 발행
- 성능 향상 및 GPU 효율성 개선

## 파일 구조

```
carla_camera_publisher/
├── scripts/
│   ├── publish_multiple_carla_cam.py      # 3대 차량 BEV 카메라 퍼블리셔 (자율주행 개선)
│   ├── yolop_batch_inference_node.py      # 배치 추론 노드
│   ├── test_batch_system.py               # 시스템 테스트 도구
│   ├── carla_cleanup.py                   # CARLA 액터 정리 스크립트 (신규)
│   └── yolop_inference_node.py            # 기존 단일 추론 노드
├── launch/
│   ├── multiple_bev_cameras.launch        # 3대 차량 카메라만 실행
│   ├── yolop_batch_inference.launch       # 배치 추론만 실행
│   ├── complete_batch_system.launch       # 전체 시스템 실행 (자동 cleanup 포함)
│   └── carla_cleanup.launch               # CARLA 정리 전용 launch (신규)
└── README_BATCH_INFERENCE.md              # 이 파일
```

## 시스템 구성

### 1. publish_multiple_carla_cam.py (개선된 자율주행)
- 3대의 Tesla Model 3 차량을 CARLA에 스폰 (각기 다른 색상)
- **Traffic Manager 통합**: 안전한 자율주행을 위한 고급 설정
- **안전한 스폰 포인트**: 최소 20m 거리 유지하여 충돌 방지
- **동기 모드**: 안정적인 시뮬레이션을 위한 20 FPS 고정
- 각 차량 위 30m에서 BEV 카메라 생성
- 각각 다른 토픽으로 이미지 발행:
  - `/carla/vehicle1/image_raw`
  - `/carla/vehicle2/image_raw`  
  - `/carla/vehicle3/image_raw`
- 각 차량의 odometry와 camera info도 함께 발행
- 추가 배경 차량 8대 자동 스폰 (다양한 교통 환경 조성)

### 2. yolop_batch_inference_node.py
- message_filters를 사용한 3개 이미지 토픽 시간 동기화
- 배치 텐서로 YOLOP 모델 추론 수행
- 각 차량별 결과 토픽 발행:
  - `/carla/vehicle{N}/yolop/inference` (시각화 이미지)
  - `/carla/vehicle{N}/yolop/det_grid` (Detection OccupancyGrid)
  - `/carla/vehicle{N}/yolop/da_grid` (Drivable Area OccupancyGrid)
  - `/carla/vehicle{N}/yolop/ll_grid` (Lane Line OccupancyGrid)
  - `/carla/vehicle{N}/yolop/costmap` (통합 Costmap)

## 사용법

### 1. 전체 시스템 실행 (권장)
```bash
# 1. CARLA 서버 실행 (별도 터미널)
cd /path/to/CARLA
./CarlaUE4.sh

# 2. 배치 시스템 실행
roslaunch carla_camera_publisher complete_batch_system.launch
```

⚠️ **주의사항**: 시스템이 동기 모드를 사용하므로 CARLA 서버가 먼저 실행되어야 합니다.

✨ **자동 정리**: Ctrl+C로 launch 파일을 종료하면 CARLA의 모든 차량과 센서가 자동으로 정리됩니다.

### 2. 단계별 실행

#### Step 1: 3대 차량 BEV 카메라 실행
```bash
roslaunch carla_camera_publisher multiple_bev_cameras.launch
```

#### Step 2: 배치 추론 노드 실행 (별도 터미널)
```bash
roslaunch carla_camera_publisher yolop_batch_inference.launch
```

### 3. 파라미터 조정
```bash
# 디버그 모드로 실행
roslaunch carla_camera_publisher complete_batch_system.launch debug:=true

# 이미지 모니터링과 함께 실행
roslaunch carla_camera_publisher yolop_batch_inference.launch monitor_images:=true

# 시스템 모니터링과 함께 실행
roslaunch carla_camera_publisher complete_batch_system.launch monitor_system:=true
```

### 4. CARLA 정리 (Cleanup)

CARLA 시뮬레이터에 누적된 차량과 센서를 정리하는 방법:

```bash
# 방법 1: 즉시 정리 (권장)
roslaunch carla_camera_publisher carla_cleanup.launch instant:=true

# 방법 2: 수동으로 스크립트 실행
python3 src/ros-bridge/carla_camera_publisher/scripts/carla_cleanup.py --instant

# 방법 3: Cleanup 노드 실행 (Ctrl+C로 정리)
roslaunch carla_camera_publisher carla_cleanup.launch
```

💡 **언제 사용하나요?**
- Launch 파일을 여러 번 실행한 후 차량이 누적된 경우
- 시뮬레이터를 깨끗한 상태로 초기화하고 싶은 경우
- 다른 CARLA 실험을 위해 환경을 정리하고 싶은 경우

## 토픽 구조

### 입력 토픽
- `/carla/vehicle1/image_raw` (sensor_msgs/Image)
- `/carla/vehicle2/image_raw` (sensor_msgs/Image)
- `/carla/vehicle3/image_raw` (sensor_msgs/Image)
- `/carla/vehicle{N}/camera_info` (sensor_msgs/CameraInfo)

### 출력 토픽
각 차량별로 다음 토픽들이 발행됩니다:
- `/carla/vehicle{N}/yolop/inference` - 시각화된 추론 결과 이미지
- `/carla/vehicle{N}/yolop/det_grid` - 객체 감지 occupancy grid
- `/carla/vehicle{N}/yolop/da_grid` - 주행 가능 영역 occupancy grid  
- `/carla/vehicle{N}/yolop/ll_grid` - 차선 occupancy grid
- `/carla/vehicle{N}/yolop/costmap` - 통합 비용 맵

## 성능 특징

### 배치 처리의 장점
1. **GPU 효율성**: 3개 이미지를 한 번에 처리하여 GPU 활용률 향상
2. **처리량 증가**: 단일 추론 대비 전체 처리량 향상
3. **일관성**: 시간 동기화로 일관된 결과 보장

### 성능 모니터링
- 배치 추론 시간과 FPS가 30번마다 로그로 출력
- 각 차량별 마스크 비율 정보 (debug 모드)
- ROS topic 주파수 모니터링 가능

## 주요 파라미터

### 모델 파라미터
- `weight_path`: YOLOP 모델 가중치 경로
- `conf_thres`: 객체 감지 신뢰도 임계값 (기본: 0.5)
- `iou_thres`: NMS IoU 임계값 (기본: 0.45)
- `da_thresh`: 주행 가능 영역 임계값 (기본: 0.53)
- `ll_thresh`: 차선 임계값 (기본: 0.5)

### 시스템 파라미터
- `debug`: 디버그 모드 활성화
- `enable_costmap`: 통합 costmap 발행 여부
- `save_bev`: BEV raster 저장 여부
- `bev_save_dir`: BEV raster 저장 디렉터리

## 시스템 테스트

### 자동 테스트 도구
시스템이 정상 동작하는지 확인할 수 있는 테스트 스크립트를 제공합니다:

```bash
# 배치 시스템 테스트 (별도 터미널에서)
rosrun carla_camera_publisher test_batch_system.py
```

이 테스트는 다음을 모니터링합니다:
- 각 차량의 이미지 토픽 상태
- YOLOP 추론 결과 발행 상태  
- OccupancyGrid 메시지 발행 상태
- 전체 시스템 건강도 평가

### 수동 테스트
```bash
# 토픽 발행 상태 확인
rostopic list | grep carla

# 특정 차량의 토픽 주파수 확인
rostopic hz /carla/vehicle1/image_raw
rostopic hz /carla/vehicle1/yolop/inference

# 이미지 확인
rosrun rqt_image_view rqt_image_view /carla/vehicle1/yolop/inference
```

## 문제 해결

### 일반적인 문제
1. **CARLA 연결 오류**: CARLA 서버가 실행 중인지 확인
2. **Actor 파괴 에러**: 시스템이 안전한 액터 접근 방식으로 개선됨
3. **차량 누적 문제**: Launch 파일 종료 시 자동 정리되며, 수동으로도 cleanup 가능
4. **시간 동기화 실패**: 네트워크 지연이나 처리 속도 차이로 인한 문제
5. **GPU 메모리 부족**: 배치 크기나 이미지 해상도 조정 필요

### 안전성 및 자율주행 개선사항
- **안전한 액터 접근**: 파괴된 액터에 대한 접근을 자동으로 감지하고 처리
- **강력한 예외 처리**: RuntimeError 및 기타 예외에 대한 포괄적 처리
- **자동 복구**: 액터가 파괴되어도 시스템이 계속 동작
- **상태 모니터링**: 차량 및 카메라 상태를 실시간으로 추적
- **Traffic Manager 통합**: 
  - 차량 간 안전거리 자동 유지 (4m)
  - 신호등 및 교통표지판 준수
  - 스마트 차선 변경
  - 속도 제한 준수 (10% 감속)
- **충돌 방지 시스템**: 
  - 안전한 스폰 포인트 자동 선택 (최소 20m 간격)
  - 차량 물리 엔진 최적화
  - 동기 모드로 안정적 시뮬레이션

### 디버깅 팁
```bash
# 토픽 발행 상태 확인
rostopic list | grep carla

# 토픽 주파수 확인
rostopic hz /carla/vehicle1/image_raw

# 로그 확인 (에러 메시지 포함)
roslog list

# 테스트 스크립트로 전체 상태 확인
python3 src/ros-bridge/carla_camera_publisher/scripts/test_batch_system.py

# CARLA 환경 정리 (차량이 너무 많을 때)
roslaunch carla_camera_publisher carla_cleanup.launch instant:=true
```

## 확장 가능성

- 차량 수 조정 (num_vehicles 파라미터)
- 다른 센서 추가 (LiDAR, Depth 등)
- 실시간 경로 계획과의 통합
- 멀티 GPU 지원

## 요구사항

- CARLA 0.9.13+
- ROS Noetic
- PyTorch 1.8+
- CUDA 지원 GPU (권장)
- 충분한 GPU 메모리 (배치 처리용)

---

문의사항이나 문제가 있으면 이슈를 생성해 주세요. 