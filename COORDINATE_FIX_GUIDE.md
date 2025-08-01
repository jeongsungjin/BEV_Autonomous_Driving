# 🔧 BEV-Planner 좌표계 정합 문제 해결 가이드

## 🔍 발견된 문제점들

### 1. **프레임 ID 불일치**
- **문제**: YOLOP 그리드는 `"map"` 프레임, BEV planner 출력은 `"ego_vehicle"` 프레임
- **결과**: 좌표 변환 없이 서로 다른 참조 시스템 사용

### 2. **CARLA-ROS 좌표계 차이**
- **CARLA**: 왼손 좌표계 (X 전진, Y 우측, Z 상향)
- **ROS**: 오른손 좌표계 (X 전진, Y 좌측, Z 상향)
- **문제**: Y 좌표 변환 누락

### 3. **BEV 그리드 방향 정렬 문제**
- **문제**: 카메라 BEV 그리드와 차량 좌표계 간 방향 불일치
- **결과**: 전진 방향과 그리드 방향이 맞지 않음

### 4. **궤적 좌표 변환 누락**
- **문제**: ego vehicle 상대 좌표를 절대 좌표로 변환하지 않음
- **결과**: 궤적이 현재 차량 위치를 반영하지 못함

## 🛠️ 적용된 해결책

### 1. **BEV Planner 노드 수정** (`bev_planner_node.py`)

```python
# ✅ 수정사항:
# - map 프레임으로 통일
# - ego 위치 기반 절대 좌표 변환
# - 차량 방향 고려한 회전 변환

# 좌표 변환 로직:
cos_yaw = np.cos(ego_yaw)
sin_yaw = np.sin(ego_yaw)

# 회전 변환 (ego vehicle 방향 고려)
map_x = rel_x * cos_yaw - rel_y * sin_yaw
map_y = rel_x * sin_yaw + rel_y * cos_yaw

# 절대 좌표 계산
abs_x = ego_pos.x + map_x
abs_y = ego_pos.y + map_y
```

### 2. **YOLOP 추론 노드 수정** (`yolop_inference_node.py`)

```python
# ✅ 수정사항:
# - BEV 그리드 90도 회전으로 방향 정렬
# - ego_vehicle 프레임으로 통일

# 그리드 회전:
det_mask_rotated = np.rot90(det_mask, k=-1)  # 시계방향 90도
da_bin_rotated = np.rot90(da_bin, k=-1)
ll_bin_rotated = np.rot90(ll_bin, k=-1)
```

### 3. **좌표계 설정 파일** (`coordinate_config.yaml`)
- 좌표계 변환 파라미터 중앙화
- 물리적 그리드 크기 정의
- 디버깅 옵션 제공

### 4. **좌표계 검증 도구** (`coordinate_validator.py`)
- 실시간 좌표계 정합성 검증
- 자동 문제 감지 및 리포트
- 시각화를 통한 디버깅 지원

## 🚀 테스트 방법

### 1. **기존 시스템 실행**
```bash
# Terminal 1: YOLOP 추론
roslaunch carla_camera_publisher yolop_inference.launch

# Terminal 2: BEV Planner  
roslaunch bev_planner_integration bev_planner.launch

# Terminal 3: RViz 시각화
rviz -d /path/to/your/config.rviz
```

### 2. **좌표계 검증 실행**
```bash
# Terminal 4: 좌표계 검증기
rosrun bev_planner_integration coordinate_validator.py

# 또는 한 번만 검증:
rosrun bev_planner_integration coordinate_validator.py _mode:=once
```

### 3. **검증 결과 확인**
검증기가 다음을 확인합니다:
- ✅ 프레임 ID 일관성
- ✅ BEV 그리드 방향성  
- ✅ 궤적 좌표 범위
- ✅ 좌표 변환 정확성

## 📊 예상 결과

### **수정 전 (문제 상황)**
```
❌ 궤적이 상하좌우로 왔다갔다
❌ 차량 전진 방향과 무관한 경로
❌ RViz에서 프레임 에러
❌ 비현실적인 좌표 범위
```

### **수정 후 (정상 동작)**
```
✅ 차량 전진 방향 기준 부드러운 궤적
✅ 현재 ego 위치에서 시작하는 경로
✅ 일관된 프레임 ID (map)
✅ 현실적인 좌표 범위 (±50m 이내)
```

## 🔧 추가 조정 옵션

### **그리드 회전각 조정**
만약 방향이 여전히 맞지 않으면:
```python
# yolop_inference_node.py에서 회전각 변경
det_mask_rotated = np.rot90(det_mask, k=-2)  # 180도 회전
# 또는
det_mask_rotated = np.rot90(det_mask, k=1)   # 반시계방향 90도
```

### **좌표 스케일 조정**
물리적 크기와 그리드 해상도 조정:
```yaml
# coordinate_config.yaml
bev_grid:
  physical_width: 60.0    # 더 넓은 시야각
  physical_height: 36.0   # 더 긴 전방 거리
```

## 🐛 문제 해결

### **여전히 방향이 이상한 경우**
1. 좌표계 검증기 실행: `rosrun bev_planner_integration coordinate_validator.py`
2. 그리드 방향성 확인
3. 필요시 회전각 조정

### **궤적이 너무 멀리 시작하는 경우**
1. ego odometry 토픽 확인: `/carla/ego_vehicle/odometry`
2. TF 변환 상태 확인: `rosrun tf tf_echo map ego_vehicle`

### **RViz에서 보이지 않는 경우**
1. 프레임 ID 확인: `rostopic echo /bev_planner/planned_trajectory`
2. RViz Fixed Frame을 `map`으로 설정

## 📈 성능 모니터링

수정 후 다음을 모니터링하세요:
- 궤적 생성 FPS (목표: 20-30 FPS)
- 좌표 변환 지연시간 (목표: <10ms)
- 안전성 점수 (목표: >0.7)

## 📞 지원

문제가 지속되면:
1. 좌표계 검증 결과 로그 확인
2. RViz 스크린샷 첨부
3. 궤적 데이터 분석 (`rostopic echo /bev_planner/planned_trajectory`)

---

**이 가이드를 통해 BEV planner의 좌표계 정합 문제가 해결되어 안정적이고 일관된 경로 생성이 가능해집니다!** 🎯 