#!/usr/bin/env python3
"""
BEV-Planner Integration ROS Node for CARLA

YOLOP의 BEV 마스크들을 구독하여 실시간 경로 계획을 수행하고
계획된 궤적을 발행하는 ROS 노드
"""

import os
import sys
import time
import threading
import yaml
from pathlib import Path
from typing import Dict, Optional

import rospy
import torch
import numpy as np
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, Path as RosPath, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf.transformations

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent.parent / "src"))
from adapters import YOLOPToBEVAdapter, BEVFeatureProcessor
from models import SimplifiedBEVPlanner, SafetyChecker, PlanningLoss


class BEVPlannerNode:
    """
    BEV-Planner 통합 ROS 노드
    
    주요 기능:
    1. YOLOP BEV 마스크 구독 (detection, drivable area, lane line)
    2. Ego vehicle 상태 구독
    3. 실시간 경로 계획 수행
    4. 계획된 궤적 발행
    5. 안전성 모니터링
    """
    
    def __init__(self):
        rospy.init_node('bev_planner_node', anonymous=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # GPU/CPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config['model']['device'] == 'cuda' else 'cpu')
        rospy.loginfo(f"🖥️  사용 디바이스: {self.device}")
        
        # 모델 초기화
        self._initialize_models()
        
        # ROS 통신 설정
        self._setup_ros_communication()
        
        # 상태 변수
        self.latest_det_grid = None
        self.latest_da_grid = None
        self.latest_ll_grid = None
        self.latest_ego_odometry = None
        self.last_trajectory = None
        

        
        # 통계
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.safety_scores = []
        
        # 스레드 안전성
        self.data_lock = threading.RLock()
        
        # 추론 루프 시작
        self.planning_thread = threading.Thread(target=self._planning_loop, daemon=True)
        self.planning_thread.start()
        
        rospy.loginfo("🚀 BEV-Planner 노드가 시작되었습니다!")
        
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        config_path = Path(__file__).parent.parent / "config" / "bev_planner_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            rospy.loginfo(f"✅ 설정 파일 로드: {config_path}")
        else:
            # 기본 설정
            config = {
                'adapter': {'input_height': 48, 'input_width': 80, 'embed_dim': 256},
                'planner': {'prediction_horizon': 6, 'time_step': 0.5, 'max_speed': 15.0},
                'carla': {
                    'topics': {
                        'yolop_det_grid': '/carla/yolop/det_grid',
                        'yolop_da_grid': '/carla/yolop/da_grid',
                        'yolop_ll_grid': '/carla/yolop/ll_grid',
                        'ego_odometry': '/carla/ego_vehicle/odometry',
                        'planned_trajectory': '/bev_planner/planned_trajectory',
                        'debug_visualization': '/bev_planner/debug_vis'
                    }
                },
                'model': {'device': 'cuda'},
                'performance': {'target_fps': 30}
            }
            rospy.logwarn(f"⚠️  기본 설정 사용 (설정 파일 없음: {config_path})")
            
        return config
    
    def _initialize_models(self):
        """모델 초기화"""
        rospy.loginfo("🔧 모델 초기화 중...")
        
        # YOLOP 어댑터
        self.yolop_adapter = YOLOPToBEVAdapter(
            input_height=self.config['adapter']['input_height'],
            input_width=self.config['adapter']['input_width'],
            embed_dim=self.config['adapter']['embed_dim'],
            use_positional_encoding=True
        ).to(self.device)
        
        # BEV-Planner (기존 학습된 모델과 호환)
        self.bev_planner = SimplifiedBEVPlanner(
            bev_embed_dim=self.config['adapter']['embed_dim'],
            ego_embed_dim=2,  # velocity_magnitude, yaw_rate (기존과 동일)
            hidden_dim=512,
            num_future_steps=self.config['planner']['prediction_horizon'],
            max_speed=self.config['planner']['max_speed'],
            safety_margin=2.0
        ).to(self.device)
        
        # 테스트용: 체크포인트 로드 강제 비활성화 (랜덤 가중치 테스트용)
        self.force_random_weights = rospy.get_param('~use_random_weights', False)
        
        # 안전성 검사기
        self.safety_checker = SafetyChecker()
        
        # 체크포인트 로드
        self._load_checkpoint()
        
        # 평가 모드로 설정
        self.yolop_adapter.eval()
        self.bev_planner.eval()
        
        rospy.loginfo("✅ 모델 초기화 완료")
    
    def _load_checkpoint(self):
        """학습된 체크포인트 로드"""
        # 테스트용 랜덤 가중치 강제 사용
        if self.force_random_weights:
            rospy.loginfo("🎲 테스트용: 랜덤 가중치 사용")
            return
            
        checkpoint_path = "/home/carla/capstone_2025/training_results_v2/checkpoints/best_checkpoint.pth"
        
        if os.path.exists(checkpoint_path):
            try:
                rospy.loginfo(f"🔄 체크포인트 로드 중: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # 모델 가중치 로드
                self.yolop_adapter.load_state_dict(checkpoint['adapter_state_dict'])
                self.bev_planner.load_state_dict(checkpoint['model_state_dict'])
                
                # 학습 정보 출력
                epoch = checkpoint.get('epoch', 'Unknown')
                val_loss = checkpoint.get('best_val_loss', 'Unknown')
                global_step = checkpoint.get('global_step', 'Unknown')
                
                rospy.loginfo(f"✅ 체크포인트 로드 성공!")
                rospy.loginfo(f"   - 에포크: {epoch}")
                rospy.loginfo(f"   - 검증 손실: {val_loss}")
                rospy.loginfo(f"   - 글로벌 스텝: {global_step}")
                
            except Exception as e:
                rospy.logwarn(f"⚠️  체크포인트 로드 실패: {e}")
                rospy.logwarn("   - 랜덤 가중치로 시작합니다")
        else:
            rospy.logwarn(f"⚠️  체크포인트 없음: {checkpoint_path}")
            rospy.logwarn("   - 랜덤 가중치로 시작합니다")
    
    def _setup_ros_communication(self):
        """ROS 토픽 및 서비스 설정"""
        topics = self.config['carla']['topics']
        
        # 구독자들
        self.sub_det_grid = rospy.Subscriber(
            topics['yolop_det_grid'], OccupancyGrid, 
            self._det_grid_callback, queue_size=1
        )
        self.sub_da_grid = rospy.Subscriber(
            topics['yolop_da_grid'], OccupancyGrid,
            self._da_grid_callback, queue_size=1  
        )
        self.sub_ll_grid = rospy.Subscriber(
            topics['yolop_ll_grid'], OccupancyGrid,
            self._ll_grid_callback, queue_size=1
        )
        self.sub_ego_odom = rospy.Subscriber(
            topics['ego_odometry'], Odometry,
            self._ego_odom_callback, queue_size=1
        )
        

        
        # 발행자들
        self.pub_trajectory = rospy.Publisher(
            topics['planned_trajectory'], RosPath, queue_size=1
        )
        self.pub_debug_vis = rospy.Publisher(
            topics['debug_visualization'], MarkerArray, queue_size=1
        )
        
        # 통계 발행 (디버깅용)
        self.pub_stats = rospy.Publisher(
            '/bev_planner/statistics', Marker, queue_size=1
        )
        
        rospy.loginfo("✅ ROS 통신 설정 완료")
    
    def _det_grid_callback(self, msg: OccupancyGrid):
        """Detection grid 콜백"""
        with self.data_lock:
            self.latest_det_grid = msg
    
    def _da_grid_callback(self, msg: OccupancyGrid):
        """Drivable area grid 콜백"""
        with self.data_lock:
            self.latest_da_grid = msg
            # 안전성 체크를 위해 DA 그리드 저장
            self.last_da_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
    
    def _ll_grid_callback(self, msg: OccupancyGrid):
        """Lane line grid 콜백"""
        with self.data_lock:
            self.latest_ll_grid = msg
    

    
    def _ego_odom_callback(self, msg: Odometry):
        """Ego vehicle odometry 콜백"""
        with self.data_lock:
            self.latest_ego_odometry = msg
    
    def _planning_loop(self):
        """메인 계획 루프"""
        rate = rospy.Rate(self.config['performance']['target_fps'])
        
        rospy.loginfo(f"🔄 계획 루프 시작 (목표 FPS: {self.config['performance']['target_fps']})")
        
        while not rospy.is_shutdown():
            try:
                # 필요한 데이터가 모두 있는지 확인
                with self.data_lock:
                    if not self._all_data_available():
                        rate.sleep()
                        continue
                    
                    # 데이터 복사 (스레드 안전성)
                    det_grid = self.latest_det_grid
                    da_grid = self.latest_da_grid
                    ll_grid = self.latest_ll_grid
                    ego_odom = self.latest_ego_odometry
                
                # 경로 계획 수행
                start_time = time.time()
                result = self._perform_planning(det_grid, da_grid, ll_grid, ego_odom)
                inference_time = time.time() - start_time
                
                if result is not None:
                    # 결과 발행
                    self._publish_results(result, ego_odom.header)
                    
                    # 통계 업데이트
                    self._update_statistics(inference_time, result['safety_score'])
                    
                    # 디버그 정보 발행
                    if rospy.get_param('~debug', False):
                        self._publish_debug_info(result, ego_odom.header)
                
            except Exception as e:
                rospy.logerr(f"❌ 계획 루프 오류: {e}")
                import traceback
                traceback.print_exc()
            
            rate.sleep()
    
    def _all_data_available(self) -> bool:
        """필요한 모든 데이터가 사용 가능한지 확인"""
        return (self.latest_det_grid is not None and
                self.latest_da_grid is not None and
                self.latest_ll_grid is not None and
                self.latest_ego_odometry is not None)
    
    def _perform_planning(self, det_grid: OccupancyGrid, da_grid: OccupancyGrid,
                         ll_grid: OccupancyGrid, ego_odom: Odometry) -> Optional[Dict]:
        """실제 경로 계획 수행"""
        try:
            # 1. OccupancyGrid를 텐서로 변환
            det_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                det_grid, self.config['adapter']['input_height'], 
                self.config['adapter']['input_width']
            ).unsqueeze(0)  # 배치 차원 추가
            
            da_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                da_grid, self.config['adapter']['input_height'],
                self.config['adapter']['input_width']
            ).unsqueeze(0)
            
            ll_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                ll_grid, self.config['adapter']['input_height'],
                self.config['adapter']['input_width']
            ).unsqueeze(0)
            
            # 2. Ego 상태 추출
            ego_status = self._extract_ego_status(ego_odom)
            
            # 3. 모델 추론
            with torch.no_grad():
                # YOLOP 어댑터 (ego_status 없이 실행)
                adapter_output = self.yolop_adapter(
                    det_tensor.to(self.device),
                    da_tensor.to(self.device), 
                    ll_tensor.to(self.device),
                    ego_status=None  # ego_status를 전달하지 않음
                )
                
                # Ego features를 직접 생성 (기존 학습된 모델과 호환)
                velocity_magnitude = np.sqrt(ego_status['velocity'][0]**2 + ego_status['velocity'][1]**2)
                
                # 기존 모델과 호환: 2D ego tensor (velocity_magnitude, yaw_rate)
                ego_tensor = torch.tensor([
                    [velocity_magnitude, ego_status['yaw_rate']]  # 기존과 동일한 형태
                ], dtype=torch.float32).to(self.device)
                
                # yaw 정보는 궤적 후처리에서 활용 (ego_status에 저장됨)
                
                # BEV-Planner
                planning_output = self.bev_planner(
                    adapter_output['bev_features'],
                    ego_tensor  # 직접 생성한 ego tensor 사용
                )
                
                # 안전성 평가
                collision_risks = self.safety_checker.check_collision_risk(
                    planning_output['trajectory'], det_tensor.to(self.device)
                )
                
                lane_compliance = self.safety_checker.check_lane_compliance(
                    planning_output['trajectory'], da_tensor.to(self.device)
                )
            
            # 4. 결과 구성
            safety_score = (1.0 - collision_risks.mean()).item() * lane_compliance.mean().item()
            
            result = {
                'trajectory': planning_output['trajectory'].detach().cpu(),
                'confidence': planning_output['confidence'].detach().cpu(),
                'collision_risks': collision_risks.detach().cpu(),
                'lane_compliance': lane_compliance.detach().cpu(),
                'safety_score': safety_score,
                'ego_status': ego_status
            }
            
            return result
            
        except Exception as e:
            rospy.logerr(f"❌ 경로 계획 실패: {e}")
            return None
    
    def _extract_ego_status(self, ego_odom: Odometry) -> Dict[str, float]:
        """Odometry 메시지에서 ego 상태 추출"""
        twist = ego_odom.twist.twist
        pose = ego_odom.pose.pose
        
        # 속도 (body frame)
        velocity_x = twist.linear.x
        velocity_y = twist.linear.y
        
        # 각속도 (이미 학습된 모델이므로 odometry에서 바로 사용)
        yaw_rate = twist.angular.z
        
        # 현재 차량 방향 (yaw) 추출 - 핵심 추가!
        orientation = pose.orientation
        euler = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        current_yaw = euler[2]  # Z축 회전각 (yaw)
        
        # 조향각 추정 (차량 동역학 기반)
        # 단순한 자전거 모델: steering ≈ yaw_rate * wheelbase / velocity
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        if velocity_magnitude > 0.1:  # 정지 상태가 아닐 때만
            wheelbase = 2.5  # CARLA 기본 차량 휠베이스 (m)
            steering = yaw_rate * wheelbase / velocity_magnitude
            steering = np.clip(steering, -0.5, 0.5)  # 조향각 제한
        else:
            steering = 0.0
        
        # 가속도 (이전 속도와 비교, 여기서는 0으로 설정)
        acceleration = 0.0
        
        # 디버깅용 로그 (방향 정보 모니터링)
        if abs(yaw_rate) > 0.01:  # 각속도가 있을 때만 로그
            rospy.logdebug(f"🔄 Yaw: {np.degrees(current_yaw):.1f}°, 각속도: {yaw_rate:.3f} rad/s")
        
        return {
            'velocity': [velocity_x, velocity_y],
            'steering': steering,
            'yaw_rate': yaw_rate,
            'acceleration': acceleration,
            'current_yaw': current_yaw  # 현재 차량 방향 추가
        }
    
    def _publish_results(self, result: Dict, header: Header):
        """계획 결과 발행 (안전성 체크 포함)"""
        # 궤적을 ROS Path 메시지로 변환
        raw_trajectory = result['trajectory'][0].numpy()  # [num_steps, 2]
        ego_status = result['ego_status']
        
        # 차량 방향 기준 궤적 조정
        adjusted_trajectory = self._adjust_trajectory_direction(raw_trajectory, ego_status)
        
        # 궤적 스무딩
        smoothed_trajectory = self._smooth_trajectory(adjusted_trajectory)
        
        # 안전성 체크 (차선 이탈 방지)
        safe_trajectory = self._safety_check_trajectory(smoothed_trajectory)
        
        path_msg = RosPath()
        path_msg.header = header
        path_msg.header.frame_id = "ego_vehicle"  # 차량 기준 상대 좌표계
        
        for i, (x, y) in enumerate(safe_trajectory):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            
            # 차량 기준 상대 좌표 (ego_vehicle 프레임)
            pose_stamped.pose.position = Point(x=float(x), y=float(y), z=0.0)
            pose_stamped.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            path_msg.poses.append(pose_stamped)
        
        self.pub_trajectory.publish(path_msg)
        self.last_trajectory = path_msg
    
    def _adjust_trajectory_direction(self, trajectory: np.ndarray, ego_status: Dict) -> np.ndarray:
        """차량 방향 기준 궤적 조정 및 좌표축 매핑"""
        if len(trajectory) == 0:
            return trajectory
        
        # 원본 궤적 정보 로깅 (디버그용)
        if len(trajectory) > 1:
            orig_vector = trajectory[-1] - trajectory[0]
            orig_angle = np.arctan2(orig_vector[1], orig_vector[0])
            rospy.logdebug(f"🔄 원본 궤적: 벡터({orig_vector[0]:.2f}, {orig_vector[1]:.2f}), 각도: {np.degrees(orig_angle):.1f}°")
        
        # 여러 변환을 시도해서 전진 방향(+X)에 가장 가까운 것 선택
        transformations = [
            ("원본", trajectory),
            ("X↔Y 교환", np.column_stack([trajectory[:, 1], trajectory[:, 0]])),
            ("X 반전", np.column_stack([-trajectory[:, 0], trajectory[:, 1]])),
            ("Y 반전", np.column_stack([trajectory[:, 0], -trajectory[:, 1]])),
            ("90도 회전", np.column_stack([-trajectory[:, 1], trajectory[:, 0]])),
            ("-90도 회전", np.column_stack([trajectory[:, 1], -trajectory[:, 0]])),
        ]
        
        best_transform = None
        best_score = -2  # 최대 +1이므로 -2부터 시작
        best_name = ""
        
        for name, transformed in transformations:
            if len(transformed) > 1:
                vector = transformed[-1] - transformed[0]
                # 전진 방향(+X)과의 코사인 유사도 계산
                norm = np.linalg.norm(vector)
                if norm > 0:
                    score = vector[0] / norm  # X 성분의 정규화된 값
                    rospy.logdebug(f"  {name}: 벡터({vector[0]:.2f}, {vector[1]:.2f}), 점수: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_transform = transformed
                        best_name = name
        
        if best_transform is not None:
            # 처음이거나 변환이 바뀔 때만 로그 출력
            if not hasattr(self, '_last_transform') or self._last_transform != best_name:
                rospy.loginfo(f"✅ 최적 변환 선택: {best_name} (점수: {best_score:.3f})")
                self._last_transform = best_name
            return best_transform
        else:
            rospy.logwarn("⚠️ 적절한 변환을 찾지 못함, 원본 사용")
            return trajectory
    
    def _smooth_trajectory(self, trajectory: np.ndarray, alpha: float = 0.9) -> np.ndarray:
        """궤적 스무딩 (지수 이동 평균 + 가우시안 필터)"""
        if len(trajectory) <= 1:
            return trajectory
            
        smoothed = trajectory.copy()
        
        # 1. 지수 이동 평균으로 기본 스무딩 (더 강한 스무딩)
        for i in range(1, len(trajectory)):
            smoothed[i] = alpha * trajectory[i] + (1 - alpha) * smoothed[i-1]
        
        # 2. 가우시안 필터로 추가 스무딩 (요동 제거)
        from scipy.ndimage import gaussian_filter1d
        try:
            # X, Y 좌표를 각각 필터링
            smoothed[:, 0] = gaussian_filter1d(smoothed[:, 0], sigma=0.5)
            smoothed[:, 1] = gaussian_filter1d(smoothed[:, 1], sigma=0.5)
        except ImportError:
            # scipy가 없으면 단순한 이동 평균
            if len(smoothed) >= 3:
                for i in range(1, len(smoothed) - 1):
                    smoothed[i] = (smoothed[i-1] + smoothed[i] + smoothed[i+1]) / 3
        
        # 3. 시작점을 차량 전방으로 조정 (더 보수적으로)
        if len(smoothed) > 0:
            # 더 보수적인 시작점 설정 (더 멀리, 중앙 고정)
            desired_start = np.array([2.0, 0.0])  # 차량 전방 2m (더 멀리)
            smoothed[0] = desired_start
            
            # 전체 궤적을 더 보수적으로 조정
            # Y 좌표 (횡방향)를 0에 가깝게 조정 (중앙선 유지) - 더 강화
            if len(smoothed) > 1:
                for i in range(len(smoothed)):
                    # 더 강한 중앙선 고정 (0.8 → 0.95)
                    weight = 0.95 * (1 - i / len(smoothed))  # 앞쪽일수록 강하게 중앙 고정
                    smoothed[i, 1] = smoothed[i, 1] * (1 - weight)  # Y를 0에 가깝게
                    
                    # 횡방향 제한 (차선 이탈 방지)
                    max_lateral = 1.5  # 최대 횡방향 이동 제한 (1.5m)
                    if abs(smoothed[i, 1]) > max_lateral:
                        smoothed[i, 1] = np.sign(smoothed[i, 1]) * max_lateral
                    
                    # X는 점진적으로 증가 (안전한 전진)
                    smoothed[i, 0] = max(smoothed[i, 0], 1.5 + i * 0.8)  # 최소 1.5m부터 시작
        
        return smoothed
    
    def _safety_check_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """궤적 안전성 체크 (차선 이탈 방지)"""
        if len(trajectory) == 0:
            return trajectory
        
        safe_trajectory = trajectory.copy()
        
        # 1. 횡방향 제한 (차선 이탈 방지)
        max_lateral = 1.2  # 최대 횡방향 이동 (1.2m)
        for i in range(len(safe_trajectory)):
            if abs(safe_trajectory[i, 1]) > max_lateral:
                safe_trajectory[i, 1] = np.sign(safe_trajectory[i, 1]) * max_lateral
                rospy.logdebug(f"🛡️ 횡방향 제한 적용: {trajectory[i, 1]:.2f} → {safe_trajectory[i, 1]:.2f}")
        
        # 2. 주행 가능 영역 체크 (DA 그리드 기반)
        if hasattr(self, 'last_da_grid') and self.last_da_grid is not None:
            da_grid = self.last_da_grid
            h, w = da_grid.shape
            
            # 궤적 끝점이 주행 가능 영역을 벗어나는지 확인
            if len(safe_trajectory) > 0:
                end_point = safe_trajectory[-1]
                
                # 그리드 좌표로 변환 (단순 추정)
                grid_x = int(w/2 + end_point[1] * 5)  # Y는 좌우
                grid_y = int(h/2 - end_point[0] * 5)  # X는 전후
                
                if 0 <= grid_x < w and 0 <= grid_y < h:
                    # 주행 불가능 영역이면 궤적 조정
                    if da_grid[grid_y, grid_x] > 50:  # occupied
                        rospy.logwarn("🚨 궤적 끝점이 주행 불가능 영역! 안전한 경로로 조정")
                        
                        # 안전한 방향으로 궤적 조정
                        safe_direction = np.array([1.0, 0.0])  # 전진 방향
                        safe_trajectory[-1] = safe_trajectory[-2] + safe_direction * 2.0  # 2m 전진
        
        # 3. 급격한 방향 변화 방지
        if len(safe_trajectory) > 2:
            for i in range(1, len(safe_trajectory) - 1):
                prev_seg = safe_trajectory[i] - safe_trajectory[i-1]
                next_seg = safe_trajectory[i+1] - safe_trajectory[i]
                
                if np.linalg.norm(prev_seg) > 0 and np.linalg.norm(next_seg) > 0:
                    cos_angle = np.dot(prev_seg, next_seg) / (np.linalg.norm(prev_seg) * np.linalg.norm(next_seg))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    # 60도 이상 급회전 방지
                    if angle > np.pi/3:  # 60도
                        rospy.logdebug(f"🛡️ 급회전 방지: {np.degrees(angle):.1f}° → 조정")
                        
                        # 부드러운 곡선으로 조정
                        mid_point = (safe_trajectory[i-1] + safe_trajectory[i+1]) / 2
                        safe_trajectory[i] = 0.7 * safe_trajectory[i] + 0.3 * mid_point
        
        return safe_trajectory
    
    def _publish_debug_info(self, result: Dict, header: Header):
        """디버그 정보 발행"""
        marker_array = MarkerArray()
        
        # 안전성 점수 텍스트
        text_marker = Marker()
        text_marker.header = header
        text_marker.header.frame_id = "ego_vehicle"
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = 0
        
        text_marker.pose.position = Point(x=5.0, y=0.0, z=2.0)
        text_marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        
        text_marker.text = f"Safety: {result['safety_score']:.3f}\nFPS: {self.get_avg_fps():.1f}"
        text_marker.scale.z = 1.0
        text_marker.color.r = 1.0 if result['safety_score'] > 0.5 else 0.0
        text_marker.color.g = 1.0 if result['safety_score'] > 0.5 else 0.0
        text_marker.color.b = 0.0
        text_marker.color.a = 1.0
        
        marker_array.markers.append(text_marker)
        self.pub_debug_vis.publish(marker_array)
    
    def _update_statistics(self, inference_time: float, safety_score: float):
        """통계 업데이트"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.safety_scores.append(safety_score)
        
        # 최근 100개만 유지
        if len(self.safety_scores) > 100:
            self.safety_scores.pop(0)
        
        # 주기적으로 통계 출력
        if self.inference_count % 100 == 0:
            avg_fps = self.get_avg_fps()
            avg_safety = np.mean(self.safety_scores)
            rospy.loginfo(f"📊 통계 - FPS: {avg_fps:.1f}, 평균 안전도: {avg_safety:.3f}")
    
    def get_avg_fps(self) -> float:
        """평균 FPS 계산"""
        if self.inference_count == 0:
            return 0.0
        return self.inference_count / self.total_inference_time
    
    def get_statistics(self) -> Dict:
        """현재 통계 반환"""
        return {
            'inference_count': self.inference_count,
            'avg_fps': self.get_avg_fps(),
            'avg_safety_score': np.mean(self.safety_scores) if self.safety_scores else 0.0,
            'has_latest_trajectory': self.last_trajectory is not None
        }
    



def main():
    """메인 함수"""
    try:
        node = BEVPlannerNode()
        
        # 노드 상태 모니터링
        def print_status():
            while not rospy.is_shutdown():
                stats = node.get_statistics()
                rospy.loginfo_throttle(10, 
                    f"🔍 상태 - 추론 횟수: {stats['inference_count']}, "
                    f"FPS: {stats['avg_fps']:.1f}, "
                    f"안전도: {stats['avg_safety_score']:.3f}")
                rospy.sleep(10)
        
        status_thread = threading.Thread(target=print_status, daemon=True)
        status_thread.start()
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 BEV-Planner 노드 종료")
    except Exception as e:
        rospy.logerr(f"❌ 노드 실행 오류: {e}")


if __name__ == '__main__':
    main() 