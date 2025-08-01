#!/usr/bin/env python3
"""
BEV-Planner Batch Inference ROS Node for CARLA

3대 차량의 YOLOP BEV 마스크들을 구독하여 배치로 실시간 경로 계획을 수행하고
각 차량별로 계획된 궤적을 발행하는 ROS 노드
"""

import os
import sys
import time
import threading
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple

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
import message_filters

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent.parent / "src"))
from adapters import YOLOPToBEVAdapter, BEVFeatureProcessor
from models import SimplifiedBEVPlanner, SafetyChecker, PlanningLoss


class BEVPlannerBatchNode:
    """
    BEV-Planner 배치 추론 ROS 노드
    
    주요 기능:
    1. 3대 차량의 YOLOP BEV 마스크 동시 구독 (detection, drivable area, lane line)
    2. 3대 차량의 Ego vehicle 상태 동시 구독
    3. 배치로 실시간 경로 계획 수행
    4. 각 차량별로 계획된 궤적 발행
    5. 안전성 모니터링
    """
    
    def __init__(self):
        rospy.init_node('bev_planner_batch_node', anonymous=True)
        
        # 차량 수 설정
        self.num_vehicles = 3
        
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
        
        # 상태 변수 (각 차량별로 저장)
        self.latest_det_grids = [None] * self.num_vehicles
        self.latest_da_grids = [None] * self.num_vehicles
        self.latest_ll_grids = [None] * self.num_vehicles
        self.latest_ego_odometries = [None] * self.num_vehicles
        self.last_trajectories = [None] * self.num_vehicles
        
        # 통계
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.safety_scores = [[] for _ in range(self.num_vehicles)]
        
        # 스레드 안전성
        self.data_lock = threading.RLock()
        
        # 추론 루프 시작
        self.planning_thread = threading.Thread(target=self._planning_loop, daemon=True)
        self.planning_thread.start()
        
        rospy.loginfo("🚀 BEV-Planner 배치 추론 노드가 시작되었습니다!")
        
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
                        'yolop_det_grid': '/carla/vehicle1/yolop/det_grid',
                        'yolop_da_grid': '/carla/vehicle1/yolop/da_grid',
                        'yolop_ll_grid': '/carla/vehicle1/yolop/ll_grid',
                        'ego_odometry': '/carla/vehicle1/odometry',
                        'planned_trajectory': '/bev_planner_batch/vehicle1/planned_trajectory',
                        'debug_visualization': '/bev_planner_batch/vehicle1/debug_vis'
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
        
        # === 구독자들 (각 차량별) ===
        self.sub_det_grids = []
        self.sub_da_grids = []
        self.sub_ll_grids = []
        self.sub_ego_odoms = []
        
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            
            # Detection grid 구독
            det_topic = f"/carla/vehicle{vehicle_id}/yolop/det_grid"
            sub_det = message_filters.Subscriber(det_topic, OccupancyGrid)
            self.sub_det_grids.append(sub_det)
            
            # Drivable area grid 구독
            da_topic = f"/carla/vehicle{vehicle_id}/yolop/da_grid"
            sub_da = message_filters.Subscriber(da_topic, OccupancyGrid)
            self.sub_da_grids.append(sub_da)
            
            # Lane line grid 구독
            ll_topic = f"/carla/vehicle{vehicle_id}/yolop/ll_grid"
            sub_ll = message_filters.Subscriber(ll_topic, OccupancyGrid)
            self.sub_ll_grids.append(sub_ll)
            
            # Ego odometry 구독
            odom_topic = f"/carla/vehicle{vehicle_id}/odometry"
            sub_odom = message_filters.Subscriber(odom_topic, Odometry)
            self.sub_ego_odoms.append(sub_odom)
        
        # === 메시지 동기화 ===
        # 모든 차량의 데이터를 동기화 (0.1초 허용 오차)
        all_subs = []
        for i in range(self.num_vehicles):
            all_subs.extend([self.sub_det_grids[i], self.sub_da_grids[i], 
                           self.sub_ll_grids[i], self.sub_ego_odoms[i]])
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            all_subs, queue_size=10, slop=0.1, allow_headerless=True
        )
        self.sync.registerCallback(self._synchronized_callback)
        
        # === 발행자들 (각 차량별) ===
        self.pub_trajectories = []
        self.pub_debug_vises = []
        
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            
            # 궤적 발행
            traj_topic = f"/bev_planner_batch/vehicle{vehicle_id}/planned_trajectory"
            pub_traj = rospy.Publisher(traj_topic, RosPath, queue_size=1)
            self.pub_trajectories.append(pub_traj)
            
            # 디버그 시각화 발행
            debug_topic = f"/bev_planner_batch/vehicle{vehicle_id}/debug_vis"
            pub_debug = rospy.Publisher(debug_topic, MarkerArray, queue_size=1)
            self.pub_debug_vises.append(pub_debug)
        
        # 통계 발행 (디버깅용)
        self.pub_stats = rospy.Publisher('/bev_planner_batch/statistics', Marker, queue_size=1)
        
        rospy.loginfo("✅ ROS 통신 설정 완료")
    
    def _synchronized_callback(self, *msgs):
        """동기화된 메시지들 처리"""
        if len(msgs) != self.num_vehicles * 4:  # 각 차량당 4개 메시지
            rospy.logwarn(f"Expected {self.num_vehicles * 4} messages, got {len(msgs)}")
            return
        
        try:
            # 메시지들을 차량별로 분류
            for i in range(self.num_vehicles):
                base_idx = i * 4
                det_grid = msgs[base_idx]
                da_grid = msgs[base_idx + 1]
                ll_grid = msgs[base_idx + 2]
                ego_odom = msgs[base_idx + 3]
                
                with self.data_lock:
                    self.latest_det_grids[i] = det_grid
                    self.latest_da_grids[i] = da_grid
                    self.latest_ll_grids[i] = ll_grid
                    self.latest_ego_odometries[i] = ego_odom
                    
        except Exception as e:
            rospy.logerr(f"동기화 콜백 처리 오류: {e}")
    
    def _planning_loop(self):
        """메인 계획 루프"""
        rate = rospy.Rate(self.config['performance']['target_fps'])
        
        rospy.loginfo(f"🔄 배치 계획 루프 시작 (목표 FPS: {self.config['performance']['target_fps']})")
        
        while not rospy.is_shutdown():
            try:
                # 필요한 데이터가 모두 있는지 확인
                with self.data_lock:
                    if not self._all_data_available():
                        rate.sleep()
                        continue
                    
                    # 데이터 복사 (스레드 안전성)
                    det_grids = self.latest_det_grids.copy()
                    da_grids = self.latest_da_grids.copy()
                    ll_grids = self.latest_ll_grids.copy()
                    ego_odoms = self.latest_ego_odometries.copy()
                
                # 배치 경로 계획 수행
                start_time = time.time()
                batch_results = self._perform_batch_planning(det_grids, da_grids, ll_grids, ego_odoms)
                inference_time = time.time() - start_time
                
                if batch_results is not None:
                    # 각 차량별로 결과 발행
                    for i, result in enumerate(batch_results):
                        if result is not None:
                            self._publish_results(result, ego_odoms[i].header, i)
                            self._update_statistics(inference_time, result['safety_score'], i)
                            
                            # 디버그 정보 발행
                            if rospy.get_param('~debug', False):
                                self._publish_debug_info(result, ego_odoms[i].header, i)
                
            except Exception as e:
                rospy.logerr(f"❌ 배치 계획 루프 오류: {e}")
                import traceback
                traceback.print_exc()
            
            rate.sleep()
    
    def _all_data_available(self) -> bool:
        """필요한 모든 데이터가 사용 가능한지 확인"""
        for i in range(self.num_vehicles):
            if (self.latest_det_grids[i] is None or
                self.latest_da_grids[i] is None or
                self.latest_ll_grids[i] is None or
                self.latest_ego_odometries[i] is None):
                return False
        return True
    
    def _perform_batch_planning(self, det_grids: List[OccupancyGrid], 
                               da_grids: List[OccupancyGrid],
                               ll_grids: List[OccupancyGrid], 
                               ego_odoms: List[Odometry]) -> Optional[List[Dict]]:
        """배치 경로 계획 수행"""
        try:
            # 1. 배치 텐서 준비
            batch_det_tensors = []
            batch_da_tensors = []
            batch_ll_tensors = []
            batch_ego_tensors = []
            batch_ego_statuses = []
            
            for i in range(self.num_vehicles):
                # OccupancyGrid를 텐서로 변환
                det_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                    det_grids[i], self.config['adapter']['input_height'], 
                    self.config['adapter']['input_width']
                )
                da_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                    da_grids[i], self.config['adapter']['input_height'],
                    self.config['adapter']['input_width']
                )
                ll_tensor = BEVFeatureProcessor.occupancy_grid_to_tensor(
                    ll_grids[i], self.config['adapter']['input_height'],
                    self.config['adapter']['input_width']
                )
                
                batch_det_tensors.append(det_tensor)
                batch_da_tensors.append(da_tensor)
                batch_ll_tensors.append(ll_tensor)
                
                # Ego 상태 추출
                ego_status = self._extract_ego_status(ego_odoms[i])
                batch_ego_statuses.append(ego_status)
                
                # Ego features 생성 (기존 단일 노드와 동일한 방식)
                velocity_magnitude = np.sqrt(ego_status['velocity'][0]**2 + ego_status['velocity'][1]**2)
                ego_tensor = torch.tensor([
                    velocity_magnitude, ego_status['yaw_rate']
                ], dtype=torch.float32)  # [2] 형태로 생성
                batch_ego_tensors.append(ego_tensor)
            
            # 배치 텐서 생성
            batch_det = torch.stack(batch_det_tensors).to(self.device)  # [B, H, W]
            batch_da = torch.stack(batch_da_tensors).to(self.device)    # [B, H, W]
            batch_ll = torch.stack(batch_ll_tensors).to(self.device)    # [B, H, W]
            batch_ego = torch.stack(batch_ego_tensors).to(self.device)  # [B, 2]
            
            # 2. 배치 모델 추론
            with torch.no_grad():
                # YOLOP 어댑터 (배치 처리)
                adapter_output = self.yolop_adapter(
                    batch_det, batch_da, batch_ll, ego_status=None
                )
                
                # BEV-Planner (배치 처리) - 기존 단일 노드와 동일한 방식
                planning_output = self.bev_planner(
                    adapter_output['bev_features'],
                    batch_ego  # 직접 [B, 2] 형태 사용
                )
                
                # 안전성 평가 (배치 처리)
                # batch_det, batch_da는 [B, H, W] 형태여야 함
                collision_risks = self.safety_checker.check_collision_risk(
                    planning_output['trajectory'], batch_det
                )
                
                lane_compliance = self.safety_checker.check_lane_compliance(
                    planning_output['trajectory'], batch_da
                )
            
            # 3. 결과 구성 (각 차량별로)
            batch_results = []
            for i in range(self.num_vehicles):
                safety_score = (1.0 - collision_risks[i].mean()).item() * lane_compliance[i].mean().item()
                
                # 궤적 데이터를 올바른 형태로 추출
                trajectory = planning_output['trajectory'][i].detach().cpu().numpy()  # [num_steps, 2]
                
                result = {
                    'trajectory': trajectory,  # numpy 배열로 저장
                    'confidence': planning_output['confidence'][i].detach().cpu(),
                    'collision_risks': collision_risks[i].detach().cpu(),
                    'lane_compliance': lane_compliance[i].detach().cpu(),
                    'safety_score': safety_score,
                    'ego_status': batch_ego_statuses[i]
                }
                batch_results.append(result)
            
            return batch_results
            
        except Exception as e:
            rospy.logerr(f"❌ 배치 경로 계획 실패: {e}")
            return None
    
    def _extract_ego_status(self, ego_odom: Odometry) -> Dict[str, float]:
        """Odometry 메시지에서 ego 상태 추출"""
        twist = ego_odom.twist.twist
        pose = ego_odom.pose.pose
        
        # 속도 (body frame)
        velocity_x = twist.linear.x
        velocity_y = twist.linear.y
        
        # 각속도
        yaw_rate = twist.angular.z
        
        # 현재 차량 방향 (yaw) 추출
        orientation = pose.orientation
        euler = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        current_yaw = euler[2]  # Z축 회전각 (yaw)
        
        # 조향각 추정
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        if velocity_magnitude > 0.1:
            wheelbase = 2.5
            steering = yaw_rate * wheelbase / velocity_magnitude
            steering = np.clip(steering, -0.5, 0.5)
        else:
            steering = 0.0
        
        return {
            'velocity': [velocity_x, velocity_y],
            'steering': steering,
            'yaw_rate': yaw_rate,
            'acceleration': 0.0,
            'current_yaw': current_yaw
        }
    
    def _publish_results(self, result: Dict, header: Header, vehicle_id: int):
        """계획 결과 발행 (안전성 체크 포함)"""
        # 궤적을 ROS Path 메시지로 변환
        raw_trajectory = result['trajectory']  # 이미 numpy 배열 [num_steps, 2]
        ego_status = result['ego_status']
        
        # 차량 방향 기준 궤적 조정
        adjusted_trajectory = self._adjust_trajectory_direction(raw_trajectory, ego_status)
        
        # 궤적 스무딩
        smoothed_trajectory = self._smooth_trajectory(adjusted_trajectory, vehicle_id)
        
        # 안전성 체크
        safe_trajectory = self._safety_check_trajectory(smoothed_trajectory, vehicle_id)
        
        path_msg = RosPath()
        path_msg.header = header
        path_msg.header.frame_id = f"ego_vehicle_{vehicle_id + 1}"  # CARLA 실제 프레임 ID
        
        for i, (x, y) in enumerate(safe_trajectory):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            
            pose_stamped.pose.position = Point(x=float(x), y=float(y), z=0.0)
            pose_stamped.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            path_msg.poses.append(pose_stamped)
        
        self.pub_trajectories[vehicle_id].publish(path_msg)
        self.last_trajectories[vehicle_id] = path_msg
    
    def _adjust_trajectory_direction(self, trajectory: np.ndarray, ego_status: Dict) -> np.ndarray:
        """차량 방향 기준 궤적 조정 및 좌표축 매핑"""
        if len(trajectory) == 0:
            return trajectory
        
        # 궤적 데이터 형태 확인 및 수정
        if trajectory.ndim == 1:
            # 1차원 배열인 경우 2차원으로 변환
            trajectory = trajectory.reshape(-1, 2)
        elif trajectory.ndim > 2:
            # 3차원 이상인 경우 첫 번째 배치만 사용
            trajectory = trajectory[0]
        
        if len(trajectory) == 0:
            return trajectory
        
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
        best_score = -2
        
        for name, transformed in transformations:
            if len(transformed) > 1:
                vector = transformed[-1] - transformed[0]
                norm = np.linalg.norm(vector)
                if norm > 0:
                    score = vector[0] / norm
                    if score > 0.3 and score > best_score:
                        best_score = score
                        best_transform = transformed
        
        if best_transform is not None:
            return best_transform
        else:
            return trajectory
    
    def _smooth_trajectory(self, trajectory: np.ndarray, vehicle_id: int) -> np.ndarray:
        """궤적 스무딩 (지수 이동 평균 + 가우시안 필터)"""
        if len(trajectory) <= 1:
            return trajectory
            
        smoothed = trajectory.copy()
        
        # 지수 이동 평균으로 기본 스무딩
        alpha = 0.95
        for i in range(1, len(trajectory)):
            smoothed[i] = alpha * trajectory[i] + (1 - alpha) * smoothed[i-1]
        
        # 이전 궤적과의 연속성 보장
        if hasattr(self, 'last_smoothed_trajectories'):
            if (self.last_smoothed_trajectories[vehicle_id] is not None and 
                len(self.last_smoothed_trajectories[vehicle_id]) > 0 and len(smoothed) > 0):
                continuity_weight = 0.8
                smoothed[0] = (continuity_weight * self.last_smoothed_trajectories[vehicle_id][-1] + 
                              (1 - continuity_weight) * smoothed[0])
        
        # 현재 궤적 저장
        if not hasattr(self, 'last_smoothed_trajectories'):
            self.last_smoothed_trajectories = [None] * self.num_vehicles
        self.last_smoothed_trajectories[vehicle_id] = smoothed.copy()
        
        # 가우시안 필터로 추가 스무딩
        from scipy.ndimage import gaussian_filter1d
        try:
            smoothed[:, 0] = gaussian_filter1d(smoothed[:, 0], sigma=1.0)
            smoothed[:, 1] = gaussian_filter1d(smoothed[:, 1], sigma=1.0)
        except ImportError:
            if len(smoothed) >= 3:
                for i in range(1, len(smoothed) - 1):
                    smoothed[i] = (smoothed[i-1] + smoothed[i] + smoothed[i+1]) / 3
        
        # 시작점 조정
        if len(smoothed) > 0:
            desired_start = np.array([2.0, 0.0])
            smoothed[0] = desired_start
            
            if len(smoothed) > 1:
                for i in range(len(smoothed)):
                    weight = 0.95 * (1 - i / len(smoothed))
                    smoothed[i, 1] = smoothed[i, 1] * (1 - weight)
                    
                    max_lateral = 1.5
                    if abs(smoothed[i, 1]) > max_lateral:
                        smoothed[i, 1] = np.sign(smoothed[i, 1]) * max_lateral
                    
                    smoothed[i, 0] = max(smoothed[i, 0], 1.5 + i * 0.8)
        
        return smoothed
    
    def _safety_check_trajectory(self, trajectory: np.ndarray, vehicle_id: int) -> np.ndarray:
        """궤적 안전성 체크"""
        if len(trajectory) == 0:
            return trajectory
        
        safe_trajectory = trajectory.copy()
        
        # 횡방향 제한
        max_lateral = 1.2
        for i in range(len(safe_trajectory)):
            if abs(safe_trajectory[i, 1]) > max_lateral:
                safe_trajectory[i, 1] = np.sign(safe_trajectory[i, 1]) * max_lateral
        
        # 급격한 방향 변화 방지
        if len(safe_trajectory) > 2:
            for i in range(1, len(safe_trajectory) - 1):
                prev_seg = safe_trajectory[i] - safe_trajectory[i-1]
                next_seg = safe_trajectory[i+1] - safe_trajectory[i]
                
                if np.linalg.norm(prev_seg) > 0 and np.linalg.norm(next_seg) > 0:
                    cos_angle = np.dot(prev_seg, next_seg) / (np.linalg.norm(prev_seg) * np.linalg.norm(next_seg))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    if angle > np.pi/4:
                        mid_point = (safe_trajectory[i-1] + safe_trajectory[i+1]) / 2
                        safe_trajectory[i] = 0.5 * safe_trajectory[i] + 0.5 * mid_point
        
        return safe_trajectory
    
    def _publish_debug_info(self, result: Dict, header: Header, vehicle_id: int):
        """디버그 정보 발행"""
        marker_array = MarkerArray()
        
        # 안전성 점수 텍스트
        text_marker = Marker()
        text_marker.header = header
        text_marker.header.frame_id = f"ego_vehicle_{vehicle_id + 1}"  # CARLA 실제 프레임 ID
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = vehicle_id
        
        text_marker.pose.position = Point(x=5.0, y=0.0, z=2.0)
        text_marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        
        text_marker.text = f"V{vehicle_id + 1} Safety: {result['safety_score']:.3f}\nFPS: {self.get_avg_fps():.1f}"
        text_marker.scale.z = 1.0
        text_marker.color.r = 1.0 if result['safety_score'] > 0.5 else 0.0
        text_marker.color.g = 1.0 if result['safety_score'] > 0.5 else 0.0
        text_marker.color.b = 0.0
        text_marker.color.a = 1.0
        
        marker_array.markers.append(text_marker)
        self.pub_debug_vises[vehicle_id].publish(marker_array)
    
    def _update_statistics(self, inference_time: float, safety_score: float, vehicle_id: int):
        """통계 업데이트"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.safety_scores[vehicle_id].append(safety_score)
        
        # 최근 100개만 유지
        if len(self.safety_scores[vehicle_id]) > 100:
            self.safety_scores[vehicle_id].pop(0)
        
        # 주기적으로 통계 출력
        if self.inference_count % 100 == 0:
            avg_fps = self.get_avg_fps()
            avg_safety = np.mean(self.safety_scores[vehicle_id])
            rospy.loginfo(f"📊 V{vehicle_id + 1} 통계 - FPS: {avg_fps:.1f}, 평균 안전도: {avg_safety:.3f}")
    
    def get_avg_fps(self) -> float:
        """평균 FPS 계산"""
        if self.inference_count == 0:
            return 0.0
        return self.inference_count / self.total_inference_time
    
    def get_statistics(self) -> Dict:
        """현재 통계 반환"""
        avg_safety_scores = []
        for scores in self.safety_scores:
            avg_safety_scores.append(np.mean(scores) if scores else 0.0)
        
        return {
            'inference_count': self.inference_count,
            'avg_fps': self.get_avg_fps(),
            'avg_safety_scores': avg_safety_scores,
            'has_latest_trajectories': [traj is not None for traj in self.last_trajectories]
        }


def main():
    """메인 함수"""
    try:
        node = BEVPlannerBatchNode()
        
        # 노드 상태 모니터링
        def print_status():
            while not rospy.is_shutdown():
                stats = node.get_statistics()
                rospy.loginfo_throttle(10, 
                    f"🔍 배치 상태 - 추론 횟수: {stats['inference_count']}, "
                    f"FPS: {stats['avg_fps']:.1f}, "
                    f"평균 안전도: {np.mean(stats['avg_safety_scores']):.3f}")
                rospy.sleep(10)
        
        status_thread = threading.Thread(target=print_status, daemon=True)
        status_thread.start()
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 BEV-Planner 배치 노드 종료")
    except Exception as e:
        rospy.logerr(f"❌ 노드 실행 오류: {e}")


if __name__ == '__main__':
    main() 