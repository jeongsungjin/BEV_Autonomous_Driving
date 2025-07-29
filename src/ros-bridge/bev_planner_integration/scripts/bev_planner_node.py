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
        
        # BEV-Planner
        self.bev_planner = SimplifiedBEVPlanner(
            bev_embed_dim=self.config['adapter']['embed_dim'],
            ego_embed_dim=2,  # 실제 ego 데이터 차원에 맞춤
            hidden_dim=512,
            num_future_steps=self.config['planner']['prediction_horizon'],
            max_speed=self.config['planner']['max_speed'],
            safety_margin=2.0
        ).to(self.device)
        
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
                
                # Ego features를 직접 생성 (학습 시와 동일한 형태)
                velocity_magnitude = np.sqrt(ego_status['velocity'][0]**2 + ego_status['velocity'][1]**2)
                ego_tensor = torch.tensor([
                    [velocity_magnitude, ego_status['yaw_rate']]  # [velocity_magnitude, angular_velocity]
                ], dtype=torch.float32).to(self.device)
                
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
        
        # 속도 (body frame)
        velocity_x = twist.linear.x
        velocity_y = twist.linear.y
        
        # 각속도 (이미 학습된 모델이므로 odometry에서 바로 사용)
        yaw_rate = twist.angular.z
        
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
        
        # 디버깅용 로그 (각속도 모니터링)
        if abs(yaw_rate) > 0.01:  # 각속도가 있을 때만 로그
            rospy.logdebug(f"🔄 각속도 감지: {yaw_rate:.3f} rad/s, 조향각: {steering:.3f}")
        
        return {
            'velocity': [velocity_x, velocity_y],
            'steering': steering,
            'yaw_rate': yaw_rate,
            'acceleration': acceleration
        }
    
    def _publish_results(self, result: Dict, header: Header):
        """계획 결과 발행"""
        # 궤적을 ROS Path 메시지로 변환
        trajectory = result['trajectory'][0].numpy()  # [num_steps, 2]
        
        path_msg = RosPath()
        path_msg.header = header
        path_msg.header.frame_id = "ego_vehicle"  # Ego vehicle 기준
        
        for i, (x, y) in enumerate(trajectory):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            
            # 상대 좌표 (ego vehicle 기준)
            pose_stamped.pose.position = Point(x=float(x), y=float(y), z=0.0)
            pose_stamped.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)  # 방향은 단순화
            
            path_msg.poses.append(pose_stamped)
        
        self.pub_trajectory.publish(path_msg)
        self.last_trajectory = path_msg
    
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