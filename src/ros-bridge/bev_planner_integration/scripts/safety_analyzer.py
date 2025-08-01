#!/usr/bin/env python3
"""
BEV-Planner 실제 안전성 분석기

수치적 성능이 아닌 실제 주행 안전성을 중심으로 궤적을 분석합니다.
"""

import rospy
import numpy as np
import time
from nav_msgs.msg import Path as RosPath, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import cv2


class SafetyAnalyzer:
    """실제 안전성 분석기"""
    
    def __init__(self):
        rospy.init_node('safety_analyzer', anonymous=True)
        
        self.trajectories = []
        self.ego_positions = []
        self.da_grids = []  # 주행 가능 영역
        self.det_grids = []  # 장애물 감지
        self.collecting = False
        
        # 구독자 설정
        rospy.Subscriber('/bev_planner/planned_trajectory', RosPath,
                        self._trajectory_callback, queue_size=1)
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry,
                        self._ego_callback, queue_size=1)
        rospy.Subscriber('/carla/yolop/da_grid', OccupancyGrid,
                        self._da_grid_callback, queue_size=1)
        rospy.Subscriber('/carla/yolop/det_grid', OccupancyGrid,
                        self._det_grid_callback, queue_size=1)
        
        rospy.loginfo("🛡️ 실제 안전성 분석기 시작!")
        
    def _trajectory_callback(self, msg: RosPath):
        """궤적 콜백"""
        if self.collecting and msg.poses:
            trajectory_points = []
            for pose in msg.poses:
                trajectory_points.append([
                    pose.pose.position.x,
                    pose.pose.position.y
                ])
            
            self.trajectories.append({
                'timestamp': time.time(),
                'points': np.array(trajectory_points),
                'frame_id': msg.header.frame_id
            })
    
    def _ego_callback(self, msg: Odometry):
        """Ego vehicle 콜백"""
        if self.collecting:
            pos = msg.pose.pose.position
            vel = msg.twist.twist.linear
            speed = np.sqrt(vel.x**2 + vel.y**2)
            
            self.ego_positions.append({
                'timestamp': time.time(),
                'x': pos.x, 'y': pos.y, 'z': pos.z,
                'speed': speed
            })
    
    def _da_grid_callback(self, msg: OccupancyGrid):
        """주행 가능 영역 콜백"""
        if self.collecting:
            grid_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            self.da_grids.append({
                'timestamp': time.time(),
                'data': grid_data,
                'resolution': msg.info.resolution
            })
    
    def _det_grid_callback(self, msg: OccupancyGrid):
        """장애물 감지 콜백"""
        if self.collecting:
            grid_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            self.det_grids.append({
                'timestamp': time.time(),
                'data': grid_data,
                'resolution': msg.info.resolution
            })
    
    def start_collection(self, duration: int = 30):
        """데이터 수집 시작"""
        rospy.loginfo(f"🛡️ {duration}초간 안전성 데이터 수집 시작...")
        rospy.loginfo("   다양한 상황에서 주행해주세요:")
        rospy.loginfo("   - 직진, 좌우 회전")
        rospy.loginfo("   - 장애물 근처 주행")
        rospy.loginfo("   - 차선 변경 상황")
        
        self.trajectories.clear()
        self.ego_positions.clear()
        self.da_grids.clear()
        self.det_grids.clear()
        
        self.collecting = True
        time.sleep(duration)
        self.collecting = False
        
        rospy.loginfo(f"✅ 안전성 분석 데이터 수집 완료")
        rospy.loginfo(f"   궤적: {len(self.trajectories)}개")
        rospy.loginfo(f"   DA 그리드: {len(self.da_grids)}개")
        rospy.loginfo(f"   장애물 그리드: {len(self.det_grids)}개")
    
    def analyze_safety(self) -> Dict:
        """실제 안전성 분석"""
        if not self.trajectories:
            rospy.logwarn("⚠️ 분석할 데이터가 없습니다")
            return {}
        
        safety_metrics = {
            'collision_risk': self._analyze_collision_risk(),
            'lane_departure_risk': self._analyze_lane_departure(),
            'emergency_situations': self._detect_emergency_situations(),
            'conservative_score': self._measure_conservativeness(),
            'predictability': self._measure_predictability(),
            'reaction_time': self._measure_reaction_time()
        }
        
        return safety_metrics
    
    def _analyze_collision_risk(self) -> Dict:
        """충돌 위험 분석"""
        high_risk_count = 0
        total_analyzed = 0
        
        for i, traj in enumerate(self.trajectories):
            # 해당 시간대의 장애물 그리드 찾기
            traj_time = traj['timestamp']
            closest_det = min(self.det_grids, 
                            key=lambda x: abs(x['timestamp'] - traj_time),
                            default=None)
            
            if closest_det is None:
                continue
                
            total_analyzed += 1
            
            # 궤적 포인트들이 장애물과 너무 가까운지 확인
            points = traj['points']
            det_grid = closest_det['data']
            
            # 궤적을 그리드 좌표로 변환 (단순화)
            h, w = det_grid.shape
            
            min_distance_to_obstacle = float('inf')
            
            for point in points:
                # 상대 좌표를 그리드 인덱스로 변환 (단순 추정)
                grid_x = int(w/2 + point[1] * 5)  # Y는 좌우
                grid_y = int(h/2 - point[0] * 5)  # X는 전후 (뒤집어짐)
                
                if 0 <= grid_x < w and 0 <= grid_y < h:
                    # 주변 영역에서 장애물 검사
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            check_y = grid_y + dy
                            check_x = grid_x + dx
                            
                            if (0 <= check_x < w and 0 <= check_y < h and 
                                det_grid[check_y, check_x] > 50):  # 장애물 감지
                                distance = np.sqrt(dx**2 + dy**2) * 0.2  # 그리드 해상도 추정
                                min_distance_to_obstacle = min(min_distance_to_obstacle, distance)
            
            # 안전 거리 임계값 (2m)
            if min_distance_to_obstacle < 2.0:
                high_risk_count += 1
        
        return {
            'high_risk_ratio': high_risk_count / max(total_analyzed, 1),
            'analyzed_count': total_analyzed
        }
    
    def _analyze_lane_departure(self) -> Dict:
        """차선 이탈 위험 분석"""
        departure_count = 0
        total_analyzed = 0
        
        for traj in self.trajectories:
            # 해당 시간대의 주행가능영역 그리드 찾기
            traj_time = traj['timestamp']
            closest_da = min(self.da_grids,
                           key=lambda x: abs(x['timestamp'] - traj_time),
                           default=None)
            
            if closest_da is None:
                continue
                
            total_analyzed += 1
            
            points = traj['points']
            da_grid = closest_da['data']
            h, w = da_grid.shape
            
            # 궤적 끝점이 주행 가능 영역을 벗어나는지 확인
            if len(points) > 0:
                end_point = points[-1]
                
                # 그리드 좌표로 변환
                grid_x = int(w/2 + end_point[1] * 5)
                grid_y = int(h/2 - end_point[0] * 5)
                
                if 0 <= grid_x < w and 0 <= grid_y < h:
                    # 주행 가능 영역 밖이면 (100은 occupied, 0은 free)
                    if da_grid[grid_y, grid_x] > 50:
                        departure_count += 1
        
        return {
            'departure_ratio': departure_count / max(total_analyzed, 1),
            'analyzed_count': total_analyzed
        }
    
    def _detect_emergency_situations(self) -> Dict:
        """비상 상황 감지"""
        emergency_count = 0
        
        for traj in self.trajectories:
            points = traj['points']
            if len(points) < 2:
                continue
            
            # 급격한 조향 감지
            for i in range(len(points) - 1):
                segment = points[i+1] - points[i]
                if np.linalg.norm(segment) > 0:
                    # 90도 이상 급회전 감지
                    if i > 0:
                        prev_segment = points[i] - points[i-1]
                        if np.linalg.norm(prev_segment) > 0:
                            cos_angle = np.dot(segment, prev_segment) / (
                                np.linalg.norm(segment) * np.linalg.norm(prev_segment)
                            )
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            
                            if angle > np.pi/2:  # 90도 이상
                                emergency_count += 1
                                break
        
        return {
            'emergency_ratio': emergency_count / max(len(self.trajectories), 1),
            'emergency_count': emergency_count
        }
    
    def _measure_conservativeness(self) -> float:
        """보수적 주행 점수 (안전 마진)"""
        conservative_scores = []
        
        for traj in self.trajectories:
            points = traj['points']
            if len(points) == 0:
                continue
            
            # 궤적의 최대 횡방향 이동 측정
            y_values = points[:, 1]
            max_lateral = np.max(np.abs(y_values))
            
            # 보수적 점수 (횡방향 이동이 작을수록 보수적)
            conservative_score = 1.0 / (1.0 + max_lateral)
            conservative_scores.append(conservative_score)
        
        return np.mean(conservative_scores) if conservative_scores else 0.0
    
    def _measure_predictability(self) -> float:
        """예측 가능성 측정"""
        if len(self.trajectories) < 10:
            return 0.0
        
        # 최근 10개 궤적의 유사성 측정
        recent_trajs = self.trajectories[-10:]
        
        similarity_scores = []
        for i in range(len(recent_trajs) - 1):
            traj1 = recent_trajs[i]['points']
            traj2 = recent_trajs[i+1]['points']
            
            if len(traj1) > 0 and len(traj2) > 0:
                # 시작점 기준 정규화
                traj1_norm = traj1 - traj1[0]
                traj2_norm = traj2 - traj2[0]
                
                # 길이 맞춤
                min_len = min(len(traj1_norm), len(traj2_norm))
                if min_len > 1:
                    traj1_norm = traj1_norm[:min_len]
                    traj2_norm = traj2_norm[:min_len]
                    
                    # 유사성 계산 (MSE 기반)
                    mse = np.mean((traj1_norm - traj2_norm)**2)
                    similarity = 1.0 / (1.0 + mse)
                    similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _measure_reaction_time(self) -> Dict:
        """반응 시간 측정 (환경 변화에 대한 반응)"""
        # 간단한 구현: 궤적 변화의 빈도 측정
        change_count = 0
        total_comparisons = 0
        
        for i in range(len(self.trajectories) - 1):
            traj1 = self.trajectories[i]['points']
            traj2 = self.trajectories[i+1]['points']
            
            if len(traj1) > 0 and len(traj2) > 0:
                total_comparisons += 1
                
                # 궤적 방향의 변화 감지
                if len(traj1) > 1 and len(traj2) > 1:
                    dir1 = traj1[-1] - traj1[0]
                    dir2 = traj2[-1] - traj2[0]
                    
                    if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                        cos_sim = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
                        
                        if cos_sim < 0.9:  # 25도 이상 변화
                            change_count += 1
        
        return {
            'reaction_frequency': change_count / max(total_comparisons, 1),
            'total_comparisons': total_comparisons
        }
    
    def print_safety_report(self, metrics: Dict):
        """안전성 리포트 출력"""
        rospy.loginfo("🛡️ 실제 안전성 분석 결과")
        rospy.loginfo("=" * 60)
        
        if not metrics:
            rospy.logwarn("⚠️ 분석할 데이터가 부족합니다")
            return
        
        # 충돌 위험
        collision = metrics.get('collision_risk', {})
        risk_ratio = collision.get('high_risk_ratio', 0)
        rospy.loginfo(f"⚠️ 충돌 위험 비율: {risk_ratio:.3f} ({risk_ratio*100:.1f}%)")
        if risk_ratio > 0.1:
            rospy.logwarn("   🚨 높은 충돌 위험 감지!")
        
        # 차선 이탈
        lane_dep = metrics.get('lane_departure_risk', {})
        dep_ratio = lane_dep.get('departure_ratio', 0)
        rospy.loginfo(f"🛣️ 차선 이탈 위험: {dep_ratio:.3f} ({dep_ratio*100:.1f}%)")
        if dep_ratio > 0.2:
            rospy.logwarn("   🚨 높은 차선 이탈 위험!")
        
        # 비상 상황
        emergency = metrics.get('emergency_situations', {})
        emg_ratio = emergency.get('emergency_ratio', 0)
        rospy.loginfo(f"🚨 비상 상황 비율: {emg_ratio:.3f} ({emg_ratio*100:.1f}%)")
        
        # 보수성
        conservative = metrics.get('conservative_score', 0)
        rospy.loginfo(f"🛡️ 보수적 주행 점수: {conservative:.3f} (높을수록 안전)")
        
        # 예측 가능성
        predictable = metrics.get('predictability', 0)
        rospy.loginfo(f"🔮 예측 가능성: {predictable:.3f} (높을수록 일관됨)")
        
        # 전체 안전성 점수 계산
        safety_score = (
            (1 - risk_ratio) * 0.3 +      # 충돌 위험 (낮을수록 좋음)
            (1 - dep_ratio) * 0.2 +       # 차선 이탈 (낮을수록 좋음)
            (1 - emg_ratio) * 0.2 +       # 비상 상황 (낮을수록 좋음)
            conservative * 0.15 +          # 보수성 (높을수록 좋음)
            predictable * 0.15             # 예측가능성 (높을수록 좋음)
        )
        
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"🏆 종합 안전성 점수: {safety_score:.3f} / 1.000")
        
        # 안전성 등급 판정
        if safety_score >= 0.8:
            rospy.loginfo("✅ 안전성 등급: 우수 (A)")
        elif safety_score >= 0.7:
            rospy.loginfo("🟡 안전성 등급: 양호 (B)")
        elif safety_score >= 0.6:
            rospy.loginfo("🟠 안전성 등급: 보통 (C)")
        else:
            rospy.loginfo("🔴 안전성 등급: 위험 (D)")
            rospy.logwarn("🚨 긴급 개선 필요!")
        
        # 개선 권장사항
        rospy.loginfo("\n🔧 개선 권장사항:")
        
        if risk_ratio > 0.1:
            rospy.loginfo("  - 장애물 회피 로직 강화")
            rospy.loginfo("  - 안전 거리 증가")
        
        if dep_ratio > 0.2:
            rospy.loginfo("  - 차선 유지 능력 개선")
            rospy.loginfo("  - 주행 가능 영역 인식 향상")
        
        if conservative < 0.5:
            rospy.loginfo("  - 더 보수적인 주행 전략 적용")
            rospy.loginfo("  - 안전 마진 확대")
        
        if predictable < 0.5:
            rospy.loginfo("  - 궤적 일관성 향상")
            rospy.loginfo("  - 스무딩 파라미터 조정")


def main():
    """메인 함수"""
    try:
        analyzer = SafetyAnalyzer()
        
        # 사용자 설정
        duration = rospy.get_param('~duration', 30)
        
        rospy.loginfo("🛡️ 실제 안전성 분석을 시작합니다...")
        rospy.loginfo("   다양한 주행 상황을 시도해주세요!")
        
        # 데이터 수집
        analyzer.start_collection(duration)
        
        # 안전성 분석
        metrics = analyzer.analyze_safety()
        
        # 결과 출력
        analyzer.print_safety_report(metrics)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 안전성 분석 종료")
    except Exception as e:
        rospy.logerr(f"❌ 안전성 분석 오류: {e}")


if __name__ == '__main__':
    main() 