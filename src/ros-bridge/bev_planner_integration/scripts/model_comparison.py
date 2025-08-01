#!/usr/bin/env python3
"""
BEV-Planner 모델 성능 비교 스크립트

기존 학습된 모델 vs 랜덤 가중치 모델의 성능을 비교하여
재학습 필요성을 판단합니다.
"""

import rospy
import numpy as np
import time
from nav_msgs.msg import Path as RosPath, Odometry
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from typing import List, Dict
import threading


class ModelComparison:
    """모델 성능 비교기"""
    
    def __init__(self):
        rospy.init_node('model_comparison', anonymous=True)
        
        self.trajectories = []
        self.ego_positions = []
        self.timestamps = []
        self.collecting = False
        self.collection_duration = 60  # 60초간 데이터 수집
        
        # 구독자 설정
        rospy.Subscriber('/bev_planner/planned_trajectory', RosPath,
                        self._trajectory_callback, queue_size=1)
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry,
                        self._ego_callback, queue_size=1)
        
        rospy.loginfo("🔍 모델 성능 비교기 시작!")
        
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
            self.ego_positions.append({
                'timestamp': time.time(),
                'x': pos.x,
                'y': pos.y,
                'z': pos.z
            })
    
    def start_collection(self, duration: int = 60):
        """데이터 수집 시작"""
        rospy.loginfo(f"📊 {duration}초간 데이터 수집 시작...")
        
        self.trajectories.clear()
        self.ego_positions.clear()
        self.timestamps.clear()
        
        self.collecting = True
        time.sleep(duration)
        self.collecting = False
        
        rospy.loginfo(f"✅ 데이터 수집 완료: {len(self.trajectories)}개 궤적")
        
    def analyze_performance(self) -> Dict:
        """성능 분석"""
        if not self.trajectories:
            rospy.logwarn("⚠️ 분석할 데이터가 없습니다")
            return {}
        
        metrics = {
            'trajectory_count': len(self.trajectories),
            'consistency': self._measure_consistency(),
            'smoothness': self._measure_smoothness(),
            'direction_stability': self._measure_direction_stability(),
            'realistic_bounds': self._check_realistic_bounds()
        }
        
        return metrics
    
    def _measure_consistency(self) -> float:
        """궤적 일관성 측정"""
        if len(self.trajectories) < 2:
            return 0.0
        
        # 연속된 궤적들의 시작점 비교
        start_points = []
        for traj in self.trajectories:
            if len(traj['points']) > 0:
                start_points.append(traj['points'][0])
        
        if len(start_points) < 2:
            return 0.0
        
        start_points = np.array(start_points)
        
        # 시작점들의 표준편차 (낮을수록 일관성 높음)
        std_x = np.std(start_points[:, 0])
        std_y = np.std(start_points[:, 1])
        
        # 일관성 점수 (표준편차가 작을수록 높은 점수)
        consistency_score = 1.0 / (1.0 + std_x + std_y)
        
        return consistency_score
    
    def _measure_smoothness(self) -> float:
        """궤적 부드러움 측정"""
        smoothness_scores = []
        
        for traj in self.trajectories:
            points = traj['points']
            if len(points) < 3:
                continue
            
            # 연속된 3점의 각도 변화량 계산
            angle_changes = []
            for i in range(len(points) - 2):
                v1 = points[i+1] - points[i]
                v2 = points[i+2] - points[i+1]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle_change = np.arccos(np.clip(cos_angle, -1, 1))
                    angle_changes.append(angle_change)
            
            if angle_changes:
                # 각도 변화의 평균 (작을수록 부드러움)
                avg_angle_change = np.mean(angle_changes)
                smoothness_score = 1.0 / (1.0 + avg_angle_change)
                smoothness_scores.append(smoothness_score)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def _measure_direction_stability(self) -> float:
        """방향 안정성 측정"""
        direction_scores = []
        
        for traj in self.trajectories:
            points = traj['points']
            if len(points) < 2:
                continue
            
            # 전체 궤적의 방향 벡터
            overall_direction = points[-1] - points[0]
            if np.linalg.norm(overall_direction) == 0:
                continue
            
            # 전진 방향(+X)과의 유사도
            forward_direction = np.array([1.0, 0.0])
            cos_similarity = np.dot(overall_direction, forward_direction) / np.linalg.norm(overall_direction)
            
            # 0~1 사이 점수로 변환
            direction_score = (cos_similarity + 1) / 2
            direction_scores.append(direction_score)
        
        return np.mean(direction_scores) if direction_scores else 0.0
    
    def _check_realistic_bounds(self) -> float:
        """현실적 범위 확인"""
        realistic_count = 0
        total_count = 0
        
        for traj in self.trajectories:
            points = traj['points']
            if len(points) == 0:
                continue
            
            total_count += 1
            
            # 궤적 범위 확인
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            # 현실적 범위 기준 (차량 기준 상대 좌표)
            if (0.1 < x_range < 20.0 and  # 전진 방향 0.1~20m
                y_range < 10.0):          # 횡방향 10m 이내
                realistic_count += 1
        
        return realistic_count / total_count if total_count > 0 else 0.0
    
    def print_report(self, metrics: Dict):
        """성능 리포트 출력"""
        rospy.loginfo("📋 BEV-Planner 성능 분석 결과")
        rospy.loginfo("=" * 50)
        
        if not metrics:
            rospy.logwarn("⚠️ 분석할 데이터가 부족합니다")
            return
        
        rospy.loginfo(f"📊 수집된 궤적 수: {metrics['trajectory_count']}")
        rospy.loginfo(f"🎯 일관성 점수: {metrics['consistency']:.3f} (높을수록 좋음)")
        rospy.loginfo(f"🌊 부드러움 점수: {metrics['smoothness']:.3f} (높을수록 좋음)")
        rospy.loginfo(f"➡️ 방향 안정성: {metrics['direction_stability']:.3f} (높을수록 좋음)")
        rospy.loginfo(f"📏 현실적 범위: {metrics['realistic_bounds']:.3f} (높을수록 좋음)")
        
        # 종합 점수 계산
        overall_score = (
            metrics['consistency'] * 0.3 +
            metrics['smoothness'] * 0.3 +
            metrics['direction_stability'] * 0.2 +
            metrics['realistic_bounds'] * 0.2
        )
        
        rospy.loginfo(f"🏆 종합 점수: {overall_score:.3f} / 1.000")
        
        # 재학습 권장 여부
        if overall_score < 0.6:
            rospy.loginfo("🚨 권장: 재학습이 필요합니다")
        elif overall_score < 0.8:
            rospy.loginfo("⚠️ 권장: 재학습을 고려해보세요")
        else:
            rospy.loginfo("✅ 현재 모델 성능이 양호합니다")
    
    def save_visualization(self, metrics: Dict, filename: str = "/tmp/trajectory_analysis.png"):
        """시각화 저장"""
        if not self.trajectories:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 궤적 시각화
        ax1 = axes[0, 0]
        for i, traj in enumerate(self.trajectories[:10]):  # 최근 10개만
            points = traj['points']
            if len(points) > 0:
                ax1.plot(points[:, 0], points[:, 1], alpha=0.7, label=f'Traj {i+1}')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('궤적 시각화 (최근 10개)')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 성능 지표 바 차트
        ax2 = axes[0, 1]
        metric_names = ['일관성', '부드러움', '방향성', '현실성']
        metric_values = [
            metrics['consistency'],
            metrics['smoothness'],
            metrics['direction_stability'],
            metrics['realistic_bounds']
        ]
        bars = ax2.bar(metric_names, metric_values)
        ax2.set_ylabel('점수 (0-1)')
        ax2.set_title('성능 지표')
        ax2.set_ylim(0, 1)
        
        # 색상 설정
        for bar, val in zip(bars, metric_values):
            if val < 0.5:
                bar.set_color('red')
            elif val < 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 3. 시작점 분포
        ax3 = axes[1, 0]
        start_points = []
        for traj in self.trajectories:
            if len(traj['points']) > 0:
                start_points.append(traj['points'][0])
        
        if start_points:
            start_points = np.array(start_points)
            ax3.scatter(start_points[:, 0], start_points[:, 1], alpha=0.6)
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_title('궤적 시작점 분포')
            ax3.grid(True)
        
        # 4. 종합 점수
        ax4 = axes[1, 1]
        overall_score = (
            metrics['consistency'] * 0.3 +
            metrics['smoothness'] * 0.3 +
            metrics['direction_stability'] * 0.2 +
            metrics['realistic_bounds'] * 0.2
        )
        
        # 원형 게이지
        theta = np.linspace(0, 2*np.pi, 100)
        r_outer = 1
        r_inner = 0.7
        
        # 배경 원
        ax4.fill_between(theta, r_inner, r_outer, alpha=0.3, color='lightgray')
        
        # 점수에 해당하는 호
        score_theta = theta[:int(overall_score * 100)]
        if overall_score < 0.5:
            color = 'red'
        elif overall_score < 0.7:
            color = 'orange'
        else:
            color = 'green'
            
        ax4.fill_between(score_theta, r_inner, r_outer, alpha=0.8, color=color)
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-1.2, 1.2)
        ax4.set_aspect('equal')
        ax4.text(0, 0, f'{overall_score:.3f}', ha='center', va='center', fontsize=20)
        ax4.set_title('종합 점수')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        rospy.loginfo(f"📸 시각화 저장: {filename}")


def main():
    """메인 함수"""
    try:
        comparator = ModelComparison()
        
        # 사용자 선택
        duration = rospy.get_param('~duration', 30)  # 기본 30초
        
        rospy.loginfo("🚀 모델 성능 측정을 시작합니다...")
        rospy.loginfo(f"   CARLA에서 차량을 다양한 상황에서 주행시켜주세요")
        rospy.loginfo(f"   {duration}초 후 분석 결과를 출력합니다")
        
        # 데이터 수집
        comparator.start_collection(duration)
        
        # 성능 분석
        metrics = comparator.analyze_performance()
        
        # 결과 출력
        comparator.print_report(metrics)
        
        # 시각화 저장
        comparator.save_visualization(metrics)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 성능 비교 종료")
    except Exception as e:
        rospy.logerr(f"❌ 성능 비교 오류: {e}")


if __name__ == '__main__':
    main() 