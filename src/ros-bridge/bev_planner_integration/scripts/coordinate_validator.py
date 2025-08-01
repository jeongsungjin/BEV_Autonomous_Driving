#!/usr/bin/env python3
"""
좌표계 정합 검증 및 디버깅 스크립트

BEV-Planner와 YOLOP 간의 좌표계 정합을 검증하고
문제가 있을 경우 시각화를 통해 디버깅을 도와주는 스크립트
"""

import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid, Path as RosPath, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import tf.transformations as tf_trans
from typing import Dict, List, Optional, Tuple


class CoordinateValidator:
    """좌표계 정합 검증기"""
    
    def __init__(self):
        rospy.init_node('coordinate_validator', anonymous=True)
        
        # 데이터 저장
        self.latest_det_grid = None
        self.latest_da_grid = None
        self.latest_ll_grid = None
        self.latest_trajectory = None
        self.latest_ego_odom = None
        
        # 구독자 설정
        self._setup_subscribers()
        
        # 발행자 설정 (디버깅 시각화용)
        self.pub_debug_markers = rospy.Publisher(
            '/coordinate_debug/markers', MarkerArray, queue_size=1
        )
        
        # 검증 결과 저장
        self.validation_results = {}
        
        rospy.loginfo("🔧 좌표계 검증기 시작!")
        
    def _setup_subscribers(self):
        """구독자 설정"""
        rospy.Subscriber('/carla/yolop/det_grid', OccupancyGrid, 
                        self._det_grid_callback, queue_size=1)
        rospy.Subscriber('/carla/yolop/da_grid', OccupancyGrid,
                        self._da_grid_callback, queue_size=1)
        rospy.Subscriber('/carla/yolop/ll_grid', OccupancyGrid,
                        self._ll_grid_callback, queue_size=1)
        rospy.Subscriber('/bev_planner/planned_trajectory', RosPath,
                        self._trajectory_callback, queue_size=1)
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry,
                        self._ego_odom_callback, queue_size=1)
    
    def _det_grid_callback(self, msg: OccupancyGrid):
        self.latest_det_grid = msg
        
    def _da_grid_callback(self, msg: OccupancyGrid):
        self.latest_da_grid = msg
        
    def _ll_grid_callback(self, msg: OccupancyGrid):
        self.latest_ll_grid = msg
        
    def _trajectory_callback(self, msg: RosPath):
        self.latest_trajectory = msg
        self._validate_trajectory_coordinates()
        
    def _ego_odom_callback(self, msg: Odometry):
        self.latest_ego_odom = msg
    
    def validate_all(self) -> Dict[str, bool]:
        """전체 좌표계 검증 수행"""
        results = {}
        
        # 1. 프레임 ID 일관성 검증
        results['frame_consistency'] = self._validate_frame_consistency()
        
        # 2. BEV 그리드 방향성 검증
        results['grid_orientation'] = self._validate_grid_orientation()
        
        # 3. 궤적 좌표 범위 검증
        results['trajectory_bounds'] = self._validate_trajectory_bounds()
        
        # 4. 좌표 변환 정확성 검증
        results['coordinate_transform'] = self._validate_coordinate_transform()
        
        self.validation_results = results
        self._print_validation_report()
        
        return results
    
    def _validate_frame_consistency(self) -> bool:
        """프레임 ID 일관성 검증"""
        rospy.loginfo("🔍 프레임 ID 일관성 검증 중...")
        
        frames = []
        if self.latest_det_grid:
            frames.append(('det_grid', self.latest_det_grid.header.frame_id))
        if self.latest_da_grid:
            frames.append(('da_grid', self.latest_da_grid.header.frame_id))
        if self.latest_ll_grid:
            frames.append(('ll_grid', self.latest_ll_grid.header.frame_id))
        if self.latest_trajectory:
            frames.append(('trajectory', self.latest_trajectory.header.frame_id))
        if self.latest_ego_odom:
            frames.append(('ego_odom', self.latest_ego_odom.header.frame_id))
        
        # 프레임 출력
        for name, frame_id in frames:
            rospy.loginfo(f"  {name}: {frame_id}")
        
        # YOLOP 그리드들이 모두 같은 프레임인지 확인
        grid_frames = [f[1] for f in frames if 'grid' in f[0]]
        grid_consistent = len(set(grid_frames)) <= 1 if grid_frames else True
        
        if not grid_consistent:
            rospy.logwarn("⚠️ YOLOP 그리드 프레임 불일치!")
        
        return grid_consistent
    
    def _validate_grid_orientation(self) -> bool:
        """BEV 그리드 방향성 검증"""
        rospy.loginfo("🔍 BEV 그리드 방향성 검증 중...")
        
        if not self.latest_da_grid or not self.latest_ego_odom:
            rospy.logwarn("⚠️ DA 그리드 또는 ego odometry 데이터 없음")
            return False
        
        # DA 그리드에서 주행 가능 영역 분석
        grid_data = np.array(self.latest_da_grid.data).reshape(
            self.latest_da_grid.info.height, 
            self.latest_da_grid.info.width
        )
        
        # 자유 공간(0)이 차량 전진 방향에 더 많이 있는지 확인
        h, w = grid_data.shape
        center_y, center_x = h // 2, w // 2
        
        # 전진 방향 (위쪽)과 후진 방향 (아래쪽) 비교
        forward_region = grid_data[:center_y, :]
        backward_region = grid_data[center_y:, :]
        
        forward_free = np.sum(forward_region == 0)
        backward_free = np.sum(backward_region == 0)
        
        orientation_correct = forward_free > backward_free * 0.5  # 전진 방향에 더 많은 자유공간
        
        rospy.loginfo(f"  전진 방향 자유공간: {forward_free}")
        rospy.loginfo(f"  후진 방향 자유공간: {backward_free}")
        rospy.loginfo(f"  방향성 올바름: {orientation_correct}")
        
        return orientation_correct
    
    def _validate_trajectory_bounds(self) -> bool:
        """궤적 좌표 범위 검증"""
        rospy.loginfo("🔍 궤적 좌표 범위 검증 중...")
        
        if not self.latest_trajectory or not self.latest_trajectory.poses:
            rospy.logwarn("⚠️ 궤적 데이터 없음")
            return False
        
        # 궤적 좌표 범위 분석
        positions = []
        for pose_stamped in self.latest_trajectory.poses:
            pos = pose_stamped.pose.position
            positions.append([pos.x, pos.y])
        
        positions = np.array(positions)
        
        # 통계 계산
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        x_mean = positions[:, 0].mean()
        y_mean = positions[:, 1].mean()
        
        rospy.loginfo(f"  X 범위: {x_range:.2f}m")
        rospy.loginfo(f"  Y 범위: {y_range:.2f}m")
        rospy.loginfo(f"  X 평균: {x_mean:.2f}m")
        rospy.loginfo(f"  Y 평균: {y_mean:.2f}m")
        
        # 합리적인지 검증 (너무 극단적이지 않은지)
        reasonable_bounds = (
            x_range < 50.0 and y_range < 20.0 and  # 궤적이 너무 크지 않음
            abs(x_mean) < 100.0 and abs(y_mean) < 100.0  # 중심이 너무 멀지 않음
        )
        
        if not reasonable_bounds:
            rospy.logwarn("⚠️ 궤적 좌표 범위가 비합리적!")
        
        return reasonable_bounds
    
    def _validate_coordinate_transform(self) -> bool:
        """좌표 변환 정확성 검증"""
        rospy.loginfo("🔍 좌표 변환 정확성 검증 중...")
        
        if not self.latest_trajectory or not self.latest_ego_odom:
            rospy.logwarn("⚠️ 궤적 또는 ego odometry 데이터 없음")
            return False
        
        # 궤적의 첫 번째 점이 ego 근처에 있는지 확인
        if not self.latest_trajectory.poses:
            return False
        
        trajectory_frame = self.latest_trajectory.header.frame_id
        first_pose = self.latest_trajectory.poses[0].pose.position
        
        # 프레임에 따라 다른 검증 로직 적용
        if trajectory_frame == "ego_vehicle":
            # ego_vehicle 프레임: 상대 좌표이므로 원점 근처에서 시작해야 함
            distance_from_origin = np.sqrt(first_pose.x**2 + first_pose.y**2)
            rospy.loginfo(f"  궤적 시작점과 원점 거리: {distance_from_origin:.2f}m (ego_vehicle 프레임)")
            
            # ego_vehicle 프레임에서는 10m 이내에서 시작하는 것이 합리적
            close_to_origin = distance_from_origin < 10.0
            
            if not close_to_origin:
                rospy.logwarn("⚠️ ego_vehicle 프레임 궤적이 원점에서 너무 멀리 시작!")
            
            return close_to_origin
            
        elif trajectory_frame == "map":
            # map 프레임: 절대 좌표이므로 ego 위치 근처에서 시작해야 함
            ego_pos = self.latest_ego_odom.pose.pose.position
            
            dx = first_pose.x - ego_pos.x
            dy = first_pose.y - ego_pos.y
            distance = np.sqrt(dx**2 + dy**2)
            
            rospy.loginfo(f"  궤적 시작점과 ego 거리: {distance:.2f}m (map 프레임)")
            
            close_to_ego = distance < 5.0  # 5m 이내
            
            if not close_to_ego:
                rospy.logwarn("⚠️ map 프레임 궤적이 ego vehicle에서 너무 멀리 시작!")
            
            return close_to_ego
        
        else:
            rospy.logwarn(f"⚠️ 알 수 없는 궤적 프레임: {trajectory_frame}")
            return False
    
    def _validate_trajectory_coordinates(self):
        """궤적 좌표 실시간 검증"""
        if not self.latest_trajectory:
            return
        
        # 급격한 방향 변화 감지
        if len(self.latest_trajectory.poses) >= 3:
            positions = []
            for pose in self.latest_trajectory.poses[:3]:
                pos = pose.pose.position
                positions.append([pos.x, pos.y])
            
            positions = np.array(positions)
            
            # 연속된 3점의 각도 변화 계산
            v1 = positions[1] - positions[0]
            v2 = positions[2] - positions[1]
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_change = np.arccos(np.clip(cos_angle, -1, 1))
                
                # 급격한 방향 변화 (90도 이상) 감지
                if angle_change > np.pi/2:
                    rospy.logwarn(f"⚠️ 급격한 방향 변화 감지: {np.degrees(angle_change):.1f}도")
                    self._publish_debug_markers()
    
    def _publish_debug_markers(self):
        """디버그 시각화 마커 발행"""
        if not self.latest_trajectory:
            return
        
        marker_array = MarkerArray()
        
        # 궤적 문제점 강조 마커
        for i, pose_stamped in enumerate(self.latest_trajectory.poses):
            marker = Marker()
            marker.header = self.latest_trajectory.header
            marker.ns = "coordinate_debug"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose = pose_stamped.pose
            marker.pose.position.z += 0.5  # 위로 올려서 강조
            
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            
            # 문제가 있는 포인트는 빨간색
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            
            marker_array.markers.append(marker)
        
        self.pub_debug_markers.publish(marker_array)
    
    def _print_validation_report(self):
        """검증 결과 리포트 출력"""
        rospy.loginfo("📋 좌표계 검증 결과:")
        rospy.loginfo("=" * 40)
        
        for test_name, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            rospy.loginfo(f"  {test_name}: {status}")
        
        overall_pass = all(self.validation_results.values())
        overall_status = "✅ 전체 통과" if overall_pass else "❌ 문제 발견"
        rospy.loginfo(f"\n총 결과: {overall_status}")
        
        if not overall_pass:
            rospy.loginfo("\n🔧 권장 조치:")
            rospy.loginfo("  1. launch 파일에서 프레임 ID 확인")
            rospy.loginfo("  2. YOLOP BEV 그리드 회전 확인")
            rospy.loginfo("  3. BEV planner 좌표 변환 로직 확인")
    
    def run_continuous_validation(self):
        """연속 검증 모드 실행"""
        rate = rospy.Rate(1)  # 1Hz
        
        while not rospy.is_shutdown():
            if (self.latest_det_grid and self.latest_da_grid and 
                self.latest_ll_grid and self.latest_trajectory and 
                self.latest_ego_odom):
                
                self.validate_all()
                rospy.sleep(5)  # 5초 대기
            
            rate.sleep()


def main():
    """메인 함수"""
    try:
        validator = CoordinateValidator()
        
        # 인터랙티브 모드 선택
        mode = rospy.get_param('~mode', 'continuous')
        
        if mode == 'once':
            # 한 번만 검증
            rospy.sleep(2)  # 데이터 수집 대기
            validator.validate_all()
        else:
            # 연속 검증
            validator.run_continuous_validation()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 좌표계 검증기 종료")
    except Exception as e:
        rospy.logerr(f"❌ 검증기 오류: {e}")


if __name__ == '__main__':
    main() 