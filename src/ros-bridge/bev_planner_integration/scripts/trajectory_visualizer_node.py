#!/usr/bin/env python3
"""
BEV-Planner 궤적 시각화 노드

계획된 궤적을 RViz에서 시각화하고 안전성 지표를 표시
"""

import rospy
import numpy as np
from nav_msgs.msg import Path as RosPath, Odometry
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class TrajectoryVisualizerNode:
    """
    BEV-Planner 궤적 시각화 노드
    """
    
    def __init__(self):
        rospy.init_node('trajectory_visualizer_node', anonymous=True)
        
        # 파라미터
        self.trajectory_color = rospy.get_param('~trajectory_color', [0.0, 1.0, 0.0])  # 녹색
        self.confidence_visualization = rospy.get_param('~show_confidence', True)
        self.safety_visualization = rospy.get_param('~show_safety', True)
        
        # 상태 변수
        self.latest_trajectory = None
        self.latest_ego_odom = None
        
        # TF2 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ROS 통신 설정
        self._setup_ros_communication()
        
        rospy.loginfo("🎨 궤적 시각화 노드 시작!")
        
    def _setup_ros_communication(self):
        """ROS 토픽 설정"""
        # 구독자
        self.sub_trajectory = rospy.Subscriber(
            'planned_trajectory', RosPath, 
            self._trajectory_callback, queue_size=1
        )
        self.sub_ego_odom = rospy.Subscriber(
            'ego_odometry', Odometry,
            self._ego_odom_callback, queue_size=1
        )
        
        # 발행자
        self.pub_trajectory_markers = rospy.Publisher(
            'trajectory_markers', MarkerArray, queue_size=1
        )
        self.pub_safety_indicators = rospy.Publisher(
            'safety_indicators', MarkerArray, queue_size=1
        )
    
    def _trajectory_callback(self, msg: RosPath):
        """계획된 궤적 콜백"""
        self.latest_trajectory = msg
        self._visualize_trajectory()
    
    def _ego_odom_callback(self, msg: Odometry):
        """Ego vehicle odometry 콜백"""
        self.latest_ego_odom = msg
    
    def _visualize_trajectory(self):
        """궤적 시각화"""
        if self.latest_trajectory is None:
            return
            
        try:
            # 궤적 마커 생성
            trajectory_markers = self._create_trajectory_markers()
            self.pub_trajectory_markers.publish(trajectory_markers)
            
            # 안전성 지표 생성
            if self.safety_visualization:
                safety_markers = self._create_safety_markers()
                self.pub_safety_indicators.publish(safety_markers)
                
        except Exception as e:
            rospy.logerr(f"❌ 시각화 오류: {e}")
    
    def _create_trajectory_markers(self) -> MarkerArray:
        """궤적 마커 생성"""
        marker_array = MarkerArray()
        
        if not self.latest_trajectory.poses:
            return marker_array
        
        # 1. 궤적 라인 마커
        line_marker = Marker()
        line_marker.header = self.latest_trajectory.header
        line_marker.ns = "trajectory_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        # 라인 속성
        line_marker.scale.x = 0.2  # 라인 두께
        line_marker.color = ColorRGBA(
            r=self.trajectory_color[0],
            g=self.trajectory_color[1], 
            b=self.trajectory_color[2],
            a=0.8
        )
        
        # 궤적 포인트 추가
        for pose_stamped in self.latest_trajectory.poses:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = pose_stamped.pose.position.z + 0.1  # 약간 위로
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        
        # 2. 궤적 포인트 마커들
        for i, pose_stamped in enumerate(self.latest_trajectory.poses):
            point_marker = Marker()
            point_marker.header = self.latest_trajectory.header
            point_marker.ns = "trajectory_points"
            point_marker.id = i
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            
            # 위치
            point_marker.pose = pose_stamped.pose
            point_marker.pose.position.z += 0.1
            
            # 크기 (시간에 따라 작아짐)
            scale = 0.3 * (1.0 - i * 0.1)
            point_marker.scale = Vector3(x=scale, y=scale, z=scale)
            
            # 색상 (시간에 따라 투명해짐)
            alpha = 1.0 - i * 0.15
            point_marker.color = ColorRGBA(
                r=self.trajectory_color[0],
                g=self.trajectory_color[1],
                b=self.trajectory_color[2], 
                a=max(0.3, alpha)
            )
            
            marker_array.markers.append(point_marker)
        
        # 3. 시작점 특별 마커
        if self.latest_trajectory.poses:
            start_marker = Marker()
            start_marker.header = self.latest_trajectory.header
            start_marker.ns = "trajectory_start"
            start_marker.id = 0
            start_marker.type = Marker.ARROW
            start_marker.action = Marker.ADD
            
            start_marker.pose = self.latest_trajectory.poses[0].pose
            start_marker.pose.position.z += 0.2
            
            start_marker.scale = Vector3(x=0.5, y=0.1, z=0.1)
            start_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # 노란색
            
            marker_array.markers.append(start_marker)
        
        return marker_array
    
    def _create_safety_markers(self) -> MarkerArray:
        """안전성 지표 마커 생성"""
        marker_array = MarkerArray()
        
        if not self.latest_trajectory.poses or self.latest_ego_odom is None:
            return marker_array
        
        # 안전 영역 시각화 (ego vehicle 주변)
        safety_zone = Marker()
        safety_zone.header = self.latest_trajectory.header
        safety_zone.ns = "safety_zone"
        safety_zone.id = 0
        safety_zone.type = Marker.CYLINDER
        safety_zone.action = Marker.ADD
        
        # Ego vehicle 위치 기준
        safety_zone.pose.position.x = 0.0
        safety_zone.pose.position.y = 0.0
        safety_zone.pose.position.z = 0.05
        safety_zone.pose.orientation.w = 1.0
        
        # 안전 영역 크기 (차량 크기 + 여유거리)
        safety_zone.scale = Vector3(x=6.0, y=3.0, z=0.1)  # 6m x 3m
        safety_zone.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.2)  # 반투명 청록색
        
        marker_array.markers.append(safety_zone)
        
        # 속도 벡터 시각화
        if self.latest_ego_odom:
            velocity_marker = Marker()
            velocity_marker.header = self.latest_trajectory.header
            velocity_marker.ns = "velocity_vector"
            velocity_marker.id = 0
            velocity_marker.type = Marker.ARROW
            velocity_marker.action = Marker.ADD
            
            # 속도 크기 계산
            vel = self.latest_ego_odom.twist.twist.linear
            speed = np.sqrt(vel.x**2 + vel.y**2)
            
            if speed > 0.1:  # 최소 속도
                # 화살표 방향 (속도 방향)
                velocity_marker.pose.position.x = 0.0
                velocity_marker.pose.position.y = 0.0
                velocity_marker.pose.position.z = 1.0
                
                # 방향 설정 (단순화)
                velocity_marker.pose.orientation.w = 1.0
                
                # 크기 (속도에 비례)
                arrow_length = min(speed * 0.2, 2.0)  # 최대 2m
                velocity_marker.scale = Vector3(
                    x=arrow_length, 
                    y=0.2, 
                    z=0.2
                )
                
                # 색상 (속도에 따라)
                if speed < 5.0:
                    color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # 녹색 (저속)
                elif speed < 10.0:
                    color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # 노란색 (중속)
                else:
                    color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # 빨간색 (고속)
                
                velocity_marker.color = color
                
                marker_array.markers.append(velocity_marker)
        
        # 정보 텍스트
        info_marker = Marker()
        info_marker.header = self.latest_trajectory.header
        info_marker.ns = "trajectory_info"
        info_marker.id = 0
        info_marker.type = Marker.TEXT_VIEW_FACING
        info_marker.action = Marker.ADD
        
        info_marker.pose.position.x = 2.0
        info_marker.pose.position.y = 0.0
        info_marker.pose.position.z = 2.0
        info_marker.pose.orientation.w = 1.0
        
        # 텍스트 내용
        num_points = len(self.latest_trajectory.poses)
        info_text = f"Trajectory Points: {num_points}\n"
        if self.latest_ego_odom:
            vel = self.latest_ego_odom.twist.twist.linear
            speed = np.sqrt(vel.x**2 + vel.y**2) * 3.6  # km/h
            info_text += f"Speed: {speed:.1f} km/h"
        
        info_marker.text = info_text
        info_marker.scale.z = 0.3
        info_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        
        marker_array.markers.append(info_marker)
        
        return marker_array
    
    def run(self):
        """메인 실행 루프"""
        rate = rospy.Rate(10)  # 10Hz 시각화 업데이트
        
        while not rospy.is_shutdown():
            # 주기적으로 시각화 업데이트
            if self.latest_trajectory is not None:
                self._visualize_trajectory()
            
            rate.sleep()


def main():
    """메인 함수"""
    try:
        visualizer = TrajectoryVisualizerNode()
        visualizer.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 궤적 시각화 노드 종료")
    except Exception as e:
        rospy.logerr(f"❌ 시각화 노드 오류: {e}")


if __name__ == '__main__':
    main() 