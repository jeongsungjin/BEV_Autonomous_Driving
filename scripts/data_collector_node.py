#!/usr/bin/env python3
"""
BEV-Planner 학습용 데이터 수집 노드
CARLA 환경에서 YOLOP + Ego Status + Expert Trajectory 데이터를 수집합니다.
"""

import rospy
import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import threading

# ROS 메시지
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs

class BEVPlannerDataCollector:
    def __init__(self):
        rospy.init_node('bev_planner_data_collector', anonymous=True)
        
        # 데이터 저장 설정
        self.data_dir = os.path.expanduser("~/capstone_2025/training_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=10000)  # 최대 10,000개 샘플
        self.current_sample = {}
        
        # 동기화를 위한 타임스탬프 허용 오차 (초)
        self.time_tolerance = 0.1
        
        # 데이터 저장 주기 (초)
        self.save_interval = 30
        
        # 최소 속도 (정지 상태 데이터는 제외)
        self.min_velocity = 0.5  # m/s
        
        rospy.loginfo("🗂️  BEV-Planner 데이터 수집기 시작")
        rospy.loginfo(f"📁 데이터 저장 경로: {self.data_dir}")
        
        # ROS 구독자 설정
        self.setup_subscribers()
        
        # TF 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 데이터 저장 타이머
        self.save_timer = rospy.Timer(rospy.Duration(self.save_interval), self.save_data_callback)
        
        rospy.loginfo("✅ 데이터 수집기 초기화 완료")
        rospy.loginfo("🎯 CARLA에서 운전하면 데이터가 자동 수집됩니다!")
        
    def setup_subscribers(self):
        """ROS 구독자 설정"""
        # YOLOP 출력
        rospy.Subscriber('/carla/yolop/det_grid', OccupancyGrid, self.det_callback)
        rospy.Subscriber('/carla/yolop/da_grid', OccupancyGrid, self.da_callback)
        rospy.Subscriber('/carla/yolop/ll_grid', OccupancyGrid, self.ll_callback)
        
        # 차량 상태
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odometry_callback)
        
        # Expert trajectory (실제 운전 경로)
        # 현재는 odometry 기반으로 미래 경로를 추정
        self.ego_history = deque(maxlen=20)  # 최근 20개 포즈 저장
        
        rospy.loginfo("📡 ROS 구독자 설정 완료")
        
    def det_callback(self, msg):
        """검출 그리드 콜백"""
        self.current_sample['det_grid'] = self.occupancy_grid_to_array(msg)
        self.current_sample['det_timestamp'] = msg.header.stamp.to_sec()
        self.check_and_save_sample()
        
    def da_callback(self, msg):
        """주행 가능 영역 콜백"""
        self.current_sample['da_grid'] = self.occupancy_grid_to_array(msg)
        self.current_sample['da_timestamp'] = msg.header.stamp.to_sec()
        self.check_and_save_sample()
        
    def ll_callback(self, msg):
        """차선 그리드 콜백"""
        self.current_sample['ll_grid'] = self.occupancy_grid_to_array(msg)
        self.current_sample['ll_timestamp'] = msg.header.stamp.to_sec()
        self.check_and_save_sample()
        
    def odometry_callback(self, msg):
        """차량 상태 콜백"""
        # 현재 속도 계산
        velocity = np.sqrt(
            msg.twist.twist.linear.x**2 + 
            msg.twist.twist.linear.y**2
        )
        
        # 정지 상태면 데이터 수집 안함
        if velocity < self.min_velocity:
            return
            
        # Ego status 저장
        self.current_sample['ego_velocity'] = velocity
        self.current_sample['ego_angular_velocity'] = msg.twist.twist.angular.z
        self.current_sample['ego_pose'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        self.current_sample['ego_timestamp'] = msg.header.stamp.to_sec()
        
        # 포즈 히스토리 업데이트
        current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.header.stamp.to_sec()
        )
        self.ego_history.append(current_pose)
        
        # 미래 궤적 생성 (expert trajectory 대신)
        if len(self.ego_history) >= 10:
            future_trajectory = self.generate_expert_trajectory()
            self.current_sample['expert_trajectory'] = future_trajectory
            
        self.check_and_save_sample()
        
    def occupancy_grid_to_array(self, grid_msg):
        """OccupancyGrid를 numpy 배열로 변환"""
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.float32)
        
        # -1(unknown)을 0으로, 100(occupied)을 1로 정규화
        data = np.where(data == -1, 0, data / 100.0)
        
        return data.reshape((height, width))
        
    def generate_expert_trajectory(self):
        """현재까지의 이동 패턴을 기반으로 미래 궤적 생성"""
        if len(self.ego_history) < 10:
            return None
            
        # 최근 포즈들로부터 속도와 방향 추정
        recent_poses = list(self.ego_history)[-10:]
        
        # 평균 속도 계산
        velocities = []
        for i in range(1, len(recent_poses)):
            dt = recent_poses[i][2] - recent_poses[i-1][2]
            if dt > 0:
                dx = recent_poses[i][0] - recent_poses[i-1][0]
                dy = recent_poses[i][1] - recent_poses[i-1][1]
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(v)
                
        if not velocities:
            return None
            
        avg_velocity = np.mean(velocities)
        
        # 현재 방향 추정
        if len(recent_poses) >= 2:
            dx = recent_poses[-1][0] - recent_poses[-2][0]
            dy = recent_poses[-1][1] - recent_poses[-2][1]
            current_heading = np.arctan2(dy, dx)
        else:
            current_heading = 0.0
            
        # 미래 6스텝 궤적 생성 (각 스텝은 0.2초)
        dt = 0.2
        trajectory = []
        current_x, current_y = recent_poses[-1][0], recent_poses[-1][1]
        
        for i in range(6):
            # 직진 가정 (실제로는 더 복잡한 예측 모델 필요)
            next_x = current_x + avg_velocity * dt * np.cos(current_heading)
            next_y = current_y + avg_velocity * dt * np.sin(current_heading)
            
            # 차량 기준 좌표로 변환
            rel_x = next_x - recent_poses[-1][0]
            rel_y = next_y - recent_poses[-1][1]
            
            trajectory.append([rel_x, rel_y])
            current_x, current_y = next_x, next_y
            
        return np.array(trajectory)
        
    def check_and_save_sample(self):
        """모든 데이터가 수집되면 샘플 저장"""
        required_keys = [
            'det_grid', 'da_grid', 'll_grid',
            'ego_velocity', 'ego_angular_velocity', 'ego_pose',
            'expert_trajectory'
        ]
        
        # 모든 필수 데이터가 있는지 확인
        if not all(key in self.current_sample for key in required_keys):
            return
            
        # 타임스탬프 동기화 확인
        timestamps = [
            self.current_sample.get('det_timestamp', 0),
            self.current_sample.get('da_timestamp', 0),
            self.current_sample.get('ego_timestamp', 0)
        ]
        
        if max(timestamps) - min(timestamps) > self.time_tolerance:
            return  # 동기화되지 않음
            
        # 샘플 복사 및 버퍼에 추가
        sample = dict(self.current_sample)
        sample['timestamp'] = rospy.Time.now().to_sec()
        
        self.data_buffer.append(sample)
        
        # 로그 출력
        if len(self.data_buffer) % 100 == 0:
            rospy.loginfo(f"📊 수집된 샘플 수: {len(self.data_buffer)}")
            
        # 현재 샘플 초기화
        self.current_sample = {}
        
    def save_data_callback(self, event):
        """주기적으로 데이터 저장"""
        if not self.data_buffer:
            return
            
        # 현재 시간 기반 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bev_planner_data_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        # 데이터 저장
        data_to_save = list(self.data_buffer)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
                
            rospy.loginfo(f"💾 데이터 저장 완료: {filename} ({len(data_to_save)} 샘플)")
            
            # 버퍼 절반 정도 클리어 (메모리 관리)
            for _ in range(len(self.data_buffer) // 2):
                self.data_buffer.popleft()
                
        except Exception as e:
            rospy.logerr(f"❌ 데이터 저장 실패: {e}")
            
    def run(self):
        """메인 루프"""
        rospy.loginfo("🚀 데이터 수집 시작!")
        rospy.loginfo("💡 팁: CARLA에서 수동 또는 자동으로 운전하세요")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("🛑 수집 중단됨")
            
        # 종료 시 마지막 데이터 저장
        if self.data_buffer:
            self.save_data_callback(None)
            rospy.loginfo(f"✅ 총 {len(self.data_buffer)} 샘플 수집 완료")

if __name__ == '__main__':
    try:
        collector = BEVPlannerDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass 