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
        self.current_sample = {}
        self.data_buffer = deque(maxlen=10000)  # 메인 데이터 버퍼
        self.samples_buffer = []
        self.buffer_size = 100
        self.min_velocity = 1.0  # 최소 속도 (m/s) - 정지 상태 데이터 배제 강화
        
        # 궤적 추적용 히스토리
        self.ego_history = deque(maxlen=50)  # 최대 50개 포즈 저장
        
        # 각속도 계산용: 이전 프레임 정보 저장
        self.prev_pose = None
        self.prev_timestamp = None
        
        # IMU에서 받은 각속도 (더 정확함)
        self.imu_angular_velocity = 0.0
        self.use_imu_angular_velocity = False  # 수치적 계산 우선 사용 (IMU 토픽 없음)
        
        # 데이터 동기화를 위한 잠금
        self.data_lock = threading.Lock()
        
        # ROS 구독자들
        self.setup_subscribers()
        
        # TF 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 데이터 저장 타이머
        self.save_timer = rospy.Timer(rospy.Duration(30), self.save_data_callback)
        
        rospy.loginfo("✅ 데이터 수집기 초기화 완료")
        rospy.loginfo("🎯 CARLA에서 운전하면 데이터가 자동 수집됩니다!")
        
    def setup_subscribers(self):
        """ROS 구독자 설정"""
        # YOLOP BEV 그리드 구독
        self.det_sub = rospy.Subscriber(
            "/carla/yolop/det_grid", OccupancyGrid, self.det_callback, queue_size=1
        )
        self.da_sub = rospy.Subscriber(
            "/carla/yolop/da_grid", OccupancyGrid, self.da_callback, queue_size=1
        )
        self.ll_sub = rospy.Subscriber(
            "/carla/yolop/ll_grid", OccupancyGrid, self.ll_callback, queue_size=1
        )
        
        # Ego vehicle 상태 구독
        self.odom_sub = rospy.Subscriber(
            "/carla/ego_vehicle/odometry", Odometry, self.odometry_callback, queue_size=1
        )
        
        # IMU 센서 구독 (각속도 정보용)
        from sensor_msgs.msg import Imu
        self.imu_sub = rospy.Subscriber(
            "/carla/ego_vehicle/imu", Imu, self.imu_callback, queue_size=1
        )
        
        # Expert trajectory (실제 운전 경로)
        # 현재는 odometry 기반으로 미래 경로를 추정
        # self.ego_history = deque(maxlen=20)  # 최근 20개 포즈 저장
        
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
        
    def imu_callback(self, msg):
        """IMU 센서 콜백 - 각속도 정보 업데이트"""
        # IMU의 gyroscope z축이 yaw rate (각속도)
        self.imu_angular_velocity = msg.angular_velocity.z
        
    def odometry_callback(self, msg):
        """차량 상태 콜백"""
        # 현재 속도 계산
        velocity = np.sqrt(
            msg.twist.twist.linear.x**2 + 
            msg.twist.twist.linear.y**2
        )
        
        # 정지 상태나 너무 느린 상태면 데이터 수집 안함
        if velocity < self.min_velocity:
            return
            
        # 각속도 계산 (IMU 우선, 수치적 방법 백업)
        if self.use_imu_angular_velocity:
            angular_velocity = self.imu_angular_velocity
        else:
            angular_velocity = self.calculate_angular_velocity(msg)
            
        # 각속도 값이 비정상적으로 클 때도 제외 (센서 오류)
        if abs(angular_velocity) > 2.0:  # 2 rad/s 이상은 비현실적
            rospy.logwarn(f"⚠️  비정상적인 각속도 감지: {angular_velocity:.3f} rad/s - 샘플 제외")
            return
        
        # Ego status 저장
        self.current_sample['ego_velocity'] = velocity
        self.current_sample['ego_angular_velocity'] = angular_velocity
        self.current_sample['ego_pose'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        self.current_sample['ego_timestamp'] = msg.header.stamp.to_sec()
        
        # 데이터 품질 로그 출력 (odometry_callback에서만)
        if len(self.data_buffer) % 25 == 0 and len(self.data_buffer) > 0:
            rospy.loginfo(
                f"🚗 주행 상태 - 속도: {velocity:.2f} m/s, "
                f"각속도: {angular_velocity:.3f} rad/s, "
                f"수집된 샘플: {len(self.data_buffer)}"
            )
        
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
        
    def calculate_angular_velocity(self, msg):
        """
        이전 프레임과 현재 프레임의 orientation 차이로부터 각속도를 수치적으로 계산
        
        Args:
            msg: nav_msgs/Odometry 메시지
            
        Returns:
            float: 계산된 각속도 (rad/s)
        """
        current_timestamp = msg.header.stamp.to_sec()
        
        # 현재 yaw angle 계산 (quaternion -> euler)
        import tf.transformations as tft
        orientation = msg.pose.pose.orientation
        current_yaw = tft.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])[2]  # yaw는 index 2
        
        # 이전 프레임이 없으면 0 반환
        if self.prev_pose is None or self.prev_timestamp is None:
            self.prev_pose = current_yaw
            self.prev_timestamp = current_timestamp
            return 0.0
        
        # 시간 차이 계산
        dt = current_timestamp - self.prev_timestamp
        
        if dt <= 0:  # 시간이 역행하거나 같으면 0 반환
            return 0.0
        
        # 각도 차이 계산 (-π ~ π 범위로 정규화)
        angle_diff = current_yaw - self.prev_pose
        
        # 각도 차이를 -π ~ π 범위로 정규화
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # 각속도 계산
        angular_velocity = angle_diff / dt
        
        # 현재 값을 이전 값으로 저장
        self.prev_pose = current_yaw
        self.prev_timestamp = current_timestamp
        
        return angular_velocity
        
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
        
        if max(timestamps) - min(timestamps) > 0.1: # 오차 허용 시간 조정
            return  # 동기화되지 않음
            
        # 샘플 복사 및 버퍼에 추가
        sample = dict(self.current_sample)
        sample['timestamp'] = rospy.Time.now().to_sec()
        
        self.data_buffer.append(sample)
        
        # 로그 출력 (간단한 정보만)
        if len(self.data_buffer) % 50 == 0:  # 더 자주 로그 출력
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