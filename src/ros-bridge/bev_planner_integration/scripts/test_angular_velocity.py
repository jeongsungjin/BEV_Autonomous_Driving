#!/usr/bin/env python3
"""
각속도 계산 테스트 스크립트
CARLA에서 각속도가 제대로 계산되는지 확인
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import tf.transformations as tft
from collections import deque
import matplotlib.pyplot as plt
import time

class AngularVelocityTester:
    def __init__(self):
        rospy.init_node('angular_velocity_tester', anonymous=True)
        
        # 데이터 저장용
        self.odometry_angular_velocities = deque(maxlen=100)
        self.imu_angular_velocities = deque(maxlen=100)
        self.numerical_angular_velocities = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        
        # 수치 계산용
        self.prev_yaw = None
        self.prev_timestamp = None
        
        # 구독자 설정
        self.odom_sub = rospy.Subscriber(
            "/carla/ego_vehicle/odometry", Odometry, self.odom_callback, queue_size=1
        )
        self.imu_sub = rospy.Subscriber(
            "/carla/ego_vehicle/imu", Imu, self.imu_callback, queue_size=1
        )
        
        rospy.loginfo("🧪 각속도 테스터 시작!")
        rospy.loginfo("CARLA에서 차량을 움직여서 각속도를 테스트하세요")
        
    def odom_callback(self, msg):
        """Odometry에서 각속도 추출 및 수치 계산"""
        timestamp = msg.header.stamp.to_sec()
        
        # 1. Odometry에서 직접 추출 (보통 0.0)
        odom_angular_vel = msg.twist.twist.angular.z
        
        # 2. 수치적 계산
        numerical_angular_vel = self.calculate_numerical_angular_velocity(msg)
        
        # 데이터 저장
        self.odometry_angular_velocities.append(odom_angular_vel)
        self.numerical_angular_velocities.append(numerical_angular_vel)
        self.timestamps.append(timestamp)
        
        # 실시간 출력 (1초마다)
        if len(self.timestamps) % 10 == 0:
            rospy.loginfo(
                f"📊 각속도 비교 - "
                f"Odom: {odom_angular_vel:.4f}, "
                f"수치: {numerical_angular_vel:.4f}, "
                f"IMU: {self.imu_angular_velocities[-1] if self.imu_angular_velocities else 0:.4f}"
            )
    
    def imu_callback(self, msg):
        """IMU에서 각속도 추출"""
        imu_angular_vel = msg.angular_velocity.z
        self.imu_angular_velocities.append(imu_angular_vel)
    
    def calculate_numerical_angular_velocity(self, msg):
        """수치적 각속도 계산"""
        current_timestamp = msg.header.stamp.to_sec()
        
        # quaternion → yaw 변환
        orientation = msg.pose.pose.orientation
        current_yaw = tft.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])[2]
        
        if self.prev_yaw is None or self.prev_timestamp is None:
            self.prev_yaw = current_yaw
            self.prev_timestamp = current_timestamp
            return 0.0
        
        dt = current_timestamp - self.prev_timestamp
        if dt <= 0:
            return 0.0
        
        # 각도 차이 정규화
        angle_diff = current_yaw - self.prev_yaw
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        angular_velocity = angle_diff / dt
        
        # 업데이트
        self.prev_yaw = current_yaw
        self.prev_timestamp = current_timestamp
        
        return angular_velocity
    
    def generate_report(self):
        """테스트 결과 리포트 생성"""
        if len(self.odometry_angular_velocities) < 10:
            rospy.logwarn("⚠️  데이터가 부족합니다. 더 오래 테스트해주세요.")
            return
        
        odom_data = np.array(list(self.odometry_angular_velocities))
        numerical_data = np.array(list(self.numerical_angular_velocities))
        imu_data = np.array(list(self.imu_angular_velocities))
        
        print("\n" + "="*60)
        print("📈 각속도 테스트 결과 리포트")
        print("="*60)
        
        print(f"📊 데이터 포인트 수: {len(odom_data)}")
        print()
        
        print("🔍 통계 분석:")
        print(f"  Odometry 각속도:")
        print(f"    - 평균: {np.mean(odom_data):.6f} rad/s")
        print(f"    - 표준편차: {np.std(odom_data):.6f}")
        print(f"    - 최대: {np.max(odom_data):.6f}")
        print(f"    - 최소: {np.min(odom_data):.6f}")
        print(f"    - 0이 아닌 값 비율: {np.count_nonzero(odom_data)/len(odom_data)*100:.1f}%")
        print()
        
        print(f"  수치 계산 각속도:")
        print(f"    - 평균: {np.mean(numerical_data):.6f} rad/s")
        print(f"    - 표준편차: {np.std(numerical_data):.6f}")
        print(f"    - 최대: {np.max(numerical_data):.6f}")
        print(f"    - 최소: {np.min(numerical_data):.6f}")
        print()
        
        if len(imu_data) > 0:
            print(f"  IMU 각속도:")
            print(f"    - 평균: {np.mean(imu_data):.6f} rad/s")
            print(f"    - 표준편차: {np.std(imu_data):.6f}")
            print(f"    - 최대: {np.max(imu_data):.6f}")
            print(f"    - 최소: {np.min(imu_data):.6f}")
        
        print("\n✅ 결론:")
        if np.std(odom_data) < 1e-6:
            print("❌ Odometry 각속도가 거의 0입니다 (CARLA 문제 확인됨)")
        else:
            print("✅ Odometry 각속도가 변화하고 있습니다")
        
        if np.std(numerical_data) > np.std(odom_data):
            print("✅ 수치 계산 방법이 더 정확한 각속도를 제공합니다")
        else:
            print("⚠️  수치 계산 결과를 확인해보세요")
        
        print("="*60)

def main():
    try:
        tester = AngularVelocityTester()
        
        print("\n📝 테스트 방법:")
        print("1. CARLA를 실행하고 ego vehicle을 스폰하세요")
        print("2. 수동 제어 또는 자율주행으로 차량을 움직이세요")
        print("3. 특히 회전(커브)하는 상황을 만들어보세요")
        print("4. 30초 후 Ctrl+C로 종료하면 리포트가 생성됩니다")
        print()
        
        # 30초 후 자동 리포트 생성
        def auto_report():
            rospy.sleep(30)
            if not rospy.is_shutdown():
                tester.generate_report()
        
        import threading
        report_thread = threading.Thread(target=auto_report, daemon=True)
        report_thread.start()
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("\n🔍 테스트 종료 - 리포트 생성 중...")
        tester.generate_report()

if __name__ == '__main__':
    main() 