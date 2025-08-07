#!/usr/bin/env python
# -- coding: utf-8 --
import rospy
import math
import os
import numpy as np
import sys
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive

# ==== CARLA egg 경로 자동 추가 ====
def append_carla_egg():
    carla_python_path = os.getenv("CARLA_PYTHON_PATH")
    if carla_python_path is None:
        raise EnvironmentError("CARLA_PYTHON_PATH 환경변수가 설정되지 않았습니다.")

    # 예: carla-0.9.13-py3.7-linux-x86_64.egg
    for fname in os.listdir(carla_python_path):
        if fname.startswith("carla-") and fname.endswith(".egg") and "py3.7" in fname:
            full_path = os.path.join(carla_python_path, fname)
            if full_path not in sys.path:
                sys.path.append(full_path)
            break
    else:
        raise FileNotFoundError("CARLA egg 파일을 찾을 수 없습니다. py3.7에 맞는 egg가 있어야 합니다.")

append_carla_egg()

# ==== carla 모듈 임포트 ====
import carla

class GlobalPathPurePursuitController:
    def __init__(self):
        rospy.init_node('global_path_pure_pursuit_controller')
        
        # 파라미터 설정
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 1.5)  # 코너 주행에 최적화된 거리
        self.max_steer = rospy.get_param('~max_steer', 1.2)  # 조향각 완화
        self.target_speed = rospy.get_param('~target_speed', 8.0)  # 코너에서 적절한 속도
        self.wheelbase = rospy.get_param('~wheelbase', 2.8)
        self.path_file = rospy.get_param('~path_file', '/home/carla/.ros/global_path_1.txt')
        
        # 곡선 감지를 위한 변수
        self.prev_waypoint_index = 0
        self.curve_detection_enabled = True
        
        # 구독자 및 발행자 설정
        self.odom_sub = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odom_callback)
        self.ctrl_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', AckermannDrive, queue_size=1)
        
        # 시각화를 위한 발행자
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1, latch=True)
        self.vehicle_pose_pub = rospy.Publisher('/ego_vehicle_pose', PoseStamped, queue_size=1)
        
        # CARLA 클라이언트 설정
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.vehicle = None
        self.ego_vehicle_found = False
        
        # 경로 데이터 초기화
        self.waypoints = []
        self.current_pose = None
        self.current_waypoint_index = 0
        self.prev_steer = 0.0  # 이전 조향각 초기화
        
        # 경로 파일 로드
        self.load_waypoints()
        
        rospy.loginfo(f"로드된 waypoints 개수: {len(self.waypoints)}")
        if self.waypoints:
            rospy.loginfo(f"첫 번째 waypoint: {self.waypoints[0]}")
            rospy.loginfo(f"마지막 waypoint: {self.waypoints[-1]}")
            
            # 글로벌 패스 시각화 즉시 발행 (한 번만)
            if not hasattr(self, '_path_published'):
                self.publish_global_path()
                self._path_published = True
                rospy.loginfo("글로벌 패스가 RViz에 발행되었습니다.")
        
        # 차량을 찾을 때까지 대기
        rospy.loginfo("Ego vehicle을 찾는 중...")
        
        # 차량을 찾을 때까지 계속 재검색
        while not rospy.is_shutdown() and self.vehicle is None:
            self.find_ego_vehicle()
            if self.vehicle is None:
                rospy.loginfo("차량을 찾지 못했습니다. 3초 후 재검색...")
                rospy.sleep(3.0)
            else:
                rospy.loginfo("차량을 찾았습니다. Pure Pursuit 제어를 시작합니다.")
                # 주기적으로 제어 루프 실행
                self.start_control_loop()
                break
        
        

    def load_waypoints(self):
        """waypoints를 로드합니다."""
        try:
            # waypoints 파일 경로 - 실제 파일 경로로 수정
            waypoints_file = rospy.get_param('~waypoints_file', '/home/carla/.ros/global_path_1.txt')
            
            rospy.loginfo(f"Waypoints 파일 경로: {waypoints_file}")
            
            if os.path.exists(waypoints_file):
                rospy.loginfo(f"Waypoints 파일을 찾았습니다: {waypoints_file}")
                with open(waypoints_file, 'r') as f:
                    lines = f.readlines()
                
                rospy.loginfo(f"파일에서 읽은 라인 수: {len(lines)}")
                
                self.waypoints = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                x = float(parts[0].strip())
                                y = float(parts[1].strip())
                                self.waypoints.append((x, y))
                                
                                # 처음 10개와 마지막 10개 waypoint 출력
                                # if i < 10 or i >= len(lines) - 10:
                                #     rospy.loginfo(f"Waypoint {len(self.waypoints)-1}: ({x:.2f}, {y:.2f})")
                        except ValueError as e:
                            rospy.logwarn(f"라인 {i+1} 파싱 실패: {line} - {e}")
                            continue
                
                rospy.loginfo(f"총 {len(self.waypoints)}개의 waypoint를 로드했습니다.")
                
                # 중복 waypoint 제거
                unique_waypoints = []
                seen = set()
                for wp in self.waypoints:
                    if wp not in seen:
                        unique_waypoints.append(wp)
                        seen.add(wp)
                
                if len(unique_waypoints) != len(self.waypoints):
                    rospy.logwarn(f"중복 waypoint 제거: {len(self.waypoints)}개 → {len(unique_waypoints)}개")
                    self.waypoints = unique_waypoints
                
                # waypoint 데이터 검증
                if len(self.waypoints) > 0:
                    # 처음, 중간, 마지막 waypoint 확인
                    first_wp = self.waypoints[0]
                    mid_wp = self.waypoints[len(self.waypoints)//2]
                    last_wp = self.waypoints[-1]
                    
                    rospy.loginfo(f"첫 번째 waypoint: {first_wp}")
                    rospy.loginfo(f"중간 waypoint: {mid_wp}")
                    rospy.loginfo(f"마지막 waypoint: {last_wp}")
                    
                    # 좌표 범위 확인
                    x_coords = [wp[0] for wp in self.waypoints]
                    y_coords = [wp[1] for wp in self.waypoints]
                    rospy.loginfo(f"X 좌표 범위: {min(x_coords):.2f} ~ {max(x_coords):.2f}")
                    rospy.loginfo(f"Y 좌표 범위: {min(y_coords):.2f} ~ {max(y_coords):.2f}")
                else:
                    rospy.logwarn("로드된 waypoint가 없습니다!")
                    
            else:
                rospy.logwarn(f"Waypoints 파일을 찾을 수 없습니다: {waypoints_file}")
                rospy.loginfo("샘플 경로를 생성합니다.")
                self.create_sample_path()
                
        except Exception as e:
            rospy.logerr(f"Waypoints 로드 중 오류: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            rospy.loginfo("샘플 경로를 생성합니다.")
            self.create_sample_path()

    def create_sample_path(self):
        """예시 경로를 생성합니다 (원형 경로)"""
        rospy.loginfo("예시 경로를 생성합니다.")
        center_x, center_y = 0, 0
        radius = 50
        num_points = 100
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.waypoints.append((x, y))

    def odom_callback(self, msg):
        """Odometry 콜백 함수"""
        try:
            # 차량 위치 업데이트
            self.current_pose.position.x = msg.pose.pose.position.x
            self.current_pose.position.y = msg.pose.pose.position.y
            self.current_pose.position.z = msg.pose.pose.position.z
            
            # Yaw 계산 - CARLA transform 우선 사용
            yaw_rad = 0.0
            if self.vehicle is not None:
                try:
                    carla_transform = self.vehicle.get_transform()
                    yaw_rad = math.radians(carla_transform.rotation.yaw)
                    rospy.loginfo(f"CARLA Transform Yaw: {carla_transform.rotation.yaw:.2f}도 ({yaw_rad:.4f} 라디안)")
                    
                    # Quaternion으로 변환하여 저장
                    self.current_pose.orientation.x = 0.0
                    self.current_pose.orientation.y = 0.0
                    self.current_pose.orientation.z = math.sin(yaw_rad / 2.0)
                    self.current_pose.orientation.w = math.cos(yaw_rad / 2.0)
                    
                except Exception as e:
                    rospy.logwarn(f"CARLA transform 오류: {e}")
                    # Fallback: 기존 quaternion 사용
                    self.current_pose.orientation = msg.pose.pose.orientation
                    yaw_rad = self.get_yaw(msg.pose.pose.orientation)
            else:
                # Fallback: 기존 quaternion 사용
                self.current_pose.orientation = msg.pose.pose.orientation
                yaw_rad = self.get_yaw(msg.pose.pose.orientation)
            
            # 속도 정보 업데이트
            self.current_velocity = msg.twist.twist.linear
            
            # 디버깅 정보
            rospy.loginfo(f"차량 위치: ({self.current_pose.position.x:.3f}, {self.current_pose.position.y:.3f})")
            rospy.loginfo(f"차량 yaw: {yaw_rad:.4f} 라디안 ({math.degrees(yaw_rad):.2f}도)")
            
        except Exception as e:
            rospy.logerr(f"Odometry 콜백 오류: {e}")

    def detect_curve(self):
        """곡선 구간을 감지하고 lookahead distance를 조정합니다."""
        if not self.curve_detection_enabled or self.current_waypoint_index >= len(self.waypoints) - 2:
            return self.lookahead_distance
        
        # 현재 waypoint와 다음 waypoint들의 방향 변화 계산
        current_wp = self.waypoints[self.current_waypoint_index]
        next_wp = self.waypoints[min(self.current_waypoint_index + 1, len(self.waypoints) - 1)]
        future_wp = self.waypoints[min(self.current_waypoint_index + 2, len(self.waypoints) - 1)]
        
        # 방향 벡터 계산
        vec1 = (next_wp[0] - current_wp[0], next_wp[1] - current_wp[1])
        vec2 = (future_wp[0] - next_wp[0], future_wp[1] - next_wp[1])
        
        # 벡터 정규화
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            vec1_norm = (vec1[0]/mag1, vec1[1]/mag1)
            vec2_norm = (vec2[0]/mag2, vec2[1]/mag2)
            
            # 내적 계산 (방향 변화 정도)
            dot_product = vec1_norm[0] * vec2_norm[0] + vec1_norm[1] * vec2_norm[1]
            angle_change = math.acos(max(-1, min(1, dot_product)))
            
            # 곡선 감지 (각도 변화가 0.4 라디안 이상으로 완화)
            if angle_change > 0.4:  # 0.3에서 0.4로 완화
                rospy.loginfo(f"곡선 감지: 각도 변화 {angle_change:.3f} 라디안, lookahead distance 조정")
                return self.lookahead_distance * 0.6  # 곡선에서는 60% 감소 (50%에서 60%로 완화)
            elif angle_change > 0.2:  # 0.15에서 0.2로 완화
                rospy.loginfo(f"약한 곡선 감지: 각도 변화 {angle_change:.3f} 라디안")
                return self.lookahead_distance * 0.8  # 약한 곡선에서는 80% (70%에서 80%로 완화)
        
        return self.lookahead_distance

    def find_target_point_sequential(self):
        """순서대로 waypoint를 강제하는 새로운 타겟 찾기 알고리즘"""
        if not self.waypoints or self.current_pose is None:
            return None
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        # 강화된 waypoint 진행 검사: 현재 위치가 현재 waypoint를 지나쳤는지 확인
        if self.current_waypoint_index < len(self.waypoints) - 1:
            current_wx, current_wy = self.waypoints[self.current_waypoint_index]
            next_wx, next_wy = self.waypoints[self.current_waypoint_index + 1]
            
            # 현재 waypoint와 다음 waypoint 사이의 선분에서 현재 위치의 투영점 계산
            waypoint_vec_x = next_wx - current_wx
            waypoint_vec_y = next_wy - current_wy
            vehicle_vec_x = x - current_wx
            vehicle_vec_y = y - current_wy
            
            # 내적 계산으로 투영
            waypoint_mag = math.sqrt(waypoint_vec_x**2 + waypoint_vec_y**2)
            if waypoint_mag > 0:
                projection = (vehicle_vec_x * waypoint_vec_x + vehicle_vec_y * waypoint_vec_y) / waypoint_mag
                
                # 투영이 음수이거나 waypoint를 지나쳤으면 다음 waypoint로 진행
                if projection < 0 or projection > waypoint_mag:
                    rospy.logwarn(f"Waypoint {self.current_waypoint_index}를 지나쳤습니다. 다음 waypoint로 진행.")
                    if self.current_waypoint_index < len(self.waypoints) - 1:
                        self.current_waypoint_index += 1
                        rospy.loginfo(f"다음 waypoint로 진행: 인덱스 {self.current_waypoint_index}")
        
        # 현재 waypoint에 충분히 가까졌는지 확인 (완화된 조건)
        if self.current_waypoint_index < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint_index]
            dist_to_current = math.hypot(wx - x, wy - y)
            
            # 거리가 너무 멀면 다음 waypoint로 점프
            if dist_to_current > 25.0:  # 20.0에서 25.0으로 완화
                rospy.logwarn(f"현재 waypoint가 너무 멀리 떨어져 있습니다. 거리: {dist_to_current:.2f}m")
                
                # 다음 20개 waypoint 중 가장 가까운 것 찾기
                min_dist = float('inf')
                best_index = self.current_waypoint_index
                
                for i in range(self.current_waypoint_index, min(self.current_waypoint_index + 20, len(self.waypoints))):
                    wx, wy = self.waypoints[i]
                    dist = math.hypot(wx - x, wy - y)
                    if dist < min_dist:
                        min_dist = dist
                        best_index = i
                
                if best_index != self.current_waypoint_index:
                    rospy.loginfo(f"가장 가까운 waypoint로 점프: {self.current_waypoint_index} -> {best_index}")
                    self.current_waypoint_index = best_index
                else:
                    # 가까운 waypoint가 없으면 한 칸씩 진행
                    self.current_waypoint_index += 1
                    rospy.loginfo(f"다음 waypoint로 진행: {self.current_waypoint_index}")
            
            # 현재 waypoint에 가까우면 다음으로 진행
            elif dist_to_current < 3.0:  # 5.0에서 3.0으로 엄격하게
                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    rospy.loginfo(f"Waypoint {self.current_waypoint_index-1} 도달. 다음 waypoint로 진행: {self.current_waypoint_index}")
            
            rospy.loginfo(f"현재 waypoint까지 거리: {dist_to_current:.2f}m")
            
            # 현재 waypoint 반환
            wx, wy = self.waypoints[self.current_waypoint_index]
            return wx, wy
        
        # 현재 waypoint 인덱스가 유효한지 확인
        if self.current_waypoint_index >= len(self.waypoints):
            rospy.loginfo("모든 waypoints를 완료했습니다!")
            return self.waypoints[-1] if self.waypoints else None
        
        # 순서대로 타겟 찾기: 현재 waypoint부터 시작하여 lookahead distance만큼 앞의 waypoint 찾기
        target_index = self.current_waypoint_index
        target_distance = 0
        
        # 현재 waypoint부터 순서대로 검색
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            wx, wy = self.waypoints[i]
            dist = math.hypot(wx - x, wy - y)
            
            # lookahead distance에 도달하면 해당 waypoint를 타겟으로 선택
            if dist >= self.lookahead_distance:
                target_index = i
                target_distance = dist
                break
            elif i == len(self.waypoints) - 1:
                # 마지막 waypoint에 도달한 경우
                target_index = i
                target_distance = dist
                break
        
        wx, wy = self.waypoints[target_index]
        rospy.loginfo(f"순서 타겟: 인덱스 {target_index}, 위치 ({wx:.2f}, {wy:.2f}), 거리 {target_distance:.2f}m")
        return wx, wy

    def find_target_point(self):
        """현재 위치에서 lookahead_distance만큼 앞의 waypoint를 찾습니다."""
        # 순서대로 강제하는 새로운 알고리즘 사용
        return self.find_target_point_sequential()

    def update_waypoint_progress(self):
        """waypoint 진행도를 업데이트합니다."""
        if not self.waypoints or self.current_pose is None:
            return
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        # 현재 waypoint에 충분히 가까졌는지 확인
        if self.current_waypoint_index < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint_index]
            dist_to_current = math.hypot(wx - x, wy - y)
            
            # 현재 waypoint에 가까워졌으면 다음 waypoint로 진행
            if dist_to_current < 0.5:  # 0.8에서 0.5로 더 엄격하게
                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    rospy.loginfo(f"다음 waypoint로 진행: 인덱스 {self.current_waypoint_index}")
                else:
                    rospy.loginfo("모든 waypoints를 완료했습니다!")
            else:
                rospy.loginfo(f"현재 waypoint까지 거리: {dist_to_current:.2f}m")



    def get_yaw(self, orientation):
        """쿼터니언에서 yaw 각도를 추출합니다."""
        # 쿼터니언에서 yaw 각도 계산 (수학적 방법)
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # 디버깅을 위한 로그 추가
        rospy.loginfo(f"쿼터니언: w={qw:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}")
        rospy.loginfo(f"계산된 Yaw: {yaw:.4f} 라디안 ({math.degrees(yaw):.2f}도)")
        
        return yaw

    def find_ego_vehicle(self):
        """ego vehicle을 찾습니다."""
        try:
            # 모든 차량 목록 출력
            all_actors = list(self.world.get_actors())
            vehicle_actors = [actor for actor in all_actors if actor.type_id.startswith('vehicle.')]
            rospy.loginfo(f"총 {len(all_actors)}개의 actor 중 {len(vehicle_actors)}개의 차량 발견")
            
            for actor in vehicle_actors:
                rospy.loginfo(f"차량 ID: {actor.id}, Type: {actor.type_id}")
                try:
                    role_name = actor.attributes.get('role_name', '')
                    rospy.loginfo(f"  - Role name: {role_name}")
                    if role_name == 'ego_vehicle':
                        self.vehicle = actor
                        self.ego_vehicle_found = True
                        rospy.loginfo(f"Ego vehicle found: {actor.id}")
                        # autopilot 비활성화하고 직접 제어
                        self.vehicle.set_autopilot(False)
                        rospy.loginfo("Autopilot을 비활성화하고 Pure Pursuit 제어를 시작합니다.")
                        return
                except Exception as e:
                    rospy.logwarn(f"차량 {actor.id} 속성 읽기 실패: {e}")
                    continue
            
            # role_name이 없으면 모든 차량 중에서 첫 번째 차량을 ego vehicle로 설정
            if not self.ego_vehicle_found and vehicle_actors:
                self.vehicle = vehicle_actors[0]
                self.ego_vehicle_found = True
                rospy.loginfo(f"첫 번째 차량을 ego vehicle으로 설정: {self.vehicle.id}")
                # autopilot 비활성화하고 직접 제어
                self.vehicle.set_autopilot(False)
                rospy.loginfo("Autopilot을 비활성화하고 Pure Pursuit 제어를 시작합니다.")
                return
            
            # 여전히 못 찾으면 가장 최근에 생성된 차량을 찾기 (ID가 큰 차량)
            if not self.ego_vehicle_found and vehicle_actors:
                max_id = max(actor.id for actor in vehicle_actors)
                latest_vehicle = next(actor for actor in vehicle_actors if actor.id == max_id)
                
                self.vehicle = latest_vehicle
                self.ego_vehicle_found = True
                rospy.loginfo(f"최신 차량을 ego vehicle으로 설정: {latest_vehicle.id}")
                # autopilot 비활성화하고 직접 제어
                self.vehicle.set_autopilot(False)
                rospy.loginfo("Autopilot을 비활성화하고 Pure Pursuit 제어를 시작합니다.")
                return
            
            if not self.ego_vehicle_found:
                rospy.logwarn("Ego vehicle을 찾을 수 없습니다. 주기적으로 재검색합니다.")
        except Exception as e:
            rospy.logerr(f"Ego vehicle 검색 중 오류: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def check_and_find_ego_vehicle(self):
        """ego vehicle이 없으면 다시 찾습니다."""
        if self.vehicle is None or not self.ego_vehicle_found:
            self.find_ego_vehicle()
            # ego vehicle을 찾았으면 로그 출력
            if self.vehicle is not None:
                rospy.loginfo(f"Ego vehicle 검색 성공: ID {self.vehicle.id}")
                # autopilot이 활성화되어 있으면 비활성화
                try:
                    if self.vehicle.get_autopilot():
                        self.vehicle.set_autopilot(False)
                        rospy.loginfo("Autopilot을 비활성화했습니다.")
                except:
                    pass
            else:
                rospy.logwarn("Ego vehicle 검색 실패 - 제어 불가")

    def control_carla_vehicle(self, steer, speed):
        """CARLA 차량을 직접 제어합니다."""
        if self.vehicle is None:
            rospy.logwarn("제어할 차량이 없습니다.")
            return
        
        try:
            # 차량의 현재 속도 가져오기
            current_velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)
            
            # CARLA 차량 제어
            control = carla.VehicleControl()
            control.steer = steer
            
            # 속도 기반 throttle/brake 계산
            if speed > 0:
                # 목표 속도와 현재 속도 비교
                speed_diff = speed - current_speed
                
                if speed_diff > 0:
                    # 가속이 필요한 경우
                    control.throttle = min(1.0, speed / 5.0)  # throttle 계산
                    control.brake = 0.0
                else:
                    # 감속이 필요한 경우 - 브레이킹 완화
                    control.throttle = 0.0
                    # 속도 차이가 클 때만 브레이크 적용 (임계값 증가)
                    if abs(speed_diff) > 2.0:  # 2m/s 이상 차이날 때만 브레이크
                        brake_intensity = min(0.5, abs(speed_diff) / 10.0)  # 브레이크 강도 완화
                        control.brake = brake_intensity
                        rospy.loginfo(f"브레이크 적용: 현재속도={current_speed:.2f}m/s, 목표속도={speed:.2f}m/s, 브레이크={brake_intensity:.2f}")
                    else:
                        control.brake = 0.0  # 작은 차이는 브레이크 없이 자연 감속
            else:
                # 정지
                control.throttle = 0.0
                control.brake = 0.5  # 1.0에서 0.5로 완화
            
            # 속도 모니터링 로그
            rospy.loginfo(f"속도 모니터링: 현재={current_speed:.2f}m/s, 목표={speed:.2f}m/s, throttle={control.throttle:.2f}, brake={control.brake:.2f}")
            
            # 차량에 제어 명령 적용
            self.vehicle.apply_control(control)
            rospy.loginfo(f"차량 제어 명령: steer={steer:.3f}, throttle={control.throttle:.3f}, brake={control.brake:.3f}")
            rospy.loginfo(f"차량 제어 명령이 적용되었습니다.")
            
        except Exception as e:
            rospy.logerr(f"차량 제어 중 오류: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def calculate_distance_to_path(self):
        """현재 위치에서 경로까지의 최단 거리를 계산합니다."""
        if not self.waypoints or self.current_pose is None:
            return float('inf')
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        min_dist = float('inf')
        for wx, wy in self.waypoints:
            dist = math.hypot(wx - x, wy - y)
            min_dist = min(min_dist, dist)
        
        return min_dist

    def start_control_loop(self):
        """주기적으로 제어 루프를 실행합니다."""
        rate = rospy.Rate(10)  # 10Hz
        rospy.loginfo("Pure Pursuit 제어 루프를 시작합니다.")
        
        while not rospy.is_shutdown():
            try:
                # ego vehicle 확인
                self.check_and_find_ego_vehicle()
                
                # ego vehicle이 없으면 제어하지 않음
                if self.vehicle is None:
                    rospy.logwarn("Ego vehicle이 없어서 제어를 건너뜁니다.")
                    rate.sleep()
                    continue
                
                # autopilot이 활성화되어 있으면 비활성화
                try:
                    if self.vehicle.get_autopilot():
                        self.vehicle.set_autopilot(False)
                        rospy.loginfo("Autopilot을 비활성화하고 Pure Pursuit 제어를 시작합니다.")
                except:
                    pass
                
                # 현재 차량 위치를 가져와서 제어
                vehicle_transform = self.vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                
                # 임시로 current_pose 설정
                from geometry_msgs.msg import Pose, Point, Quaternion
                self.current_pose = Pose()
                self.current_pose.position.x = vehicle_location.x
                self.current_pose.position.y = vehicle_location.y
                self.current_pose.position.z = vehicle_location.z
                
                # 제어 루프 실행
                self.control_loop()
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"제어 루프 중 오류: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                rate.sleep()

    def publish_global_path(self):
        """글로벌 패스를 RViz에서 시각화하기 위해 발행합니다."""
        if not self.waypoints:
            return
            
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        rospy.loginfo(f"글로벌 패스 발행: {len(self.waypoints)}개의 waypoints")

    def publish_vehicle_pose(self, location, rotation):
        """차량 위치를 RViz에서 시각화하기 위해 발행합니다."""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = location.x
            pose_msg.pose.position.y = location.y
            pose_msg.pose.position.z = location.z
            
            # 쿼터니언으로 변환
            import tf
            quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(rotation.yaw))
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            
            self.vehicle_pose_pub.publish(pose_msg)
            rospy.loginfo(f"차량 pose 발행: ({location.x:.2f}, {location.y:.2f})")
            
        except Exception as e:
            rospy.logerr(f"차량 pose 발행 중 오류: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def control_loop(self):
        """메인 제어 루프"""
        # ego vehicle 확인
        self.check_and_find_ego_vehicle()
        
        # ego vehicle이 없으면 제어하지 않음
        if self.vehicle is None:
            rospy.logwarn("Ego vehicle이 없어서 제어를 건너뜁니다.")
            return
        
        # autopilot이 활성화되어 있으면 비활성화
        try:
            if self.vehicle.get_autopilot():
                self.vehicle.set_autopilot(False)
                rospy.loginfo("Autopilot을 비활성화하고 Pure Pursuit 제어를 시작합니다.")
        except:
            pass
        
        # 현재 차량 위치와 방향
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        # CARLA transform에서 직접 yaw 가져오기
        try:
            carla_transform = self.vehicle.get_transform()
            yaw = math.radians(carla_transform.rotation.yaw)
            rospy.loginfo(f"CARLA Yaw: {carla_transform.rotation.yaw:.2f}도 ({yaw:.4f} 라디안)")
        except Exception as e:
            rospy.logwarn(f"CARLA transform 오류: {e}")
            # Fallback: 기존 방식
            yaw = self.get_yaw(self.current_pose.orientation)
        
        # waypoint 진행도 업데이트
        self.update_waypoint_progress()
        
        # 현재 waypoint까지의 거리 계산 (경로 이탈 감지용)
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        dist_to_current = 0.0
        if self.current_waypoint_index < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint_index]
            dist_to_current = math.hypot(wx - x, wy - y)
        
        # 경로 이탈 감지 및 리셋
        path_dist = self.calculate_distance_to_path()
        if path_dist > 10.0 or (self.current_waypoint_index < len(self.waypoints) and dist_to_current > 30.0):  # 15.0에서 10.0으로 엄격하게
            rospy.logwarn(f"차량이 경로를 많이 벗어났습니다 (Path Dist: {path_dist:.2f}m, Waypoint Dist: {dist_to_current:.2f}m). 가장 가까운 waypoint로 리셋합니다.")
            
            # 가장 가까운 waypoint 찾기
            min_dist = float('inf')
            closest_index = 0
            
            for i, (wx, wy) in enumerate(self.waypoints):
                dist = math.hypot(wx - x, wy - y)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
            
            self.current_waypoint_index = closest_index
            rospy.loginfo(f"타겟을 가장 가까운 waypoint로 리셋: 인덱스 {closest_index}, 위치 ({self.waypoints[closest_index][0]:.6f}, {self.waypoints[closest_index][1]:.6f}), 거리 {min_dist:.2f}m")
        
        # 좌표계 변환 - CARLA와 ROS 좌표계 차이 해결
        # x, y는 이미 위에서 정의됨
        
        # CARLA transform에서 직접 yaw 가져오기
        try:
            carla_transform = self.vehicle.get_transform()
            yaw = math.radians(carla_transform.rotation.yaw)
            rospy.loginfo(f"CARLA Yaw: {carla_transform.rotation.yaw:.2f}도 ({yaw:.4f} 라디안)")
        except Exception as e:
            rospy.logwarn(f"CARLA transform 오류: {e}")
            # Fallback: 기존 방식
            yaw = self.get_yaw(self.current_pose.orientation)
        
        # 타겟 waypoint 찾기
        target_point = self.find_target_point_sequential()
        if target_point is None:
            rospy.logwarn("타겟 포인트를 찾을 수 없습니다.")
            return
        
        tx, ty = target_point
        
        # 디버깅: 차량 위치와 방향 정보 출력
        rospy.loginfo(f"차량 위치: ({x:.3f}, {y:.3f})")
        rospy.loginfo(f"차량 yaw: {yaw:.4f} 라디안 ({math.degrees(yaw):.2f}도)")
        rospy.loginfo(f"타겟 위치: ({tx:.3f}, {ty:.3f})")
        
        # CARLA 좌표계에서 ROS 좌표계로 변환 (필요한 경우)
        # CARLA: X(forward), Y(right), Z(up)
        # ROS: X(forward), Y(left), Z(up)
        # y = -y  # Y축 반전이 필요한 경우
        # ty = -ty  # 타겟 Y축도 반전
        
        # 목표점까지의 거리와 각도 계산
        dx = tx - x
        dy = ty - y
        alpha = math.atan2(dy, dx) - yaw
        
        # 각도 정규화 (-pi ~ pi)
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        Ld = math.hypot(dx, dy)
        
        # 디버깅: 목표 각도 정보 출력
        rospy.loginfo(f"목표 각도: {math.degrees(alpha):.2f}도, 거리: {Ld:.2f}m")
        rospy.loginfo(f"dx: {dx:.3f}, dy: {dy:.3f}")
        
        # Pure Pursuit 공식 적용 (개선된 버전)
        if Ld > 0:
            # 기본 Pure Pursuit 공식
            steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
            
            # alpha가 π에 가까우면 강한 조향 적용
            if abs(alpha) > 2.5:  # 거의 반대 방향
                steer = math.copysign(self.max_steer, alpha)
        else:
            steer = 0
        
        # 스티어링 각도 제한 (코너 주행에 적합하게)
        max_steer_safe = self.max_steer * 0.8  # 60%에서 80%로 완화
        steer = max(-max_steer_safe, min(max_steer_safe, steer))
        
        # 급격한 조향 변화 방지 (완화)
        if hasattr(self, 'prev_steer'):
            steer_diff = abs(steer - self.prev_steer)
            if steer_diff > 0.2:  # 0.15에서 0.2로 완화
                rospy.logwarn(f"급격한 조향 변화 감지: {steer_diff:.3f}, 조향각 제한")
                if steer > self.prev_steer:
                    steer = self.prev_steer + 0.2
                else:
                    steer = self.prev_steer - 0.2
        self.prev_steer = steer
        
        # 속도 조정 (안정적인 주행에 최적화)
        speed = self.target_speed
        
        # 현재 waypoint까지의 거리에 따른 속도 조정
        if self.current_waypoint_index < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint_index]
            dist_to_current = math.hypot(wx - x, wy - y)
            if dist_to_current < 2.0:  # 3.0에서 2.0으로 엄격하게
                speed *= 0.6  # 70%에서 60%로 엄격하게
        
        # 조향각에 따른 속도 조정 (안정적인 주행에 적합하게)
        if abs(steer) > 0.15:  # 0.1에서 0.15로 엄격하게
            speed *= (1.0 - abs(steer) / self.max_steer * 0.4)  # 0.5에서 0.4로 엄격하게
        
        # 급격한 조향 시 추가 감속 (엄격하게)
        if abs(steer) > 0.25:  # 0.3에서 0.25로 엄격하게
            speed *= 0.5  # 70%에서 50%로 엄격하게
            rospy.logwarn(f"급격한 조향 감지: steer={steer:.3f}, 속도 감속: {speed:.2f} m/s")
        
        # 곡선 감지에 따른 추가 감속 (완화)
        dynamic_lookahead = self.detect_curve()
        if dynamic_lookahead < self.lookahead_distance * 0.8:  # 0.7에서 0.8로 완화
            speed *= 0.9  # 곡선에서 90% 추가 감속 (80%에서 90%로 완화)
            rospy.loginfo(f"곡선 구간 감지: 속도 추가 감속 {speed:.2f} m/s")
        
        # 최소 속도 보장
        speed = max(2.0, speed)  # 1.5에서 2.0으로 증가
        
        # 최대 속도 제한
        speed = min(speed, self.target_speed)
        
        # 디버그 정보 추가
        rospy.loginfo(f"Pure Pursuit 계산: dx={dx:.3f}, dy={dy:.3f}, alpha={alpha:.3f}, Ld={Ld:.3f}, steer={steer:.3f}")
        rospy.loginfo(f"차량 yaw: {yaw:.3f}, 목표 각도: {math.atan2(dy, dx):.3f}")
        rospy.loginfo(f"차량 위치: ({x:.3f}, {y:.3f}), 타겟 위치: ({tx:.3f}, {ty:.3f})")
        
        # CARLA 차량 직접 제어
        self.control_carla_vehicle(steer, speed)
        
        # ROS 메시지도 발행 (다른 노드들이 사용할 수 있도록)
        ctrl = AckermannDrive()
        ctrl.steering_angle = steer
        ctrl.speed = speed
        self.ctrl_pub.publish(ctrl)
        
        # 차량 정보 출력
        if self.vehicle is not None:
            try:
                vehicle_transform = self.vehicle.get_transform()
                vehicle_velocity = self.vehicle.get_velocity()
                vehicle_location = vehicle_transform.location
                vehicle_rotation = vehicle_transform.rotation
                
                # 경로까지의 거리 계산
                path_dist = self.calculate_distance_to_path()
                
                rospy.loginfo(f"=== 차량 정보 ===")
                rospy.loginfo(f"위치: x={vehicle_location.x:.3f}, y={vehicle_location.y:.3f}, z={vehicle_location.z:.3f}")
                rospy.loginfo(f"속도: {math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2):.2f} m/s")
                rospy.loginfo(f"조향각: {vehicle_rotation.yaw:.2f}도")
                rospy.loginfo(f"Target: ({tx:.2f}, {ty:.2f}), Steer: {steer:.3f}, Speed: {speed:.2f}")
                rospy.loginfo(f"Path Dist: {path_dist:.2f}")
                rospy.loginfo(f"현재 Waypoint 인덱스: {self.current_waypoint_index}/{len(self.waypoints)}")
                
                # 현재 waypoint까지의 거리 계산
                if self.current_waypoint_index < len(self.waypoints):
                    wx, wy = self.waypoints[self.current_waypoint_index]
                    dist_to_current = math.hypot(wx - vehicle_location.x, wy - vehicle_location.y)
                    rospy.loginfo(f"현재 waypoint까지 거리: {dist_to_current:.2f}m")
                
                rospy.loginfo(f"==================")
                
                # 차량 위치 시각화
                try:
                    self.publish_vehicle_pose(vehicle_location, vehicle_rotation)
                except Exception as e:
                    rospy.logerr(f"차량 pose 발행 중 오류: {e}")
                
            except Exception as e:
                rospy.logerr(f"차량 정보 출력 중 오류: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
        else:
            rospy.logwarn("차량 정보를 가져올 수 없습니다.")
            rospy.logwarn("Ego vehicle이 설정되지 않았습니다.")

if __name__ == '__main__':
    try:
        GlobalPathPurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 