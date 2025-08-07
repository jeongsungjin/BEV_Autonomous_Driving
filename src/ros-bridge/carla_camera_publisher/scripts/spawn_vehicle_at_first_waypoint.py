#!/usr/bin/env python
# -- coding: utf-8 --
import rospy
import os
import sys
import time
import math
import random
from carla_msgs.msg import CarlaEgoVehicleInfo
from geometry_msgs.msg import Pose
from std_msgs.msg import String

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

class VehicleSpawner:
    def __init__(self):
        rospy.init_node('vehicle_spawner')
        
        # 파라미터 설정
        self.path_file = rospy.get_param('~path_file', '/home/carla/.ros/global_path_1.txt')
        self.vehicle_model = rospy.get_param('~vehicle_model', 'vehicle.tesla.model3')
        
        # 첫 번째 waypoint 로드
        self.first_waypoint = self.load_first_waypoint()
        
        if self.first_waypoint:
            rospy.loginfo(f"첫 번째 waypoint 위치: ({self.first_waypoint[0]:.3f}, {self.first_waypoint[1]:.3f})")
            self.spawn_vehicle()
        else:
            rospy.logerr("첫 번째 waypoint를 로드할 수 없습니다.")

    def load_first_waypoint(self):
        """경로 파일에서 첫 번째 waypoint를 로드합니다."""
        try:
            if not os.path.exists(self.path_file):
                rospy.logerr(f"경로 파일이 존재하지 않습니다: {self.path_file}")
                return None
            
            with open(self.path_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # x, y, z, qx, qy, qz, qw 형식으로 파싱
                        parts = line.split(',')
                        if len(parts) >= 2:
                            x = float(parts[0].strip())
                            y = float(parts[1].strip())
                            return (x, y)
                    except ValueError:
                        continue
            
            rospy.logerr("유효한 waypoint를 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            rospy.logerr(f"경로 파일 로드 중 오류: {e}")
            return None

    def spawn_vehicle(self):
        """첫 번째 waypoint 위치에 차량을 스폰합니다."""
        try:
            x, y = self.first_waypoint
            
            # CARLA 클라이언트 연결
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            world = client.get_world()
            
            # 차량 모델 설정
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(self.vehicle_model)
            
            if vehicle_bp is None:
                rospy.logerr(f"차량 모델을 찾을 수 없습니다: {self.vehicle_model}")
                return
            
            # 차량을 ego_vehicle로 설정 (blueprint에서 설정)
            vehicle_bp.set_attribute('role_name', 'ego_vehicle')
            
            # 차량 스폰 후 role_name 확인
            rospy.loginfo(f"Blueprint role_name: {vehicle_bp.get_attribute('role_name')}")
            
            # 스폰 위치 설정 (첫 번째 waypoint 위치)
            spawn_point = carla.Transform(
                carla.Location(x=x, y=y, z=1.0),
                carla.Rotation(pitch=0, yaw=0, roll=0)  # 180도로 차량을 반대로 회전
            )
            
            # 차량 스폰 시도
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if vehicle is not None:
                # 차량 스폰 후 autopilot 비활성화 (Pure Pursuit 제어를 위해)
                vehicle.set_autopilot(False)
                rospy.loginfo("차량 autopilot이 비활성화되었습니다. Pure Pursuit 제어 대기 중...")
            
            if vehicle is None:
                rospy.logwarn("첫 번째 위치에서 스폰 실패. 다른 위치를 시도합니다.")
                
                # 다른 스폰 포인트들 시도
                spawn_points = world.get_map().get_spawn_points()
                for sp in spawn_points:
                    # 첫 번째 waypoint와 가까운 스폰 포인트 찾기
                    sp_location = sp.location
                    distance = math.sqrt((sp_location.x - x)**2 + (sp_location.y - y)**2)
                    
                    if distance < 50:  # 50m 이내의 스폰 포인트만 시도
                        vehicle = world.try_spawn_actor(vehicle_bp, sp)
                        if vehicle is not None:
                            rospy.loginfo(f"대체 위치에서 스폰 성공: ({sp_location.x:.2f}, {sp_location.y:.2f})")
                            break
                
                # 여전히 실패하면 기존 차량을 찾아서 사용
                if vehicle is None:
                    rospy.logwarn("새 차량 스폰 실패. 기존 차량을 찾습니다.")
                    for actor in world.get_actors():
                        if actor.type_id.startswith('vehicle.'):
                            vehicle = actor
                            rospy.loginfo(f"기존 차량 사용: {actor.id}")
                            break
                    
                    if vehicle is None:
                        rospy.logerr("차량을 찾을 수 없습니다.")
                        return
            
            rospy.loginfo(f"차량이 성공적으로 스폰되었습니다!")
            rospy.loginfo(f"차량 모델: {self.vehicle_model}")
            rospy.loginfo(f"스폰 위치: x={x:.3f}, y={y:.3f}")
            rospy.loginfo(f"첫 번째 waypoint: ({self.first_waypoint[0]:.3f}, {self.first_waypoint[1]:.3f})")
            rospy.loginfo(f"차량 ID: {vehicle.id}")
            
            # 관찰자 카메라를 차량 위치로 이동
            spectator = world.get_spectator()
            spectator_location = carla.Location(x=x, y=y, z=x+20)  # 차량 위 20m
            spectator_rotation = carla.Rotation(pitch=-45, yaw=0)  # 45도 아래를 보도록
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            rospy.loginfo("관찰자 카메라가 차량 위치로 이동했습니다.")
            
        except Exception as e:
            rospy.logerr(f"차량 스폰 중 오류: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    try:
        VehicleSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 