#!/usr/bin/env python3

import os
import sys
import math
import time
import random
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped

# for static optical frame
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# === Odometry message ===
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

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


class MultipleCarlaCamera:
    def __init__(self):
        rospy.init_node("multiple_carla_camera_publisher", anonymous=True)
        
        # 차량 수
        self.num_vehicles = 3
        
        # 각 차량별 퍼블리셔 초기화
        self.image_pubs = []
        self.cam_info_pubs = []
        self.odom_pubs = []
        
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            self.image_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/image_raw", Image, queue_size=10)
            )
            self.cam_info_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/camera_info", CameraInfo, queue_size=10, latch=True)
            )
            self.odom_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/odometry", Odometry, queue_size=10)
            )

        # Static transform broadcaster
        self.static_broad = StaticTransformBroadcaster()
        self.setup_static_transforms()
        
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # CARLA 연결
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Traffic Manager 설정
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(3.0)  # 앞차와의 거리 3m
        self.traffic_manager.set_synchronous_mode(True)
        
        # 동기 모드 설정 (더 안정적인 시뮬레이션)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # 차량 및 카메라 저장용
        self.vehicles = []
        self.cameras = []
        self.camera_infos = []
        self.additional_vehicles = []  # 추가 배경 차량들 저장
        
        self.setup_vehicles_and_cameras()
        
    def setup_static_transforms(self):
        """각 차량별 static transform 설정"""
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            tf_opt = TransformStamped()
            tf_opt.header.stamp = rospy.Time.now()
            tf_opt.header.frame_id = f"carla_camera_{vehicle_id}"
            tf_opt.child_frame_id = f"carla_camera_optical_{vehicle_id}"
            tf_opt.transform.translation.x = 0.0
            tf_opt.transform.translation.y = 0.0
            tf_opt.transform.translation.z = 0.0
            # rotation matrix [[0,0,1],[-1,0,0],[0,-1,0]] -> quaternion (w,x,y,z)
            tf_opt.transform.rotation.w = 0.5
            tf_opt.transform.rotation.x = -0.5
            tf_opt.transform.rotation.y = 0.5
            tf_opt.transform.rotation.z = -0.5
            self.static_broad.sendTransform(tf_opt)

    def setup_vehicles_and_cameras(self):
        """차량과 카메라 설정"""
        # 스폰 포인트 가져오기 및 필터링
        all_spawn_points = self.world.get_map().get_spawn_points()
        
        # 안전한 스폰 포인트 선택 (서로 최소 거리 유지)
        safe_spawn_points = self._select_safe_spawn_points(all_spawn_points, min_distance=20.0)
        
        if len(safe_spawn_points) < self.num_vehicles:
            rospy.logerr(f"안전한 스폰 포인트가 부족합니다. 필요: {self.num_vehicles}, 사용 가능: {len(safe_spawn_points)}")
            return

        # 차량 블루프린트
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        # 차량 색상을 다르게 설정
        colors = ['255,0,0', '0,255,0', '0,0,255']  # 빨강, 초록, 파랑
        
        # 카메라 블루프린트
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '384')
        camera_bp.set_attribute('fov', '90')
        
        cam_width = int(camera_bp.get_attribute('image_size_x').as_int())
        cam_height = int(camera_bp.get_attribute('image_size_y').as_int())
        cam_fov_deg = float(camera_bp.get_attribute('fov').as_float())

        # 각 차량과 카메라 생성
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            
            # 차량 블루프린트 설정
            if i < len(colors):
                vehicle_bp.set_attribute('color', colors[i])
            
            # 차량 생성 (안전한 스폰 포인트 사용)
            spawn_point = safe_spawn_points[i]
            vehicle = None
            
            # 여러 번 시도하여 차량 스폰
            for attempt in range(3):
                try:
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    break
                except RuntimeError as e:
                    rospy.logwarn(f"Vehicle {vehicle_id} spawn attempt {attempt + 1} failed: {e}")
                    if attempt < 2:  # 마지막 시도가 아니면 다른 포인트 시도
                        if i + attempt + 1 < len(safe_spawn_points):
                            spawn_point = safe_spawn_points[i + attempt + 1]
                        time.sleep(0.1)
            
            if vehicle is None:
                rospy.logerr(f"Failed to spawn vehicle {vehicle_id}")
                continue
                
            # 차량의 물리 속성 설정
            physics_control = vehicle.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            vehicle.apply_physics_control(physics_control)
            
            # Traffic Manager에 차량 등록 및 자율주행 설정
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
            # 개별 차량 설정
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)  # 앞차와 4m 거리
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -10)  # 10% 느리게
            self.traffic_manager.auto_lane_change(vehicle, True)  # 차선 변경 허용
            self.traffic_manager.ignore_lights_percentage(vehicle, 0)  # 신호등 준수
            self.traffic_manager.ignore_signs_percentage(vehicle, 0)  # 표지판 준수
            
            self.vehicles.append(vehicle)
            rospy.loginfo(f"Vehicle {vehicle_id} spawned with id: {vehicle.id} at {spawn_point.location}")

            # 카메라 정보 생성
            cam_info = self.build_camera_info(cam_width, cam_height, cam_fov_deg, vehicle_id)
            self.camera_infos.append(cam_info)

            # BEV 카메라 변환 (차량 위 30m에서 아래를 내려다봄)
            camera_transform = carla.Transform(
                carla.Location(x=0, y=0, z=30),
                carla.Rotation(pitch=-90)
            )

            # 카메라 생성
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            self.cameras.append(camera)
            rospy.loginfo(f"Camera {vehicle_id} spawned with id: {camera.id}")

            # 카메라 콜백 등록 (올바른 vehicle_id 캡처를 위한 클로저)
            def make_callback(v_id):
                return lambda image: self.camera_callback(image, v_id)
            camera.listen(make_callback(vehicle_id))

        # 추가 자율주행 차량들 (배경용) - 남은 안전한 스폰 포인트 사용
        remaining_points = safe_spawn_points[self.num_vehicles:]
        self.additional_vehicles = self.spawn_additional_vehicles(remaining_points)

    def _select_safe_spawn_points(self, all_spawn_points, min_distance=20.0):
        """최소 거리를 유지하는 안전한 스폰 포인트들을 선택"""
        if not all_spawn_points:
            return []
            
        safe_points = [all_spawn_points[0]]  # 첫 번째 포인트는 항상 선택
        
        for point in all_spawn_points[1:]:
            is_safe = True
            for safe_point in safe_points:
                distance = point.location.distance(safe_point.location)
                if distance < min_distance:
                    is_safe = False
                    break
            
            if is_safe:
                safe_points.append(point)
                
            # 충분한 포인트를 찾으면 중단 (최대 20개)
            if len(safe_points) >= 20:
                break
        
        random.shuffle(safe_points)  # 무작위로 섞어서 다양성 확보
        rospy.loginfo(f"Selected {len(safe_points)} safe spawn points from {len(all_spawn_points)} total points")
        return safe_points

    def build_camera_info(self, width, height, fov_deg, vehicle_id):
        """카메라 정보 생성"""
        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        info.K = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.binning_x = 0
        info.binning_y = 0
        info.roi.x_offset = 0
        info.roi.y_offset = 0
        info.roi.height = 0
        info.roi.width = 0
        info.roi.do_rectify = False
        info.header.frame_id = f"carla_camera_optical_{vehicle_id}"
        return info

    def spawn_additional_vehicles(self, remaining_spawn_points):
        """추가 배경 차량들 스폰"""
        additional_vehicles = []
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*.*')
        
        # 트럭이나 너무 큰 차량 제외
        filtered_bps = []
        for bp in vehicle_blueprints:
            if 'truck' not in bp.id.lower() and 'bus' not in bp.id.lower():
                filtered_bps.append(bp)
        
        spawn_count = min(len(remaining_spawn_points), 8)  # 최대 8대로 제한
        
        for i in range(spawn_count):
            sp = remaining_spawn_points[i]
            vehicle_bp_rand = random.choice(filtered_bps)
            
            # 랜덤 색상 설정
            if vehicle_bp_rand.has_attribute('color'):
                color = random.choice(vehicle_bp_rand.get_attribute('color').recommended_values)
                vehicle_bp_rand.set_attribute('color', color)
            
            try:
                v = self.world.spawn_actor(vehicle_bp_rand, sp)
                if v is not None:
                    # Traffic Manager에 등록하여 자율주행 설정
                    v.set_autopilot(True, self.traffic_manager.get_port())
                    
                    # 배경 차량들은 좀 더 빠르게 설정
                    self.traffic_manager.vehicle_percentage_speed_difference(v, random.randint(-20, 10))
                    self.traffic_manager.distance_to_leading_vehicle(v, 2.5) 
                    self.traffic_manager.auto_lane_change(v, True)
                    self.traffic_manager.ignore_lights_percentage(v, 30)  # 가끔 신호등 무시
                    
                    additional_vehicles.append(v)
                    rospy.logdebug(f"Background vehicle spawned: {v.id}")
                    
            except RuntimeError as e:
                rospy.logdebug(f"Failed to spawn background vehicle at {sp.location}: {e}")
                continue
                
        rospy.loginfo(f"총 {len(additional_vehicles)}대의 추가 배경 차량을 스폰했습니다.")
        return additional_vehicles

    def euler_to_quaternion(self, roll, pitch, yaw):
        """오일러 각을 쿼터니언으로 변환"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def publish_odometry(self, vehicle, vehicle_id, stamp):
        """차량의 odometry 발행 (안전한 액터 접근)"""
        # 차량의 transform과 velocity를 안전하게 가져오기
        transform = self._safe_actor_operation(
            vehicle, f"vehicle_transform_v{vehicle_id}",
            lambda: vehicle.get_transform()
        )
        
        vel = self._safe_actor_operation(
            vehicle, f"vehicle_velocity_v{vehicle_id}",
            lambda: vehicle.get_velocity()
        )
        
        if transform is None or vel is None:
            rospy.logdebug(f"Failed to get vehicle data for vehicle {vehicle_id}")
            return

        # Carla rotation은 도 단위; 라디안으로 변환
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = f"ego_vehicle_{vehicle_id}"

        odom.pose.pose = Pose(
            position=Point(transform.location.x, transform.location.y, transform.location.z),
            orientation=self.euler_to_quaternion(roll, pitch, yaw),
        )

        odom.twist.twist = Twist(
            linear=Vector3(vel.x, vel.y, vel.z),
            angular=Vector3(0.0, 0.0, 0.0),
        )

        try:
            self.odom_pubs[vehicle_id - 1].publish(odom)
        except Exception as e:
            rospy.logwarn(f"Failed to publish odometry for vehicle {vehicle_id}: {e}")
            return

        # Transform 발행: map -> ego_vehicle_N
        tf_ev = TransformStamped()
        tf_ev.header.stamp = stamp
        tf_ev.header.frame_id = "map"
        tf_ev.child_frame_id = f"ego_vehicle_{vehicle_id}"
        tf_ev.transform.translation.x = transform.location.x
        tf_ev.transform.translation.y = transform.location.y
        tf_ev.transform.translation.z = transform.location.z
        q_ev = self.euler_to_quaternion(roll, pitch, yaw)
        tf_ev.transform.rotation = q_ev
        
        try:
            self.tf_broadcaster.sendTransform(tf_ev)
        except Exception as e:
            rospy.logwarn(f"Failed to broadcast transform for vehicle {vehicle_id}: {e}")

    def _safe_actor_operation(self, actor, operation_name, operation_func):
        """액터 조작을 안전하게 수행하는 헬퍼 함수"""
        try:
            if actor is None:
                return None
            if hasattr(actor, 'is_alive') and not actor.is_alive:
                return None
            return operation_func()
        except RuntimeError as e:
            if "destroyed actor" in str(e).lower():
                rospy.logwarn_once(f"Actor destroyed during {operation_name}")
                return None
            else:
                raise e
        except Exception as e:
            rospy.logerr(f"Error in {operation_name}: {e}")
            return None

    def camera_callback(self, image, vehicle_id):
        """카메라 콜백 - 각 차량별로 호출됨"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

            stamp = rospy.Time.now()
            
            # 이미지 발행
            ros_image = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
            ros_image.header.stamp = stamp
            ros_image.header.frame_id = f"carla_camera_optical_{vehicle_id}"
            self.image_pubs[vehicle_id - 1].publish(ros_image)

            # 카메라 transform 발행 (안전한 액터 접근)
            camera = self.cameras[vehicle_id - 1] if vehicle_id <= len(self.cameras) else None
            cam_tf = self._safe_actor_operation(
                camera, f"camera_transform_v{vehicle_id}", 
                lambda: camera.get_transform()
            )
            
            if cam_tf is not None:
                cam_roll = math.radians(cam_tf.rotation.roll)
                cam_pitch = math.radians(cam_tf.rotation.pitch)
                cam_yaw = math.radians(cam_tf.rotation.yaw)

                tf_cam = TransformStamped()
                tf_cam.header.stamp = stamp
                tf_cam.header.frame_id = "map"
                tf_cam.child_frame_id = f"carla_camera_{vehicle_id}"
                tf_cam.transform.translation.x = cam_tf.location.x
                tf_cam.transform.translation.y = cam_tf.location.y
                tf_cam.transform.translation.z = cam_tf.location.z
                q_cam = self.euler_to_quaternion(cam_roll, cam_pitch, cam_yaw)
                tf_cam.transform.rotation = q_cam
                self.tf_broadcaster.sendTransform(tf_cam)

            # 카메라 정보 발행
            if vehicle_id <= len(self.camera_infos):
                cam_msg = self.camera_infos[vehicle_id - 1]
                cam_msg.header.stamp = stamp
                self.cam_info_pubs[vehicle_id - 1].publish(cam_msg)

            # Odometry 발행 (안전한 액터 접근)
            vehicle = self.vehicles[vehicle_id - 1] if vehicle_id <= len(self.vehicles) else None
            if vehicle is not None:
                self._safe_actor_operation(
                    vehicle, f"odometry_v{vehicle_id}",
                    lambda: self.publish_odometry(vehicle, vehicle_id, stamp)
                )

        except Exception as e:
            rospy.logerr(f"Camera callback error for vehicle {vehicle_id}: {e}")

    def run(self):
        """메인 실행 루프"""
        try:
            spectator = self.world.get_spectator()
        except Exception as e:
            rospy.logerr(f"Failed to get spectator: {e}")
            spectator = None
            
        try:
            rate = rospy.Rate(20)  # 20 Hz (동기 모드와 맞춤)
            while not rospy.is_shutdown():
                try:
                    # 동기 모드에서 world tick 실행
                    try:
                        self.world.tick()
                    except Exception as e:
                        rospy.logwarn_once(f"Failed to tick world: {e}")
                    
                    # 첫 번째 차량을 따라가도록 spectator 설정 (안전한 액터 접근)
                    if self.vehicles and spectator is not None:
                        vehicle_transform = self._safe_actor_operation(
                            self.vehicles[0], "vehicle_transform_for_spectator",
                            lambda: self.vehicles[0].get_transform()
                        )
                        
                        if vehicle_transform is not None:
                            vehicle_location = vehicle_transform.location
                            vehicle_rotation = vehicle_transform.rotation

                            # 차량 위 50m에서 pitch -60도로 조망
                            spectator_location = carla.Location(
                                x=vehicle_location.x - 20,
                                y=vehicle_location.y,
                                z=vehicle_location.z + 50
                            )
                            spectator_rotation = carla.Rotation(
                                pitch=-60,
                                yaw=vehicle_rotation.yaw,
                                roll=0
                            )

                            try:
                                spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
                            except Exception as e:
                                rospy.logwarn_once(f"Failed to set spectator transform: {e}")
                    
                    # 차량들의 상태 확인 (안전한 방식)
                    alive_count = 0
                    for i, vehicle in enumerate(self.vehicles):
                        is_alive = self._safe_actor_operation(
                            vehicle, f"vehicle_alive_check_{i+1}",
                            lambda: vehicle.is_alive
                        )
                        if is_alive:
                            alive_count += 1
                        elif is_alive is False:  # None이 아닌 False인 경우만
                            rospy.logwarn_throttle(5.0, f"Vehicle {i+1} is no longer alive")
                    
                    # 모든 차량이 죽었으면 종료
                    if alive_count == 0 and len(self.vehicles) > 0:
                        rospy.logwarn("All vehicles are destroyed. Shutting down...")
                        break

                except Exception as e:
                    rospy.logwarn(f"Error in main loop: {e}")
                    # 계속 실행하되 에러 발생 시 잠시 대기
                    rospy.sleep(1.0)

                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
                    
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        rospy.loginfo("Cleaning up actors...")
        
        # 카메라 정리 (안전한 방식)
        for i, camera in enumerate(self.cameras):
            try:
                if camera is not None:
                    # 카메라 리스닝 중지
                    self._safe_actor_operation(
                        camera, f"camera_stop_{i+1}",
                        lambda: camera.stop()
                    )
                    
                    # 카메라 파괴
                    self._safe_actor_operation(
                        camera, f"camera_destroy_{i+1}",
                        lambda: camera.destroy()
                    )
                    
            except Exception as e:
                rospy.logwarn(f"Error cleaning up camera {i+1}: {e}")
        
        # 메인 차량 정리 (안전한 방식)
        for i, vehicle in enumerate(self.vehicles):
            try:
                if vehicle is not None:
                    self._safe_actor_operation(
                        vehicle, f"vehicle_destroy_{i+1}",
                        lambda: vehicle.destroy()
                    )
                    
            except Exception as e:
                rospy.logwarn(f"Error cleaning up vehicle {i+1}: {e}")
        
        # 추가 배경 차량들 정리
        for i, vehicle in enumerate(self.additional_vehicles):
            try:
                if vehicle is not None:
                    self._safe_actor_operation(
                        vehicle, f"bg_vehicle_destroy_{i+1}",
                        lambda: vehicle.destroy()
                    )
                    
            except Exception as e:
                rospy.logwarn(f"Error cleaning up background vehicle {i+1}: {e}")
                
        # 동기 모드 해제
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            rospy.loginfo("Synchronous mode disabled")
        except Exception as e:
            rospy.logwarn(f"Failed to disable synchronous mode: {e}")
        
        # Traffic Manager 정리
        try:
            if hasattr(self, 'traffic_manager'):
                self.traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            rospy.logwarn(f"Failed to cleanup traffic manager: {e}")
                
        # 리스트 초기화
        self.cameras.clear()
        self.vehicles.clear()
        self.additional_vehicles.clear()
                
        rospy.loginfo("Actor cleanup completed.")


def main():
    try:
        camera_system = MultipleCarlaCamera()
        camera_system.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 