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


def main():
    rospy.init_node("carla_camera_publisher", anonymous=True)
    pub = rospy.Publisher("/carla/yolop/image_raw", Image, queue_size=10)
    cam_info_pub = rospy.Publisher("/carla/yolop/camera_info", CameraInfo, queue_size=10, latch=True)

    # --- static transform carla_camera -> carla_camera_optical (REP103) ---
    static_broad = StaticTransformBroadcaster()
    tf_opt = TransformStamped()
    tf_opt.header.stamp = rospy.Time.now()
    tf_opt.header.frame_id = "carla_camera"
    tf_opt.child_frame_id = "carla_camera_optical"
    tf_opt.transform.translation.x = 0.0
    tf_opt.transform.translation.y = 0.0
    tf_opt.transform.translation.z = 0.0
    # rotation matrix [[0,0,1],[-1,0,0],[0,-1,0]] -> quaternion (w,x,y,z)
    tf_opt.transform.rotation.w = 0.5
    tf_opt.transform.rotation.x = -0.5
    tf_opt.transform.rotation.y = 0.5
    tf_opt.transform.rotation.z = -0.5
    static_broad.sendTransform(tf_opt)

    odom_pub = rospy.Publisher("/carla/ego_vehicle/odometry", Odometry, queue_size=10)
    bridge = CvBridge()
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # ==== 차량 생성 ====
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    rospy.loginfo(f"Spawned vehicle id: {vehicle.id}")

    # ==== 추가 자율주행 차량 10대 스폰 ====
    additional_vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        if len(additional_vehicles) >= 10:
            break  # 10대만 생성

        # ego 차량과 같은 스폰 포인트는 건너뜀
        if sp.location.distance(spawn_point.location) < 1.0:
            continue

        # 다양한 차종 중 무작위 선택
        vehicle_bp_rand = random.choice(blueprint_library.filter('vehicle.*.*'))
        v = world.try_spawn_actor(vehicle_bp_rand, sp)
        if v is not None:
            v.set_autopilot(True)
            additional_vehicles.append(v)
            rospy.loginfo(f"Spawned additional vehicle id: {v.id}")

    rospy.loginfo(f"총 {len(additional_vehicles)}대의 추가 차량을 스폰했습니다.")

    # ==== 카메라 생성 ====
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '384')
    camera_bp.set_attribute('fov', '90')
    cam_width = int(camera_bp.get_attribute('image_size_x').as_int())
    cam_height = int(camera_bp.get_attribute('image_size_y').as_int())
    cam_fov_deg = float(camera_bp.get_attribute('fov').as_float())

    # --- build static CameraInfo ---
    def build_camera_info():
        info = CameraInfo()
        info.width = cam_width
        info.height = cam_height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = cam_width / (2.0 * math.tan(math.radians(cam_fov_deg) / 2.0))
        fy = fx
        cx = cam_width / 2.0
        cy = cam_height / 2.0
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
        info.header.frame_id = "carla_camera_optical"
        return info

    static_cam_info = build_camera_info()

    camera_transform = carla.Transform(
        carla.Location(x=0, y=0, z=30),
        carla.Rotation(pitch=-90)
    )

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    rospy.loginfo(f"Spawned camera id: {camera.id}")

    # ==== 이미지 콜백 ====
    def euler_to_quaternion(roll, pitch, yaw):
        """Convert Euler angles (rad) to geometry_msgs/Quaternion"""
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

    def publish_odometry(stamp):
        transform = vehicle.get_transform()
        vel = vehicle.get_velocity()

        # Carla rotation is degrees; convert to radians
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "ego_vehicle"

        odom.pose.pose = Pose(
            position=Point(transform.location.x, transform.location.y, transform.location.z),
            orientation=euler_to_quaternion(roll, pitch, yaw),
        )

        odom.twist.twist = Twist(
            linear=Vector3(vel.x, vel.y, vel.z),
            angular=Vector3(0.0, 0.0, 0.0),
        )

        odom_pub.publish(odom)

        # Broadcast transform map -> ego_vehicle
        tf_ev = TransformStamped()
        tf_ev.header.stamp = stamp
        tf_ev.header.frame_id = "map"
        tf_ev.child_frame_id = "ego_vehicle"
        tf_ev.transform.translation.x = transform.location.x
        tf_ev.transform.translation.y = transform.location.y
        tf_ev.transform.translation.z = transform.location.z
        q_ev = euler_to_quaternion(roll, pitch, yaw)
        tf_ev.transform.rotation = q_ev
        tf_broadcaster.sendTransform(tf_ev)

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

        stamp = rospy.Time.now()
        # Get camera world transform
        cam_tf = camera.get_transform()
        cam_roll = math.radians(cam_tf.rotation.roll)
        cam_pitch = math.radians(cam_tf.rotation.pitch)
        cam_yaw = math.radians(cam_tf.rotation.yaw)

        ros_image = bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        ros_image.header.stamp = stamp
        ros_image.header.frame_id = "carla_camera_optical"
        pub.publish(ros_image)

        # Broadcast transform map -> carla_camera
        tf_cam = TransformStamped()
        tf_cam.header.stamp = stamp
        tf_cam.header.frame_id = "map"
        tf_cam.child_frame_id = "carla_camera"
        tf_cam.transform.translation.x = cam_tf.location.x
        tf_cam.transform.translation.y = cam_tf.location.y
        tf_cam.transform.translation.z = cam_tf.location.z
        q_cam = euler_to_quaternion(cam_roll, cam_pitch, cam_yaw)
        tf_cam.transform.rotation = q_cam
        tf_broadcaster.sendTransform(tf_cam)

        # Publish CameraInfo with matching timestamp
        cam_msg = static_cam_info
        cam_msg.header.stamp = stamp
        cam_info_pub.publish(cam_msg)

        # publish odometry with the same timestamp
        publish_odometry(stamp)

    camera.listen(camera_callback)

    # ==== spectator 설정 ====
    spectator = world.get_spectator()
    try:
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            try:
                # actor가 이미 파괴된 경우 예외 발생
                if not vehicle.is_alive:
                    rospy.logwarn("[publish_carla_cam] vehicle actor no longer alive. Exiting loop.")
                    break

                vehicle_transform = vehicle.get_transform()
            except RuntimeError as e:
                rospy.logwarn(f"[publish_carla_cam] actor error: {e}. Exiting loop.")
                break

            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation

            # 차량 위 30m에서 pitch -90도로 아래를 보는 시점 설정
            spectator_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + 30
            )
            spectator_rotation = carla.Rotation(
                pitch=-90,
                yaw=vehicle_rotation.yaw,
                roll=0
            )

            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

            rate.sleep()
    finally:
        camera.stop()
        if vehicle.is_alive:
            vehicle.destroy()
        if camera.is_alive:
            camera.destroy()
        # 추가로 생성된 차량 정리
        for v in additional_vehicles:
            if v.is_alive:
                v.destroy()
        rospy.loginfo("Actors destroyed.")

if __name__ == "__main__":
    main()
