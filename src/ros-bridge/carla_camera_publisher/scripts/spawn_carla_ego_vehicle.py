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
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from ackermann_msgs.msg import AckermannDrive

def append_carla_egg():
    carla_python_path = os.getenv("CARLA_PYTHON_PATH")
    if carla_python_path is None:
        raise EnvironmentError("CARLA_PYTHON_PATH 환경변수가 설정되지 않았습니다.")
    for fname in os.listdir(carla_python_path):
        if fname.startswith("carla-") and fname.endswith(".egg") and "py3.7" in fname:
            full_path = os.path.join(carla_python_path, fname)
            if full_path not in sys.path:
                sys.path.append(full_path)
            break
    else:
        raise FileNotFoundError("CARLA egg 파일을 찾을 수 없습니다. py3.7에 맞는 egg가 있어야 합니다.")

append_carla_egg()
import carla

class CarlaEgoVehicleNode:
    def __init__(self):
        rospy.init_node("carla_ego_vehicle_node", anonymous=True)
        self.pub = rospy.Publisher("/carla/yolop/image_raw", Image, queue_size=10)
        self.cam_info_pub = rospy.Publisher("/carla/yolop/camera_info", CameraInfo, queue_size=10, latch=True)
        self.odom_pub = rospy.Publisher("/carla/ego_vehicle/odometry", Odometry, queue_size=10)
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broad = StaticTransformBroadcaster()
        self.vehicle = None
        self.camera = None
        self.world = None
        self.control_cmd = None
        self.last_cmd_time = rospy.Time.now()
        self.cmd_timeout = rospy.Duration(1.0)  # 1초 동안 명령 없으면 정지
        self.setup()

    def setup(self):
        # static transform carla_camera -> carla_camera_optical
        tf_opt = TransformStamped()
        tf_opt.header.stamp = rospy.Time.now()
        tf_opt.header.frame_id = "carla_camera"
        tf_opt.child_frame_id = "carla_camera_optical"
        tf_opt.transform.translation.x = 0.0
        tf_opt.transform.translation.y = 0.0
        tf_opt.transform.translation.z = 0.0
        tf_opt.transform.rotation.w = 0.5
        tf_opt.transform.rotation.x = -0.5
        tf_opt.transform.rotation.y = 0.5
        tf_opt.transform.rotation.z = -0.5
        self.static_broad.sendTransform(tf_opt)

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        # 차량 생성 (autopilot X)
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        rospy.loginfo(f"Spawned ego vehicle id: {self.vehicle.id}")

        # 카메라 생성
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '384')
        camera_bp.set_attribute('fov', '90')
        self.cam_width = int(camera_bp.get_attribute('image_size_x').as_int())
        self.cam_height = int(camera_bp.get_attribute('image_size_y').as_int())
        self.cam_fov_deg = float(camera_bp.get_attribute('fov').as_float())
        self.static_cam_info = self.build_camera_info()
        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=30),
            carla.Rotation(pitch=-90)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        rospy.loginfo(f"Spawned camera id: {self.camera.id}")
        self.camera.listen(self.camera_callback)

        # 제어 명령 구독
        rospy.Subscriber("/carla/ego_vehicle/vehicle_control_cmd", AckermannDrive, self.control_callback)

    def build_camera_info(self):
        info = CameraInfo()
        info.width = self.cam_width
        info.height = self.cam_height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = self.cam_width / (2.0 * math.tan(math.radians(self.cam_fov_deg) / 2.0))
        fy = fx
        cx = self.cam_width / 2.0
        cy = self.cam_height / 2.0
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

    def euler_to_quaternion(self, roll, pitch, yaw):
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

    def publish_odometry(self, stamp):
        transform = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "ego_vehicle"
        odom.pose.pose = Pose(
            position=Point(transform.location.x, transform.location.y, transform.location.z),
            orientation=self.euler_to_quaternion(roll, pitch, yaw),
        )
        odom.twist.twist = Twist(
            linear=Vector3(vel.x, vel.y, vel.z),
            angular=Vector3(0.0, 0.0, 0.0),
        )
        self.odom_pub.publish(odom)
        tf_ev = TransformStamped()
        tf_ev.header.stamp = stamp
        tf_ev.header.frame_id = "map"
        tf_ev.child_frame_id = "ego_vehicle"
        tf_ev.transform.translation.x = transform.location.x
        tf_ev.transform.translation.y = transform.location.y
        tf_ev.transform.translation.z = transform.location.z
        q_ev = self.euler_to_quaternion(roll, pitch, yaw)
        tf_ev.transform.rotation = q_ev
        self.tf_broadcaster.sendTransform(tf_ev)

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        stamp = rospy.Time.now()
        cam_tf = self.camera.get_transform()
        cam_roll = math.radians(cam_tf.rotation.roll)
        cam_pitch = math.radians(cam_tf.rotation.pitch)
        cam_yaw = math.radians(cam_tf.rotation.yaw)
        ros_image = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        ros_image.header.stamp = stamp
        ros_image.header.frame_id = "carla_camera_optical"
        self.pub.publish(ros_image)
        tf_cam = TransformStamped()
        tf_cam.header.stamp = stamp
        tf_cam.header.frame_id = "map"
        tf_cam.child_frame_id = "carla_camera"
        tf_cam.transform.translation.x = cam_tf.location.x
        tf_cam.transform.translation.y = cam_tf.location.y
        tf_cam.transform.translation.z = cam_tf.location.z
        q_cam = self.euler_to_quaternion(cam_roll, cam_pitch, cam_yaw)
        tf_cam.transform.rotation = q_cam
        self.tf_broadcaster.sendTransform(tf_cam)
        cam_msg = self.static_cam_info
        cam_msg.header.stamp = stamp
        self.cam_info_pub.publish(cam_msg)
        self.publish_odometry(stamp)

    def control_callback(self, msg):
        self.control_cmd = msg
        self.last_cmd_time = rospy.Time.now()

    def apply_control(self):
        if self.control_cmd is not None and (rospy.Time.now() - self.last_cmd_time) < self.cmd_timeout:
            ctrl = carla.VehicleControl()
            # 현재 속도 계산 (m/s)
            vel = self.vehicle.get_velocity()
            current_speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            target_speed = self.control_cmd.speed
            speed_error = target_speed - current_speed

            # 간단한 비례제어(P)
            k_p = 0.5
            throttle = np.clip(k_p * speed_error, 0.0, 1.0)
            brake = 0.0
            if speed_error < -0.1:  # 목표 속도보다 0.1m/s 이상 빠르면 브레이크
                throttle = 0.0
                brake = np.clip(-k_p * speed_error, 0.0, 1.0)

            ctrl.throttle = throttle
            ctrl.steer = self.control_cmd.steering_angle / 0.6
            ctrl.brake = brake
            ctrl.hand_brake = False
            ctrl.reverse = False
            self.vehicle.apply_control(ctrl)
        else:
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.0
            ctrl.steer = 0.0
            ctrl.brake = 1.0
            self.vehicle.apply_control(ctrl)

    def run(self):
        spectator = self.world.get_spectator()
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                if not self.vehicle.is_alive:
                    rospy.logwarn("[carla_ego_vehicle_node] vehicle actor no longer alive. Exiting loop.")
                    break
                vehicle_transform = self.vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                vehicle_rotation = vehicle_transform.rotation
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
                self.apply_control()
                rate.sleep()
        finally:
            self.camera.stop()
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            if self.camera.is_alive:
                self.camera.destroy()
            rospy.loginfo("Actors destroyed.")

if __name__ == "__main__":
    node = CarlaEgoVehicleNode()
    node.run() 