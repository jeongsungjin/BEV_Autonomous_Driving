#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
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


class AttachCameraPublisher:
    def __init__(self):
        rospy.init_node("carla_camera_publisher_attach", anonymous=True)

        # params
        self.host = rospy.get_param("~host", "localhost")
        self.port = int(rospy.get_param("~port", 2000))
        self.image_topic = rospy.get_param("~image_topic", "/carla/yolop/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/carla/yolop/camera_info")
        self.role_name = rospy.get_param("~role_name", "ego_vehicle")
        self.topdown_height_m = float(rospy.get_param("~topdown_height", 30.0))
        self.topdown_fov_deg = float(rospy.get_param("~fov", 90.0))
        self.image_width = int(rospy.get_param("~image_width", 640))
        self.image_height = int(rospy.get_param("~image_height", 384))
        self.set_spectator = rospy.get_param("~set_spectator", True)
        # Spectator (관찰자) 추적 뷰 파라미터
        self.spectator_follow = rospy.get_param("~spectator_follow", True)
        self.spectator_pitch_deg = float(rospy.get_param("~spectator_pitch_deg", -60.0))
        self.spectator_height_m = float(rospy.get_param("~spectator_height_m", 20.0))
        self.spectator_distance_m = float(rospy.get_param("~spectator_distance_m", 10.0))
        self.spectator_yaw_offset_deg = float(rospy.get_param("~spectator_yaw_offset_deg", 0.0))

        self.pub_image = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.pub_caminfo = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=10, latch=True)
        self.pub_odom = rospy.Publisher("/carla/ego_vehicle/odometry", Odometry, queue_size=10)

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broad = StaticTransformBroadcaster()

        self.prev_yaw = None
        self.prev_timestamp = None

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

        # connect to CARLA and find ego vehicle
        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.vehicle = self._find_ego_vehicle(self.role_name)
        if self.vehicle is None:
            raise RuntimeError("role_name이 'ego_vehicle'인 차량을 찾지 못했습니다. 스폰 노드가 먼저 실행되어야 합니다.")

        # setup camera (attached to ego vehicle)
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', str(self.topdown_fov_deg))

        self.cam_width = int(camera_bp.get_attribute('image_size_x').as_int())
        self.cam_height = int(camera_bp.get_attribute('image_size_y').as_int())
        self.cam_fov_deg = float(camera_bp.get_attribute('fov').as_float())

        self.static_cam_info = self._build_camera_info()

        camera_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=self.topdown_height_m),
            carla.Rotation(pitch=-90.0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        rospy.loginfo(f"[attach_cam] Spawned camera id: {self.camera.id} attach_to ego_vehicle")

        self.camera.listen(self._camera_callback)

        # Cache spectator for follow camera
        self.spectator = None
        if self.set_spectator:
            try:
                self.spectator = self.world.get_spectator()
            except Exception as e:
                rospy.logwarn(f"[attach_cam] get spectator failed: {e}")

    def _find_ego_vehicle(self, role_name: str):
        actors = self.world.get_actors()
        for actor in actors:
            try:
                if actor.type_id.startswith('vehicle.'):
                    if actor.attributes.get('role_name', '') == role_name:
                        return actor
            except Exception:
                continue
        return None

    def _build_camera_info(self):
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

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        q = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def _publish_odometry(self, stamp):
        try:
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

            angular_vel = self.vehicle.get_angular_velocity()
            angular_z = -math.radians(angular_vel.z)
            if abs(angular_z) < 1e-6:
                current_time = time.time()
                if self.prev_yaw is not None and self.prev_timestamp is not None:
                    dt = current_time - self.prev_timestamp
                    if dt > 0:
                        angle_diff = yaw - self.prev_yaw
                        while angle_diff > math.pi:
                            angle_diff -= 2 * math.pi
                        while angle_diff < -math.pi:
                            angle_diff += 2 * math.pi
                        angular_z = angle_diff / dt
                self.prev_yaw = yaw
                self.prev_timestamp = current_time

            odom.twist.twist = Twist(
                linear=Vector3(vel.x, vel.y, vel.z),
                angular=Vector3(
                    math.radians(angular_vel.x),
                    -math.radians(angular_vel.y),
                    angular_z,
                ),
            )
            self.pub_odom.publish(odom)

            # map -> ego_vehicle TF
            tf_ev = TransformStamped()
            tf_ev.header.stamp = stamp
            tf_ev.header.frame_id = "map"
            tf_ev.child_frame_id = "ego_vehicle"
            tf_ev.transform.translation.x = transform.location.x
            tf_ev.transform.translation.y = transform.location.y
            tf_ev.transform.translation.z = transform.location.z
            tf_ev.transform.rotation = self.euler_to_quaternion(roll, pitch, yaw)
            self.tf_broadcaster.sendTransform(tf_ev)
        except Exception as e:
            rospy.logwarn(f"[attach_cam] publish_odometry failed: {e}")

    def _update_spectator_follow(self):
        if not (self.set_spectator and self.spectator_follow and self.spectator and self.vehicle):
            return
        try:
            vtf = self.vehicle.get_transform()
            yaw_deg = vtf.rotation.yaw + self.spectator_yaw_offset_deg
            yaw_rad = math.radians(yaw_deg)
            # 뒤쪽-아래 비스듬한 뷰: 차량 뒤쪽 distance, 위쪽 height
            dx = -self.spectator_distance_m * math.cos(yaw_rad)
            dy = -self.spectator_distance_m * math.sin(yaw_rad)
            loc = carla.Location(x=vtf.location.x + dx,
                                 y=vtf.location.y + dy,
                                 z=vtf.location.z + self.spectator_height_m)
            rot = carla.Rotation(pitch=self.spectator_pitch_deg,
                                 yaw=yaw_deg,
                                 roll=0.0)
            self.spectator.set_transform(carla.Transform(loc, rot))
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[attach_cam] spectator follow failed: {e}")

    def _camera_callback(self, image):
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
        self.pub_image.publish(ros_image)

        # map -> carla_camera TF
        tf_cam = TransformStamped()
        tf_cam.header.stamp = stamp
        tf_cam.header.frame_id = "map"
        tf_cam.child_frame_id = "carla_camera"
        tf_cam.transform.translation.x = cam_tf.location.x
        tf_cam.transform.translation.y = cam_tf.location.y
        tf_cam.transform.translation.z = cam_tf.location.z
        tf_cam.transform.rotation = self.euler_to_quaternion(cam_roll, cam_pitch, cam_yaw)
        self.tf_broadcaster.sendTransform(tf_cam)

        # publish CameraInfo
        cam_msg = self.static_cam_info
        cam_msg.header.stamp = stamp
        self.pub_caminfo.publish(cam_msg)

        # publish odom with same stamp
        self._publish_odometry(stamp)

        # update spectator follow view (tilted chase camera)
        self._update_spectator_follow()


def main():
    try:
        node = AttachCameraPublisher()
        rospy.loginfo("[attach_cam] started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[attach_cam] fatal error: {e}")


if __name__ == "__main__":
    main() 