#!/usr/bin/env python3

"""
Carla camera publisher
Publishes RGB images from a top-down camera mounted on a Tesla Model3 in CARLA
as a ROS1 sensor_msgs/Image topic.
"""

import random
import math
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import carla  # CARLA Python API
import numpy as np
import cv2


class CarlaCameraPublisher:
    """Attach an RGB camera to a spawned vehicle and publish images over ROS."""

    def __init__(self):
        rospy.init_node("carla_camera_publisher", anonymous=True)

        # ROS parameters (can be overridden via rosparam)
        self.host = rospy.get_param("~host", "localhost")
        self.port = rospy.get_param("~port", 2000)
        self.image_topic = rospy.get_param("~image_topic", "/carla/yolop/image_raw")

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(self.image_topic, Image, queue_size=10)

        self._init_carla()
        rospy.loginfo("CarlaCameraPublisher initialised – publishing on %s", self.image_topic)

    def _init_carla(self):
        """Connect to CARLA, spawn vehicle and camera sensor."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        rospy.loginfo("Spawned vehicle id=%s", self.vehicle.id)

        # Create camera sensor – top-down 30 m
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "384")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=30), carla.Rotation(pitch=-90)
        )

        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self._camera_callback)
        rospy.loginfo("Spawned camera id=%s", self.camera.id)

    def _camera_callback(self, image: "carla.Image"):
        """Convert CARLA image to ROS Image and publish."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        frame_bgr = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

        ros_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        ros_msg.header.stamp = rospy.Time.now()
        ros_msg.header.frame_id = "carla_camera"
        self.publisher.publish(ros_msg)

    def destroy(self):
        """Cleanly destroy actors."""
        rospy.loginfo("Shutting down – destroying CARLA actors…")
        try:
            if self.camera.is_listening:
                self.camera.stop()
        except Exception:
            pass
        for actor in [getattr(self, "camera", None), getattr(self, "vehicle", None)]:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass


if __name__ == "__main__":
    node = CarlaCameraPublisher()
    try:
        rospy.spin()
    finally:
        node.destroy() 