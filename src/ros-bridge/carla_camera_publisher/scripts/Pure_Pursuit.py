#!/usr/bin/env python
# -- coding: utf-8 --
import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive

class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller')
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        self.odom_sub = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odom_callback)
        self.ctrl_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', AckermannDrive, queue_size=1)
        self.path = []
        self.current_pose = None
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 2.0)
        self.max_steer = rospy.get_param('~max_steer', 0.6)
        self.target_speed = rospy.get_param('~target_speed', 1.0)  # m/s
        self.wheelbase = rospy.get_param('~wheelbase', 2.8)  # 차량에 맞게 조정

    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.control_loop()

    def find_target_point(self):
        if not self.path or self.current_pose is None:
            return None
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        for px, py in self.path:
            dist = math.hypot(px - x, py - y)
            if dist > self.lookahead_distance:
                return px, py
        return self.path[-1]  # 마지막 점

    def get_yaw(self, orientation):
        import tf
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def control_loop(self):
        target = self.find_target_point()
        if target is None:
            return
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = self.get_yaw(self.current_pose.orientation)
        tx, ty = target
        dx = tx - x
        dy = ty - y
        # 목표점까지의 각도(alpha)
        alpha = math.atan2(dy, dx) - yaw
        Ld = math.hypot(dx, dy)
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        ctrl = AckermannDrive()
        ctrl.steering_angle = steer
        ctrl.speed = self.target_speed
        self.ctrl_pub.publish(ctrl)

if __name__ == '__main__':
    PurePursuitController()
    rospy.spin()
