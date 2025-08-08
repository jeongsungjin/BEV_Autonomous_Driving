#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose  # type: ignore


class TrajectoryFrameRepublisher:
    def __init__(self):
        rospy.init_node("trajectory_frame_republisher", anonymous=True)
        self.source_topic = rospy.get_param("~source_topic", "/bev_planner/planned_trajectory")
        self.target_topic = rospy.get_param("~target_topic", "/bev_planner/planned_trajectory_map")
        self.target_frame = rospy.get_param("~target_frame", "map")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.target_topic, RosPath, queue_size=1)
        self.sub = rospy.Subscriber(self.source_topic, RosPath, self._path_cb, queue_size=1)

    def _path_cb(self, path_msg: RosPath):
        try:
            if not path_msg.poses:
                return
            src_frame = path_msg.header.frame_id
            if not src_frame:
                return
            # lookup transform target_frame <- src_frame at latest time
            tf = self.tf_buffer.lookup_transform(self.target_frame, src_frame, rospy.Time(0), rospy.Duration(1.0))

            out = RosPath()
            out.header = path_msg.header
            out.header.frame_id = self.target_frame
            out.poses = []
            for ps in path_msg.poses:
                transformed_pose = do_transform_pose(ps, tf)
                out.poses.append(transformed_pose)
            self.pub.publish(out)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[traj_repub] transform failed: {e}")


def main():
    try:
        TrajectoryFrameRepublisher()
        rospy.loginfo("[traj_repub] started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main() 