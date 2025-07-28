#!/usr/bin/env python3
import csv, rospy
from nav_msgs.msg import Odometry

out_path = rospy.get_param("~output_csv", "/home/carla/poses.csv")
csv_file = open(out_path, "w", newline='')
writer = csv.writer(csv_file)
writer.writerow(["timestamp", "x", "y"])      # header

def cb(msg):
    t = msg.header.stamp.to_sec()
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    writer.writerow([f"{t:.6f}", f"{x:.3f}", f"{y:.3f}"])

rospy.init_node("pose_csv_recorder")
rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, cb, queue_size=10)
rospy.loginfo(f"Recording poses to {out_path}")
rospy.spin()