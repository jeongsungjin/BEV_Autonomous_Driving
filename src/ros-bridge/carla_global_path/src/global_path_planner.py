#!/usr/bin/env python

import os
import sys
import rospy
import time
import math
import random
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Header
from carla_msgs.msg import CarlaEgoVehicleInfo, CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import NavSatFix
import tf
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose

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

class GlobalPathPlanner:
    def __init__(self):
        rospy.init_node('global_path_planner', anonymous=True)
        
        # Publishers
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        
        # Path storage
        self.path_points = []
        self.last_save_time = 0
        self.save_interval = 0.5  # Save every 0.5 seconds
        
        # Autonomous driving parameters
        self.target_speed = 20.0  # km/h
        self.max_speed = 30.0     # km/h
        self.min_speed = 5.0      # km/h
        
        # CARLA client setup (like publish_carla_cam.py)
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Log current map
        current_map = self.world.get_map().name
        rospy.loginfo(f"Connected to CARLA map: {current_map}")
        
        # Disable synchronous mode to avoid tick issues
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        self.vehicle = None
        
        print("Global Path Planner initialized!")
        print("Autonomous driving mode - vehicle will drive automatically")
        print("Press Ctrl+C to stop and save path")
        
    def spawn_vehicle(self):
        """Spawn ego vehicle in CARLA"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        rospy.loginfo(f"Spawned vehicle id: {self.vehicle.id}")
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        rospy.loginfo("Autopilot enabled")
        
    def update_vehicle_pose(self):
        """Update vehicle pose and add to path"""
        if self.vehicle and self.vehicle.is_alive:
            try:
                transform = self.vehicle.get_transform()
                
                # Create pose stamped message
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = transform.location.x
                pose.pose.position.y = transform.location.y
                pose.pose.position.z = transform.location.z
                
                # Convert CARLA rotation to quaternion
                roll = math.radians(transform.rotation.roll)
                pitch = math.radians(transform.rotation.pitch)
                yaw = math.radians(transform.rotation.yaw)
                
                # Euler to quaternion conversion
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cp = math.cos(pitch * 0.5)
                sp = math.sin(pitch * 0.5)
                cr = math.cos(roll * 0.5)
                sr = math.sin(roll * 0.5)
                
                pose.pose.orientation.w = cr * cp * cy + sr * sp * sy
                pose.pose.orientation.x = sr * cp * cy - cr * sp * sy
                pose.pose.orientation.y = cr * sp * cy + sr * cp * sy
                pose.pose.orientation.z = cr * cp * sy - sr * sp * cy
                
                # Add to path if enough time has passed
                current_time = rospy.Time.now().to_sec()
                if current_time - self.last_save_time > self.save_interval:
                    self.path_points.append(pose)
                    self.last_save_time = current_time
                    
                    # Publish path for visualization
                    self.publish_path()
                    
            except Exception as e:
                rospy.logwarn(f"Error getting vehicle transform: {e}")
    
    def publish_path(self):
        """Publish the current path for visualization"""
        if len(self.path_points) > 0:
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp = rospy.Time.now()
            path_msg.poses = self.path_points
            self.path_pub.publish(path_msg)
    
    def save_path_to_file(self):
        """Save the current path to global_path_1.txt"""
        if len(self.path_points) == 0:
            print("No path points to save!")
            return
        
        # Save to current working directory with absolute path
        import os
        filename = os.path.join(os.getcwd(), "global_path_1.txt")
        try:
            with open(filename, 'w') as f:
                f.write("# CARLA Global Path - Map Coordinates\n")
                f.write("# Format: x, y, z, qx, qy, qz, qw\n")
                f.write("# Timestamp: {}\n".format(rospy.Time.now().to_sec()))
                f.write("# Map: {}\n".format(self.world.get_map().name))
                f.write("\n")
                
                for i, pose in enumerate(self.path_points):
                    f.write("{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z,
                        pose.pose.orientation.w
                    ))
            
            print(f"\nPath saved to: {filename}")
            print(f"Total points: {len(self.path_points)}")
            
        except Exception as e:
            print(f"Error saving path: {e}")
            print(f"Current working directory: {os.getcwd()}")
    
    def reset_path(self):
        """Reset the current path"""
        self.path_points = []
        print("\nPath reset!")
    
    def autonomous_control_callback(self, event):
        """Autonomous control callback using ROS timer"""
        if self.vehicle and self.vehicle.is_alive:
            try:
                # Get current vehicle state
                velocity = self.vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s to km/h
                
                # Simple speed control
                if current_speed < self.target_speed:
                    # Accelerate
                    control = carla.VehicleControl()
                    control.throttle = 0.8
                    control.brake = 0.0
                    control.steer = 0.0
                    control.hand_brake = False
                    control.reverse = False
                    control.gear = 1
                    control.manual_gear_shift = False
                    self.vehicle.apply_control(control)
                elif current_speed > self.max_speed:
                    # Decelerate
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.brake = 0.3
                    control.steer = 0.0
                    control.hand_brake = False
                    control.reverse = False
                    control.gear = 1
                    control.manual_gear_shift = False
                    self.vehicle.apply_control(control)
                else:
                    # Maintain speed
                    control = carla.VehicleControl()
                    control.throttle = 0.5
                    control.brake = 0.0
                    control.steer = 0.0
                    control.hand_brake = False
                    control.reverse = False
                    control.gear = 1
                    control.manual_gear_shift = False
                    self.vehicle.apply_control(control)
                
                # Print status
                print(f"\rSpeed: {current_speed:.1f} km/h, Points: {len(self.path_points)}", end='', flush=True)
                
            except Exception as e:
                rospy.logwarn(f"Error in autonomous control: {e}")
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        print("Starting global path planner...")
        print("Spawning vehicle...")
        
        # Spawn vehicle
        self.spawn_vehicle()
        
        print("Vehicle spawned! Starting autonomous driving...")
        print("Press Ctrl+C to stop and save path")
        
        # Start autonomous control timer
        self.autonomous_timer = rospy.Timer(rospy.Duration(0.1), self.autonomous_control_callback)
        
        # Main loop (like publish_carla_cam.py)
        try:
            rate = rospy.Rate(20)  # 20 Hz
            while not rospy.is_shutdown():
                try:
                    # Check if vehicle is still alive
                    if not self.vehicle.is_alive:
                        rospy.logwarn("Vehicle actor no longer alive. Exiting loop.")
                        break
                    
                    # Update vehicle pose and path
                    self.update_vehicle_pose()
                    
                    rate.sleep()
                except RuntimeError as e:
                    rospy.logwarn(f"Actor error: {e}. Exiting loop.")
                    break
        finally:
            # Save path before cleanup
            if len(self.path_points) > 0:
                print(f"\nSaving path with {len(self.path_points)} points...")
                self.save_path_to_file()
            
            # Cleanup
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
            rospy.loginfo("Vehicle destroyed.")

if __name__ == '__main__':
    try:
        planner = GlobalPathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        print("Global path planner interrupted!")
    except KeyboardInterrupt:
        print("Global path planner stopped by user!")
    except Exception as e:
        print(f"Error: {e}") 