cmake_minimum_required(VERSION 2.8.3)
project(carla_global_path)

find_package(ros_environment REQUIRED)
set(ROS_VERSION $ENV{ROS_VERSION})

if(${ROS_VERSION} EQUAL 1)

  find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs geometry_msgs
                                          nav_msgs tf roslaunch)

  catkin_python_setup()

  catkin_package()

  if(CATKIN_ENABLE_TESTING)
    roslaunch_add_file_check(launch)
  endif()

  include_directories(${catkin_INCLUDE_DIRS})

  install(PROGRAMS src/carla_global_path/global_path_planner.py
          DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

  install(DIRECTORY launch/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

  install(DIRECTORY config/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

endif() 