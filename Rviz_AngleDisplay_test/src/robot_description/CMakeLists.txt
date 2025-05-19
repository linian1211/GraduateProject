cmake_minimum_required(VERSION 3.5)
project(robot_description)

# 尋找依賴
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(rviz2 REQUIRED)


# 安裝 URDF、SDF、launch 和 config 檔案
install(
  DIRECTORY urdf sdf launch config
  DESTINATION share/${PROJECT_NAME}
)

# 安裝 package.xml
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# 安裝Python腳本
install(PROGRAMS
  scripts/joint_angle_plotter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

