cmake_minimum_required(VERSION 3.5)
project(robot_description)

# 查找依賴
find_package(ament_cmake REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(rviz2 REQUIRED)

# 安裝 urdf 和 launch 資料夾
install(
  DIRECTORY urdf launch
  DESTINATION share/${PROJECT_NAME}
)

# 必須放在最後
ament_package()