# cmake_minimum_required(VERSION 3.8)
# project(lo_dd)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# # find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(rclpy REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(PCL REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(nav_msgs REQUIRED)
# find_package(tf2 REQUIRED)
# find_package(tf2_ros REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(rosidl_default_generators REQUIRED)



# include_directories(
#   include/lo_dd
#   include/sophus
# )

# install(
#   DIRECTORY include/
#   DESTINATION include
# )
# ament_export_include_directories(include)

# set(msg_files
#   "msg/Pose6D.msg"
# )
# rosidl_generate_interfaces(${PROJECT_NAME}
#   ${msg_files}
# )

# ament_export_dependencies(rosidl_default_runtime)

# add_executable(${PROJECT_NAME}_odometry src/odometry.cpp src/imuProcess.hpp)
# ament_target_dependencies(${PROJECT_NAME}_odometry rclcpp rclpy sensor_msgs std_msgs PCL pcl_conversions geometry_msgs nav_msgs tf2 tf2_ros visualization_msgs )
# rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_odometry "${cpp_typesupport_target}") 




# add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
# ament_target_dependencies(${PROJECT_NAME}_featureExtraction rclcpp rclpy sensor_msgs std_msgs PCL pcl_conversions geometry_msgs nav_msgs tf2 tf2_ros visualization_msgs)
# # rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_featureExtraction "${cpp_typesupport_target}") 


# install(
#   DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}/
# )

# install(
#   DIRECTORY config
#   DESTINATION share/${PROJECT_NAME}/
# )

# install(
#   TARGETS ${PROJECT_NAME}_odometry
#   DESTINATION lib/${PROJECT_NAME}
# )

# install(
#   TARGETS ${PROJECT_NAME}_featureExtraction
#   DESTINATION lib/${PROJECT_NAME}
# )



# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   # set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   # set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()

cmake_minimum_required(VERSION 3.8)
project(lo_dd)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 查找依赖项
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 包含目录
include_directories(
  include/lo_dd
  include/sophus
  include/ikd-Tree
)

# 定义消息文件
set(msg_files
  "msg/Pose6D.msg"
)

# 生成消息接口
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)

# 导出运行时依赖
ament_export_dependencies(rosidl_default_runtime)

# 类型支持库
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 定义可执行文件: odometry
add_executable(${PROJECT_NAME}_odometry src/odometry.cpp src/imuProcess.hpp)
ament_target_dependencies(${PROJECT_NAME}_odometry rclcpp rclpy sensor_msgs std_msgs PCL pcl_conversions geometry_msgs nav_msgs tf2 tf2_ros visualization_msgs)
target_link_libraries(${PROJECT_NAME}_odometry "${cpp_typesupport_target}")


# 定义可执行文件: featureExtraction
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
ament_target_dependencies(${PROJECT_NAME}_featureExtraction rclcpp rclpy sensor_msgs std_msgs PCL pcl_conversions geometry_msgs nav_msgs tf2 tf2_ros visualization_msgs)
target_link_libraries(${PROJECT_NAME}_featureExtraction "${cpp_typesupport_target}")

# 安装启动文件和配置文件
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# 安装可执行文件
install(
  TARGETS ${PROJECT_NAME}_odometry ${PROJECT_NAME}_featureExtraction
  DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
# install(
#   DIRECTORY "include/"
#   DESTINATION include
# )
# ament_export_include_directories(include)


# 测试部分
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# 声明包
ament_package()
