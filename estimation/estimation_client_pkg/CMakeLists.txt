cmake_minimum_required(VERSION 3.0.2)
project(estimation_client_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  common_msgs
  common_srvs
  cv_bridge
  pcl_ros
  util
  gazebo_model_package
  tf_package
  data_transform_pkg
)


catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES estimation_client_pkg
 CATKIN_DEPENDS gazebo_model_package
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(estimation_client src/estimation_client.cpp)
target_link_libraries(estimation_client
  ${catkin_LIBRARIES}
)

add_executable(acc_estimation_client src/acc_estimation_client.cpp)
target_link_libraries(acc_estimation_client
  ${catkin_LIBRARIES}
)

add_executable(estimation_client_use_hdf5 src/estimation_client_use_hdf5.cpp)
target_link_libraries(estimation_client_use_hdf5
  ${catkin_LIBRARIES}
)