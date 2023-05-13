cmake_minimum_required(VERSION 3.5.1)
project(lo_sam)

find_package(Boost REQUIRED COMPONENTS thread)

set(CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(GTSAM_USE_SYSTEM_EIGEN ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  nav_msgs
  sensor_msgs
  message_generation
  message_runtime
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)





add_executable(feature_extract src/feature_extract.cpp)
target_link_libraries(feature_extract ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(laserOdometry src/laserOdometry.cpp)
target_link_libraries(laserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(laserMapping src/laserMapping.cpp)
target_link_libraries(laserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})



add_executable(map_optimization src/map_optimization)
target_compile_options(map_optimization   PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(map_optimization  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}  ${OpenMP_CXX_FLAGS}  gtsam)


