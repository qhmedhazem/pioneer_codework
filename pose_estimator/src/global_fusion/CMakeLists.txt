cmake_minimum_required(VERSION 2.8.3)
project(global_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  shared
)

find_package(Ceres REQUIRED)

add_subdirectory(./ThirdParty/GeographicLib/)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ./ThirdParty/GeographicLib/include/
)

add_library(global_fusion
	src/globalopt_node.cpp
	src/globalopt.cpp)

target_link_libraries(global_fusion ${catkin_LIBRARIES} ${CERES_LIBRARIES} libGeographiccc) 

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES global_fusion 
)

# add_executable(global_fusion_node src/globalopt_node.cpp)
# target_link_libraries(global_fusion_node global_fusion) 