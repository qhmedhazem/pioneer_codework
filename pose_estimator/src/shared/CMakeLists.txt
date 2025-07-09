cmake_minimum_required(VERSION 2.8.3)
project(shared)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
)

find_package(OpenCV)
find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include  
)


add_library(shared
    src/messages.cpp
    src/parameters.cpp
    src/utility/utility.cpp
)
target_link_libraries(shared ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES shared
)