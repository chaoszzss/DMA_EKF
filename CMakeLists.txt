cmake_minimum_required(VERSION 2.8.3)
project(DMA_EKF)

set(CMAKE_CXX_FLAGS "-std=c++17 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
add_definitions(-DDATASET_PATH="${PROJECT_SOURCE_DIR}/data/")

find_package(Boost REQUIRED COMPONENTS date_time iostreams)
include_directories(${Boost_INCLUDE_DIRS})
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf cv_bridge)

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

include(cmake/FindEigen.cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "Using OpenCV version: ${OpenCV_VERSION}")
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(lib_map SHARED
    src/SystemPrimitives.cpp
    src/LocalizationEngine.cpp
    src/RosVisualizationBridge.cpp
)
target_link_libraries(lib_map ${OpenCV_LIBS})

add_executable(DMA_EKF
        src/RosNodeMain.cpp
)

target_link_libraries(DMA_EKF ${catkin_LIBRARIES} ${PCL_LIBRARIES} lib_map ${Boost_LIBRARIES})
