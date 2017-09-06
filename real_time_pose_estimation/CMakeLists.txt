project(real_time_pose_estimation)

cmake_minimum_required(VERSION 2.8)

set(target cpp-tutorial-)

#include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )
#include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
include( $ENV{OpenCV33_DIR}/OpenCVConfig.cmake )
add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")


set(sample_dir ${CMAKE_SOURCE_DIR}/src/)

set(sample_pnplib
        ${sample_dir}CsvReader.cpp
        ${sample_dir}CsvWriter.cpp
        ${sample_dir}ModelRegistration.cpp
        ${sample_dir}Mesh.cpp
        ${sample_dir}Model.cpp
        ${sample_dir}PnPProblem.cpp
        ${sample_dir}Utils.cpp
        ${sample_dir}RobustMatcher.cpp
)


add_executable( ${target}pnp_registration ${sample_dir}main_registration.cpp ${sample_pnplib} )
add_executable( ${target}pnp_detection ${sample_dir}main_detection.cpp ${sample_pnplib} )

target_link_libraries( ${target}pnp_registration ${OpenCV_LIBS})
target_link_libraries( ${target}pnp_detection ${OpenCV_LIBS})