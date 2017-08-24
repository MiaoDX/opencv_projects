cmake_minimum_required(VERSION 2.8)

project(surf_orb_cuda_opencv_3_3_0)

set(EXECUTABLE_OUTPUT_PATH ../bin)
set(LIBRARY_OUTPUT_PATH ../lib)

#find_package(OpenCV REQUIRED)
include( $ENV{OpenCV33_DIR}/OpenCVConfig.cmake )
add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)


include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

#cuda_add_executable(surf_keypoint_matcher surf_keypoint_matcher.cpp) # not necessary to add `cuda`
add_executable(surf_keypoint_matcher surf_keypoint_matcher.cpp)
target_link_libraries(surf_keypoint_matcher ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(surf_orb_cuda_cv330 surf_orb_cuda_cv330.cpp)
target_link_libraries(surf_orb_cuda_cv330 ${CUDA_LIBRARIES} ${OpenCV_LIBS})