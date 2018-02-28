cmake_minimum_required(VERSION 2.8)

set(appname kfusion)

find_package(TooN   REQUIRED)
find_package(OpenCL )
find_package(CUDA   )
find_package(GLUT   )
find_package(OpenGL )
find_package(OpenNI2)
find_package(PAPI)

find_package(PkgConfig)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5PrintSupport)

include_directories(./include ${TOON_INCLUDE_PATH} ./thirdparty)

if (CUDA_FOUND) 
    SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
endif(CUDA_FOUND)

 # ---- PREPARE COMMON DEPENDENCIES  ------------ 
if(APPLE)
    set(common_libraries     stdc++ pthread)
else(APPLE)
    set(common_libraries     rt stdc++ pthread)
endif(APPLE)

set(common_compile_flags "")

# ----------------- CPP VERSION ----------------- 
add_library(${appname}-cpp  src/cpp/kernels.cpp)
target_link_libraries(${appname}-cpp   ${common_libraries})	
add_version(${appname} cpp "" "")

# ----------------- STL VERSION -----------------
include_directories(/home/dom/range-v3/include)
set(COMPUTECPP_PACKAGE_ROOT_DIR /home/dom/computecpp-ce-0.5.1)
set(COMPUTECPP_INCLUDE_DIRECTORY /home/dom/computecpp-ce-0.5.1/include)
set(SYCL_PARALLEL_STL_INCLUDE_DIRECTORY /home/dom/sycl_parallel_stl/include)

set(CMAKE_MODULE_PATH /home/dom/computecpp-sdk/cmake/Modules/)
include(FindComputeCpp)

#include_directories("/opt/intel/pstl/include")

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})
include_directories(${SYCL_PARALLEL_STL_INCLUDE_DIRECTORY})

add_definitions(-DSYCL_PSTL_USE_OLD_ALGO)
set(COMPUTECPP_DEVICE_COMPILER_FLAGS "-std=c++14 ${COMPUTECPP_DEVICE_COMPILER_FLAGS} -DSYCL_PSTL_USE_OLD_ALGO")

# ${CMAKE_CURRENT_SOURCE_DIR}/

set(STL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/kernels.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/preprocess.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/track.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/integrate.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/raycast.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/raycasting.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/stl/render.cpp)

add_library(${appname}-stl ${STL_FILES})
target_link_libraries(${appname}-stl PRIVATE ${common_libraries})

add_sycl_to_target(${appname}-stl ${CMAKE_CURRENT_BINARY_DIR} ${STL_FILES})
add_version(${appname} stl "" "")

# ----------------- OPENMP VERSION ----------------- 
add_library(${appname}-openmp  src/cpp/kernels.cpp)
target_link_libraries(${appname}-openmp   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-openmp PROPERTIES COMPILE_FLAGS "-fopenmp")
add_version(${appname} openmp "-fopenmp" "-fopenmp")

#  ----------------- OCL VERSION -----------------  
if (OPENCL_FOUND) 
   include_directories(${OPENCL_INCLUDE_DIRS})
    add_library(${appname}-opencl  src/opencl/kernels.cpp src/opencl/common_opencl.cpp)
    target_link_libraries(${appname}-opencl   ${common_libraries} ${OPENCL_LIBRARIES})	
    add_version(${appname} opencl "" "")
endif(OPENCL_FOUND)

#  ----------------- CUDA VERSION ----------------- 
if (CUDA_FOUND) 

set(kfusion_cuda_srcs
	src/cuda/kernels.cu
	thirdparty/kfusion.h
	)
   	
    cuda_add_library(${appname}-cuda    STATIC  ${kfusion_cuda_srcs} OPTIONS -O3 )
	target_link_libraries(${appname}-cuda   ${common_libraries})	
    add_version(${appname} cuda    "" "")

endif(CUDA_FOUND)

#  ----------------- thirdparty ----------------- 
add_subdirectory(thirdparty)
