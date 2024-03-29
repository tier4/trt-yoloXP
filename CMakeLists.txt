cmake_minimum_required(VERSION 3.1) 
project(trt-yoloxp VERSION 1.0)            
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

add_compile_options(-D YOLOX_STANDALONE)

include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
find_package(OpenCV REQUIRED)

#detector lib (cuda)
file(GLOB_RECURSE sources src/tensorrt_common/tensorrt_common.cpp src/tensorrt_common/simple_profiler.cpp src/tensorrt_yolox/tensorrt_yolox.cpp src/tensorrt_yolox/preprocess.cu)
add_library(detector SHARED ${sources})

target_link_libraries(detector  nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart  "stdc++fs")
target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include )


add_executable(trt-yoloxp samples/sample_detector.cpp samples/yolo_config_parser.cpp ${sources})

target_include_directories(trt-yoloxp PRIVATE include)
target_link_libraries(trt-yoloxp  ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart gflags boost_system boost_filesystem "stdc++fs" "stdc++")




