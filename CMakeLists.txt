cmake_minimum_required(VERSION 3.11)
project(geosim)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(gpu main.cu photon_map.cu)
target_compile_features(gpu PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "80")

