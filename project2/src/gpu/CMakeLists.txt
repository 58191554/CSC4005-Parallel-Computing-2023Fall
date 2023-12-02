# Required packages and flags
## CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4")
enable_language(CUDA)  # 启用CUDA支持
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()


SET(CMAKE_CXX_COMPILER pgc++)

## CUDA
cuda_add_executable(cudaMul
        cudaMul.cu
        ../matrix.cpp ../matrix.hpp)
target_link_libraries(cudaMul cudart)

## OpenACC
find_package(OpenACC REQUIRED)
if (OpenACC_CXX_FOUND)
    message("OpenACC FOUND, Version: " ${OpenACC_CXX_VERSION})
endif ()

## OpenACC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc")
add_executable(openaccMul
        openaccMul.cpp
        ../matrix.cpp ../matrix.hpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc")