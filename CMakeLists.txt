cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(net_forward)

# Find required packages
find_package(Matlab REQUIRED MX_LIBRARY)

set(ONNXRUNTIME_PATH "${CMAKE_CURRENT_LIST_DIR}/onnxruntime-linux-x64-1.22.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_PATH}/include")
message(STATUS "ONNXRUNTIME_PATH: ${ONNXRUNTIME_PATH}")

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_PATH}/lib)

set(CMAKE_CXX_STANDARD 17)

# Create a standalone version and a MEX version
option(BUILD_STANDALONE "Build standalone executable" ON)
option(BUILD_MEX "Build MEX file" ON)

# Add MEX file target for MATLAB
matlab_add_mex(
    NAME net_forward_mex
    SRC net_forward.cpp
    LINK_TO ${ONNXRUNTIME_LIB} ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY}
)

# Set include directories for the MEX file
target_include_directories(net_forward_mex PRIVATE 
    ${Matlab_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Set RPATH for the MEX file to find ONNX Runtime libraries
if(UNIX)
    SET_TARGET_PROPERTIES(net_forward_mex PROPERTIES
        INSTALL_RPATH "$ORIGIN/../onnxruntime-linux-x64-1.22.0/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()