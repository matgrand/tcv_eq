cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(net_forward)

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to the ONNX Runtime installation")
set(ONNXRUNTIME_INCLUDE_DIRS "" CACHE PATH "Path to the ONNX Runtime include directory")
set(ONNX_NET_FORWARD_DIR "" CACHE PATH "Output directory for net_forward mex executable")
set(Matlab_ROOT "" CACHE PATH "Path to MATLAB installation")

# Make variables required
if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR is required but not set.")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIRS)
    message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIRS is required but not set.")
endif()
if(NOT ONNX_NET_FORWARD_DIR)
    message(FATAL_ERROR "ONNX_NET_FORWARD_DIR is required but not set.")
endif()
if(NOT Matlab_ROOT)
    message(FATAL_ERROR "Matlab_ROOT is required but not set.")
endif()

message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIRS: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX_NET_FORWARD_DIR: ${ONNX_NET_FORWARD_DIR}")
message(STATUS "MATLAB_ROOT: ${Matlab_ROOT}")

# --- START OF CHANGES ---

# Manually find MATLAB libraries and includes for older systems
set(Matlab_INCLUDE_DIRS "${Matlab_ROOT}/extern/include")
find_library(Matlab_MEX_LIBRARY mex HINTS "${Matlab_ROOT}/bin/glnxa64")
find_library(Matlab_MX_LIBRARY mx HINTS "${Matlab_ROOT}/bin/glnxa64")

# Check if libraries were found
if(NOT Matlab_MEX_LIBRARY OR NOT Matlab_MX_LIBRARY)
    message(FATAL_ERROR "Could not find MATLAB mex or mx libraries in ${Matlab_ROOT}/bin/glnxa64")
endif()

# Use C++11 for compatibility with older compilers like GCC 5.x
set(CMAKE_CXX_STANDARD 11)

# --- END OF CHANGES ---

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)

# Check dependencies
message(STATUS "Matlab libraries: ${Matlab_MX_LIBRARY}, ${Matlab_MEX_LIBRARY}, ${Matlab_INCLUDE_DIRS}")

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

# RPATH settings for library resolution
if(APPLE)
    # macOS: Add RPATH to locate libMatlabEngine.dylib and ONNX Runtime
    set(MATLAB_ENGINE_LIB_DIR "${Matlab_ROOT}/extern/lib/maca64/macosarm64")
    message(STATUS "macOS detected. Adding RPATH to: ${MATLAB_ENGINE_LIB_DIR}")
    target_link_options(net_forward_mex PRIVATE
        "-Wl,-rpath,@loader_path/lib"
        "-Wl,-rpath,${MATLAB_ENGINE_LIB_DIR}"
    )
elseif(UNIX)
    # Linux: Use $ORIGIN for relative library loading
    set_target_properties(net_forward_mex PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# copy the MEX file to the specified output directory
add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "net_forward_mex.*" ${ONNX_NET_FORWARD_DIR}
    COMMENT "Copying net_forward_mex to ${ONNX_NET_FORWARD_DIR}"
)

# copy lib directory to the specified output directory
add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ONNXRUNTIME_DIR}/lib" "${ONNX_NET_FORWARD_DIR}/lib"
    COMMENT "Copying ONNX Runtime libraries to ${ONNX_NET_FORWARD_DIR}/lib"
)