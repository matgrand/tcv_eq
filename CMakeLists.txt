cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(net_forward)

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to the ONNX Runtime installation")
set(ONNXRUNTIME_INCLUDE_DIRS "" CACHE PATH "Path to the ONNX Runtime include directory")
set(ONNX_NET_FORWARD_DIR "" CACHE PATH "Output directory for net_forward mex executable")
set(Matlab_ROOT "" CACHE PATH "Path to MATLAB installation")

# --- (Variable checks and manual library finding are unchanged) ---
if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR is not set.")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIRS)
    message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIRS is not set.")
endif()
if(NOT ONNX_NET_FORWARD_DIR)
    message(FATAL_ERROR "ONNX_NET_FORWARD_DIR is not set.")
endif()
if(NOT Matlab_ROOT)
    message(FATAL_ERROR "Matlab_ROOT is not set.")
endif()

message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIRS: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX_NET_FORWARD_DIR: ${ONNX_NET_FORWARD_DIR}")
message(STATUS "MATLAB_ROOT: ${Matlab_ROOT}")

set(Matlab_INCLUDE_DIRS "${Matlab_ROOT}/extern/include")
find_library(Matlab_MEX_LIBRARY mex HINTS "${Matlab_ROOT}/bin/glnxa64")
find_library(Matlab_MX_LIBRARY mx HINTS "${Matlab_ROOT}/bin/glnxa64")

if(NOT Matlab_MEX_LIBRARY OR NOT Matlab_MX_LIBRARY)
    message(FATAL_ERROR "Could not find MATLAB mex or mx libraries in ${Matlab_ROOT}/bin/glnxa64")
endif()

set(CMAKE_CXX_STANDARD 11)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)
message(STATUS "Matlab libraries: ${Matlab_MX_LIBRARY}, ${Matlab_MEX_LIBRARY}, ${Matlab_INCLUDE_DIRS}")

# --- (target_link_libraries and target_include_directories are unchanged) ---
add_library(net_forward_mex SHARED net_forward.cpp)

# --- START OF CHANGES ---

# Set properties for the MEX file: remove 'lib' prefix and set correct extension
if(UNIX AND NOT APPLE)
    set_target_properties(net_forward_mex PROPERTIES
        PREFIX ""
        SUFFIX .mexa64
    )
endif()

target_link_libraries(net_forward_mex PRIVATE
    ${ONNXRUNTIME_LIB}
    ${Matlab_MX_LIBRARY}
    ${Matlab_MEX_LIBRARY}
)
target_include_directories(net_forward_mex PRIVATE
    ${Matlab_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
    set_target_properties(net_forward_mex PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Replace the failing copy command with a robust one
add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:net_forward_mex>"
        "${ONNX_NET_FORWARD_DIR}/"
    COMMENT "Copying $<TARGET_FILE_NAME:net_forward_mex> to ${ONNX_NET_FORWARD_DIR}"
)

# --- END OF CHANGES ---

# copy lib directory to the specified output directory
add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ONNXRUNTIME_DIR}/lib" "${ONNX_NET_FORWARD_DIR}/lib"
    COMMENT "Copying ONNX Runtime libraries to ${ONNX_NET_FORWARD_DIR}/lib"
)