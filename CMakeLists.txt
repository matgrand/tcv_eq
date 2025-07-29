# Use a version compatible with the old server
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(net_forward)

# --- Variable Definitions (Unchanged) ---
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to the ONNX Runtime installation")
set(ONNXRUNTIME_INCLUDE_DIRS "" CACHE PATH "Path to the ONNX Runtime include directory")
set(ONNX_NET_FORWARD_DIR "" CACHE PATH "Output directory for net_forward mex executable")
set(Matlab_ROOT "" CACHE PATH "Path to MATLAB installation")

# --- Variable Checks (Corrected Syntax) ---
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

# Use C++11 for cross-platform compatibility
set(CMAKE_CXX_STANDARD 11)

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)

# --- Unified Manual MATLAB Finding (Corrected Syntax) ---
message(STATUS "Using manual method to find MATLAB libraries for all platforms.")

set(Matlab_INCLUDE_DIRS "${Matlab_ROOT}/extern/include")

if(APPLE)
    # macOS library paths
    find_library(Matlab_MEX_LIBRARY mex HINTS "${Matlab_ROOT}/bin/maca64")
    find_library(Matlab_MX_LIBRARY mx HINTS "${Matlab_ROOT}/bin/maca64")
else()
    # Linux library paths
    find_library(Matlab_MEX_LIBRARY mex HINTS "${Matlab_ROOT}/bin/glnxa64")
    find_library(Matlab_MX_LIBRARY mx HINTS "${Matlab_ROOT}/bin/glnxa64")
endif()

# Verify that libraries were found
if(NOT Matlab_MEX_LIBRARY OR NOT Matlab_MX_LIBRARY)
    message(FATAL_ERROR "Could NOT find Matlab mex or mx libraries. Check MATLABROOT path.")
endif()
message(STATUS "Found Matlab libraries: ${Matlab_MEX_LIBRARY};${Matlab_MX_LIBRARY}")

# --- Unified Target Creation ---
add_library(net_forward_mex SHARED net_forward.cpp)

# Set correct MEX file extension and properties for each platform
if(APPLE)
    # On macOS, prevent the 'lib' prefix and set the correct extension
    set_target_properties(net_forward_mex PROPERTIES
        PREFIX ""
        SUFFIX .mexmaca64
    )
else()
    # On Linux, also prevent 'lib' prefix and set correct extension
    set_target_properties(net_forward_mex PROPERTIES
        PREFIX ""
        SUFFIX .mexa64
    )
endif()

# Link all libraries to the MEX target
target_link_libraries(net_forward_mex PRIVATE
    ${ONNXRUNTIME_LIB}
    ${Matlab_MX_LIBRARY}
    ${Matlab_MEX_LIBRARY}
)

# --- Common Settings (Unchanged) ---
target_include_directories(net_forward_mex PRIVATE
    ${Matlab_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

if(APPLE)
    target_link_options(net_forward_mex PRIVATE
        "-Wl,-rpath,@loader_path/lib"
        "-Wl,-rpath,${Matlab_ROOT}/bin/maca64"
    )
elseif(UNIX)
    set_target_properties(net_forward_mex PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:net_forward_mex>"
        "${ONNX_NET_FORWARD_DIR}/"
    COMMENT "Copying $<TARGET_FILE_NAME:net_forward_mex> to ${ONNX_NET_FORWARD_DIR}"
)

add_custom_command(TARGET net_forward_mex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ONNXRUNTIME_DIR}/lib" "${ONNX_NET_FORWARD_DIR}/lib"
    COMMENT "Copying ONNX Runtime libraries to ${ONNX_NET_FORWARD_DIR}/lib"
)