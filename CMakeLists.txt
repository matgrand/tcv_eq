cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(net_forward)

# Find required packages
find_package(Torch REQUIRED)
find_package(Matlab REQUIRED MX_LIBRARY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# Define separate source files for standalone app and MEX
set(SHARED_SOURCE net_forward_core.cpp)
set(STANDALONE_SOURCE net_forward.cpp)
set(MEX_SOURCE net_forward_mex.cpp)

# Add regular executable
add_executable(net_forward ${STANDALONE_SOURCE} ${SHARED_SOURCE})
target_link_libraries(net_forward "${TORCH_LIBRARIES}")
set_property(TARGET net_forward PROPERTY CXX_STANDARD 17)

# Add MEX file target for MATLAB
matlab_add_mex(
    NAME net_forward_mex
    SRC ${MEX_SOURCE} ${SHARED_SOURCE}
    LINK_TO ${TORCH_LIBRARIES} ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY}
)

# Set include directories for the MEX file
target_include_directories(net_forward_mex PRIVATE 
    ${Matlab_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Define MATLAB_MEX_FILE preprocessor directive for the MEX source
target_compile_definitions(net_forward_mex PRIVATE MATLAB_MEX_FILE)

# Set RPATH for the MEX file to find libtorch libraries
if(UNIX)
    set_target_properties(net_forward_mex PROPERTIES
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# The following code block is for Windows to copy DLLs
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET net_forward
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:net_forward>)
  
  # Also copy DLLs for the MEX file
  add_custom_command(TARGET net_forward_mex
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:net_forward_mex>)
endif (MSVC)