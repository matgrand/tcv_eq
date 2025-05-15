cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(net_forward)

# Find required packages
find_package(Torch REQUIRED)
find_package(Matlab REQUIRED MX_LIBRARY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# Create a standalone version and a MEX version
option(BUILD_STANDALONE "Build standalone executable" ON)
option(BUILD_MEX "Build MEX file" ON)

# Add standalone executable if requested
if(BUILD_STANDALONE)
    add_executable(net_forward net_forward.cpp)
    target_compile_definitions(net_forward PRIVATE STANDALONE_MODE)
    target_include_directories(net_forward PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(net_forward "${TORCH_LIBRARIES}")
    set_property(TARGET net_forward PROPERTY CXX_STANDARD 17)
endif()

# Add MEX file target for MATLAB
if(BUILD_MEX)
    matlab_add_mex(
        NAME net_forward_mex
        SRC net_forward.cpp
        LINK_TO ${TORCH_LIBRARIES} ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY}
    )

    # Set include directories for the MEX file
    target_include_directories(net_forward_mex PRIVATE 
        ${Matlab_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )

    # Set RPATH for the MEX file to find libtorch libraries
    if(UNIX)
        SET_TARGET_PROPERTIES(net_forward_mex PROPERTIES
            INSTALL_RPATH "$ORIGIN/../libtorch/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()