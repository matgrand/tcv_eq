#!/bin/bash

# modifications requried for lac8
# To make the build work on an old Fedora server and macOS, we unified the system 
# around the server's limitations. We downgraded the ONNX Runtime library to v1.10.0 
# in the compile_net_forward.sh script to match the C++11-compatible GCC 5.3.1 compiler. 
# In CMakeLists.txt, we replaced the modern, newer find_package(Matlab) with a 
# robust manual find_library method that works for both platforms. We then 
# modified the net_forward.cpp code to remove the C++17 <filesystem> header, 
# using std::string instead. 
# Finally, the net.onnx model itself had to be re-exported with Opset 15 to be 
# compatible with the older ONNX library, resolving the last runtime error in MATLAB.

clear

# Check OS and architecture
OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

echo "Detected OS: $OS_TYPE"
echo "Detected ARCH: $ARCH_TYPE"
echo "---------------------------------------------------------------------------------"

# Downgraded to a version compatible with older C++11 compilers like GCC 5.x
ONNXRUNTIME_VERSION="1.10.0" # 1.22.0 is the latest version, but it requires C++17

if [[ "$OS_TYPE" == "Linux" && "$ARCH_TYPE" == "x86_64" ]]; then
    ONNXRUNTIME_NAME="onnxruntime-linux-x64-$ONNXRUNTIME_VERSION"
elif [[ "$OS_TYPE" == "Darwin" && "$ARCH_TYPE" == "arm64" ]]; then
    ONNXRUNTIME_NAME="onnxruntime-osx-arm64-$ONNXRUNTIME_VERSION"
else
    echo "Unsupported OS/Architecture: $OS_TYPE / $ARCH_TYPE"
    exit 1
fi

ONNXRUNTIME_DIR="$(pwd)/$ONNXRUNTIME_NAME"
echo "ONNX version: $ONNXRUNTIME_VERSION"
echo "ONNX full version name: $ONNXRUNTIME_NAME"
echo "ONNX full path: $ONNXRUNTIME_DIR"
echo "---------------------------------------------------------------------------------"

if [[ "$OS_TYPE" == "Linux" ]]; then
    # MATLABROOT="/usr/local/MATLAB/R2019a" # lac8 2019a
    MATLABROOT="/usr/local/MATLAB/R2021a" # lac8 2021a
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    MATLABROOT="/Applications/MATLAB_R2024b.app" # macOS
else
    echo "Unsupported OS for MATLABROOT: $OS_TYPE"
    exit 1
fi
echo "MATLAB version: $MATLABROOT"
echo "---------------------------------------------------------------------------------"

# Output directory
export ONNX_NET_FORWARD_DIR="$(pwd)/onnx_net_forward"
echo "onnx_net_forward directory: $ONNX_NET_FORWARD_DIR"
rm -rf "$ONNX_NET_FORWARD_DIR" && mkdir "$ONNX_NET_FORWARD_DIR"

# Download ONNX Runtime if not already there
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "Downloading ONNX Runtime..."
    rm -rf onnxruntime-linux-x64-* # Clean up other versions
    wget "https://github.com/microsoft/onnxruntime/releases/download/v$ONNXRUNTIME_VERSION/$ONNXRUNTIME_NAME.tgz"
    tar -xzf "$ONNXRUNTIME_NAME.tgz"
    rm "$ONNXRUNTIME_NAME.tgz"
    echo "ONNX Runtime downloaded and extracted."
    echo "---------------------------------------------------------------------------------"
fi

# Compile the C++ code
echo "Compiling..."
rm -rf build && mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" \
    -DONNXRUNTIME_INCLUDE_DIRS="$ONNXRUNTIME_DIR/include" \
    -DONNX_NET_FORWARD_DIR="$ONNX_NET_FORWARD_DIR" \
    -DMatlab_ROOT="$MATLABROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
cd ..
rm -rf build
echo "Compilation completed."

# Copy ONNX net file
# ONNX_NET_PATH="$(pwd)/data/best_old/net.onnx" # <- adjust if needed
# ONNX_NET_PATH="$(pwd)/data/local/net.onnx" # <- adjust if needed
ONNX_NET_PATH="$(pwd)/data/best/net.onnx" # <- adjust if needed
if [ ! -f "$ONNX_NET_PATH" ]; then
    echo "Error: net.onnx file not found at $ONNX_NET_PATH"
    exit 1
fi
cp "$ONNX_NET_PATH" "$ONNX_NET_FORWARD_DIR/net.onnx"
echo "$ONNX_NET_PATH copied to $ONNX_NET_FORWARD_DIR/net.onnx"

# Run MATLAB test
echo "----- Matlab --------------------------------------------------------------------"
"$MATLABROOT/bin/matlab" -nosplash -nodesktop -r "run('forward_test.m'); exit;"
echo "---------------------------------------------------------------------------------"
echo "MATLAB version test completed."
