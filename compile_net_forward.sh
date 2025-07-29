#!/bin/bash

# when running on macOS, run this script like this:
# export PATH="/Applications/MATLAB_R2024b.app/bin:$PATH" && bash compile_net_forward.sh

clear

# Check OS and architecture
OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

echo "Detected OS: $OS_TYPE"
echo "Detected ARCH: $ARCH_TYPE"
echo "---------------------------------------------------------------------------------"

# Choose the correct ONNX Runtime version and binary
ONNXRUNTIME_VERSION="1.22.0"
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

MATLABROOT="/usr/local/MATLAB/R2019a"
echo "MATLAB version: $MATLABROOT"
echo "---------------------------------------------------------------------------------"

# Output directory
export ONNX_NET_FORWARD_DIR="$(pwd)/onnx_net_forward"
echo "onnx_net_forward directory: $ONNX_NET_FORWARD_DIR"
rm -rf "$ONNX_NET_FORWARD_DIR" && mkdir "$ONNX_NET_FORWARD_DIR"

# Download ONNX Runtime if not already there
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "Downloading ONNX Runtime..."
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
    -DONNX_NET_FORWARD_DIR="$ONNX_NET_FORWARD_DIR"\
    -DMATLAB_ROOT="$MATLABROOT" \
    -DMATLAB_MEX_LIBRARY="$MATLABROOT/bin/glnxa64/libmex.so" \
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
matlab -nosplash -nodesktop -r "run('forward_test.m'); exit;"
echo "---------------------------------------------------------------------------------"
echo "MATLAB version test completed."
