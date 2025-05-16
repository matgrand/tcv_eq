#!/bin/bash

clear

# check https://github.com/microsoft/onnxruntime/releases
ONNXRUNTIME_VERSION="1.22.0" # change accordingly
ONNXRUNTIME_NAME="onnxruntime-linux-x64-$ONNXRUNTIME_VERSION" # change accordingly
ONNXRUNTIME_DIR="$(pwd)/$ONNXRUNTIME_NAME"
echo "onnx version: $ONNXRUNTIME_VERSION"
echo "onnx full version name: $ONNXRUNTIME_NAME"
echo "onnx full path: $ONNXRUNTIME_DIR"
echo "---------------------------------------------------------------------------------"

export ONNX_NET_FORWARD_DIR="$(pwd)/onnx_net_forward" # output directory 

# download onnxruntime 
if [ ! -d "$(pwd)/onnxruntime-linux-x64-1.22.0" ]; then
    wget "https://github.com/microsoft/onnxruntime/releases/download/v$ONNXRUNTIME_VERSION/$ONNXRUNTIME_NAME.tgz"
    tar -xzf "$ONNXRUNTIME_NAME.tgz"
    rm "$ONNXRUNTIME_NAME.tgz"
    echo "onnxruntime downloaded and extracted."
    echo "---------------------------------------------------------------------------------"
fi

# compile the C++ code
echo "Compiling..."
rm -rf build && mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" \
    -DONNXRUNTIME_INCLUDE_DIRS="$ONNXRUNTIME_DIR/include" \
    -DONNX_NET_FORWARD_DIR="$ONNX_NET_FORWARD_DIR" 
make
cd ..
rm -rf build
echo "Compilation completed."

# create the .net file with python
echo "Creating the .net file with python..."
echo "----- Python --------------------------------------------------------------------"
python create_net.py
echo "---------------------------------------------------------------------------------"

# # test MATLAB version
echo "----- Matlab --------------------------------------------------------------------"
matlab -nodisplay -nosplash -nodesktop -r "run('forward_test.m'); exit;"
echo "---------------------------------------------------------------------------------"
echo "MATLAB version test completed."


