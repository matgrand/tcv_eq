#!/bin/bash

clear

# check https://github.com/microsoft/onnxruntime/releases
onnxrt_version="1.22.0"
onnxrt_dir_name="onnxruntime-linux-x64-$onnxrt_version"
onnxrt_fullpath="$(pwd)/$onnxrt_dir_name"
echo "onnx version: $onnxrt_version"
echo "onnx dir name: $onnxrt_dir_name"
echo "onnx full path: $onnxrt_fullpath"
echo "---------------------------------------------------------------------------------"

export ONNX_NET_FORWARD_DIR="$(pwd)/onnx_net_forward"

# download onnxruntime 
if [ ! -d "$(pwd)/onnxruntime-linux-x64-1.22.0" ]; then
    wget "https://github.com/microsoft/onnxruntime/releases/download/v$onnxrt_version/$onnxrt_dir_name.tgz"
    tar -xzf "$onnxrt_dir_name.tgz"
    rm "$onnxrt_dir_name.tgz"
    echo "onnxruntime downloaded and extracted."
    echo "---------------------------------------------------------------------------------"
fi

# compile the C++ code
echo "Compiling..."
rm -rf build  # remove the build directory if it exists
mkdir build
cd build
cmake .. 

make
cd ..
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


