#!/bin/bash

echo "Compiling net_forward_mex.cpp..."
libtorch_path="$(pwd)/libtorch"
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$libtorch_path"
make

cd ..

# test standalone version
cp build/net_forward net_forward
echo "Testing standalone version..."
echo "---------------------------------------------------------"
./net_forward
echo "---------------------------------------------------------"
echo "Standalone version test completed."

echo "Copying mex file to main directory for MATLAB..."
# Copy the compiled mex files to the current directory
for file in build/net_forward_mex.mex*; do
    cp "$file" "net_forward.${file##*.}"
done

echo "Copy completed. You can now use the mex file in MATLAB."
